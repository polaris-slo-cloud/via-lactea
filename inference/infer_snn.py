#!/usr/bin/env python3
"""
SN-Net inference that:
- Auto-detects anchors from checkpoint (unless provided).
- If checkpoint contains a trained stitched head ('head.*'), loads it automatically.
- Otherwise:
    * For stitch_id 0/1, uses the anchor heads (build anchors with the *checkpoint’s* head out_dim).
    * For stitched paths (>=2), falls back to features and warns if no stitched head is present.
"""

import os, sys, argparse, warnings
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

warnings.filterwarnings("ignore", message=".*interpolation.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing.*")

from timm.models import create_model
from timm.models.snnet import SNNet

# ------------------ constants ------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer",
                  "dog","frog","horse","ship","truck"]

# ------------------ helpers ------------------
def build_anchors(a_names, pretrained=True, num_classes=0):
    anchors = []
    for n in a_names:
        m = create_model(n, pretrained=pretrained, num_classes=num_classes)
        m.eval()
        anchors.append(m)
    return anchors

def need_cnn_to_vit(anchor_names):
    def is_vit_like(n):
        nl = n.lower()
        return any(k in nl for k in ["swin","vit","deit","tnt","xcit","mvit"])
    L = [n.lower() for n in anchor_names]
    return any(is_vit_like(n) for n in L) and any("resnet" in n for n in L)

def make_transform(img_size, use_cifar10=False, center_crop=True):
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if use_cifar10 else (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    if center_crop:
        resize_side = max(int(round(img_size / 0.875)), img_size)
        return T.Compose([
            T.Resize(resize_side, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

# ---- anchor detection (handles anchors.* and trunk.anchors.*) ----
def _anchor_indices_from_state(sd, prefix="anchors."):
    idxs = set()
    for k in sd.keys():
        if k.startswith(prefix):
            rest = k[len(prefix):]
            head = rest.split(".", 1)[0]
            if head.isdigit():
                idxs.add(int(head))
    return sorted(idxs)

def _guess_under(sd, prefix):
    picks = []
    idxs = _anchor_indices_from_state(sd, prefix=prefix)
    for i in idxs:
        base = f"{prefix}{i}."
        # --- ResNet probe: basic vs bottleneck via conv1 kernel
        rkey = base + "layer1.0.conv1.weight"
        if rkey in sd and getattr(sd[rkey], "ndim", 0) == 4:
            kH, kW = sd[rkey].shape[-2], sd[rkey].shape[-1]
            if kH == 3 and kW == 3:
                picks.append("resnet18"); continue
            if kH == 1 and kW == 1:
                picks.append("resnet50"); continue
        # --- Swin-T probe
        pkey = base + "patch_embed.proj.weight"
        has_swin = any(k.startswith(base + "layers.") or k.startswith(base + "stages.") for k in sd.keys())
        if has_swin and pkey in sd and sd[pkey].shape[0] == 96:
            picks.append("swin_tiny_patch4_window7_224"); continue
        raise RuntimeError(f"Unknown anchor under {base} (sample: {[k for k in sd if k.startswith(base)][:5]})")
    return picks

def anchors_from_ckpt_guess(sd):
    # try plain 'anchors.*'
    try:
        picks = _guess_under(sd, "anchors.")
        if picks: return picks
    except RuntimeError:
        pass
    # try 'trunk.anchors.*' (fine-tuned checkpoints)
    try:
        picks = _guess_under(sd, "trunk.anchors.")
        if picks: return picks
    except RuntimeError:
        pass
    return []

def strip_dataparallel(sd):
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def split_trunk_head_state(sd):
    """
    Returns (trunk_sd, head_sd).
    If keys are 'trunk.*' / 'head.*', strip 'trunk.' for the trunk dict and keep 'head.*'.
    If keys are raw 'anchors.*', return sd as trunk_sd and empty head_sd.
    """
    trunk_sd, head_sd = {}, {}
    has_trunk_prefix = any(k.startswith("trunk.") for k in sd)
    if has_trunk_prefix:
        for k, v in sd.items():
            if k.startswith("trunk."):
                trunk_sd[k.replace("trunk.", "", 1)] = v
            elif k.startswith("head."):
                head_sd[k] = v
        return trunk_sd, head_sd
    else:
        return sd, {}

# ---------- robust class-count inference ----------
def infer_anchor_head_out_dim(sd):
    """
    Look ONLY at actual classifier heads on anchors:
      anchors.{i}.fc.weight
      anchors.{i}.head.weight
    and their 'trunk.' equivalents.
    Returns most common out_dim or None.
    """
    candidates = []
    for prefix in ("anchors.", "trunk.anchors."):
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not k.startswith(prefix):
                continue
            if k.endswith(".fc.weight") or k.endswith(".head.weight") or k.endswith(".classifier.weight"):
                if v.ndim == 2:
                    candidates.append(int(v.shape[0]))
    if not candidates:
        return None
    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]

def infer_num_classes_from_ckpt(sd, meta_num_classes=None, default=10):
    """
    Generic fallback: use meta if present; else look for *classifier* weights.
    Intentionally ignores 'mlp.fc1', 'mlp.fc2', 'qkv', etc.
    """
    if isinstance(meta_num_classes, int) and meta_num_classes > 0:
        return meta_num_classes

    # Prefer explicit anchor head dims if present
    anchor_nc = infer_anchor_head_out_dim(sd)
    if anchor_nc is not None:
        return anchor_nc

    # Otherwise scan for global head/classifier weights (non-anchor)
    candidates = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.ndim != 2:
            continue
        name = k.lower()
        if name.endswith("head.weight") or name.endswith("classifier.weight") or name.endswith(".fc.weight"):
            # exclude mlp heads inside transformer blocks
            if ".mlp." in name or ".blocks." in name or "qkv" in name:
                continue
            candidates.append(int(v.shape[0]))
    if candidates:
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]

    return default

# ------------------ probe ------------------
def probe_output_dim(model, device, img_size):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        y = model(x)
    d = int(y.shape[-1])
    print(f"[probe] runtime output dim = {d}")
    return d

class TrunkWithHead(nn.Module):
    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head
    def forward(self, x):
        return self.head(self.trunk(x))

# ------------------ core loader ------------------
def load_ckpt_and_maybe_wrap(ckpt_path, device, stitch_id, img_size,
                             cli_anchors=None, cli_num_classes=10):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_full = strip_dataparallel(ckpt.get("state_dict", ckpt))

    # anchors
    if "anchors" in ckpt and ckpt["anchors"]:
        anchors_names = list(ckpt["anchors"])
        print(f"[meta] anchors from ckpt metadata: {anchors_names}")
    else:
        auto = anchors_from_ckpt_guess(sd_full)
        if auto:
            anchors_names = auto
            print(f"[auto] anchors: {anchors_names}")
        else:
            anchors_names = None
    if cli_anchors is not None:
        anchors_names = list(cli_anchors)
        print(f"[info] anchors from CLI: {anchors_names}")
    if not anchors_names:
        raise SystemExit("[error] could not infer anchors; pass --anchors explicitly.")

    # stitched head present?
    has_head = ("head.weight" in sd_full) or ("head.bias" in sd_full)

    # infer classes
    inferred_nc = infer_num_classes_from_ckpt(sd_full, ckpt.get("num_classes"), default=cli_num_classes)

    # build anchors with correct classifier dims
    if has_head:
        anchor_num_classes = 0
        print("[path] stitched head present -> building headless anchors")
    else:
        if stitch_id in (0, 1):
            # Use the checkpoint's anchor head out_dim (e.g., 100), not a random guess
            anchor_num_classes = infer_anchor_head_out_dim(sd_full) or inferred_nc
            print(f"[path] no stitched head; stitch_id={stitch_id} -> using ANCHOR HEADS (num_classes={anchor_num_classes})")
        else:
            anchor_num_classes = 0
            print("[path] no stitched head; stitched path -> features only (no classifier)")

    anchors = build_anchors(anchors_names, pretrained=False, num_classes=anchor_num_classes)
    trunk = SNNet(anchors, cnn_to_vit=need_cnn_to_vit(anchors_names)).to(device).eval()

    # pick stitch config (prefer CLI, else ckpt's stored stitch_id)
    if stitch_id not in trunk.stitch_configs:
        if "stitch_id" in ckpt and ckpt["stitch_id"] in trunk.stitch_configs:
            stitch_id = ckpt["stitch_id"]
            print(f"[info] using stitch_id from ckpt: {stitch_id}")
        else:
            print("Available stitch IDs:")
            for k, v in trunk.stitch_configs.items():
                print(k, v)
            raise SystemExit(f"Invalid stitch_id {stitch_id}")
    trunk.reset_stitch_id(stitch_id)

    # split & load state dict into trunk
    trunk_sd, head_sd = split_trunk_head_state(sd_full)
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    print(f"[load] trunk missing={len(missing)} unexpected={len(unexpected)}")

    # stitched head path
    if has_head:
        W = head_sd.get("head.weight", sd_full.get("head.weight"))
        if W is None:
            print("[warn] head markers found but 'head.weight' missing; falling back to trunk-only.")
        else:
            out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
            head = nn.Linear(in_dim, out_dim, bias=("head.bias" in head_sd or "head.bias" in sd_full)).to(device)

            class Wrapper(nn.Module):
                def __init__(self, trunk, head):
                    super().__init__()
                    self.trunk = trunk
                    self.head = head
                def forward(self, x):
                    return self.head(self.trunk(x))

            model = Wrapper(trunk, head).to(device).eval()

            wrap_sd = {}
            for k, v in sd_full.items():
                if k.startswith("trunk.") or k.startswith("head."):
                    wrap_sd[k] = v
            m_missing, m_unexp = model.load_state_dict(wrap_sd, strict=False)
            print(f"[load] stitched head loaded: out_dim={out_dim}  missing={len(m_missing)} unexpected={len(m_unexp)}")
            return model, out_dim

    # no stitched head
    if anchor_num_classes > 0 and stitch_id in (0, 1):
        # anchor heads output logits of size 'anchor_num_classes'
        return trunk, anchor_num_classes

    # stitched features only; probe
    out_dim = probe_output_dim(trunk, device, img_size)
    if stitch_id >= 2:
        print("[warn] stitched path has no classifier head; "
              "use --stitch_id 0/1 for anchor heads or load a checkpoint with a stitched head.")
    return trunk, out_dim

# ------------------ inference ------------------
@torch.no_grad()
def infer_one(model, transform, image_path, device, topk=5):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = logits.softmax(-1)
    k = min(topk, probs.shape[-1])
    top = torch.topk(probs, k=k, dim=-1)
    return top.values.squeeze(0).cpu(), top.indices.squeeze(0).cpu()

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="SNNet stitched inference")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth/.tar)")
    ap.add_argument("--anchors", nargs="+", default=None, help="Anchor names (override auto)")
    ap.add_argument("--stitch_id", type=int, default=7, help="Stitch config ID")
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--num-classes", type=int, default=10, help="Expected classes (fallback only)")
    ap.add_argument("--img-size", type=int, default=224, help="Transform size")
    ap.add_argument("--cifar10-transforms", action="store_true", help="Use CIFAR-10 mean/std", default=False)
    ap.add_argument("--no-center-crop", action="store_true", help="Disable center crop")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if not Path(args.ckpt).exists():
        sys.exit(f"[error] ckpt not found: {args.ckpt}")
    if not Path(args.image).exists():
        sys.exit(f"[error] image not found: {args.image}")

    model, out_dim = load_ckpt_and_maybe_wrap(
        ckpt_path=args.ckpt,
        device=device,
        stitch_id=args.stitch_id,
        img_size=args.img_size,
        cli_anchors=args.anchors,
        cli_num_classes=args.num_classes,
    )

    labels = CIFAR10_LABELS if out_dim == 10 else None

    transform = make_transform(
        args.img_size,
        use_cifar10=args.cifar10_transforms,
        center_crop=not args.no_center_crop,
    )

    probs, idxs = infer_one(model, transform, args.image, device, topk=args.topk)
    for r in range(len(probs)):
        cls_id = int(idxs[r].item())
        name = labels[cls_id] if labels and cls_id < len(labels) else str(cls_id)
        print(f"Top-{r+1}: {name}  p={float(probs[r].item()):.4f}")

if __name__ == "__main__":
    main()
