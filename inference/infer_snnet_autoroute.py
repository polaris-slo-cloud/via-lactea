#!/usr/bin/env python3
"""
Auto-route inference for SN-Net with a stitched head.

- Loads stitched checkpoint (must contain trunk.* + head.*).
- Loads route_stats.json produced by build_route_stats.py.
- Infers head_in_dim from ckpt (head.weight.shape[1]).
- Chooses best stitch_id among entries whose feat_dim == head_in_dim
  using --criterion {acc,conf} (val acc or mean_maxprob).
- Runs single-image inference with the chosen stitch_id.

Usage:
  python3 infer_snnet_autoroute.py \
    --ckpt stitched_head_ft.pth.tar \
    --stats route_stats.json \
    --image /path/to/img.png \
    --img-size 224 --cifar10-transforms --no-center-crop --amp
"""

import os, sys, json, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*interpolation.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing.*")

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from timm.models import create_model
from timm.models.snnet import SNNet

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# ---------- helpers ----------
def build_anchors(names, pretrained=False, num_classes=0):
    xs = []
    for n in names:
        m = create_model(n, pretrained=pretrained, num_classes=num_classes)
        m.eval()
        xs.append(m)
    return xs

def need_cnn_to_vit(names):
    L = [n.lower() for n in names]
    is_vit = any(any(k in n for k in ["swin","vit","deit","tnt","xcit","mvit"]) for n in L)
    is_cnn = any("resnet" in n for n in L)
    return is_vit and is_cnn

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

def load_stats(stats_path):
    with open(stats_path, "r") as f:
        data = json.load(f)
    # Accept either:
    #  A) {"entries":[{"sid":..., "feat_dim":..., "acc1":..., "mean_maxprob":...}, ...], ...}
    #  B) [{"sid":..., "feat_dim":..., ...}, ...]
    #  C) {"<sid>":{"feat_dim":...,"acc1":...,...}, ...}
    if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], list):
        return data["entries"], data
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        entries = []
        for k, v in data.items():
            try:
                sid = int(k)
                e = dict(v)
                e["sid"] = sid
                entries.append(e)
            except Exception:
                pass
        return entries, {}
    raise RuntimeError("Unrecognized stats JSON format.")

def anchors_from_meta_or_sd(ckpt_obj):
    sd = ckpt_obj.get("state_dict", ckpt_obj)
    anchors = ckpt_obj.get("anchors", None)
    if anchors:
        return anchors
    # Fallback: infer a few common cases
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.startswith("anchors.") and k.split(".")[1].isdigit()})
    picks = []
    for i in idxs:
        base = f"anchors.{i}."
        rkey = base + "layer1.0.conv1.weight"
        if rkey in sd and getattr(sd[rkey], "ndim", 0) == 4:
            kH, kW = sd[rkey].shape[-2], sd[rkey].shape[-1]
            if kH == 3 and kW == 3: picks.append("resnet18"); continue
            if kH == 1 and kW == 1: picks.append("resnet50"); continue
        pkey = base + "patch_embed.proj.weight"
        has_swin = any(k.startswith(base + "layers.") or k.startswith(base + "stages.") for k in sd)
        if has_swin and pkey in sd and sd[pkey].shape[0] == 96:
            picks.append("swin_tiny_patch4_window7_224"); continue
        raise RuntimeError(f"Unknown anchor at anchors.{i}")
    return picks

def choose_best_sid(entries, head_in_dim, criterion="acc"):
    cand = [e for e in entries if int(e.get("feat_dim", -1)) == int(head_in_dim)]
    if not cand:
        print(f"[warn] No stats entries match head_in_dim={head_in_dim}. Considering all stitch_ids.")
        cand = entries[:]
    key = ("acc1" if criterion == "acc" else "mean_maxprob")
    def score(e):
        v = e.get(key, None)
        try:
            return float(v)
        except Exception:
            return float("-inf")
    best = max(cand, key=score)
    return int(best["sid"]), float(score(best))

class TrunkWithHead(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head
    def forward(self, x):
        return self.head(self.trunk(x))

@torch.no_grad()
def infer_one(model, transform, image_path, device, topk=5, amp=False):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    if device.startswith("cuda") and amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
    else:
        logits = model(x)
    probs = logits.softmax(-1)
    k = min(topk, probs.shape[-1])
    top = torch.topk(probs, k=k, dim=-1)
    return top.values.squeeze(0).cpu(), top.indices.squeeze(0).cpu()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="SNNet auto-route inference using precomputed route stats")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--criterion", choices=["acc","conf"], default="acc",
                    help="acc = highest validation accuracy; conf = highest mean max-prob")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--cifar10-transforms", action="store_true")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if not Path(args.ckpt).exists(): sys.exit(f"[error] ckpt not found: {args.ckpt}")
    if not Path(args.stats).exists(): sys.exit(f"[error] stats not found: {args.stats}")
    if not Path(args.image).exists(): sys.exit(f"[error] image not found: {args.image}")

    # Load ckpt (fixed: use map_location, not map_mode)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    anchors = anchors_from_meta_or_sd(ckpt)

    if "head.weight" not in sd:
        sys.exit("[error] This checkpoint has no stitched head (no head.*). Use a stitched_head_ft.pth.tar.")

    head_W = sd["head.weight"]
    head_out_dim, head_in_dim = int(head_W.shape[0]), int(head_W.shape[1])
    print(f"[ckpt] head_in_dim={head_in_dim}  head_out_dim={head_out_dim}  anchors={anchors}")

    # Load stats and pick best stitch_id for this head_in_dim
    entries, _ = load_stats(args.stats)
    best_sid, best_score = choose_best_sid(entries, head_in_dim, criterion=("acc" if args.criterion=="acc" else "conf"))
    print(f"[autoroute] chosen stitch_id={best_sid} by {args.criterion} (score={best_score:.4f})")

    # Build trunk and load trunk.*
    trunk = SNNet(
        build_anchors(anchors, pretrained=False, num_classes=0),
        cnn_to_vit=need_cnn_to_vit(anchors)
    ).to(device).eval()

    trunk_sd = {k.replace("trunk.","",1): v for k,v in sd.items() if k.startswith("trunk.")}
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] trunk load: missing={len(missing)} unexpected={len(unexpected)}")

    # Reset to chosen stitch_id and sanity-check feature dim
    trunk.reset_stitch_id(best_sid)
    with torch.no_grad():
        probe = torch.zeros(1,3,args.img_size,args.img_size, device=device)
        feat = trunk(probe)
        feat_dim = int(feat.shape[-1])
    if feat_dim != head_in_dim:
        print(f"[warn] chosen stitch_id {best_sid} produced feat_dim={feat_dim} "
              f"but head expects {head_in_dim}. Searching for a matching sid...")
        match = None
        for e in entries:
            if int(e.get("feat_dim", -1)) == head_in_dim:
                match = int(e["sid"]); break
        if match is None:
            sys.exit("[error] No stitch_id in stats matches head_in_dim. Re-train stats or head.")
        best_sid = match
        trunk.reset_stitch_id(best_sid)
        with torch.no_grad():
            feat = trunk(probe)
            feat_dim = int(feat.shape[-1])
        if feat_dim != head_in_dim:
            sys.exit(f"[error] Even fallback sid {best_sid} mismatches head_in_dim={head_in_dim} (got {feat_dim}).")

    # Build head & wrap
    head = nn.Linear(head_in_dim, head_out_dim, bias=("head.bias" in sd)).to(device).eval()
    head.load_state_dict({k.replace("head.",""): v for k,v in sd.items() if k.startswith("head.")}, strict=True)
    model = TrunkWithHead(trunk, head).to(device).eval()

    # Inference
    transform = make_transform(args.img_size, use_cifar10=args.cifar10_transforms, center_crop=not args.no_center_crop)
    probs, idxs = infer_one(model, transform, args.image, device, topk=args.topk, amp=args.amp)

    # Labels
    labels = CIFAR10_LABELS if (args.num_classes == 10 or head_out_dim == 10) else None
    print(f"\n[using] stitch_id={best_sid}")
    for r in range(len(probs)):
        cls_id = int(idxs[r].item())
        name = labels[cls_id] if labels and cls_id < len(labels) else str(cls_id)
        print(f"Top-{r+1}: {name:<10} p={float(probs[r]):.4f}")

if __name__ == "__main__":
    main()
