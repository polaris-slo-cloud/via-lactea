#!/usr/bin/env python3
"""
Train a classification head on top of a stitched SNNet trunk.

- Loads your trunk checkpoint (anchors + stitch layers; no stitched head).
- Auto-detects anchors from the checkpoint unless provided.
- Builds SNNet, selects --stitch_id, probes stitched feature dim using the
  SAME training transform used for batches.
- Attaches a Linear(feat_dim -> num_classes) head.
- By default: keeps trunk in EVAL FORWARD (stable stitched path) but still
  allows gradients on stitch/bridge when --unfreeze-stitch is set.
- Saves a checkpoint that includes BOTH trunk ("trunk.*") and head ("head.*").
"""

import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", message=".*interpolation.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

# timm
from timm.models import create_model
from timm.models.snnet import SNNet

# -------------------- constants --------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# -------------------- utilities --------------------
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

def make_train_transform(img_size, use_cifar10):
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if use_cifar10 else (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    return T.Compose([
        T.RandomResizedCrop(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

def make_val_transform(img_size, use_cifar10, center_crop=True):
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

def _anchor_indices_from_state(sd):
    idxs = set()
    prefix = "anchors."
    for k in sd.keys():
        if k.startswith(prefix):
            rest = k[len(prefix):]
            head = rest.split(".", 1)[0]
            if head.isdigit():
                idxs.add(int(head))
    return sorted(idxs)

def anchors_from_ckpt_guess(sd):
    """
    Minimal robust inference for resnet18 / resnet50 / swin_tiny.
    Extend if you add more families.
    """
    picks = []
    for i in _anchor_indices_from_state(sd):
        base = f"anchors.{i}."
        # resnet probe (basic vs bottleneck)
        rkey = base + "layer1.0.conv1.weight"
        if rkey in sd and getattr(sd[rkey], "ndim", 0) == 4:
            kH, kW = sd[rkey].shape[-2], sd[rkey].shape[-1]
            if kH == 3 and kW == 3:
                picks.append("resnet18"); continue
            if kH == 1 and kW == 1:
                picks.append("resnet50"); continue
        # swin-tiny probe
        pkey = base + "patch_embed.proj.weight"
        has_swin = any(k.startswith(base + "layers.") or k.startswith(base + "stages.") for k in sd.keys())
        if has_swin and pkey in sd and sd[pkey].shape[0] == 96:
            picks.append("swin_tiny_patch4_window7_224"); continue
        raise RuntimeError(f"Unknown anchor at anchors.{i}, sample={ [k for k in sd if k.startswith(base)][:5] }")
    return picks

def load_checkpoint_flex(model, ckpt_path, num_classes=None, drop_heads="auto"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # strip DP 'module.'
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    head_suffixes = (
        "fc.weight","fc.bias",
        "head.weight","head.bias",
        "classifier.weight","classifier.bias",
        "heads.head.weight","heads.head.bias",
    )

    to_drop = []
    if drop_heads == "auto" and num_classes is not None:
        for k, v in sd.items():
            if k.endswith(head_suffixes) and hasattr(v, "shape"):
                out_dim = v.shape[0] if v.ndim >= 1 else None
                if out_dim is not None and out_dim != num_classes:
                    to_drop.append(k)
    elif drop_heads is True:
        to_drop = [k for k in sd if k.endswith(head_suffixes)]

    for k in to_drop:
        sd.pop(k, None)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} dropped={len(to_drop)}")
    return model

class TrunkWithHead(nn.Module):
    def __init__(self, trunk: nn.Module, head: nn.Module):
        super().__init__()
        self.trunk = trunk
        self.head = head
    def forward(self, x):
        return self.head(self.trunk(x))

def set_requires_grad_for_stitch(trunk: nn.Module, requires: bool):
    """
    Enable grads for stitch/bridge layers only; keep anchors frozen.
    """
    for n, p in trunk.named_parameters():
        # default: freeze everything
        p.requires_grad = False
        # selectively enable stitch / cnn_to_vit / bridge
        if any(tag in n for tag in ["stitch", "cnn_to_vit", "bridge"]):
            p.requires_grad = requires

def train_one_epoch(model, loader, optimizer, device, scaler=None,
                    train_stitch: bool = False, eval_forward: bool = True):
    """
    If eval_forward=True (default):
      - trunk stays in eval() for forward (stable stitched path / feature dim)
      - BUT gradients still flow if its params have requires_grad=True.
    If eval_forward=False:
      - trunk.train() when train_stitch=True (only use if you’ve verified the trunk
        still returns the same feature dim in train mode).
    """
    # Head always in train mode
    model.head.train()

    # Decide trunk mode for forward
    if eval_forward:
        model.trunk.eval()
    else:
        model.trunk.train() if train_stitch else model.trunk.eval()

    loss_sum = acc_sum = n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            if train_stitch:
                # forward with grads through trunk (eval or train, as set above)
                feats = model.trunk(x)
                logits = model.head(feats)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    feats = model.trunk(x)
                logits = model.head(feats)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
        else:
            if train_stitch:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = model.trunk(x)
                    logits = model.head(feats)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                with torch.no_grad():
                    feats = model.trunk(x)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model.head(feats)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()

        b = y.size(0)
        loss_sum += loss.item() * b
        acc_sum  += (logits.argmax(1) == y).float().sum().item()
        n += b

    return loss_sum / n, acc_sum / n * 100.0

@torch.no_grad()
def evaluate(model, loader, device, eval_forward=True):
    # eval modes for metrics
    model.head.eval()
    model.trunk.eval()  # eval-forward always for validation
    loss_sum = acc_sum = n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        feats  = model.trunk(x)
        logits = model.head(feats)
        loss = F.cross_entropy(logits, y)
        b = y.size(0)
        loss_sum += loss.item() * b
        acc_sum  += (logits.argmax(1) == y).float().sum().item()
        n += b
    return loss_sum / n, acc_sum / n * 100.0

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune stitched head on SNNet")
    ap.add_argument("--ckpt", required=True, help="Path to trunk checkpoint (your existing training output)")
    ap.add_argument("--anchors", nargs="+", default=None, help="Anchor list; omit to auto-detect from ckpt")
    ap.add_argument("--stitch_id", type=int, default=7, help="Stitch config ID to train")
    ap.add_argument("--train-root", required=True, help="ImageFolder train root")
    ap.add_argument("--val-root",   required=True, help="ImageFolder val root")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--cifar10-transforms", action="store_true")
    ap.add_argument("--no-center-crop", action="store_true", help="Eval: disable center-crop")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--unfreeze-stitch", action="store_true",
                    help="Also unfreeze stitch/bridge params (in addition to head)")
    ap.add_argument("--eval-forward", dest="eval_forward", action="store_true", default=True,
                    help="Use trunk.eval() for forward (default; stable stitched path).")
    ap.add_argument("--train-forward", dest="eval_forward", action="store_false",
                    help="Use trunk.train() forward when --unfreeze-stitch (only if dims match).")
    ap.add_argument("--save", default="stitched_head_ft.pth.tar")
    ap.add_argument("--pretrained-anchors", action="store_true",
                    help="If True, anchors start from timm pretrained weights (usually False for your ckpt)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.ckpt).exists():
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")

    # Resolve anchors from checkpoint if not given
    base = torch.load(args.ckpt, map_location="cpu")
    sd = base.get("state_dict", base)
    if args.anchors is None:
        args.anchors = anchors_from_ckpt_guess(sd)
        print(f"[auto] anchors: {args.anchors}")
    else:
        print(f"[info] using anchors: {args.anchors}")

    # Build stitched trunk (headless anchors)
    anchors = build_anchors(args.anchors, pretrained=False, num_classes=0)
    trunk = SNNet(anchors, cnn_to_vit=need_cnn_to_vit(args.anchors)).to(device).eval()
    load_checkpoint_flex(trunk, args.ckpt, num_classes=None, drop_heads="auto")

    # Pick stitch config
    if args.stitch_id not in trunk.stitch_configs:
        print("Available stitch IDs:")
        for k, v in trunk.stitch_configs.items():
            print(k, v)
        raise SystemExit(f"Invalid stitch_id {args.stitch_id}")
    trunk.reset_stitch_id(args.stitch_id)

    # Build loaders FIRST (so we can probe feature dim with the exact train transform)
    train_tf = make_train_transform(args.img_size, args.cifar10_transforms)
    val_tf   = make_val_transform(args.img_size, args.cifar10_transforms, center_crop=not args.no_center_crop)

    ds_tr = datasets.ImageFolder(args.train_root, transform=train_tf)
    ds_va = datasets.ImageFolder(args.val_root,   transform=val_tf)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=(device=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=(device=="cuda"))
    print(f"[data] train samples={len(ds_tr)}  val samples={len(ds_va)}  "
          f"train iters/epoch={(len(ds_tr)+args.batch_size-1)//args.batch_size}  "
          f"val iters/epoch={(len(ds_va)+args.batch_size-1)//args.batch_size}")

    # Probe stitched feature dim using SAME transform & trunk in EVAL forward
    trunk.eval()
    with torch.no_grad():
        xb, _ = next(iter(dl_tr))
        xb = xb[:2].to(device)
        feat = trunk(xb)
        feat_dim = int(feat.shape[-1])
    print(f"[info] runtime stitched feature dim (train transform, eval-forward) = {feat_dim}")

    # Create head matching the probed dim
    head = nn.Linear(feat_dim, args.num_classes, bias=True).to(device)
    model = TrunkWithHead(trunk, head).to(device)

    # Freeze trunk by default; optionally unfreeze stitch/bridge params
    set_requires_grad_for_stitch(model.trunk, requires=args.unfreeze_stitch)

    # Optimizer (+ AMP scaler if requested)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and args.amp) else None

    # Decide training knobs
    train_stitch = args.unfreeze_stitch
    print(f"[info] training mode: head {'+ stitch/bridge' if train_stitch else 'only'}  "
          f"(forward={'eval' if args.eval_forward else 'train'})")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, dl_tr, optim, device, scaler,
            train_stitch=train_stitch, eval_forward=args.eval_forward
        )
        va_loss, va_acc = evaluate(model, dl_va, device, eval_forward=True)
        best = max(best, va_acc)
        print(f"[{epoch:02d}/{args.epochs}] train: loss {tr_loss:.4f} acc@1 {tr_acc:.2f} | "
              f"val: loss {va_loss:.4f} acc@1 {va_acc:.2f} (best {best:.2f})")

    # Save a checkpoint with BOTH trunk and head + class mapping for safer inference
    payload = {
        "state_dict": model.state_dict(),   # contains 'trunk.*' and 'head.*'
        "epoch": args.epochs,
        "best_acc1": best,
        "stitch_id": args.stitch_id,
        "anchors": args.anchors,
        "num_classes": args.num_classes,
        "class_to_idx": getattr(ds_tr, "class_to_idx", None),
    }
    torch.save(payload, args.save)
    print(f"[save] wrote: {args.save}")

if __name__ == "__main__":
    main()
