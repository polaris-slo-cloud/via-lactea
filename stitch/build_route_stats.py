#!/usr/bin/env python3
"""
OOM-safe route stats builder for SN-Net stitched models.

- Requires a checkpoint that contains BOTH trunk.* and head.* (your stitched_head_ft.pth.tar).
- Evaluates every stitch_id, accumulating accuracy and mean max-prob.
- Robust GPU memory handling: microbatching + retry on OOM + cache clearing.

Example:
  python3 build_route_stats_oomsafe.py \
    --ckpt stitched_head_ft.pth.tar \
    --val-root /path/to/cifar10/val \
    --img-size 224 --batch-size 256 --microbatch 64 \
    --workers 8 --cifar10-transforms --no-center-crop --amp \
    --save route_stats.json
"""

import os
# ---- allocator knobs MUST be set before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

import argparse, json, gc, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*interpolation.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing.*")

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader

from timm.models import create_model
from timm.models.snnet import SNNet

# ---------------- constants ----------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---------------- helpers ----------------
def build_anchors(names, pretrained=False, num_classes=0):
    xs=[]
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

def make_val_transform(img_size, use_cifar10, center_crop):
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

@torch.no_grad()
def probe_feat_dim(trunk, device, img_size):
    x = torch.zeros(1,3,img_size,img_size, device=device)
    y = trunk(x)
    return int(y.shape[-1])

def print_mem(tag=""):
    if not torch.cuda.is_available(): return
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserv = torch.cuda.memory_reserved() / (1024**2)
    peak  = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[mem]{' '+tag if tag else ''} allocated={alloc:.1f}MB reserved={reserv:.1f}MB peak={peak:.1f}MB")

def free_cuda(*objs):
    for o in objs:
        try: del o
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --------------- microbatch runner ---------------
@torch.no_grad()
def forward_in_microbatches(trunk, head, x, device, use_amp=False, microbatch=64):
    """
    Returns logits for the whole batch `x` by splitting it into microbatches.
    Retries with smaller microbatch if OOM is raised.
    """
    B = x.size(0)
    mb = min(microbatch, B)
    out_chunks = []
    while True:
        try:
            out_chunks.clear()
            for s in range(0, B, mb):
                xb = x[s:s+mb].to(device, non_blocking=True)
                if device.startswith("cuda") and use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        feats = trunk(xb)
                        logits = head(feats)
                else:
                    feats = trunk(xb)
                    logits = head(feats)
                out_chunks.append(logits.detach().float().cpu())
                # free per-chunk
                free_cuda(xb, feats, logits)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and mb > 1:
                print(f"[oom] microbatch {mb} too big → halving and retrying...")
                free_cuda()
                mb = max(1, mb // 2)
                continue
            raise
    return torch.cat(out_chunks, dim=0)

# --------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="OOM-safe SNNet route stats")
    ap.add_argument("--ckpt", required=True, help="stitched_head_ft.pth.tar (contains trunk.* + head.*)")
    ap.add_argument("--val-root", required=True, help="ImageFolder validation root")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--microbatch", type=int, default=64,
                    help="Per-forward split inside a batch to reduce VRAM")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cifar10-transforms", action="store_true")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save", default="route_stats.json")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if not Path(args.ckpt).exists():
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")
    if not Path(args.val_root).exists():
        raise SystemExit(f"[error] val_root not found: {args.val_root}")

    # -- load ckpt & metadata
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    anchors = ckpt.get("anchors", None)
    if not anchors:
        raise SystemExit("[error] anchors metadata missing in ckpt payload; re-train/save with anchors.")

    if "head.weight" not in sd:
        raise SystemExit("[error] stitched head not found in ckpt (no head.*). Use your stitched_head_ft.pth.tar.")

    # -- build trunk (headless anchors) and load trunk.*
    trunk = SNNet(build_anchors(anchors, pretrained=False, num_classes=0),
                  cnn_to_vit=need_cnn_to_vit(anchors)).to(device).eval()
    trunk_sd = {k.replace("trunk.","",1):v for k,v in sd.items() if k.startswith("trunk.")}
    m_missing, m_unexp = trunk.load_state_dict(trunk_sd, strict=False)
    if m_missing or m_unexp:
        print(f"[warn] trunk load: missing={len(m_missing)} unexpected={len(m_unexp)}")

    # -- build head from head.*
    W = sd["head.weight"]
    out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
    head = nn.Linear(in_dim, out_dim, bias=("head.bias" in sd)).to(device).eval()
    head.load_state_dict({k.replace("head.",""):v for k,v in sd.items() if k.startswith("head.")}, strict=True)

    # -- data
    tf = make_val_transform(args.img_size, args.cifar10_transforms, center_crop=not args.no_center_crop)
    loader = DataLoader(
        datasets.ImageFolder(args.val_root, transform=tf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=(device=="cuda"), persistent_workers=False
    )

    # -- route stats
    results = {}
    num_routes = len(trunk.stitch_configs)
    print("stitch_id | feat_dim | acc@1   | mean_maxprob")
    print("-----------------------------------------------")

    for sid in sorted(trunk.stitch_configs.keys()):
        # Select route
        trunk.reset_stitch_id(sid)
        # Probe feat dim; skip if head doesn't match
        with torch.inference_mode():
            feat_dim = probe_feat_dim(trunk, device, args.img_size)
        if feat_dim != in_dim:
            print(f"{sid:9d} | {feat_dim:8d} | SKIP (head expects {in_dim})")
            results[sid] = {"feat_dim": feat_dim, "acc1": None, "mean_maxprob": None}
            # free any lingering cache between routes
            free_cuda()
            continue

        # Eval this route (OOM-safe microbatching)
        total = correct = 0
        maxprob_sum = 0.0

        with torch.inference_mode():
            for xb, yb in loader:
                try:
                    logits = forward_in_microbatches(
                        trunk, head, xb, device,
                        use_amp=args.amp, microbatch=args.microbatch
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"[oom] even after shrinking microbatch; reducing dataloader batch size might help.")
                        raise
                    raise
                probs = torch.softmax(logits.float(), dim=1)
                maxprob, pred = probs.max(dim=1)
                correct += (pred == yb).sum().item()
                total   += yb.size(0)
                maxprob_sum += float(maxprob.sum().item())

                # free per-batch
                free_cuda(xb, yb, logits, probs, maxprob, pred)

        acc1 = 100.0 * correct / total if total else 0.0
        mean_maxprob = (maxprob_sum / total) if total else 0.0
        print(f"{sid:9d} | {feat_dim:8d} | {acc1:6.2f}% | {mean_maxprob:12.4f}")

        results[sid] = {"feat_dim": feat_dim, "acc1": acc1, "mean_maxprob": mean_maxprob}

        # free between routes to fight fragmentation
        free_cuda()

    # Save stats
    out_path = Path(args.save)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] wrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
