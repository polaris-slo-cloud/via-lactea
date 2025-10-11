#!/usr/bin/env python3
import argparse, warnings, io, random
from pathlib import Path

warnings.filterwarnings("ignore", message=".*interpolation.*deprecated.*")
warnings.filterwarnings("ignore", message="torch.meshgrid:.*indexing.*")

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter, ImageEnhance

from timm.models import create_model
from timm.models.snnet import SNNet

# ---------------- constants ----------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD  = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---------------- small helpers ----------------
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

# --- corruption transforms (pure-PIL/Numpy, no extra deps) ---
class AddGaussianNoiseT:
    def __init__(self, sigma):
        self.sigma = float(sigma)
    def __call__(self, img: Image.Image):
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, self.sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

class MotionBlurT:
    """Simple horizontal motion blur using PIL kernel (no OpenCV)."""
    def __init__(self, k=7):
        k = max(3, int(k) | 1)  # odd
        self.k = k
        # horizontal line kernel
        kernel = [0.0]*(k*k)
        center = (k//2)*k
        for i in range(k):
            kernel[center + i] = 1.0 / k
        self.kernel = kernel
    def __call__(self, img: Image.Image):
        return img.filter(ImageFilter.Kernel((self.k, self.k), self.kernel, scale=None))

class JPEGCompressionT:
    def __init__(self, quality=30):
        self.quality = int(quality)
    def __call__(self, img: Image.Image):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class PixelateT:
    def __init__(self, scale=0.5):
        self.scale = float(scale)
    def __call__(self, img: Image.Image):
        w, h = img.size
        down_w = max(1, int(w * self.scale))
        down_h = max(1, int(h * self.scale))
        img_small = img.resize((down_w, down_h), resample=Image.BILINEAR)
        return img_small.resize((w, h), resample=Image.NEAREST)

class FogT:
    """Crude fog: blend with a bright blurred version."""
    def __init__(self, alpha=0.4, radius=3):
        self.alpha = float(alpha)
        self.radius = int(radius)
    def __call__(self, img: Image.Image):
        bright = ImageEnhance.Brightness(img).enhance(1.2)
        blur = bright.filter(ImageFilter.GaussianBlur(self.radius))
        return Image.blend(img, blur, self.alpha)

class ContrastT:
    def __init__(self, factor=0.6):
        self.factor = float(factor)
    def __call__(self, img: Image.Image):
        return ImageEnhance.Contrast(img).enhance(self.factor)

class CutoutBoxT:
    def __init__(self, frac=0.25):
        self.frac = float(frac)
    def __call__(self, img: Image.Image):
        w, h = img.size
        bw, bh = int(w * self.frac), int(h * self.frac)
        if bw <= 0 or bh <= 0:
            return img
        x = random.randint(0, max(0, w - bw))
        y = random.randint(0, max(0, h - bh))
        arr = np.array(img)
        arr[y:y+bh, x:x+bw, :] = 0
        return Image.fromarray(arr)

def corruption_transform(name, severity):
    """Map severity (1..5) to parameters and return a PIL transform or None."""
    if name is None:
        return None
    name = name.lower()
    if name == "gauss_noise":
        sig = {1:5,2:10,3:20,4:30,5:45}[severity]
        return AddGaussianNoiseT(sig)
    if name == "motion_blur":
        k = {1:3,2:5,3:7,4:9,5:11}[severity]
        return MotionBlurT(k)
    if name == "jpeg":
        q = {1:80,2:60,3:40,4:25,5:15}[severity]
        return JPEGCompressionT(q)
    if name == "pixelate":
        s = {1:0.8,2:0.6,3:0.5,4:0.35,5:0.25}[severity]
        return PixelateT(s)
    if name == "fog":
        a = {1:0.15,2:0.25,3:0.35,4:0.45,5:0.55}[severity]
        r = {1:1,2:2,3:3,4:4,5:5}[severity]
        return FogT(alpha=a, radius=r)
    if name == "contrast":
        f = {1:0.9,2:0.8,3:0.7,4:0.6,5:0.5}[severity]
        return ContrastT(f)
    return None

def make_val_transform(img_size, use_cifar10, center_crop, corrupt=None, severity=3, cutout_frac=0.0):
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if use_cifar10 else (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    pre = []
    t_cor = corruption_transform(corrupt, severity) if corrupt else None
    if t_cor is not None:
        pre.append(t_cor)
    if cutout_frac and cutout_frac > 0:
        pre.append(CutoutBoxT(cutout_frac))
    if center_crop:
        resize_side = max(int(round(img_size / 0.875)), img_size)
        base = [
            T.Resize(resize_side, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
        ]
    else:
        base = [T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC)]
    return T.Compose(pre + base + [T.ToTensor(), T.Normalize(mean, std)])

def _stage_name(anchor_name: str, stage_idx: int) -> str:
    n = anchor_name.lower()
    if "resnet" in n:
        names = ["layer1 (C2)", "layer2 (C3)", "layer3 (C4)", "layer4 (C5)"]
        return names[stage_idx] if 0 <= stage_idx < 4 else f"layer{stage_idx}"
    if "swin" in n:
        names = ["stage0 (P2)", "stage1 (P3)", "stage2 (P4)", "stage3 (P5)"]
        return names[stage_idx] if 0 <= stage_idx < 4 else f"stage{stage_idx}"
    return f"stage{stage_idx}"

def _resnet_stage_channels(anchor_name: str, stage_idx: int) -> int:
    return [64, 128, 256, 512][stage_idx]

def _swin_stage_embed(anchor_name: str, stage_idx: int) -> int:
    n = anchor_name.lower()
    if "swin_tiny" in n:   dims = [96, 192, 384, 768]
    elif "swin_small" in n: dims = [96, 192, 384, 768]
    elif "swin_base" in n:  dims = [128, 256, 512, 1024]
    else:                   dims = [96, 192, 384, 768]
    return dims[stage_idx]

def _down_factor_for_stage(anchor_name: str, stage_idx: int) -> int:
    n = anchor_name.lower()
    if "resnet" in n:
        return [4, 8, 16, 32][stage_idx]
    if "swin" in n:
        base = 4
        return base * (2 ** stage_idx)
    return 4 * (2 ** stage_idx)

def _grid(img_size: int, factor: int) -> str:
    return f"{img_size//factor}×{img_size//factor}"

def _get_anchor_model(trunk, idx: int):
    m = getattr(trunk, "anchors", None)
    if isinstance(m, (list, tuple)) and 0 <= idx < len(m):
        return m[idx]
    return None

def _swin_stage_depth_from_model(anchor_model, stage_idx: int):
    try:
        layers = getattr(anchor_model, "layers", None) or getattr(anchor_model, "stages", None)
        if layers is not None:
            blkseq = getattr(layers[stage_idx], "blocks", None)
            if blkseq is not None:
                return len(blkseq)
    except Exception:
        pass
    n = anchor_model.__class__.__name__.lower() if anchor_model else ""
    if "tiny" in n:   return [2,2,6,2][stage_idx]
    if "small" in n:  return [2,2,18,2][stage_idx]
    if "base" in n:   return [2,2,18,2][stage_idx]
    return [2,2,6,2][stage_idx]

def print_stitch_table(trunk, anchors_names, img_size=224):
    cfgs = getattr(trunk, "stitch_configs", {})
    if not cfgs:
        print("[stitch] no stitch configs found on model.")
        return

    print("\nstitch_id | route     | target stage                       | inner | stitch_cfgs")
    print("-----------------------------------------------------------------------------------")
    for sid in sorted(cfgs.keys()):
        c = cfgs[sid]
        comb = c.get("comb_id")
        slayers = c.get("stitch_layers", [])
        scfgs = c.get("stitch_cfgs", [])
        stage_id = c.get("stage_id", None)

        if isinstance(comb, (list, tuple)) and len(comb) == 1:
            route = f"{comb[0]}"
            target_str = "-"
        elif isinstance(comb, (list, tuple)) and len(comb) == 2:
            a_src, a_tgt = comb
            route = f"{a_src}→{a_tgt}"
            if stage_id is not None and 0 <= a_tgt < len(anchors_names):
                target_str = f"{anchors_names[a_tgt]} / {_stage_name(anchors_names[a_tgt], stage_id)}"
            else:
                target_str = f"anchor {a_tgt} / stage {stage_id}"
        else:
            route = str(comb)
            target_str = "-"

        inner = ",".join(map(str, slayers)) if slayers else "-"
        print(f"{sid:9d} | {route:<9} | {target_str:<30} | {inner:^5} | {scfgs}")

        if isinstance(comb, (list, tuple)) and len(comb) == 2 and stage_id is not None:
            a_src, a_tgt = comb
            src_name = anchors_names[a_src]
            tgt_name = anchors_names[a_tgt]

            src_model = _get_anchor_model(trunk, a_src)
            tgt_model = _get_anchor_model(trunk, a_tgt)
            tgt_depth = _swin_stage_depth_from_model(tgt_model, stage_id) if "swin" in tgt_name.lower() else None

            t_factor = _down_factor_for_stage(tgt_name, stage_id)
            t_hw = _grid(img_size, t_factor)
            if "swin" in tgt_name.lower():
                t_embed = _swin_stage_embed(tgt_name, stage_id)
            else:
                t_embed = _resnet_stage_channels(tgt_name, stage_id)

            for (src_stage, tgt_block) in scfgs:
                s_factor = _down_factor_for_stage(src_name, src_stage)
                s_hw = _grid(img_size, s_factor)
                if "resnet" in src_name.lower():
                    s_c = _resnet_stage_channels(src_name, src_stage)
                else:
                    s_c = _swin_stage_embed(src_name, src_stage)

                valid_note = ""
                if tgt_depth is not None and (tgt_block < 0 or tgt_block >= tgt_depth):
                    valid_note = f"  [!] block idx out of range (valid 0..{tgt_depth-1})"

                print(f"           ↳  {src_name} / {_stage_name(src_name, src_stage)} "
                      f"[C={s_c}, {s_hw}]  →  {tgt_name} / {_stage_name(tgt_name, stage_id)}.block{tgt_block} "
                      f"[E={t_embed}, {t_hw}]{valid_note}")
        print()

@torch.no_grad()
def probe_feat_dim(trunk, device, img_size):
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    y = trunk(x)
    return int(y.shape[-1])

# --- AMP context compatible with older PyTorch (no device_type kw) ---
from contextlib import contextmanager
from torch.cuda.amp import autocast
@contextmanager
def amp_ctx(enabled: bool):
    if enabled and torch.cuda.is_available():
        with autocast():
            yield
    else:
        yield

@torch.no_grad()
def eval_one_id(trunk, head, loader, device, amp=False, stitch_noise_std=0.0):
    trunk.eval(); head.eval()
    total = correct = 0
    use_amp = (device.startswith("cuda") and amp)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx(use_amp):
            feats = trunk(x)
            if stitch_noise_std and stitch_noise_std > 0:
                feats = feats + torch.randn_like(feats) * float(stitch_noise_std)
            logits = head(feats)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total if total else 0.0

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate SNNet stitched head across stitch_ids (with stress options)")
    ap.add_argument("--ckpt", required=True, help="stitched_head_ft.pth.tar (contains trunk.* + head.*)")
    ap.add_argument("--val-root", required=True, help="ImageFolder root for validation")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cifar10-transforms", action="store_true")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")

    # --- stress flags to widen variance (no retraining) ---
    ap.add_argument("--corrupt", choices=[
        "gauss_noise","motion_blur","jpeg","pixelate","fog","contrast"
    ], default=None, help="Apply an ImageNet-C style corruption before resize/crop.")
    ap.add_argument("--severity", type=int, default=3, help="Corruption severity 1..5 (default 3).")
    ap.add_argument("--cutout", type=float, default=0.0, help="Cutout fraction of image side (e.g., 0.2).")
    ap.add_argument("--stitch-noise-std", type=float, default=0.0,
                    help="Add Gaussian noise to stitched features before head (e.g., 0.02).")

    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if not Path(args.ckpt).exists():
        raise SystemExit(f"[error] ckpt not found: {args.ckpt}")
    if not Path(args.val_root).exists():
        raise SystemExit(f"[error] val_root not found: {args.val_root}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    anchors = ckpt.get("anchors", None)
    if not anchors:
        raise SystemExit("[error] anchors metadata missing in ckpt; re-train/save with anchors in payload.")

    # Build trunk (headless anchors) & load trunk.*
    trunk = SNNet(
        build_anchors(anchors, pretrained=False, num_classes=0),
        cnn_to_vit=need_cnn_to_vit(anchors)
    ).to(device).eval()

    trunk_sd = {k.replace("trunk.", "", 1): v for k, v in sd.items() if k.startswith("trunk.")}
    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    if missing or unexpected:
        print(f"[warn] trunk load: missing={len(missing)} unexpected={len(unexpected)}")

    # Print a readable map of all stitch routes (with shapes)
    print_stitch_table(trunk, anchors, img_size=args.img_size)

    # Build stitched head from head.*
    if "head.weight" not in sd:
        raise SystemExit("[error] stitched head not found in ckpt (no head.*). Use your stitched_head_ft.pth.tar.")
    W = sd["head.weight"]
    out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
    head = nn.Linear(in_dim, out_dim, bias=("head.bias" in sd)).to(device).eval()
    head.load_state_dict({k.replace("head.", ""): v for k, v in sd.items() if k.startswith("head.")}, strict=True)

    # Data (with optional corruption/cutout)
    tf = make_val_transform(
        args.img_size,
        args.cifar10_transforms,
        center_crop=not args.no_center_crop,
        corrupt=args.corrupt,
        severity=max(1, min(5, args.severity)),
        cutout_frac=args.cutout
    )
    loader = DataLoader(
        datasets.ImageFolder(args.val_root, transform=tf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=(device == "cuda")
    )

    # Evaluate per stitch_id, skip mismatches
    results = []
    print("\nstitch_id | feat_dim | eval_status")
    print("-----------------------------------")
    for sid in sorted(trunk.stitch_configs.keys()):
        trunk.reset_stitch_id(sid)
        feat_dim = probe_feat_dim(trunk, device, args.img_size)
        if feat_dim != in_dim:
            print(f"{sid:9d} | {feat_dim:8d} | SKIP (head expects {in_dim})")
            results.append((sid, None, feat_dim))
            continue
        acc = eval_one_id(trunk, head, loader, device, amp=args.amp, stitch_noise_std=args.stitch_noise_std)
        print(f"{sid:9d} | {feat_dim:8d} | acc@1 = {acc:5.2f}%")
        results.append((sid, acc, feat_dim))

    # Summary
    valid = [(sid, acc) for sid, acc, _ in results if acc is not None]
    if valid:
        best_sid, best_acc = max(valid, key=lambda t: t[1])
        worst_sid, worst_acc = min(valid, key=lambda t: t[1])
        print("\nBEST / WORST:")
        print(f"  best:  stitch_id {best_sid}  acc@1 = {best_acc:.2f}%")
        print(f"  worst: stitch_id {worst_sid}  acc@1 = {worst_acc:.2f}%")
        print(f"  spread: {best_acc - worst_acc:.2f} pts")
    else:
        print("\nNo stitch_id produced features matching head_in_dim; "
              "your head was trained for a different route.")

if __name__ == "__main__":
    main()
