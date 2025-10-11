#!/usr/bin/env python3
import os, json, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*deprecated.*")

import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torchvision import transforms as T
from timm.models import create_model

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---------------- utils ----------------
def make_transform(img_size, cifar10=False, center_crop=True):
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if cifar10 else ((0.485,0.456,0.406),(0.229,0.224,0.225))
    if center_crop:
        resize_side = max(int(round(img_size/0.875)), img_size)
        t = [T.Resize(resize_side, interpolation=T.InterpolationMode.BICUBIC),
             T.CenterCrop(img_size)]
    else:
        t = [T.Resize((img_size,img_size), interpolation=T.InterpolationMode.BICUBIC)]
    t += [T.ToTensor(), T.Normalize(mean,std)]
    return T.Compose(t)

def ddp_init():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    # default init_method is env://, so we just need env vars set
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def load_json(path): return json.loads(Path(path).read_text())

def strip_prefix(d, prefix):
    out = {}
    for k,v in d.items():
        nk = k[len(prefix):] if k.startswith(prefix) else k
        out[nk] = v
    return out

def load_shard(path): return torch.load(path, map_location="cpu") if Path(path).exists() else {}

# ---------------- ResNet (rank 0) ----------------
class ResNetTapFromShards(nn.Module):
    """
    Load only the ResNet parts we need and expose taps C2 (layer1) and C3 (layer2).
    """
    def __init__(self, shards_dir, manifest, device="cpu"):
        super().__init__()
        self.m = create_model("resnet18", pretrained=False, num_classes=0)
        # map keys: remove anchors prefix + resnet anchor idx
        pref = f"{manifest['anchors']['prefix']}anchors.{manifest['anchors']['resnet_idx']}."
        own = self.m.state_dict()
        merged = {}
        for name in ["resnet_stem", "resnet_layer1", "resnet_layer2"]:
            d = load_shard(Path(shards_dir)/f"{name}.pt")
            d = strip_prefix(d, pref)
            for k,v in d.items():
                if k in own and own[k].shape == v.shape:
                    merged[k] = v
        self.m.load_state_dict({**own, **merged}, strict=False)
        self.to(device).eval()

    @torch.no_grad()
    def forward(self, x, tap="C2"):
        m = self.m
        x = m.conv1(x); x = m.bn1(x); x = m.act1(x)
        if hasattr(m, "maxpool"): x = m.maxpool(x)
        c2 = m.layer1(x)        # [B,64,56,56]
        if tap == "C2": return c2
        c3 = m.layer2(c2)       # [B,128,28,28]
        if tap == "C3": return c3
        raise ValueError("tap must be C2 or C3")

# ---------------- Swin tail (rank 1) ----------------
class SwinTailFromShards(nn.Module):
    """
    Enter Swin at S1B1 or S2B{k}, continue through remaining stages, then Linear head.
    Uses simple 1x1+pool adapters; swap in your learned adapters from snnet_trunk.pt if needed.
    """
    def __init__(self, shards_dir, manifest, entry, num_classes, device="cpu"):
        super().__init__()
        self.entry = entry  # "S1B1" or "S2B1"... "S2B5"
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
        own = self.swin.state_dict()

        swin_pref = f"{manifest['anchors']['prefix']}anchors.{manifest['anchors']['swin_idx']}."
        def load_into(d):
            merged = {}
            d = strip_prefix(d, swin_pref)
            for k,v in d.items():
                if k in own and own[k].shape == v.shape:
                    merged[k] = v
            return merged

        # load needed shards
        merged = {}
        # stage1 pieces if entry S1B1
        if self.entry == "S1B1":
            merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage1_b0.pt")))
            merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage1_b1.pt")))
            merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage1_downsample.pt")))
            # stage2 blocks 0..5
            for b in range(6):
                merged.update(load_into(load_shard(Path(shards_dir)/f"swin_stage2_b{b}.pt")))
            merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage2_downsample.pt")))
        else:
            # S2B{k} entry -> need stage2 blocks k..5 + stage2 downsample
            kmap = {"S2B1":0,"S2B2":1,"S2B3":2,"S2B4":3,"S2B5":4}
            b0 = kmap[self.entry]
            for b in range(b0,6):
                merged.update(load_into(load_shard(Path(shards_dir)/f"swin_stage2_b{b}.pt")))
            merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage2_downsample.pt")))
        # full stage3 & stage4
        merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage3.pt")))
        merged.update(load_into(load_shard(Path(shards_dir)/"swin_stage4.pt")))
        # norm/head inside swin (if present)
        merged.update(load_into(load_shard(Path(shards_dir)/"swin_tail_extras.pt")))

        self.swin.load_state_dict({**own, **merged}, strict=False)

        # adapters (placeholder). Replace with learned adapters from snnet_trunk.pt if desired.
        if self.entry == "S1B1":
            self.adapt = nn.Sequential(
                nn.Conv2d(64, 192, 1, bias=False),
                nn.AvgPool2d(2)  # 56->28
            )
        else:
            self.adapt_c2 = nn.Sequential(nn.Conv2d(64, 384, 1, bias=False), nn.AvgPool2d(4))   # 56->14
            self.adapt_c3 = nn.Sequential(nn.Conv2d(128,384, 1, bias=False), nn.AvgPool2d(2))   # 28->14

        # load head
        head_sd = torch.load(Path(shards_dir)/"head.pt", map_location="cpu")
        self.head = nn.Linear(768, head_sd["weight"].shape[0], bias=("bias" in head_sd))
        self.head.load_state_dict(head_sd, strict=True)

        self.to(device).eval()

    @torch.no_grad()
    def forward(self, feat):
        # feat: [B,C,H,W] from ResNet tap
        if self.entry == "S1B1":
            x = self.adapt(feat)       # [B,192,28,28]
            tokens = self._run_from(stage=1, block_idx=0, x=x)
        else:
            C = feat.shape[1]
            if C == 64:    x = self.adapt_c2(feat)   # [B,384,14,14]
            elif C == 128: x = self.adapt_c3(feat)   # [B,384,14,14]
            else: raise ValueError(f"unexpected channels {C}")
            kmap = {"S2B1":0,"S2B2":1,"S2B3":2,"S2B4":3,"S2B5":4}
            tokens = self._run_from(stage=2, block_idx=kmap[self.entry], x=x)

        # final norm + pool + head (use timm’s forward_features tail)
        if hasattr(self.swin, 'norm'):
            tokens = self.swin.norm(tokens)
        # tokens: [B, L, 768]
        x = tokens.mean(dim=1)  # global avg over tokens
        logits = self.head(x)
        return logits

    def _run_from(self, stage:int, block_idx:int, x:torch.Tensor):
        m = self.swin
        def NCHW_to_tokens(y):
            B,C,H,W = y.shape
            return y.flatten(2).transpose(1,2)  # [B, HW, C]

        def run_layer(layer, tokens, start_block=0):
            # run blocks start..end, then downsample if present
            for i, blk in enumerate(layer.blocks):
                if i < start_block: continue
                tokens = blk(tokens)
            if layer.downsample is not None:
                tokens = layer.downsample(tokens)
            return tokens

        # timm: layers[0]=stage1(96@56), [1]=stage2(192@28), [2]=stage3(384@14), [3]=stage4(768@7)
        if stage == 1:
            # entering at P3 (192@28)
            t = NCHW_to_tokens(x)
            t = run_layer(m.layers[1], t, start_block=block_idx)  # stage2 192@28 -> 384@14 (with its downsample)
            t = run_layer(m.layers[2], t, start_block=0)          # stage3 384@14 -> 768@7
            t = run_layer(m.layers[3], t, start_block=0)          # stage4 768@7 -> 768@7
            return t
        elif stage == 2:
            # entering at P4 (384@14)
            t = NCHW_to_tokens(x)
            t = run_layer(m.layers[2], t, start_block=block_idx)  # finish stage3
            t = run_layer(m.layers[3], t, start_block=0)          # stage4
            return t
        else:
            raise ValueError("stage must be 1 or 2")

# ---------------- route helper ----------------
def choose_sid(stats_path, head_in_dim, criterion="acc"):
    data = load_json(stats_path)
    entries = data["entries"] if isinstance(data,dict) and "entries" in data else (data if isinstance(data,list) else [])
    cand = [e for e in entries if int(e.get("feat_dim",-1)) == int(head_in_dim)]
    key = "acc1" if criterion=="acc" else "mean_maxprob"
    pick = max(cand, key=lambda e: float(e.get(key,-1e9)))
    return int(pick["sid"])

def sid_to_tap_entry(sid:int):
    if sid == 2: return "C2","S1B1"
    if sid == 3: return "C2","S2B1"
    if sid == 4: return "C2","S2B2"
    if sid == 5: return "C3","S2B3"
    if sid == 6: return "C3","S2B4"
    if sid == 7: return "C3","S2B5"
    if sid == 1: return None, None  # pure Swin (not covered in 2-rank pipeline)
    raise ValueError(f"unsupported sid {sid}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--manifest", default=None, help="defaults to shards-dir/manifest.json")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--cifar10-transforms", action="store_true")
    ap.add_argument("--no-center-crop", action="store_true")
    ap.add_argument("--criterion", choices=["acc","conf"], default="acc")
    ap.add_argument("--amp", action="store_true")
    # testing-only flags to avoid torchrun
    ap.add_argument("--set-test-env", action="store_true",
                    help="set RANK/WORLD_SIZE/MASTER_* and CUDA_VISIBLE_DEVICES inside this process")
    ap.add_argument("--rank-override", type=int, default=None,
                    help="force this process to act as rank 0 or 1 when --set-test-env is used")
    args = ap.parse_args()

    # --- testing-only: self-contained env so you can run without torchrun ---
    if args.set_test_env:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("WORLD_SIZE", "2")
        if args.rank_override is not None:
            os.environ["RANK"] = str(args.rank_override)
        else:
            os.environ.setdefault("RANK", "0")
        os.environ["LOCAL_RANK"] = os.environ["RANK"]  # map rank->GPU index
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    rank, world, local_rank = ddp_init()
    assert world == 2, "This script expects exactly 2 ranks (0=ResNet, 1=Swin)."
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    manifest_path = args.manifest or str(Path(args.shards_dir)/"manifest.json")
    manifest = load_json(manifest_path)

    # Get head dims from head.pt
    head_pt = torch.load(Path(args.shards_dir)/"head.pt", map_location="cpu")
    head_in_dim = head_pt["weight"].shape[1]

    # Choose stitch id (rank0 decides)
    if rank == 0:
        sid = choose_sid(args.stats, head_in_dim, criterion=args.criterion)
        print(f"[autoroute] chose sid={sid}")
    else:
        sid = -1
    obj = [sid]
    dist.broadcast_object_list(obj, src=0)
    sid = obj[0]

    if sid == 1:
        if rank == 0:
            print("[warn] sid=1 (pure Swin) not supported by this 2-rank pipeline. Choose sid in {2..7}.")
        dist.barrier(); return

    tap, entry = sid_to_tap_entry(sid)

    # AMP
    amp_ok = args.amp and device.startswith("cuda")
    autocast = torch.autocast("cuda", dtype=torch.float16) if amp_ok else nullcontext()

    # Rank 0: run ResNet tap and send the feature
    if rank == 0:
        transform = make_transform(args.img_size, cifar10=args.cifar10_transforms, center_crop=not args.no_center_crop)
        img = Image.open(args.image).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        tapper = ResNetTapFromShards(args.shards_dir, manifest, device=device)
        with torch.no_grad():
            if amp_ok:
                with autocast:
                    feat = tapper(x, tap=tap)
            else:
                feat = tapper(x, tap=tap)
        feat_cpu = feat.cpu().contiguous()
        dist.send(feat_cpu, dst=1)
        dist.barrier()

    # Rank 1: receive feature, run Swin tail + head
    elif rank == 1:
        recv_shape = (1,64,56,56) if tap=="C2" and entry=="S1B1" else ((1,64,56,56) if tap=="C2" else (1,128,28,28))
        buf = torch.empty(recv_shape, dtype=torch.float16 if amp_ok else torch.float32)
        dist.recv(buf, src=0)
        buf = buf.to(device)

        tail = SwinTailFromShards(args.shards_dir, manifest, entry, num_classes=head_pt["weight"].shape[0], device=device)
        with torch.no_grad():
            if amp_ok:
                with autocast:
                    logits = tail(buf)
            else:
                logits = tail(buf)

        probs = logits.softmax(-1).squeeze(0).cpu()
        vals, idxs = torch.topk(probs, k=min(5, probs.numel()), dim=-1)
        print(f"[using] sid={sid} ({tap} -> {entry})")
        for r in range(vals.numel()):
            print(f"Top-{r+1}: class={int(idxs[r])} p={float(vals[r]):.4f}")

        dist.barrier()

# small helper for cpu autocast noop
from contextlib import contextmanager
@contextmanager
def nullcontext():
    yield

if __name__ == "__main__":
    main()
