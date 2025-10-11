# =========================
# Ray Serve CIFAR-10 pipeline (no timm)
# =========================
import os, io, base64
from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.models as tv
import torchvision.transforms as T
from torchvision.models import ResNet18_Weights
from PIL import Image

from ray import serve
from torch.cuda.amp import autocast

# ---------- config via env ----------
CKPT_PATH = os.getenv("CKPT_PATH", "")  # optional: path to your checkpoint
USE_IMAGENET_PRETRAINED = os.getenv("RESNET_IMAGENET", "1") == "1"  # use torchvision pretrained ResNet18
FREEZE_RESC3 = os.getenv("FREEZE_RESC3", "1") == "1"  # freeze ResC3 when using pretrained
DEVICE_GPU = torch.cuda.is_available()
CPU, GPU = "cpu", "cuda"
CPU_DTYPE, GPU_DTYPE = torch.float32, torch.float16

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ---------- CIFAR-10 ----------
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

_preprocess = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def preprocess_batch(pils: List[Image.Image]) -> torch.Tensor:
    xs = [_preprocess(p.convert("RGB")) for p in pils]
    x = torch.stack(xs, dim=0)  # [N,3,32,32]
    return x.to(dtype=CPU_DTYPE, device=CPU).contiguous()


# ---------- utils ----------
def load_prefixed(module: nn.Module, ckpt_path: str, prefix: str) -> None:
    """
    Load a sub-state-dict from `ckpt_path` into `module` using keys that start with `prefix`.
    Example expected ckpt keys: 'adapter.0.weight', 'head.weight', etc.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print(f"[load_prefixed] skip: ckpt not found: {ckpt_path}")
        return
    sd_all = torch.load(ckpt_path, map_location="cpu")
    # accept either raw dict or {state_dict: ...}
    if isinstance(sd_all, dict) and "state_dict" in sd_all:
        sd_all = sd_all["state_dict"]

    sub = {}
    for k, v in sd_all.items():
        if k.startswith(prefix):
            sub[k[len(prefix):]] = v
    if not sub:
        print(f"[load_prefixed] no keys with prefix '{prefix}' in {ckpt_path}")
        return
    missing, unexpected = module.load_state_dict(sub, strict=False)
    print(f"[load_prefixed] loaded '{prefix}' from {ckpt_path} (missing={len(missing)} unexpected={len(unexpected)})")


# =========================
# Serve deployments
# =========================

@serve.deployment
class Ingress:
    """
    HTTP entrypoint.
    - multipart/form-data: one or multiple 'file'
    - application/json: {'b64': '<...>'} or {'b64_batch': ['<...>', ...]}
    Fallback: random batch with ?b=...
    """

    def __init__(self, next_handle):
        self.next = next_handle

    async def __call__(self, request):
        x: Optional[torch.Tensor] = None
        ct = (request.headers.get("content-type", "") or "").lower()

        try:
            if "multipart/form-data" in ct:
                form = await request.form()
                files = []
                if hasattr(form, "getlist"): files = form.getlist("file")
                if not files and form.get("file") is not None: files = [form.get("file")]
                if files:
                    pils = []
                    for f in files:
                        data = await f.read()
                        pils.append(Image.open(io.BytesIO(data)))
                    x = preprocess_batch(pils)

            if x is None and "application/json" in ct:
                payload = await request.json()
                if "b64_batch" in payload and isinstance(payload["b64_batch"], list):
                    pils = []
                    for b64s in payload["b64_batch"]:
                        data = base64.b64decode(b64s)
                        pils.append(Image.open(io.BytesIO(data)))
                    x = preprocess_batch(pils)
                elif "b64" in payload:
                    data = base64.b64decode(payload["b64"])
                    pil = Image.open(io.BytesIO(data))
                    x = preprocess_batch([pil])
        except Exception as e:
            print(f"[Ingress] parse error: {e}", flush=True)

        if x is None:
            b = int(request.query_params.get("b", "1"))
            x = torch.randn(b, 3, 32, 32, dtype=CPU_DTYPE, device=CPU).contiguous()

        print(f"[Ingress] out: {tuple(x.shape)} {x.dtype}", flush=True)
        return await self.next.remote(x)


@serve.deployment
class ResC3:
    """
    ResNet18 front-half (stem → layer1 → layer2) on CPU.
    32x32 -> conv7x7 s=2 -> 16 -> maxpool s=2 -> 8 -> layer1 -> 8 -> layer2 s=2 -> 4
    Output: [N,128,4,4]
    """

    def __init__(self, next_handle):
        base = tv.resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if USE_IMAGENET_PRETRAINED else None
        )
        self.m = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2
        ).eval().to(CPU).to(CPU_DTYPE)

        # optionally freeze when using pretrained
        if USE_IMAGENET_PRETRAINED and FREEZE_RESC3:
            for p in self.m.parameters(): p.requires_grad = False

        # optional: load resc3.* from your checkpoint
        if CKPT_PATH:
            load_prefixed(self.m, CKPT_PATH, "resc3.")

        self.next = next_handle

    async def __call__(self, x_t: torch.Tensor):
        with torch.no_grad():
            assert isinstance(x_t, torch.Tensor), "[ResC3] expected torch.Tensor"
            x = x_t.to(dtype=CPU_DTYPE, device=CPU, non_blocking=True)
            c3 = self.m(x).contiguous()  # [N,128,4,4]
            print(f"[ResC3] out: {tuple(c3.shape)} {c3.dtype}", flush=True)
        return await self.next.remote(c3)


@serve.deployment(max_ongoing_requests=1)
class AdapterC3toE384Dep:
    """
    GPU adapter ('stitch'): [N,128,H,W] -> [N,384,H/2,W/2]
    With CIFAR path (H=W=4) -> [N,384,2,2]
    """

    def __init__(self, next_handle):
        self.m = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(192, 384, 1),
        ).eval().to(GPU if DEVICE_GPU else CPU).to(GPU_DTYPE if DEVICE_GPU else CPU_DTYPE)

        # optional: load adapter.* from your checkpoint
        if CKPT_PATH:
            load_prefixed(self.m.float() if not DEVICE_GPU else self.m, CKPT_PATH, "adapter.")

        self.next = next_handle

    async def __call__(self, c3_t: torch.Tensor):
        assert isinstance(c3_t, torch.Tensor) and c3_t.ndim == 4 and c3_t.shape[1] == 128, \
            f"[Adapter] expected [N,128,H,W], got {getattr(c3_t, 'shape', None)}"
        with torch.no_grad():
            if DEVICE_GPU and torch.cuda.is_available():
                with autocast(dtype=GPU_DTYPE):
                    c3 = c3_t.to(device=GPU, dtype=torch.float32, non_blocking=True)
                    e = self.m(c3).contiguous()  # GPU
                e_cpu = e.detach().to(CPU, non_blocking=True).contiguous()
            else:
                c3 = c3_t.to(device=CPU, dtype=CPU_DTYPE, non_blocking=True)
                e = self.m.float().to(CPU)(c3).contiguous()
                e_cpu = e
            print(f"[Adapter] out: {tuple(e_cpu.shape)} {e_cpu.dtype}", flush=True)
        return await self.next.remote(e_cpu)


@serve.deployment(max_ongoing_requests=1)
class SwinStage2Block3Dep:
    """
    Placeholder 'transformer' stage. Currently identity; replace with real block if you have one.
    Contract: [N,384,H,W] -> [N,384,H,W]
    """

    def __init__(self, next_handle):
        self.block = nn.Identity().eval().to(GPU if DEVICE_GPU else CPU).to(GPU_DTYPE if DEVICE_GPU else CPU_DTYPE)

        # optional: load swin.* from your checkpoint (if you eventually add a real module)
        if CKPT_PATH:
            load_prefixed(self.block, CKPT_PATH, "swin.")

        self.next = next_handle

    async def __call__(self, e_t: torch.Tensor):
        assert isinstance(e_t, torch.Tensor) and e_t.ndim == 4 and e_t.shape[1] == 384, \
            f"[Swin] expected [N,384,H,W], got {getattr(e_t, 'shape', None)}"
        with torch.no_grad():
            if DEVICE_GPU and torch.cuda.is_available():
                with autocast(dtype=GPU_DTYPE):
                    e = e_t.to(device=GPU, dtype=torch.float32, non_blocking=True)
                    y = self.block(e).contiguous()
                y_cpu = y.detach().to(CPU, non_blocking=True).contiguous()
            else:
                e = e_t.to(device=CPU, dtype=CPU_DTYPE, non_blocking=True)
                y = self.block.float().to(CPU)(e).contiguous()
                y_cpu = y
            print(f"[Swin] out: {tuple(y_cpu.shape)} {y_cpu.dtype}", flush=True)
        return await self.next.remote(y_cpu)


@serve.deployment
class Head:
    """
    CIFAR-10 head: GAP → Linear(384->10). Returns top-5 with labels.
    """

    def __init__(self):
        self.m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, 10),
        ).eval().to(CPU).to(CPU_DTYPE)

        # optional: load head.* from your checkpoint
        if CKPT_PATH:
            load_prefixed(self.m, CKPT_PATH, "head.")

    async def __call__(self, y_t: torch.Tensor):
        with torch.no_grad():
            assert isinstance(y_t, torch.Tensor) and y_t.ndim == 4 and y_t.shape[1] == 384, \
                f"[Head] expected [N,384,*,*], got {getattr(y_t, 'shape', None)}"
            y = y_t.to(dtype=CPU_DTYPE, device=CPU, non_blocking=True)
            logits = self.m(y).detach().cpu().contiguous()  # [N,10]
            probs = torch.softmax(logits.float(), dim=-1)  # [N,10]
            top_p, top_i = probs.topk(k=min(5, probs.shape[-1]), dim=-1)

            batch_top5 = []
            for p_row, i_row in zip(top_p, top_i):
                items = []
                for p, i in zip(p_row.tolist(), i_row.tolist()):
                    label = CIFAR10_LABELS[i] if 0 <= i < 10 else f"class_{i}"
                    items.append({"index": int(i), "label": label, "prob": float(p)})
                batch_top5.append(items)

            print(f"[Head] out: logits={tuple(logits.shape)} top5", flush=True)
        return {"top5": batch_top5, "labels": CIFAR10_LABELS}


# ---------- build DAG ----------
def build_app():
    head = Head.options(
        ray_actor_options={"resources": {"node:front": 0.01}, "num_gpus": 0}
    ).bind()

    swin = SwinStage2Block3Dep.options(
        ray_actor_options={"num_gpus": 0.5 if DEVICE_GPU else 0}
    ).bind(head)

    adapter = AdapterC3toE384Dep.options(
        ray_actor_options={"num_gpus": 0.5 if DEVICE_GPU else 0}
    ).bind(swin)

    resc3 = ResC3.options(
        ray_actor_options={"resources": {"node:front": 0.01}, "num_gpus": 0}
    ).bind(adapter)

    ingress = Ingress.options(
        ray_actor_options={"resources": {"node:front": 0.01}, "num_gpus": 0}
    ).bind(resc3)

    return ingress


graph = build_app()
