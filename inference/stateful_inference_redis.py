#!/usr/bin/env python3
"""
SN-Net stitched inference (module-glue) + targeted "exchange" summary + Redis I/O

Additions:
- Dynamic boundary discovery: hooks for ALL Swin blocks (stage1..4, all indices)
- Optional bridge/adapter discovery (cnn_to_vit/bridge/adapter/proj/token/merge/repatch)
- --redis-write to store each executed boundary's activation to Redis (key printed)
- --redis-read-first KEY to inject the FIRST boundary's input tensor from Redis
- --redis-start-key KEY to parse the boundary label from the key and inject AT THAT BOUNDARY
- --redis-url, --redis-prefix, --redis-ttl for connection & key mgmt
- --no-bridge-scan to disable bridge discovery (on by default)

New:
- FIGURE_SIMILARITY_THRESHOLD constant (used by external similarity logic; no similarity code here)
- Shallow reuse: given candidate Redis keys from a similar figure, inject the deepest shallow boundary
"""

import os, sys, argparse, json, warnings, io, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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

# ---- Figure similarity cutoff (used by an external checker; no similarity here) ----
# If an external similarity tool says "distance <= FIGURE_SIMILARITY_THRESHOLD * bits",
# we allow shallow-layer reuse by injecting a cached activation from Redis.
FIGURE_SIMILARITY_THRESHOLD = 0.10  # 10% of perceptual-hash bits (tune externally)

# ------------------ Redis bridge ------------------
class RedisBridge:
    """
    Minimal Redis I/O for tensors.
    - put_tensor(key, tensor)
    - get_tensor(key) -> tensor (CPU tensor)
    - key_for(sid, part, order)
    """
    def __init__(self, url: str, prefix: str, ttl: Optional[int]):
        self.enabled = False
        self.err = None
        self.prefix = prefix
        self.ttl = ttl
        try:
            import redis  # lazy requirement
            self._redis_mod = redis
            self.r = redis.from_url(url)
            try:
                self.r.ping()
                self.enabled = True
            except Exception as e:
                self.err = f"redis ping failed: {e}"
        except Exception as e:
            self.err = f"redis import/connect failed: {e}"
            self.r = None

    def key_for(self, sid: int, part: str, order: int) -> str:
        return f"{self.prefix}:sid{sid}:{part}:o{order}"

    def put_tensor(self, key: str, tensor: torch.Tensor) -> bool:
        if not self.enabled or self.r is None:
            return False
        buf = io.BytesIO()
        torch.save(tensor.detach().cpu(), buf)  # store as CPU tensor
        data = buf.getvalue()
        try:
            if self.ttl and self.ttl > 0:
                self.r.setex(key, self.ttl, data)
            else:
                self.r.set(key, data)
            return True
        except Exception:
            return False

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        if not self.enabled or self.r is None:
            return None
        try:
            data = self.r.get(key)
            if data is None:
                return None
            buf = io.BytesIO(data)
            t = torch.load(buf, map_location="cpu")
            if not isinstance(t, torch.Tensor):
                return None
            return t
        except Exception:
            return None

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

def strip_dataparallel(sd):
    if any(k.startswith("module.") for k in sd):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

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
        rkey = base + "layer1.0.conv1.weight"
        if rkey in sd and getattr(sd[rkey], "ndim", 0) == 4:
            kH, kW = sd[rkey].shape[-2], sd[rkey].shape[-1]
            if kH == 3 and kW == 3:
                picks.append("resnet18"); continue
            if kH == 1 and kW == 1:
                picks.append("resnet50"); continue
        pkey = base + "patch_embed.proj.weight"
        has_swin = any(k.startswith(base + "layers.") or k.startswith(base + "stages.") for k in sd.keys())
        if has_swin and pkey in sd and sd[pkey].shape[0] == 96:
            picks.append("swin_tiny_patch4_window7_224"); continue
    return picks

def anchors_from_ckpt_guess(sd):
    for prefix in ("anchors.", "trunk.anchors."):
        picks = _guess_under(sd, prefix)
        if picks:
            return picks
    return []

def infer_anchor_head_out_dim(sd):
    candidates = []
    for prefix in ("anchors.", "trunk.anchors."):
        for k, v in sd.items():
            if not isinstance(v, torch.Tensor): continue
            if not k.startswith(prefix): continue
            if k.endswith(".fc.weight") or k.endswith(".head.weight") or k.endswith(".classifier.weight"):
                if v.ndim == 2:
                    candidates.append(int(v.shape[0]))
    if not candidates:
        return None
    from collections import Counter
    return Counter(candidates).most_common(1)[0][0]

def infer_num_classes_from_ckpt(sd, meta_num_classes=None, default=10):
    if isinstance(meta_num_classes, int) and meta_num_classes > 0:
        return meta_num_classes
    anchor_nc = infer_anchor_head_out_dim(sd)
    if anchor_nc is not None:
        return anchor_nc
    candidates = []
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2: continue
        name = k.lower()
        if name.endswith("head.weight") or name.endswith("classifier.weight") or name.endswith(".fc.weight"):
            if ".mlp." in name or ".blocks." in name or "qkv" in name:
                continue
            candidates.append(int(v.shape[0]))
    if candidates:
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]
    return default

# ------------------ module gluing ------------------
def load_manifest(manifest_path: Path) -> Optional[dict]:
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None

def resolve_parts_for_sid(ckpt_dir: Path, sid: int, manifest: Optional[dict]):
    use_head = False
    anchor_names = None
    manifest_nc = None
    part_files: List[Path] = []

    if manifest:
        sid_map = manifest.get("sid_to_parts") or manifest.get("sid_map") or {}
        key = str(sid)
        if key in sid_map:
            names = sid_map[key]
            part_files = [ckpt_dir / n for n in names]
            use_head = manifest.get("use_head", {}).get(key, False) or ("head.pt" in names)
            anchor_names = manifest.get("anchors")
            manifest_nc = manifest.get("num_classes")
            return part_files, use_head, anchor_names, manifest_nc
        default_names = manifest.get("default_parts")
        if default_names:
            part_files = [ckpt_dir / n for n in default_names]
            use_head = "head.pt" in default_names or manifest.get("use_head_default", False)
            anchor_names = manifest.get("anchors")
            manifest_nc = manifest.get("num_classes")
            return part_files, use_head, anchor_names, manifest_nc

    mono = ckpt_dir / "snnet_trunk.pt"
    if mono.exists():
        use_head = (ckpt_dir / "head.pt").exists()
        return [mono], use_head, None, None

    all_parts = sorted(p for p in ckpt_dir.glob("*.pt") if p.name != "head.pt")
    use_head = (ckpt_dir / "head.pt").exists()
    return all_parts, use_head, None, None

def merge_state_dicts(files: List[Path]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for fp in files:
        obj = torch.load(fp, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
        else:
            sd = obj
        sd = strip_dataparallel(sd)
        for k, v in sd.items():
            merged[k] = v
    return merged

def split_trunk_head_state(sd):
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

def probe_output_dim(model, device, img_size):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, img_size, img_size, device=device)
        y = model(x)
    d = int(y.shape[-1])
    print(f"[probe] runtime output dim = {d}")
    return d

# ------------------ EXCHANGE PROBES ------------------
class ExchangeProbe:
    """Attach hooks only on boundary modules (the parts you ship), with optional Redis writes."""
    def __init__(self, device: str, redis: Optional[RedisBridge], sid: int, enable_write: bool):
        self.device = device
        self.records: List[Dict[str, Any]] = []
        self.order = 0
        self.redis = redis
        self.sid = sid
        self.enable_write = enable_write and (redis is not None) and redis.enabled

    def _tensor_bytes(self, t: torch.Tensor) -> int:
        return t.numel() * t.element_size()

    def _shape_str(self, t: torch.Tensor) -> str:
        return "x".join(str(int(d)) for d in t.shape)

    def _first_tensor(self, output) -> Optional[torch.Tensor]:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    return o
        return None

    def _make_hook(self, part_name: str, module_path: str):
        def hook(_module, _inputs, output):
            self.order += 1
            out_t = self._first_tensor(output)
            if out_t is not None:
                rec = dict(order=self.order, part=part_name, module=module_path,
                           shape=self._shape_str(out_t),
                           dtype=str(out_t.dtype).replace("torch.", ""),
                           bytes=self._tensor_bytes(out_t))
                if self.enable_write:
                    key = self.redis.key_for(self.sid, part_name, self.order)
                    ok = self.redis.put_tensor(key, out_t)
                    if ok:
                        rec["redis_key"] = key
                        print(f"[redis] wrote activation -> {key}")
                    else:
                        print(f"[redis] WARN: failed writing key for {part_name}")
            else:
                rec = dict(order=self.order, part=part_name, module=module_path,
                           shape="-", dtype="-", bytes=0)
            self.records.append(rec)
        return hook

def get_root_prefix(model: nn.Module) -> str:
    if isinstance(model, SNNet) or hasattr(model, "anchors"):
        return ""       # names: anchors.*
    if hasattr(model, "trunk") and isinstance(getattr(model, "trunk"), SNNet):
        return "trunk."
    if hasattr(model, "trunk") and hasattr(model.trunk, "anchors"):
        return "trunk."
    return ""

def _get_if_exists(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    cur = root
    for part in dotted.split("."):
        if not part:
            continue
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

# ---------- dynamic boundary discovery ----------
def _maybe(model: nn.Module, path: str) -> Optional[str]:
    return path if _get_if_exists(model, path) is not None else None

def _discover_bridge_module_paths(model: nn.Module) -> List[str]:
    kws = ("cnn_to_vit", "bridge", "adapter", "proj", "token", "merge", "repatch")
    picks = []
    for name, _m in model.named_modules():
        ln = name.lower()
        if any(k in ln for k in kws):
            picks.append(name)
    seen, out = set(), []
    for p in picks:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def build_boundary_map(model: nn.Module, include_bridges: bool = True) -> List[Tuple[str, str]]:
    base = get_root_prefix(model)
    parts: List[Tuple[str, str]] = []

    # ResNet anchor (index 0)
    for name in ["stem", "layer1", "layer2", "layer3", "layer4"]:
        p = _maybe(model, f"{base}anchors.0.{name}")
        if p: parts.append((f"resnet_{name}", p))

    # Swin anchor (index 1)
    pe = _maybe(model, f"{base}anchors.1.patch_embed")
    if pe: parts.append(("swin_patch_embed", pe))

    # Swin stages & blocks
    layers_root = _maybe(model, f"{base}anchors.1.layers") or _maybe(model, f"{base}anchors.1.stages")
    if layers_root:
        for si in range(4):
            stage_path = f"{layers_root}.{si}"
            if _get_if_exists(model, stage_path) is None:
                continue
            blocks_path = f"{stage_path}.blocks"
            blist = _get_if_exists(model, blocks_path)
            if blist is not None and hasattr(blist, "__len__"):
                for bi in range(len(blist)):
                    bp = f"{blocks_path}.{bi}"
                    if _get_if_exists(model, bp) is None:
                        continue
                    label = f"swin_stage{si+1}_b{bi}"
                    parts.append((label, bp))
        normp = _maybe(model, f"{base}anchors.1.norm")
        if normp:
            parts.append(("swin_tail_extras", normp))

    # Heads
    hp = _maybe(model, f"{base}head")
    if hp:
        parts.append(("head", hp))
    hp2 = _maybe(model, "anchors.1.head")
    if hp2:
        parts.append(("head_fallback", hp2))

    # Bridges
    if include_bridges:
        for path in _discover_bridge_module_paths(model):
            label = f"bridge_{path.replace('.', '_')}"
            parts.append((label, path))

    return parts

def parse_label_from_redis_key(key: str, expected_prefix: Optional[str] = None) -> Optional[str]:
    """
    Expected format: <prefix>:sid<sid>:<label>:o<order>
    Returns <label> or None.
    """
    try:
        parts = key.split(":")
        if len(parts) < 4:
            return None
        if expected_prefix and parts[0] != expected_prefix:
            return None
        return parts[2]
    except Exception:
        return None

def normalize_boundary_label(raw_label: str, label_to_module: Dict[str, nn.Module]) -> Optional[str]:
    """
    Normalize common/legacy labels to actual attached labels.
      - 'swin_stage4'      -> 'swin_stage4_b0'
      - 'swin_stage2_5'    -> 'swin_stage2_b5'
      - 'swin_stage2b5'    -> 'swin_stage2_b5'
      - exact match        -> itself
      - prefix fallback    -> first label starting with raw_label
    """
    if not raw_label:
        return None
    if raw_label in label_to_module:
        return raw_label

    m = re.fullmatch(r"(swin_stage[1-4])", raw_label)
    if m:
        cand = f"{m.group(1)}_b0"
        if cand in label_to_module:
            return cand

    m = re.fullmatch(r"(swin_stage[1-4])[_]?b?(\d+)", raw_label)
    if m:
        cand = f"{m.group(1)}_b{int(m.group(2))}"
        if cand in label_to_module:
            return cand

    for cand in ("head", "head_fallback", "swin_tail_extras", "swin_patch_embed"):
        if raw_label == cand and cand in label_to_module:
            return cand

    for cand in label_to_module.keys():
        if cand.startswith(raw_label):
            return cand

    return None

# ---- shallow reuse helpers ----
def shallow_reuse_labels(all_labels: List[str]) -> List[str]:
    """
    Which boundaries are 'shallow' and good candidates for reuse.
    Order matters: we will prefer the deepest shallow label available.
    """
    shallow = []
    for name in ["resnet_stem", "resnet_layer1"]:
        if name in all_labels:
            shallow.append(name)
    if "swin_patch_embed" in all_labels:
        shallow.append("swin_patch_embed")
    s1 = sorted([l for l in all_labels if l.startswith("swin_stage1_b")],
                key=lambda s: int(s.rsplit("b", 1)[-1]))
    shallow.extend(s1)
    return shallow

def pick_deepest_shallow_available(redis_keys: List[str],
                                   label_to_module: Dict[str, nn.Module],
                                   expected_prefix: Optional[str] = None) -> Optional[str]:
    """
    From a list of Redis keys (for a similar figure), pick the key whose label
    is the deepest shallow boundary. Return that key.
    Keys: <prefix>:sid<sid>:<label>:o<order>
    """
    all_labels = list(label_to_module.keys())
    shallow = shallow_reuse_labels(all_labels)
    if not shallow:
        return None

    label_to_key: Dict[str, str] = {}
    for k in redis_keys:
        lbl = parse_label_from_redis_key(k, expected_prefix=expected_prefix)
        if lbl:
            label_to_key[lbl] = k

    for lbl in reversed(shallow):  # deepest first
        if lbl in label_to_key:
            return label_to_key[lbl]
    return None

# ---------- injected tensor adapter ----------
def _adapt_injected_tensor_for_module(t: torch.Tensor, module: nn.Module, device: str) -> torch.Tensor:
    """
    Make a loaded tensor shape & dtype compatible with the boundary module.
    Target: Swin blocks expect (B, L, C).
    """
    try:
        p = next(module.parameters())
        dtype = p.dtype
    except StopIteration:
        dtype = torch.float32

    t = t.to(device=device, dtype=dtype)

    if t.ndim == 4 and t.shape[1] == 1:  # (B,1,L,C) -> (B,L,C)
        t = t.squeeze(1)

    expected_c = getattr(module, "dim", None)

    if t.ndim == 4:
        B = t.shape[0]
        if expected_c is not None and t.shape[-1] == expected_c:  # (B,H,W,C)
            L = int(t.shape[1] * t.shape[2])
            t = t.view(B, L, expected_c)
        elif expected_c is not None and t.shape[1] == expected_c:  # (B,C,H,W)
            L = int(t.shape[2] * t.shape[3])
            t = t.view(B, expected_c, L).transpose(1, 2)  # (B, L, C)
        else:
            C = t.shape[-1]
            L = int(torch.tensor(t.shape[1:-1]).prod().item())
            t = t.view(B, L, C)

    if t.ndim == 2:
        t = t.unsqueeze(0)

    if t.ndim > 3:
        B = t.shape[0]
        C = t.shape[-1] if expected_c is None else expected_c
        L = int(torch.tensor(t.shape[1:-1]).prod().item())
        t = t.view(B, L, C)

    if t.ndim == 3 and expected_c is not None:
        if t.shape[1] == expected_c and t.shape[2] != expected_c:  # (B, C, L)
            t = t.transpose(1, 2)  # -> (B, L, C)

    return t

def attach_exchange_probes(model: nn.Module,
                           redis: Optional[RedisBridge],
                           sid: int,
                           enable_write: bool,
                           include_bridges: bool = True,
                           start_at_label: Optional[str] = None
                           ) -> Tuple[ExchangeProbe, List[str], Optional[nn.Module], Optional[str], Dict[str, nn.Module]]:
    """
    Attach hooks to dynamically discovered boundaries.
    If start_at_label is provided, only attach hooks from that label onward.
    Returns: probe, attached_part_names, first_module_obj, first_part_name, label->module map
    """
    boundary_parts = build_boundary_map(model, include_bridges=include_bridges)

    # Build label map
    label_to_module: Dict[str, nn.Module] = {}
    for part_name, mod_path in boundary_parts:
        m = _get_if_exists(model, mod_path)
        if m is not None:
            label_to_module[part_name] = m

    # Normalize requested start label
    normalized_start: Optional[str] = None
    if start_at_label:
        normalized_start = normalize_boundary_label(start_at_label, label_to_module)
        if normalized_start:
            if normalized_start != start_at_label:
                print(f"[redis] start label normalized: '{start_at_label}' -> '{normalized_start}'")
        else:
            print(f"[redis] WARN: start label '{start_at_label}' not found; using full path from beginning.")

    # Filter parts
    filtered_parts: List[Tuple[str, str]] = []
    skipping = normalized_start is not None
    for part_name, mod_path in boundary_parts:
        if skipping:
            if part_name == normalized_start:
                skipping = False
                filtered_parts.append((part_name, mod_path))
        else:
            filtered_parts.append((part_name, mod_path))

    if not filtered_parts:
        filtered_parts = boundary_parts

    probe = ExchangeProbe(
        device="cuda" if next(model.parameters()).is_cuda else "cpu",
        redis=redis, sid=sid, enable_write=enable_write
    )

    attached_names: List[str] = []
    first_module: Optional[nn.Module] = None
    first_part_name: Optional[str] = None

    for part_name, mod_path in filtered_parts:
        m = _get_if_exists(model, mod_path)
        if m is None:
            continue
        h = probe._make_hook(part_name, mod_path)
        m.register_forward_hook(lambda mod, inp, out, h=h: h(mod, inp, out))
        attached_names.append(f"{part_name} -> {mod_path}")
        if first_module is None:
            first_module = m
            first_part_name = part_name

    if attached_names:
        print("[exchange] attached boundaries:")
        for s in attached_names:
            print("  -", s)
    else:
        print("[exchange] WARNING: no boundaries were attached (naming mismatch?)")

    return probe, [name.split(" -> ")[0] for name in attached_names], first_module, first_part_name, label_to_module

def summarize_exchanges(records: List[Dict[str, Any]]):
    recs = sorted(records, key=lambda r: r["order"])
    if len(recs) < 2:
        print("[exchange] Not enough boundaries executed to form handoffs.")
        return

    print("\n[exchange] ===== Handoffs (A -> B) =====")
    total = 0
    for i in range(len(recs) - 1):
        A, B = recs[i], recs[i + 1]
        bytes_AB = int(A["bytes"])
        total += bytes_AB
        extra = f"  [key: {A['redis_key']}]" if "redis_key" in A else ""
        print(f"[exchange] {A['part']} -> {B['part']} : {bytes_AB:,} B  ({A['shape']} {A['dtype']}){extra}")
    print(f"[exchange] ===== Total along executed chain: {total:,} B =====\n")

# ------------------ core loader ------------------
def load_from_parts_and_wrap(ckpt_dir: Path, device: str, sid: int, img_size: int,
                             cli_anchors: Optional[List[str]], cli_num_classes: int,
                             manifest: Optional[dict]):
    part_files, want_head, manifest_anchors, manifest_nc = resolve_parts_for_sid(ckpt_dir, sid, manifest)
    if not part_files:
        sys.exit("[error] No checkpoint parts were found to assemble a trunk.")

    print(f"[glue] SID {sid}: assembling trunk from {len(part_files)} parts")
    for p in part_files:
        print("       -", p.name)

    sd_full = merge_state_dicts(part_files)
    trunk_sd, head_sd_embedded = split_trunk_head_state(sd_full)

    anchors_names = cli_anchors or manifest_anchors
    if not anchors_names:
        auto = anchors_from_ckpt_guess(sd_full)
        anchors_names = auto if auto else ["resnet50", "swin_tiny_patch4_window7_224"]
        print(f"[auto] anchors: {anchors_names}" if auto else f"[fallback] anchors: {anchors_names}")

    stitched_head_present = ("head.weight" in sd_full) or ("head.bias" in sd_full)
    inferred_nc = infer_num_classes_from_ckpt(sd_full, manifest_nc, default=cli_anchors and cli_num_classes or 10)

    # Decide anchor head usage (only relevant if we *don't* end up using a stitched head)
    if stitched_head_present or want_head:
        anchor_num_classes = 0
        print("[path] stitched head present/required -> building headless anchors")
    else:
        if sid in (0, 1):
            anchor_num_classes = infer_anchor_head_out_dim(sd_full) or (manifest_nc or cli_num_classes)
            print(f"[path] no stitched head; sid={sid} -> using ANCHOR HEADS (num_classes={anchor_num_classes})")
        else:
            anchor_num_classes = 0
            print("[path] no stitched head; stitched path -> features only (no classifier)")

    # Build trunk
    anchors = build_anchors(anchors_names, pretrained=False, num_classes=anchor_num_classes)
    trunk = SNNet(anchors, cnn_to_vit=need_cnn_to_vit(anchors_names)).to(device).eval()

    if sid not in trunk.stitch_configs:
        print("Available stitch IDs:")
        for k, v in trunk.stitch_configs.items():
            print(k, v)
        sys.exit(f"[error] Invalid stitch_id {sid}")
    trunk.reset_stitch_id(sid)

    missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
    print(f"[load] trunk missing={len(missing)} unexpected={len(unexpected)}")

    # stitched head candidates
    head_sd_external = {}
    head_file = ckpt_dir / "head.pt"
    if head_file.exists() and want_head and not stitched_head_present:
        obj = torch.load(head_file, map_location="cpu")
        head_sd_external = obj.get("state_dict", obj) if isinstance(obj, dict) else {}
        print("[info] external stitched head.pt found and requested")

    final_head_sd = None
    if stitched_head_present:
        final_head_sd = {k: v for k, v in sd_full.items() if k.startswith("head.")}
    elif head_sd_external:
        final_head_sd = {k: v for k, v in head_sd_external.items() if k.startswith("head.")}

    # --- Robust head/feature compatibility check ---
    if final_head_sd and "head.weight" in final_head_sd:
        stitched_in = int(final_head_sd["head.weight"].shape[1])
        # Probe feature dim produced by this SID
        feat_dim = probe_output_dim(trunk, device, img_size)

        if stitched_in != feat_dim:
            print(f"[path] stitched head in_dim={stitched_in} != trunk feat_dim={feat_dim} for sid={sid} -> disabling stitched head.")
            final_head_sd = None
            # If we can, fall back to anchor heads (SID 0/1)
            if sid in (0, 1):
                anchor_num_classes = infer_anchor_head_out_dim(sd_full) or (manifest_nc or cli_num_classes)
                anchors = build_anchors(anchors_names, pretrained=False, num_classes=anchor_num_classes)
                trunk = SNNet(anchors, cnn_to_vit=need_cnn_to_vit(anchors_names)).to(device).eval()
                trunk.reset_stitch_id(sid)
                missing, unexpected = trunk.load_state_dict(trunk_sd, strict=False)
                print(f"[load] trunk (anchor-heads) missing={len(missing)} unexpected={len(missing)}")
                return trunk, anchor_num_classes
            else:
                print("[warn] stitched path has no compatible classifier head; "
                      "provide a matching head.pt or use --sid 0/1 for anchor heads.")
                return trunk, feat_dim

    # If we still have a compatible stitched head, wrap it
    if final_head_sd:
        W = final_head_sd.get("head.weight")
        if W is None:
            print("[warn] head markers found but 'head.weight' missing; falling back to trunk-only.")
        else:
            out_dim, in_dim = int(W.shape[0]), int(W.shape[1])
            head = nn.Linear(in_dim, out_dim, bias=("head.bias" in final_head_sd)).to(device)

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
            for k, v in final_head_sd.items():
                wrap_sd[k] = v

            m_missing, m_unexp = model.load_state_dict(wrap_sd, strict=False)
            print(f"[load] stitched head loaded: out_dim={out_dim}  missing={len(m_missing)} unexpected={len(m_unexp)}")
            return model, out_dim

    # If we didn't return yet, either we have anchor heads (sid 0/1) or feature-only stitched path
    if anchor_num_classes > 0 and sid in (0, 1):
        return trunk, anchor_num_classes

    out_dim = probe_output_dim(trunk, device, img_size)
    if sid >= 2:
        print("[warn] stitched path has no classifier head; "
              "use --sid 0/1 for anchor heads or provide head.pt / manifest to load a stitched head.")
    return trunk, out_dim

# ------------------ inference ------------------
@torch.no_grad()
def infer_and_exchange(model, transform, image_path, device,
                       exchange_only=False, io_csv: Optional[str]=None,
                       redis: Optional[RedisBridge]=None, sid: int=0,
                       enable_redis_write: bool=False,
                       redis_read_first_key: Optional[str]=None,
                       redis_start_key: Optional[str]=None,
                       redis_prefix: Optional[str]=None,
                       include_bridges: bool = True,
                       # NEW: pass candidate Redis keys from a *similar* figure (external decision)
                       redis_candidate_reuse_keys: Optional[List[str]] = None):
    # Parse a start label from --redis-start-key, if provided
    start_label = None
    parsed_label = None
    if redis_start_key:
        parsed_label = parse_label_from_redis_key(redis_start_key, expected_prefix=redis_prefix)
        if parsed_label is None:
            print(f"[redis] WARN: could not parse label from key: {redis_start_key}")
        else:
            start_label = parsed_label
            print(f"[redis] will start at boundary label parsed from key: '{start_label}'")

    # Attach probes (optionally from start_label onward)
    probe, attached_parts, first_module, first_part_name, label_to_module = attach_exchange_probes(
        model, redis=redis, sid=sid, enable_write=enable_redis_write,
        include_bridges=include_bridges, start_at_label=start_label
    )

    # Choose injection point
    inject_key = None
    inject_module = None
    inject_label = None

    if redis_start_key and parsed_label:
        if attached_parts:
            normalized_first = attached_parts[0]
            inject_label = normalized_first
            inject_key = redis_start_key
            inject_module = label_to_module.get(normalized_first)
    elif redis_read_first_key and first_module is not None:
        inject_key = redis_read_first_key
        inject_module = first_module
        inject_label = first_part_name

    # ---- Shallow reuse (EXTERNAL similarity said: "similar") ----
    # If caller supplies candidate keys from a similar figure, reuse the deepest shallow boundary.
    if (inject_key is None and redis_candidate_reuse_keys and
            redis is not None and redis.enabled):
        chosen = pick_deepest_shallow_available(redis_candidate_reuse_keys,
                                                label_to_module,
                                                expected_prefix=redis_prefix)
        if chosen:
            parsed = parse_label_from_redis_key(chosen, expected_prefix=redis_prefix)
            if parsed:
                # Re-attach starting from this boundary to log/write from here onward
                print(f"[reuse] injecting cached shallow boundary '{parsed}' from similar figure")
                probe, attached_parts, first_module, first_part_name, label_to_module = attach_exchange_probes(
                    model, redis=redis, sid=sid, enable_write=enable_redis_write,
                    include_bridges=include_bridges, start_at_label=parsed
                )
                inject_key = chosen
                inject_label = parsed
                inject_module = label_to_module.get(parsed)
            else:
                print(f"[reuse] WARN: cannot parse label from reuse key: {chosen}")

    # Install pre-hook (if any injection decided)
    if inject_key and inject_module is not None and redis is not None and redis.enabled:
        def _pre_inject(_module, inputs):
            t = redis.get_tensor(inject_key)
            if t is None:
                print(f"[redis] WARN: key not found -> {inject_key}; proceeding without injection.")
                return None
            t = _adapt_injected_tensor_for_module(t, inject_module, device)
            print(f"[redis] injected tensor at boundary '{inject_label}' from key: {inject_key}  shape={tuple(t.shape)}")
            return (t,)
        inject_module.register_forward_pre_hook(lambda mod, inp: _pre_inject(mod, inp))
    elif redis_start_key:
        print("[redis] WARN: cannot inject at parsed boundary (label missing or redis unavailable).")
    elif redis_read_first_key:
        print("[redis] WARN: cannot set read-first hook (no first boundary or redis unavailable).")

    # Forward (image still used to drive the graph; injected module overrides its input)
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)

    summarize_exchanges(probe.records)

    if io_csv:
        import csv
        cols = ["order", "part", "module", "shape", "dtype", "bytes", "redis_key"]
        with open(io_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(probe.records, key=lambda r: r["order"]):
                row = {k: r.get(k, "") for k in cols}
                w.writerow(row)
        print(f"[exchange] CSV saved: {io_csv}")

    if exchange_only:
        return None, None

    probs = logits.softmax(-1)
    k = min(5, probs.shape[-1])
    top = torch.topk(probs, k=k, dim=-1)
    return top.values.squeeze(0).cpu(), top.indices.squeeze(0).cpu()

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser(description="SNNet stitched inference (glue from parts) + exchange summary + Redis I/O")
    ap.add_argument("--checkpoints-dir", default="checkpoints", help="Directory with *.pt parts and optional manifest.json")
    ap.add_argument("--sid", type=int, default=7, help="Stitch config ID")
    ap.add_argument("--anchors", nargs="+", default=None, help="Anchor names (override auto/manifest)")
    ap.add_argument("--image", required=True, help="Path to RGB image")
    ap.add_argument("--num-classes", type=int, default=10, help="Expected classes (fallback only)")
    ap.add_argument("--img-size", type=int, default=224, help="Transform size")
    ap.add_argument("--cifar10-transforms", action="store_true", default=False, help="Use CIFAR-10 mean/std")
    ap.add_argument("--no-center-crop", action="store_true", help="Disable center crop")
    ap.add_argument("--cpu", action="store_true")

    # Exchange / logging
    ap.add_argument("--exchange-only", action="store_true", help="Print only part-to-part exchange summary (no per-leaf spam)")
    ap.add_argument("--io-csv", type=str, default=None, help="Optional CSV of boundary records")

    # Redis options
    ap.add_argument("--redis-write", action="store_true", help="Write each boundary activation to Redis")
    ap.add_argument("--redis-read-first", type=str, default=None, help="Read & inject FIRST boundary input tensor from this Redis KEY")
    ap.add_argument("--redis-start-key", type=str, default=None, help="Read & inject tensor at the boundary encoded in this Redis KEY")
    ap.add_argument("--redis-url", type=str, default="redis://localhost:6379/0", help="Redis URL")
    ap.add_argument("--redis-prefix", type=str, default="snex", help="Redis key prefix")
    ap.add_argument("--redis-ttl", type=int, default=0, help="TTL seconds for keys (0 = no expiry)")

    # Boundary discovery
    ap.add_argument("--no-bridge-scan", action="store_true",
                    help="Disable automatic scan for adapter/bridge modules")

    # Optional: accept reuse keys from a file (JSON array). Keep optional.
    ap.add_argument("--reuse-keys-file", type=str, default=None,
                    help="JSON file with an array of Redis keys from a similar figure to enable shallow reuse")

    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    ckpt_dir = Path(args.checkpoints_dir)
    if not ckpt_dir.exists():
        sys.exit(f"[error] checkpoints dir not found: {ckpt_dir}")

    manifest = load_manifest(ckpt_dir / "manifest.json")
    if manifest:
        print(f"[info] manifest loaded with keys: {list(manifest.keys())}")

    model, out_dim = load_from_parts_and_wrap(
        ckpt_dir=ckpt_dir,
        device=device,
        sid=args.sid,
        img_size=args.img_size,
        cli_anchors=args.anchors,
        cli_num_classes=args.num_classes,
        manifest=manifest,
    )

    transform = make_transform(
        args.img_size,
        use_cifar10=args.cifar10_transforms,
        center_crop=not args.no_center_crop,
    )

    # Build Redis bridge (non-fatal if unavailable)
    redis = None
    if args.redis_write or args.redis_read_first or args.redis_start_key or args.reuse_keys_file:
        redis = RedisBridge(url=args.redis_url, prefix=args.redis_prefix,
                            ttl=(args.redis_ttl if args.redis_ttl > 0 else None))
        if not redis.enabled:
            print(f"[redis] NOTE: Redis disabled ({self.err if hasattr(redis,'err') else 'unknown error'}); continuing without Redis.")

    # Optionally load candidate reuse keys (external similarity decision)
    candidate_keys_from_similar = None
    if args.reuse_keys_file:
        try:
            with open(args.reuse_keys_file, "r") as f:
                candidate_keys_from_similar = json.load(f)
            if not isinstance(candidate_keys_from_similar, list):
                print("[reuse] WARN: --reuse-keys-file must contain a JSON array of strings; ignoring.")
                candidate_keys_from_similar = None
            else:
                print(f"[reuse] loaded {len(candidate_keys_from_similar)} candidate keys for shallow reuse")
        except Exception as e:
            print(f"[reuse] WARN: could not read --reuse-keys-file: {e}")

    probs, idxs = infer_and_exchange(
        model, transform, args.image, device,
        exchange_only=args.exchange_only, io_csv=args.io_csv,
        redis=redis, sid=args.sid,
        enable_redis_write=args.redis_write,
        redis_read_first_key=args.redis_read_first,
        redis_start_key=args.redis_start_key,
        redis_prefix=args.redis_prefix,
        include_bridges=not args.no_bridge_scan,
        redis_candidate_reuse_keys=candidate_keys_from_similar
    )

    if probs is not None:
        labels = CIFAR10_LABELS if out_dim == 10 else None
        for r in range(len(probs)):
            cls_id = int(idxs[r].item())
            name = labels[cls_id] if labels and cls_id < len(labels) else str(cls_id)
            print(f"Top-{r+1}: {name}  p={float(probs[r].item()):.4f}")

if __name__ == "__main__":
    main()
