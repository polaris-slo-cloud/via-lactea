#!/usr/bin/env python3
import re, json, argparse, torch
from pathlib import Path
from collections import Counter

# ---------- utilities ----------
def pick(sd, patterns):
    rx = re.compile("|".join([f"^(?:{p})" for p in patterns]))
    return {k: v for k, v in sd.items() if rx.match(k)}

def sample_keys(sd, prefix="", n=50):
    out = [k for k in sd.keys() if k.startswith(prefix)]
    return out[:n]

def find_anchors_prefix(sd):
    """
    Find the common prefix before 'anchors.<idx>.'.
    E.g., '', 'trunk.', 'module.trunk.', 'model.', etc.
    """
    m = re.compile(r"^(.*?)(anchors\.\d+\.)")
    prefixes = []
    for k in sd.keys():
        mo = m.match(k)
        if mo:
            prefixes.append(mo.group(1))
    if not prefixes:
        return ""  # fall back
    # choose most common shortest prefix
    counts = Counter(prefixes)
    pref = min([p for p,_ in counts.most_common()], key=len)
    return pref

def list_anchor_indices(sd, anchors_prefix):
    rx = re.compile(rf"^{re.escape(anchors_prefix)}anchors\.(\d+)\.")
    idxs = set()
    for k in sd.keys():
        m = rx.match(k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)

def detect_backbone_kind(sd, base):
    """Return 'resnet' or 'swin' or 'unknown' for a given anchors.<i>. base."""
    # Heuristics
    has_res = any(sd_k.startswith(base + p) for sd_k in sd.keys()
                  for p in ("layer1.", "layer2.", "layer3.", "layer4.", "conv1.", "bn1."))
    has_patch = any(k.startswith(base + "patch_embed.") for k in sd.keys())
    has_layers = any(k.startswith(base + p) for k in sd.keys() for p in ("layers.", "stages.", "stage1.", "stage2.", "stage3.", "stage4."))
    if has_patch or has_layers:
        return "swin"
    if has_res:
        return "resnet"
    return "unknown"

def detect_swin_roots(sd, swin_base):
    """
    Returns {'style': <style>, 'stage_map': {'stage1': pref, ...}}
    where style in {'layers','stages','stageN','backbone.layers','encoder.layers',...}
    """
    candidates = [
        "layers.", "stages.", "stage1.",
        "backbone.layers.", "backbone.stages.",
        "encoder.layers.", "encoder.stages.",
        "module.layers.", "module.stages."
    ]
    for cand in candidates:
        root = swin_base + cand
        if any(k.startswith(root) for k in sd.keys()):
            stage_map = {}
            if "layers" in cand or "stages" in cand or "module.layers" in cand or "encoder.layers" in cand or "backbone.layers" in cand:
                for n in range(4):
                    pref = f"{root}{n}."
                    if any(k.startswith(pref) for k in sd.keys()):
                        stage_map[f"stage{n+1}"] = pref
            else:
                # stageN.* format
                found = False
                for s in (1,2,3,4):
                    pref = swin_base + f"stage{s}."
                    if any(k.startswith(pref) for k in sd.keys()):
                        stage_map[f"stage{s}"] = pref
                        found = True
                if not found:
                    continue
            return {"style": cand.rstrip("."), "stage_map": stage_map}
    return {"style": None, "stage_map": {}}

def save_shard(outdir, name, d, manifest):
    if not d: return
    p = Path(outdir) / f"{name}.pt"
    torch.save(d, p)
    manifest["shards"][name] = {"count": len(d), "path": str(p)}

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Split SN-Net stitched checkpoint into shard files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--granularity", choices=["stage","block"], default="block",
                    help="split Swin by whole stages or by individual blocks in stage1/2")
    ap.add_argument("--keep-trunk", action="store_true",
                    help="write trunk.* to a separate shard (snnet_trunk.pt)")
    ap.add_argument("--resnet-index", dest="resnet_index", type=int, default=None)
    ap.add_argument("--swin-index", dest="swin_index", type=int, default=None)
    ap.add_argument("--dump-keys", dest="dump_keys", type=int, default=0,
                    help="print N state_dict keys (debug) and exit")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    if args.dump_keys > 0:
        print("\n".join(list(sd.keys())[:args.dump_keys]))
        return

    # 1) Find anchors prefix ('' or 'trunk.' or 'module.trunk.' etc.)
    anchors_prefix = find_anchors_prefix(sd)  # may be ''
    # 2) List anchor indices under that prefix
    idxs = list_anchor_indices(sd, anchors_prefix)
    if not idxs:
        # show some keys so user can see what's inside
        keys = "\n".join(sample_keys(sd, "", 80))
        raise RuntimeError(f"No 'anchors.<i>.' found (prefix='{anchors_prefix}'). "
                           f"First keys:\n{keys}")

    # 3) Detect resnet/swin indices
    res_idx = args.resnet_index
    swin_idx = args.swin_index
    if res_idx is None or swin_idx is None:
        kinds = {}
        for i in idxs:
            base = f"{anchors_prefix}anchors.{i}."
            kinds[i] = detect_backbone_kind(sd, base)
        # choose by kind
        if res_idx is None:
            cands = [i for i,k in kinds.items() if k == "resnet"]
            if cands: res_idx = cands[0]
        if swin_idx is None:
            cands = [i for i,k in kinds.items() if k == "swin"]
            if cands: swin_idx = cands[0]
    if res_idx is None or swin_idx is None:
        raise RuntimeError(f"Could not detect resnet/swin anchors under prefix '{anchors_prefix}'. "
                           f"indices={idxs}, kinds={ {i:detect_backbone_kind(sd, f'{anchors_prefix}anchors.{i}.') for i in idxs} }")

    res_base  = f"{anchors_prefix}anchors.{res_idx}."
    swin_base = f"{anchors_prefix}anchors.{swin_idx}."

    # 4) Detect Swin stage roots
    swin_roots = detect_swin_roots(sd, swin_base)
    if not swin_roots["style"]:
        ex = "\n".join(sample_keys(sd, swin_base, 80))
        raise RuntimeError("Cannot find Swin stages under "
                           f"{swin_base}{{layers.|stages.|stage1.|backbone.layers.|encoder.layers.|module.layers.}}.\n"
                           f"Example keys under {swin_base}:\n{ex}")

    manifest = {"shards": {}, "anchors": {
        "prefix": anchors_prefix,
        "resnet_idx": res_idx,
        "swin_idx": swin_idx,
        "swin_style": swin_roots["style"]
    }, "notes": []}

    # ---------- ResNet shards ----------
    save_shard(outdir, "resnet_stem",   pick(sd, [rf"{re.escape(res_base)}conv1\.", rf"{re.escape(res_base)}bn1\."]), manifest)
    save_shard(outdir, "resnet_layer1", pick(sd, [rf"{re.escape(res_base)}layer1\."]), manifest)
    save_shard(outdir, "resnet_layer2", pick(sd, [rf"{re.escape(res_base)}layer2\."]), manifest)
    save_shard(outdir, "resnet_layer3", pick(sd, [rf"{re.escape(res_base)}layer3\."]), manifest)
    save_shard(outdir, "resnet_layer4", pick(sd, [rf"{re.escape(res_base)}layer4\."]), manifest)

    # ---------- Swin shards ----------
    save_shard(outdir, "swin_patch_embed", pick(sd, [rf"{re.escape(swin_base)}patch_embed\."]), manifest)

    stage_map = swin_roots["stage_map"]  # e.g., 'stage1' -> '...layers.0.'
    def blocks_prefix(stage_name):
        pref = stage_map.get(stage_name, "")
        return pref + "blocks." if pref else None

    if args.granularity == "stage":
        for s in [1,2,3,4]:
            stg = f"stage{s}"
            pref = stage_map.get(stg, None)
            if pref:
                save_shard(outdir, f"swin_{stg}", pick(sd, [rf"{re.escape(pref)}"]), manifest)
    else:
        # stage1/2 block-level, stage3/4 full
        for stg, max_try in [("stage1", 16), ("stage2", 16)]:
            bpref = blocks_prefix(stg)
            if not bpref:
                manifest["notes"].append(f"No blocks found for {stg}")
                continue
            for b in range(max_try):
                pat = rf"{re.escape(bpref)}{b}\."
                d = pick(sd, [pat])
                if d:
                    save_shard(outdir, f"swin_{stg}_b{b}", d, manifest)
        for s in [3,4]:
            stg = f"stage{s}"
            pref = stage_map.get(stg, None)
            if pref:
                save_shard(outdir, f"swin_{stg}", pick(sd, [rf"{re.escape(pref)}"]), manifest)

    # Swin extras (norm/head inside Swin backbone)
    save_shard(outdir, "swin_tail_extras", pick(sd, [rf"{re.escape(swin_base)}norm\.", rf"{re.escape(swin_base)}head\."]), manifest)

    # ---------- Classification head (outside anchors) ----------
    # Prefer exact 'head.' at root; also accept 'module.head.' etc., but exclude Swin's internal head
    head_global = {}
    for k, v in sd.items():
        if k.endswith("head.weight") or k.endswith("head.bias"):
            # exclude Swin-internal heads
            if k.startswith(swin_base):
                continue
            # keep only top-level or prefixed variants (module./model.)
            if re.match(r"^(?:module\.|model\.)?head\.(?:weight|bias)$", k):
                head_global[k.split(".",1)[1]] = v  # store as 'weight'/'bias'
    if head_global:
        torch.save(head_global, Path(outdir) / "head.pt")
        manifest["shards"]["head"] = {"count": len(head_global), "path": str(Path(outdir)/"head.pt")}

    # ---------- trunk adapters ----------
    if args.keep_trunk:
        trunk_sd = {}
        for k, v in sd.items():
            if re.match(r"^(?:module\.|model\.)?trunk\.", k):
                trunk_sd[k] = v
        if trunk_sd:
            torch.save(trunk_sd, Path(outdir)/"snnet_trunk.pt")
            manifest["shards"]["snnet_trunk"] = {"count": len(trunk_sd), "path": str(Path(outdir)/"snnet_trunk.pt")}
        else:
            manifest["notes"].append("No trunk.* found (prefix might differ).")

    # ---------- metadata ----------
    if "head.weight" in sd:
        W = sd["head.weight"]
        manifest["head_in_dim"]  = int(W.shape[1])
        manifest["head_out_dim"] = int(W.shape[0])

    mpath = Path(outdir)/"manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] wrote {len(manifest['shards'])} shards to {outdir}")
    print(f"[ok] manifest: {mpath}")

if __name__ == "__main__":
    main()
