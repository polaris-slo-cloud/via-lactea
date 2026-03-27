# selection.py
"""
Stitch-selection policies (SLO-first, Best-Acc, Lowest-Latency, Random, Round-Robin, Full-model).
Adds selection_time_ms to report wall-clock policy runtime, plus ssp_calls for DP selector.

Key correctness rule:
- Do NOT modify shared candidate-pair helpers in a way that changes other strategies.
  Any pruning / caps must be confined to SLO-first-only code paths.

This version:
- Keeps _allowed_pairs_for_edge() fully unpruned (shared).
- Adds _allowed_pairs_for_edge_slofirst() used only by SLO-first DP.
- Ensures SLO-first ALWAYS returns a finite row (never blank/empty group), even on failure.
- Fixes SLO-first "3e15" by fail-open + adaptive relaxation when SLO-first pruning/forward-filter removes all feasible pairs.
- Adds TOP-K E2E recheck for SLO-first only.
- Node-aware caching for DP cached totals:
    If the layer output is already cached on the chosen src_node, then cached payload/link contribution is 0.
    Otherwise, cached contribution equals uncached contribution.
  This affects ONLY payload_mb_cached/link_mb_cached (uncached fields unchanged).

Expected (recommended) cache state:
  placement.LAYER_CACHE: Dict[str, Set[str]]
    node_id -> set of cached layer/module names

Fallback behavior:
  If placement.LAYER_CACHE is missing, uses pattern policy via CACHEABLE_LAYER_PATTERNS.

Important note about E2E overwrite:
- We overwrite payload_mb/link_mb/hop_count from E2E (for reporting).
- We DO NOT overwrite payload_mb_cached/link_mb_cached unless E2E explicitly provides cached fields.
  Otherwise, we keep the DP cached totals (node-aware caching result).
"""

import math
import random
import time
import fnmatch
import re
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
from dataclasses import dataclass

from . import config, placement
from . import profiles
from .runtime import e2e_metrics_for_stitch, _nodes_for_module
from .selection_algo import _pair_shortest_cached
from .topology import Node, module_output_mb, Topology, _filtered_topology_view


def _cfg(name: str, default):
    return getattr(config, name, default)


# ----------------------------------------------------------------------------- #
# Soft SLO knobs                                                                #
# ----------------------------------------------------------------------------- #

SOFT_SLO_ENABLE = bool(_cfg("SOFT_SLO_ENABLE", True))

HARD_LATENCY_CONSTRAINT = bool(_cfg("HARD_LATENCY_CONSTRAINT", False))
HARD_ACCURACY_CONSTRAINT = bool(_cfg("HARD_ACCURACY_CONSTRAINT", False))

SOFT_W_LAT = float(_cfg("SOFT_W_LAT", 1.0))
SOFT_W_ACC = float(_cfg("SOFT_W_ACC", 1.0))

SOFT_P_LAT = float(_cfg("SOFT_P_LAT", 2.0))
SOFT_P_ACC = float(_cfg("SOFT_P_ACC", 2.0))

SOFT_SSSP_SLACK_FACTOR = _cfg("SOFT_SSSP_SLACK_FACTOR", None)

SOFT_W_HOP = float(_cfg("SOFT_W_HOP", 0.0))
SOFT_HOP_TARGET = float(_cfg("SOFT_HOP_TARGET", 50.0))


# ----------------------------------------------------------------------------- #
# SLO-first speed knobs (SLO-first ONLY)                                        #
# ----------------------------------------------------------------------------- #

SLO_FIRST_MAX_NODES_PER_MODULE = int(_cfg("SLO_FIRST_MAX_NODES_PER_MODULE", 2))
if SLO_FIRST_MAX_NODES_PER_MODULE < 1:
    SLO_FIRST_MAX_NODES_PER_MODULE = 1

SLO_FIRST_MAX_PAIRS_PER_EDGE = int(_cfg("SLO_FIRST_MAX_PAIRS_PER_EDGE", 64))
if SLO_FIRST_MAX_PAIRS_PER_EDGE < 1:
    SLO_FIRST_MAX_PAIRS_PER_EDGE = 1

SLO_FIRST_BEAM_WIDTH = int(_cfg("SLO_FIRST_BEAM_WIDTH", 32))
if SLO_FIRST_BEAM_WIDTH < 1:
    SLO_FIRST_BEAM_WIDTH = 1

HOP_PENALTY_MS = float(_cfg("HOP_PENALTY_MS", 0.0))

# SLO-first robustness knobs (SLO-first ONLY)
SLO_FIRST_FAILOPEN_FORWARD_FILTER = bool(_cfg("SLO_FIRST_FAILOPEN_FORWARD_FILTER", True))
SLO_FIRST_ADAPTIVE_RELAX = bool(_cfg("SLO_FIRST_ADAPTIVE_RELAX", True))

# Try these (nodes_per_module, pairs_per_edge) if the strict caps fail
SLO_FIRST_RELAX_STEPS = _cfg("SLO_FIRST_RELAX_STEPS", [(2, 64), (4, 256), (8, 1024)])
if not isinstance(SLO_FIRST_RELAX_STEPS, (list, tuple)) or len(SLO_FIRST_RELAX_STEPS) == 0:
    SLO_FIRST_RELAX_STEPS = [(2, 64), (4, 256), (8, 1024)]

# SLO-first latency-accuracy closeness knob (SLO-first ONLY)
SLO_FIRST_E2E_TOPK = int(_cfg("SLO_FIRST_E2E_TOPK", 5))
if SLO_FIRST_E2E_TOPK < 1:
    SLO_FIRST_E2E_TOPK = 1

SLO_FIRST_PICK_BY_E2E_IN_TOPK = bool(_cfg("SLO_FIRST_PICK_BY_E2E_IN_TOPK", True))


# ----------------------------------------------------------------------------- #
# Forward filtering knobs (used by SLO-first DP only)                            #
# ----------------------------------------------------------------------------- #

FORWARD_FILTER_ENABLE = bool(_cfg("FORWARD_FILTER_ENABLE", True))
FORWARD_FORBID_EDGE_TO_SAT = bool(_cfg("FORWARD_FORBID_EDGE_TO_SAT", True))
FORWARD_FORBID_EDGE_TO_EDGE = bool(_cfg("FORWARD_FORBID_EDGE_TO_EDGE", True))
FORWARD_FORBID_CLOUD_TO_CLOUD = bool(_cfg("FORWARD_FORBID_CLOUD_TO_CLOUD", False))

FORWARD_SATS_PER_RING = _cfg("SATS_PER_RING", None)
FORWARD_ISL_NEIGHBOR_SPAN = _cfg("ISL_NEIGHBOR_SPAN", 1)
FORWARD_MAX_RING_DELTA = int(_cfg("FORWARD_MAX_RING_DELTA", 1))


@dataclass
class StitchEvalConfig:
    per_edge_prop_cap_ms: Optional[float] = None
    stage_slo_ms: Optional[float] = None
    enable_caching: bool = True
    stitch_ids: Optional[List[int]] = None
    shuffle: bool = False
    terminal_objective: str = "latency"  # "latency" | "hops"


# ----------------------------------------------------------------------------- #
# SLO helpers                                                                    #
# ----------------------------------------------------------------------------- #

def _stage_slo_fields(total_net_ms: float):
    slo_ms = getattr(config, "SLO_MS_STAGE", None)
    if slo_ms is None or not math.isfinite(slo_ms) or slo_ms <= 0:
        return True, 0.0, float("nan"), slo_ms
    excess = max(0.0, float(total_net_ms) - float(slo_ms))
    met = (excess <= 1e-9)
    pct = 100.0 * excess / float(slo_ms) if slo_ms > 0 else float("nan")
    return met, excess, pct, slo_ms


def _workflow_latency_target_ms() -> Optional[float]:
    slo = getattr(config, "SLO_MS_WORKFLOW", None)
    if slo is None or not math.isfinite(float(slo)) or float(slo) <= 0:
        slo = getattr(config, "SLO_MS_STAGE", None)
    if slo is None:
        return None
    slo = float(slo)
    return slo if math.isfinite(slo) and slo > 0 else None


def _accuracy_target(acc_min: Optional[float]) -> Optional[float]:
    if acc_min is not None and math.isfinite(float(acc_min)):
        return float(acc_min)
    cfg_acc = getattr(config, "SLO_ACC_MIN", None)
    if cfg_acc is None:
        return None
    try:
        cfg_acc = float(cfg_acc)
    except Exception:
        return None
    return cfg_acc if math.isfinite(cfg_acc) else None


def _hinge(x: float, p: float) -> float:
    if x <= 0.0:
        return 0.0
    if p == 1.0:
        return x
    return x ** p


def _soft_slo_penalty(
    *,
    net_ms: float,
    acc: float,
    lat_target_ms: Optional[float],
    acc_target: Optional[float],
) -> Tuple[float, float, float, float, float]:
    if lat_target_ms is None or not math.isfinite(lat_target_ms) or lat_target_ms <= 0:
        v_lat_ms = 0.0
        v_lat_norm = 0.0
    else:
        v_lat_ms = max(0.0, float(net_ms) - float(lat_target_ms))
        v_lat_norm = v_lat_ms / float(lat_target_ms)

    if acc_target is None or not math.isfinite(acc_target):
        v_acc = 0.0
        v_acc_norm = 0.0
    else:
        v_acc = max(0.0, float(acc_target) - float(acc))
        denom = float(acc_target) if float(acc_target) > 0 else 1.0
        v_acc_norm = v_acc / denom

    score = (SOFT_W_LAT * _hinge(v_lat_norm, SOFT_P_LAT)) + (SOFT_W_ACC * _hinge(v_acc_norm, SOFT_P_ACC))
    return float(score), float(v_lat_ms), float(v_lat_norm), float(v_acc), float(v_acc_norm)


def _soft_score_with_hops(score: float, hops: float) -> float:
    if SOFT_W_HOP <= 0.0:
        return float(score)
    denom = SOFT_HOP_TARGET if SOFT_HOP_TARGET > 0 else 1.0
    return float(score) + float(SOFT_W_HOP) * (float(hops) / denom)


def _effective_sssp_prune_ms() -> Optional[float]:
    base = getattr(config, "SLO_MS_STAGE", None)
    if base is None:
        return None
    try:
        base = float(base)
    except Exception:
        return None
    if not math.isfinite(base) or base <= 0:
        return None

    if not SOFT_SLO_ENABLE:
        return base

    if SOFT_SSSP_SLACK_FACTOR is None:
        return None
    try:
        f = float(SOFT_SSSP_SLACK_FACTOR)
    except Exception:
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return base * f


# ----------------------------------------------------------------------------- #
# Caching helpers (SLO-first DP only)                                            #
# ----------------------------------------------------------------------------- #

def _is_cacheable_layer(layer_name: str) -> bool:
    pats = getattr(config, "CACHEABLE_LAYER_PATTERNS", [])
    if not pats:
        return False
    for pat in pats:
        if fnmatch.fnmatch(layer_name, pat):
            return True
    return False


def _cached_payload_mb_for_layer(layer_name: str, uncached_mb: float) -> float:
    first_run = bool(getattr(config, "CACHE_FIRST_RUN", True))
    if first_run:
        return float(uncached_mb)
    return 0.0 if _is_cacheable_layer(layer_name) else float(uncached_mb)


def _node_has_layer_cache(node: Node, layer_name: str) -> bool:
    """
    Node-aware cache check.

    Preferred: placement.LAYER_CACHE: Dict[node_id -> Set[layer_name]]
    If missing, fallback to pattern-based policy (_is_cacheable_layer).
    If CACHE_FIRST_RUN=True, treat as miss everywhere.
    """
    if bool(getattr(config, "CACHE_FIRST_RUN", True)):
        return False

    cache = getattr(placement, "LAYER_CACHE", None)
    if isinstance(cache, dict):
        s = cache.get(node.nid)
        if s is None:
            return False
        try:
            return layer_name in s
        except Exception:
            return False

    return _is_cacheable_layer(layer_name)


# ----------------------------------------------------------------------------- #
# Forward-only filter (SLO-first DP only)                                        #
# ----------------------------------------------------------------------------- #

_sat_re = re.compile(r"^sat_r(\d+)_i(\d+)$")


def _parse_sat(nid: str) -> Optional[Tuple[int, int]]:
    m = _sat_re.match(nid)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _forward_ok(src: Node, dst: Node) -> bool:
    if not FORWARD_FILTER_ENABLE:
        return True

    if FORWARD_FORBID_EDGE_TO_SAT and src.kind == "edge" and dst.kind == "sat":
        return False
    if FORWARD_FORBID_EDGE_TO_EDGE and src.kind == "edge" and dst.kind == "edge":
        return False
    if FORWARD_FORBID_CLOUD_TO_CLOUD and src.kind == "cloud" and dst.kind == "cloud":
        return False

    if src.kind == "sat" and dst.kind == "sat":
        ps = _parse_sat(src.nid)
        pd = _parse_sat(dst.nid)
        if ps is None or pd is None:
            return True

        r1, i1 = ps
        r2, i2 = pd
        if abs(r1 - r2) > FORWARD_MAX_RING_DELTA:
            return False

        spr = FORWARD_SATS_PER_RING
        span = FORWARD_ISL_NEIGHBOR_SPAN
        if spr is not None:
            try:
                spr_i = int(spr)
            except Exception:
                spr_i = None
            if spr_i is not None and spr_i > 0:
                d = abs(i1 - i2)
                d = min(d, spr_i - d)
                if d > int(span):
                    return False

    return True


def _filter_forward_pairs(pairs: List[Tuple[Node, Node]]) -> List[Tuple[Node, Node]]:
    if not FORWARD_FILTER_ENABLE:
        return pairs
    return [(s, d) for (s, d) in pairs if _forward_ok(s, d)]


# ----------------------------------------------------------------------------- #
# Pair selection (SHARED, do not prune here)                                     #
# ----------------------------------------------------------------------------- #

def _allowed_pairs_for_edge(placement_dict: Dict, m_src: str, m_dst: str) -> List[Tuple[Node, Node]]:
    key = (m_src, m_dst)
    if key in placement_dict:
        pairs = placement_dict[key]
        out: List[Tuple[Node, Node]] = []
        for p in pairs:
            if isinstance(p, tuple) and len(p) == 2 and p[0] is not None and p[1] is not None:
                out.append((p[0], p[1]))
        return out

    src_nodes = _nodes_for_module(placement_dict, m_src)
    dst_nodes = _nodes_for_module(placement_dict, m_dst)
    return [(s, d) for s in src_nodes for d in dst_nodes]


# ----------------------------------------------------------------------------- #
# Pair selection (SLO-first ONLY pruning)                                        #
# ----------------------------------------------------------------------------- #

def _limit_nodes(nodes: Iterable[Node], k: int) -> List[Node]:
    nn = sorted(list(nodes), key=lambda n: n.nid)
    return nn[:k] if len(nn) > k else nn


def _nodes_for_module_limited(placement_dict: Dict, module_name: str, k_nodes: int) -> List[Node]:
    nodes = _nodes_for_module(placement_dict, module_name)
    if not nodes:
        return []
    return _limit_nodes(nodes, k_nodes)


def _cap_pairs(pairs: List[Tuple[Node, Node]], k_pairs: int) -> List[Tuple[Node, Node]]:
    if len(pairs) <= k_pairs:
        return pairs
    return pairs[:k_pairs]


def _allowed_pairs_for_edge_slofirst(
    placement_dict: Dict,
    m_src: str,
    m_dst: str,
    *,
    k_nodes: int,
    k_pairs: int,
) -> List[Tuple[Node, Node]]:
    key = (m_src, m_dst)
    if key in placement_dict:
        pairs = placement_dict[key]
        out: List[Tuple[Node, Node]] = []
        for p in pairs:
            if isinstance(p, tuple) and len(p) == 2 and p[0] is not None and p[1] is not None:
                out.append((p[0], p[1]))

        out_capped = _cap_pairs(out, k_pairs)
        fwd = _filter_forward_pairs(out_capped)
        if fwd:
            return _cap_pairs(fwd, k_pairs)

        if SLO_FIRST_FAILOPEN_FORWARD_FILTER:
            return out_capped
        return []

    src_nodes = _nodes_for_module_limited(placement_dict, m_src, k_nodes)
    dst_nodes = _nodes_for_module_limited(placement_dict, m_dst, k_nodes)
    if not src_nodes or not dst_nodes:
        return []

    out: List[Tuple[Node, Node]] = []
    for s in src_nodes:
        for d in dst_nodes:
            out.append((s, d))
            if len(out) >= k_pairs:
                break
        if len(out) >= k_pairs:
            break

    fwd = _filter_forward_pairs(out)
    if fwd:
        return _cap_pairs(fwd, k_pairs)

    if SLO_FIRST_FAILOPEN_FORWARD_FILTER:
        return out
    return []


# ----------------------------------------------------------------------------- #
# DP internals (SLO-first DP only)                                               #
# ----------------------------------------------------------------------------- #

def _hop_penalized_lat(lat_ms: float, hops: int) -> float:
    if HOP_PENALTY_MS <= 0.0:
        return float(lat_ms)
    return float(lat_ms) + float(HOP_PENALTY_MS) * float(hops)


DPState = Tuple[float, int, float, float, float, float, Optional[str], float]


def _beam_prune(dp: Dict[str, DPState], beam_width: int) -> Dict[str, DPState]:
    if len(dp) <= beam_width:
        return dp
    items = sorted(dp.items(), key=lambda kv: kv[1][0])[:beam_width]
    return dict(items)


def _dp_net_for_stitch(
    sid: int,
    placement_dict: Dict,
    ftopo: Topology,
    *,
    stage_prune_ms: Optional[float],
    enable_caching: bool,
    dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]],
    k_nodes: int,
    k_pairs: int,
) -> Tuple[Optional[Dict], int]:
    spec = profiles.CANDIDATE_STITCHES.get(sid)
    if spec is None:
        return None, 0

    acc = float(spec["acc"])
    mods: List[str] = spec["modules"]
    ssp_calls = 0

    if len(mods) < 2:
        met_slo, exc_ms, exc_pct, slo_used = _stage_slo_fields(0.0)
        return ({
            "stitch_id": sid,
            "acc": acc,
            "latency_ms": 0.0,
            "net_base_ms": 0.0,
            "hop_count": 0,
            "payload_mb": 0.0,
            "link_mb": 0.0,
            "payload_mb_cached": 0.0,
            "link_mb_cached": 0.0,
            "compute_ms": 0.0,
            "net_latency_ms": 0.0,
            "met_slo": met_slo,
            "slo_ms": slo_used,
            "slo_excess_ms": exc_ms,
            "slo_excess_pct": exc_pct,
            "met_dual_slo": met_slo,
            "net_obj_ms": 0.0,
        }, 0)

    dp_prev: Dict[str, DPState] = {}

    first_nodes = _nodes_for_module_limited(placement_dict, mods[0], k_nodes)
    if not first_nodes:
        pairs01 = _allowed_pairs_for_edge_slofirst(placement_dict, mods[0], mods[1], k_nodes=k_nodes, k_pairs=k_pairs)
        first_nodes = sorted({p[0] for p in pairs01}, key=lambda n: n.nid)[:k_nodes]
    if not first_nodes:
        return None, 0

    for n in first_nodes:
        dp_prev[n.nid] = (0.0, 0, 0.0, 0.0, 0.0, 0.0, None, 0.0)

    for i in range(len(mods) - 1):
        m_src, m_dst = mods[i], mods[i + 1]
        payload_i = float(module_output_mb(m_src))

        allowed_pairs = _allowed_pairs_for_edge_slofirst(
            placement_dict, m_src, m_dst, k_nodes=k_nodes, k_pairs=k_pairs
        )
        if not allowed_pairs:
            return None, ssp_calls

        allowed_pairs = [(s, d) for (s, d) in allowed_pairs if s.nid in dp_prev]
        if not allowed_pairs:
            return None, ssp_calls

        dp_curr: Dict[str, DPState] = {}

        for src_node, dst_node in allowed_pairs:
            prev = dp_prev.get(src_node.nid)
            if prev is None:
                continue

            (_prev_obj, prev_hops,
             prev_payload, prev_link,
             prev_payload_cached, prev_link_cached,
             _parent, prev_raw) = prev

            ssp_calls += 1
            seg_lat_ms, path_nodes = _pair_shortest_cached(
                ftopo, src_node.nid, dst_node.nid, dijk_cache, stage_prune_ms
            )
            if (seg_lat_ms is None) or (not math.isfinite(seg_lat_ms)):
                continue

            seg_hops = max(0, (len(path_nodes) - 1) if path_nodes is not None else 0)

            raw_net = float(prev_raw) + float(seg_lat_ms)
            hops = int(prev_hops) + int(seg_hops)

            # Uncached totals
            payload = float(prev_payload) + payload_i
            link = float(prev_link) + payload_i * float(seg_hops)

            # Cached totals (node-aware)
            if enable_caching and _node_has_layer_cache(src_node, m_src):
                payload_i_cached_node = 0.0
            else:
                payload_i_cached_node = payload_i

            payload_c = float(prev_payload_cached) + float(payload_i_cached_node)
            link_c = float(prev_link_cached) + float(payload_i_cached_node) * float(seg_hops)

            obj = _hop_penalized_lat(raw_net, hops)
            cand: DPState = (obj, hops, payload, link, payload_c, link_c, src_node.nid, raw_net)

            best = dp_curr.get(dst_node.nid)
            if best is None or cand < best:
                dp_curr[dst_node.nid] = cand

        if not dp_curr:
            return None, ssp_calls

        dp_prev = _beam_prune(dp_curr, SLO_FIRST_BEAM_WIDTH)

    _end_nid, end_state = min(dp_prev.items(), key=lambda kv: kv[1][0])
    (obj_ms, hops,
     payload, link,
     payload_c, link_c,
     _parent, raw_net_ms) = end_state

    met_slo, exc_ms, exc_pct, slo_used = _stage_slo_fields(raw_net_ms)

    return ({
        "stitch_id": sid,
        "acc": acc,
        "latency_ms": float(raw_net_ms),
        "net_base_ms": float(raw_net_ms),
        "hop_count": int(hops),
        "payload_mb": float(payload),
        "link_mb": float(link),
        "payload_mb_cached": float(payload_c),
        "link_mb_cached": float(link_c),
        "compute_ms": 0.0,
        "net_latency_ms": float(raw_net_ms),
        "met_slo": met_slo,
        "slo_ms": slo_used,
        "slo_excess_ms": exc_ms,
        "slo_excess_pct": exc_pct,
        "met_dual_slo": met_slo,
        "net_obj_ms": float(obj_ms),
    }, ssp_calls)


def _dp_net_for_stitch_adaptive(
    sid: int,
    placement_dict: Dict,
    ftopo: Topology,
    *,
    stage_prune_ms: Optional[float],
    enable_caching: bool,
    dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]],
) -> Tuple[Optional[Dict], int]:
    total_calls = 0

    steps = list(SLO_FIRST_RELAX_STEPS)
    strict = (int(SLO_FIRST_MAX_NODES_PER_MODULE), int(SLO_FIRST_MAX_PAIRS_PER_EDGE))
    if strict in steps:
        steps.remove(strict)
    steps.insert(0, strict)

    for (k_nodes, k_pairs) in steps:
        dp, calls = _dp_net_for_stitch(
            sid, placement_dict, ftopo,
            stage_prune_ms=stage_prune_ms,
            enable_caching=enable_caching,
            dijk_cache=dijk_cache,
            k_nodes=int(max(1, k_nodes)),
            k_pairs=int(max(1, k_pairs)),
        )
        total_calls += int(calls)
        if dp is not None:
            return dp, total_calls
        if not SLO_FIRST_ADAPTIVE_RELAX:
            break

    return None, total_calls


# ----------------------------------------------------------------------------- #
# Public reporting helpers                                                       #
# ----------------------------------------------------------------------------- #

def eval_stitch_net_metrics(
    sid: int,
    placement: Dict,
    topo: Topology,
    rng: random.Random,
    *,
    acc_min: Optional[float] = None,
    cfg: Optional[StitchEvalConfig] = None,
) -> Optional[Dict]:
    if cfg is None:
        cfg = StitchEvalConfig(
            per_edge_prop_cap_ms=None,
            stage_slo_ms=_effective_sssp_prune_ms(),
            enable_caching=True,
            stitch_ids=None,
            shuffle=False,
            terminal_objective="latency",
        )

    t0 = time.perf_counter()
    ftopo = _filtered_topology_view(topo, cfg.per_edge_prop_cap_ms)
    dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]] = {}

    dp, ssp_calls = _dp_net_for_stitch_adaptive(
        sid, placement, ftopo,
        stage_prune_ms=cfg.stage_slo_ms,
        enable_caching=cfg.enable_caching,
        dijk_cache=dijk_cache,
    )
    if dp is None:
        return None

    lat_target = _workflow_latency_target_ms()
    acc_target = _accuracy_target(acc_min)

    score, vL_ms, vL_norm, vA_abs, vA_norm = _soft_slo_penalty(
        net_ms=float(dp.get("net_base_ms", float("inf"))),
        acc=float(dp.get("acc", float("nan"))),
        lat_target_ms=lat_target,
        acc_target=acc_target,
    )
    score = _soft_score_with_hops(score, float(dp.get("hop_count", 0)))

    out = dict(dp)
    out.update({
        "soft_slo_score": float(score),
        "v_lat_ms": float(vL_ms),
        "v_lat_norm": float(vL_norm),
        "v_acc": float(vA_abs),
        "v_acc_norm": float(vA_norm),
        "lat_target_ms": lat_target,
        "acc_target": acc_target,
        "selection_time_ms": (time.perf_counter() - t0) * 1000.0,
        "ssp_calls": int(ssp_calls),
    })
    return out


def eval_all_stitches_net_metrics(
    placement: Dict,
    topo: Topology,
    rng: random.Random,
    *,
    acc_min: Optional[float] = None,
    cfg: Optional[StitchEvalConfig] = None,
) -> List[Dict]:
    if cfg is None:
        cfg = StitchEvalConfig(
            per_edge_prop_cap_ms=None,
            stage_slo_ms=_effective_sssp_prune_ms(),
            enable_caching=True,
            stitch_ids=None,
            shuffle=False,
            terminal_objective="latency",
        )

    stitch_ids = cfg.stitch_ids if cfg.stitch_ids is not None else list(profiles.CANDIDATE_STITCHES.keys())
    if cfg.shuffle:
        tmp = list(stitch_ids)
        rng.shuffle(tmp)
        stitch_ids = tmp

    out: List[Dict] = []
    for sid in stitch_ids:
        mets = eval_stitch_net_metrics(sid, placement, topo, rng, acc_min=acc_min, cfg=cfg)
        if mets is None:
            mets = _reject_row(stitch_id=sid, acc=float(profiles.CANDIDATE_STITCHES.get(sid, {}).get("acc", float("nan"))))
        out.append(mets)
    return out


# ----------------------------------------------------------------------------- #
# Rejection row                                                                  #
# ----------------------------------------------------------------------------- #

def _reject_row(stitch_id=None, acc=np.nan, slo_hit=False):
    return {
        "stitch_id": stitch_id,
        "latency_ms": float("inf"),
        "compute_ms": 0.0,
        "net_latency_ms": float("inf"),
        "payload_mb": 0.0,
        "link_mb": 0.0,
        "payload_mb_cached": 0.0,
        "link_mb_cached": 0.0,
        "hop_count": 0,
        "acc": acc,
        "met_slo": slo_hit,
        "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
        "slo_excess_ms": float("inf") if not slo_hit else 0.0,
        "slo_excess_pct": float("nan"),
        "met_dual_slo": slo_hit,
        "selection_time_ms": 0.0,
        "ssp_calls": 0,
        "soft_slo_score": float("inf"),
        "v_lat_ms": float("inf"),
        "v_lat_norm": float("inf"),
        "v_acc": float("inf"),
        "v_acc_norm": float("inf"),
        "lat_target_ms": _workflow_latency_target_ms(),
        "acc_target": _accuracy_target(None),
    }


def _reject_row_finite(stitch_id=None, acc=np.nan, slo_hit=False, huge=1e15):
    r = _reject_row(stitch_id=stitch_id, acc=acc, slo_hit=slo_hit)
    r["latency_ms"] = float(huge)
    r["net_latency_ms"] = float(huge)
    r["soft_slo_score"] = float(huge)
    r["selection_time_ms"] = float(r.get("selection_time_ms", 0.0))
    r["ssp_calls"] = int(r.get("ssp_calls", 0))
    r["valid"] = False
    return r


# ----------------------------------------------------------------------------- #
# SLO-first selector                                                             #
# ----------------------------------------------------------------------------- #

def choose_stitch_for_task(
    placement: Dict,
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
    *,
    slo_ms: Optional[float] = None,
    acc_min: Optional[float] = None,
    per_edge_prop_cap_ms: Optional[float] = None,
) -> Dict:
    """
    IMPORTANT: signature uses `placement` to match simulator calls that pass placement=...
    """
    t0 = time.perf_counter()

    ftopo = _filtered_topology_view(topo, per_edge_prop_cap_ms)
    stage_prune_ms = _effective_sssp_prune_ms()

    lat_target = _workflow_latency_target_ms()
    acc_target = _accuracy_target(acc_min)

    dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]] = {}

    candidates: List[Tuple[Tuple[float, float, float], Dict]] = []
    ssp_calls_total = 0

    sid_ids = sorted(
        profiles.CANDIDATE_STITCHES,
        key=lambda sid: profiles.CANDIDATE_STITCHES[sid]["acc"],
        reverse=True
    )[:3]

    for sid in sid_ids:
        dp, ssp_calls = _dp_net_for_stitch_adaptive(
            sid, placement, ftopo,
            stage_prune_ms=stage_prune_ms,
            enable_caching=True,
            dijk_cache=dijk_cache,
        )
        ssp_calls_total += int(ssp_calls)
        if dp is None:
            continue

        acc = float(dp.get("acc", float("nan")))
        net = float(dp.get("net_base_ms", float("inf")))
        hops = float(dp.get("hop_count", 0))

        score, vL_ms, vL_norm, vA_abs, vA_norm = _soft_slo_penalty(
            net_ms=net, acc=acc, lat_target_ms=lat_target, acc_target=acc_target
        )
        score = _soft_score_with_hops(score, hops)

        if HARD_LATENCY_CONSTRAINT and vL_ms > 0.0:
            continue
        if HARD_ACCURACY_CONSTRAINT and vA_abs > 0.0:
            continue

        key = (float(score), float(net), -float(acc))

        dp2 = dict(dp)
        dp2.update({
            "soft_slo_score": float(score),
            "v_lat_ms": float(vL_ms),
            "v_lat_norm": float(vL_norm),
            "v_acc": float(vA_abs),
            "v_acc_norm": float(vA_norm),
            "lat_target_ms": lat_target,
            "acc_target": acc_target,
        })

        candidates.append((key, dp2))

    if not candidates:
        out = _reject_row_finite(stitch_id=None, acc=float("nan"), slo_hit=False)
        out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
        out["ssp_calls"] = int(ssp_calls_total)
        return out

    candidates.sort(key=lambda kv: kv[0])

    best_dp = candidates[0][1]
    best_sid = int(best_dp["stitch_id"])

    K = min(int(SLO_FIRST_E2E_TOPK), len(candidates))
    topk = [candidates[i][1] for i in range(K)]

    best_e2e: Optional[Dict] = None
    best_e2e_lat = float("inf")
    best_e2e_dp: Optional[Dict] = None

    for dp_cand in topk:
        sid = int(dp_cand["stitch_id"])
        try:
            e2e = e2e_metrics_for_stitch(
                sid, placement, topo, rng, task_profile_name,
                greedy_objective="latency"
            )
        except Exception:
            continue

        lat = float(e2e.get("latency_ms", float("inf")))
        if not math.isfinite(lat):
            continue

        if lat < best_e2e_lat:
            best_e2e_lat = lat
            best_e2e = e2e
            best_e2e_dp = dp_cand

    if SLO_FIRST_PICK_BY_E2E_IN_TOPK and (best_e2e is not None) and (best_e2e_dp is not None):
        out = dict(best_e2e_dp)
        out["valid"] = True

        # Keep DP cached totals unless E2E explicitly provides cached totals
        dp_cached_payload = float(out.get("payload_mb_cached", out.get("payload_mb", 0.0)))
        dp_cached_link = float(out.get("link_mb_cached", out.get("link_mb", 0.0)))

        out.update({
            "latency_ms": float(best_e2e.get("latency_ms", out.get("latency_ms", 1e15))),
            "compute_ms": float(best_e2e.get("compute_ms", 0.0)),
            "net_latency_ms": float(best_e2e.get("net_latency_ms", out.get("net_latency_ms", 1e15))),
            "payload_mb": float(best_e2e.get("payload_mb", out.get("payload_mb", 0.0))),
            "link_mb": float(best_e2e.get("link_mb", out.get("link_mb", 0.0))),
            "hop_count": int(best_e2e.get("hop_count", out.get("hop_count", 0))),
            "stitch_id": int(out.get("stitch_id", best_sid)),
        })

        out["payload_mb_cached"] = dp_cached_payload
        out["link_mb_cached"] = dp_cached_link
        if "payload_mb_cached" in best_e2e:
            out["payload_mb_cached"] = float(best_e2e["payload_mb_cached"])
        if "link_mb_cached" in best_e2e:
            out["link_mb_cached"] = float(best_e2e["link_mb_cached"])

    else:
        out = dict(best_dp)
        out["valid"] = True

        dp_cached_payload = float(out.get("payload_mb_cached", out.get("payload_mb", 0.0)))
        dp_cached_link = float(out.get("link_mb_cached", out.get("link_mb", 0.0)))

        try:
            e2e = e2e_metrics_for_stitch(
                best_sid, placement, topo, rng, task_profile_name,
                greedy_objective="latency"
            )
            out.update({
                "latency_ms": float(e2e.get("latency_ms", out.get("latency_ms", 1e15))),
                "compute_ms": float(e2e.get("compute_ms", 0.0)),
                "net_latency_ms": float(e2e.get("net_latency_ms", out.get("net_latency_ms", 1e15))),
                "payload_mb": float(e2e.get("payload_mb", out.get("payload_mb", 0.0))),
                "link_mb": float(e2e.get("link_mb", out.get("link_mb", 0.0))),
                "hop_count": int(e2e.get("hop_count", out.get("hop_count", 0))),
            })

            out["payload_mb_cached"] = dp_cached_payload
            out["link_mb_cached"] = dp_cached_link
            if "payload_mb_cached" in e2e:
                out["payload_mb_cached"] = float(e2e["payload_mb_cached"])
            if "link_mb_cached" in e2e:
                out["link_mb_cached"] = float(e2e["link_mb_cached"])

        except Exception:
            out["latency_ms"] = float(out.get("latency_ms", 1e15))
            out["net_latency_ms"] = float(out.get("net_latency_ms", 1e15))
            out["valid"] = False
            # Keep DP cached totals even on E2E failure
            out["payload_mb_cached"] = dp_cached_payload
            out["link_mb_cached"] = dp_cached_link

    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    out["ssp_calls"] = int(ssp_calls_total)
    return out


# ----------------------------------------------------------------------------- #
# Other strategies (leave as-is)                                                 #
# ----------------------------------------------------------------------------- #

def _mirror_cached_fields(mets: Dict) -> Dict:
    out = dict(mets)
    out["payload_mb_cached"] = float(out.get("payload_mb", 0.0))
    out["link_mb_cached"] = float(out.get("link_mb", 0.0))
    return out


def always_best_accuracy(placement: Dict[str, Node], topo: Topology, rng: random.Random, task_profile_name: str) -> Dict:
    t0 = time.perf_counter()
    top_acc = max(spec["acc"] for spec in profiles.CANDIDATE_STITCHES.values())
    best = None
    candidate_stitches = list(profiles.CANDIDATE_STITCHES.items())
    random.shuffle(candidate_stitches)
    for sid, spec in candidate_stitches:
        if spec["acc"] + 1e-9 < top_acc:
            continue
        mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="accuracy_first_fit")
        cand = {"stitch_id": sid, **mets}
        if (best is None) or (cand["latency_ms"] < best["latency_ms"]):
            best = cand
    if best is None:
        best = _reject_row()
    best = _mirror_cached_fields(best)
    best["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return best


def lowest_latency(placement: Dict[str, Node], topo: Topology, rng: random.Random, task_profile_name: str) -> Dict:
    t0 = time.perf_counter()
    best = None
    candidates = sorted(
        profiles.CANDIDATE_STITCHES.keys(),
        key=lambda sid: profiles.CANDIDATE_STITCHES[sid]["acc"]
    )[:3]
    for sid in candidates:
        mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="latency")
        cand = {"stitch_id": sid, **mets}
        if (best is None) or (cand["latency_ms"] < best["latency_ms"]):
            best = cand
    if best is None:
        best = _reject_row()
    best = _mirror_cached_fields(best)
    best["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return best


def random_pick_stitch(placement: Dict[str, Node], topo: Topology, rng: random.Random, task_profile_name: str) -> Dict:
    t0 = time.perf_counter()
    sid = rng.choice(list(profiles.CANDIDATE_STITCHES.keys()))
    mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="random2", random_k=2)
    out = {"stitch_id": sid, **mets}
    out = _mirror_cached_fields(out)
    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


def round_robin_pick_stitch(
    index: int,
    placement: Dict[str, Node],
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
    offset: int = config.RR_START_OFFSET
) -> Dict:
    t0 = time.perf_counter()
    rr_order = sorted(profiles.CANDIDATE_STITCHES.keys())
    sid = rr_order[(index + offset) % len(rr_order)]
    mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="rr", rr_index=index, rr_offset=offset)
    out = {"stitch_id": sid, **mets}
    out = _mirror_cached_fields(out)
    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


def full_model_stitch(
    placement: Dict[str, Node],
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
) -> Dict:
    t0 = time.perf_counter()
    sid = min(profiles.CANDIDATE_STITCHES.keys())
    mets = e2e_metrics_for_stitch(
        sid, placement, topo, rng, task_profile_name,
        greedy_objective="full-model"
    )
    out = {"stitch_id": sid, **mets}
    out = _mirror_cached_fields(out)
    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return out