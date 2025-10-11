"""
Compute-time sampling and end-to-end stitch evaluation.
"""

import math, random
from typing import Dict, List, Optional, Tuple

from . import config
from .profiles import TASK_PROFILES, OUTPUT_SIZES_MB, CANDIDATE_STITCHES
from .topology import (
    Node, _hop_latency_ms, route_hops_nodes, Topology,
    dijkstra_base_path_between_nodes
)

# ---------------------------
# Compute-time sampling
# ---------------------------

def _sample_runtime_lognormal(mean_ms: float, cv: float, rng: random.Random) -> float:
    if mean_ms <= 0:
        return 0.0
    sigma2 = math.log(1.0 + cv * cv)
    sigma  = math.sqrt(sigma2)
    mu     = math.log(mean_ms) - 0.5 * sigma2
    return rng.lognormvariate(mu, sigma)

def compute_time_ms(module: str, node: Node, rng: random.Random, task_profile_name: str) -> float:
    prof = TASK_PROFILES.get(task_profile_name, TASK_PROFILES["object-det"])
    is_prefix = module.startswith("resnet_")
    bucket = "prefix" if is_prefix else "suffix"
    base_mean = prof[node.kind][bucket]
    cv = config.CV_PREFIX if is_prefix else config.CV_SUFFIX
    return _sample_runtime_lognormal(base_mean, cv, rng)

def module_output_mb(module: str) -> float:
    return OUTPUT_SIZES_MB.get(module, 0.0)

# ---------------------------
# Helpers
# ---------------------------

def _nodes_for_module(placement: Dict[str, object], m: str) -> List[Node]:
    v = placement.get(m)
    if v is None:
        return []
    if isinstance(v, Node):
        return [v]
    if isinstance(v, list):
        return [n for n in v if n is not None]
    if isinstance(v, dict) and "nodes" in v and isinstance(v["nodes"], list):
        return [n for n in v["nodes"] if n is not None]
    return []

def _first_reachable_pair(
    topo: Topology,
    src_candidates: List[Node],
    dst_candidates: List[Node],
) -> Optional[Tuple[Node, Node]]:
    for s in src_candidates:
        for d in dst_candidates:
            ms, _ = dijkstra_base_path_between_nodes(topo, s.nid, d.nid, float("inf"))
            if (ms is not None) and math.isfinite(ms):
                return (s, d)
    return None

def _best_latency_pair(
    topo: Topology,
    src_candidates: List[Node],
    dst_candidates: List[Node],
) -> Optional[Tuple[Node, Node]]:
    if not src_candidates or not dst_candidates:
        return None
    best = None  # (ms, s, d)
    for s in src_candidates:
        for d in dst_candidates:
            ms, _ = dijkstra_base_path_between_nodes(topo, s.nid, d.nid, float("inf"))
            if (ms is None) or (not math.isfinite(ms)):
                continue
            ms = float(ms)
            if (best is None) or (ms < best[0]):
                best = (ms, s, d)
    return None if best is None else (best[1], best[2])

def _rr_reachable_dst(
    topo: Topology,
    src: Node,
    dst_list: List[Node],
    start_idx: int
) -> Optional[Node]:
    tried = 0
    n = len(dst_list)
    while tried < n:
        d = dst_list[(start_idx + tried) % n]
        ms, _ = dijkstra_base_path_between_nodes(topo, src.nid, d.nid, float("inf"))
        if (ms is not None) and math.isfinite(ms):
            return d
        tried += 1
    return None

def _randomk_reachable_dst(
    topo: Topology,
    rng: random.Random,
    src: Node,
    dst_list: List[Node],
    k: int
) -> Optional[Node]:
    if not dst_list:
        return None
    dsts = dst_list[:]
    rng.shuffle(dsts)
    k = max(1, min(k, len(dsts)))
    samples = dsts[:k]
    reach = []
    for d in samples:
        ms, _ = dijkstra_base_path_between_nodes(topo, src.nid, d.nid, float("inf"))
        if (ms is not None) and math.isfinite(ms):
            reach.append(d)
    if not reach:
        return None
    return rng.choice(reach)

def _pick_pair(
    topo: Topology,
    objective: str,
    rng: random.Random,
    src_candidates: List[Node],
    dst_candidates: List[Node],
    *,
    rr_index: int = 0,
    rr_offset: int = 0,
    hop_idx: int = 0,
    random_k: int = 2
) -> Optional[Tuple[Node, Node]]:
    """
    Unified picker used for both the first hop (src_candidates many) and subsequent hops
    (src_candidates is length 1).
    Returns (src_node, dst_node) or None if no reachable pair given the strategy.
    """
    if not src_candidates or not dst_candidates:
        return None

    if objective == "latency":
        return _best_latency_pair(topo, src_candidates, dst_candidates)

    if objective == "accuracy_first_fit":
        return _first_reachable_pair(topo, src_candidates, dst_candidates)

    if objective == "rr":
        s = src_candidates[(rr_index + rr_offset + hop_idx) % len(src_candidates)]
        start_idx = (rr_index + rr_offset + hop_idx) % len(dst_candidates)
        d = _rr_reachable_dst(topo, s, dst_candidates, start_idx)
        return None if d is None else (s, d)

    if objective == "random2":
        s = rng.choice(src_candidates)
        d = _randomk_reachable_dst(topo, rng, s, dst_candidates, random_k)
        return None if d is None else (s, d)

    raise ValueError(f"Unknown greedy_objective: {objective}")

# ---------------------------
# SLO helpers
# ---------------------------

def _slo_fields(total_ms: float, slo_ms: Optional[float]):
    """Return (met_slo, excess_ms, excess_pct)."""
    if slo_ms is None or not math.isfinite(slo_ms) or slo_ms <= 0:
        return True, 0.0, float("nan")
    excess = max(0.0, total_ms - slo_ms)
    met = (excess <= 1e-9)
    pct = (100.0 * excess / slo_ms) if slo_ms > 0 else float("nan")
    return met, excess, pct

# ---------------------------
# E2E metrics
# ---------------------------

def e2e_metrics_for_stitch(
    stitch_id: int,
    placement: Dict[str, object],   # Node or List[Node] per module
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
    *,
    greedy_objective: str = "latency",      # "latency" | "accuracy_first_fit" | "rr" | "random2"
    acc_min: Optional[float] = None,        # used for accuracy_first_fit (gate)
    rr_index: int = 0,
    rr_offset: int = 0,
    random_k: int = 2,
) -> Dict[str, float]:

    spec     = CANDIDATE_STITCHES[stitch_id]
    modules  = spec["modules"]
    acc_val  = float(spec["acc"])

    if greedy_objective == "accuracy_first_fit":
        if acc_min is None:
            acc_min = 0.0
        if acc_val < acc_min:
            return {
                "latency_ms": float("inf"),
                "compute_ms": 0.0,
                "net_latency_ms": float("inf"),
                "payload_mb": 0.0,
                "link_mb": 0.0,
                "hop_count": 0,
                "acc": acc_val,
                "per_stage": [],
                "met_slo": False,
                "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
                "slo_excess_ms": float("inf"),
                "slo_excess_pct": float("nan"),
            }

    # Candidate layers
    layers: List[List[Node]] = []
    for m in modules:
        cand = _nodes_for_module(placement, m)
        if not cand:
            return {
                "latency_ms": float("inf"),
                "compute_ms": 0.0,
                "net_latency_ms": float("inf"),
                "payload_mb": 0.0,
                "link_mb": 0.0,
                "hop_count": 0,
                "acc": acc_val,
                "per_stage": [],
                "met_slo": False,
                "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
                "slo_excess_ms": float("inf"),
                "slo_excess_pct": float("nan"),
            }
        layers.append(cand)

    # 0 or 1 module edge case
    if len(modules) <= 1:
        compute_ms = 0.0
        if modules:
            host = layers[0][0]
            compute_ms += compute_time_ms(modules[0], host, rng, task_profile_name)
        total = compute_ms
        slo_wf = getattr(config, "SLO_MS_WORKFLOW", None)
        met, exc_ms, exc_pct = _slo_fields(total, slo_wf)
        return {
            "latency_ms": total,
            "compute_ms": compute_ms,
            "net_latency_ms": 0.0,
            "payload_mb": 0.0,
            "link_mb": 0.0,
            "hop_count": 0,
            "acc": acc_val,
            "per_stage": [{"module": modules[0], "compute_ms": compute_ms, "net_ms": 0.0, "seg_hops": 0, "payload_mb": 0.0}] if modules else [],
            "met_slo": met,
            "slo_ms": slo_wf,
            "slo_excess_ms": exc_ms,
            "slo_excess_pct": exc_pct,
        }

    # ---------- Select a concrete node chain ----------
    chosen: List[Optional[Node]] = [None] * len(modules)

    # First hop: any src in layer 0 to any dst in layer 1
    pair = _pick_pair(
        topo, greedy_objective, rng,
        layers[0], layers[1],
        rr_index=rr_index, rr_offset=rr_offset, hop_idx=0, random_k=random_k
    )
    if pair is None:
        return {
            "latency_ms": float("inf"),
            "compute_ms": 0.0,
            "net_latency_ms": float("inf"),
            "payload_mb": 0.0,
            "link_mb": 0.0,
            "hop_count": 0,
            "acc": acc_val,
            "per_stage": [],
            "met_slo": False,
            "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
            "slo_excess_ms": float("inf"),
            "slo_excess_pct": float("nan"),
        }
    chosen[0], chosen[1] = pair

    # Next hops: src is fixed (chosen[hop]) to any dst in next layer
    for hop in range(1, len(modules) - 1):
        pair = _pick_pair(
            topo, greedy_objective, rng,
            [chosen[hop]], layers[hop + 1],
            rr_index=rr_index, rr_offset=rr_offset, hop_idx=hop, random_k=random_k
        )
        if pair is None:
            return {
                "latency_ms": float("inf"),
                "compute_ms": 0.0,
                "net_latency_ms": float("inf"),
                "payload_mb": 0.0,
                "link_mb": 0.0,
                "hop_count": 0,
                "acc": acc_val,
                "per_stage": [],
                "met_slo": False,
                "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
                "slo_excess_ms": float("inf"),
                "slo_excess_pct": float("nan"),
            }
        _, chosen[hop + 1] = pair

    # ---------- Compute metrics ----------
    compute_ms       = 0.0
    net_latency_ms   = 0.0
    payload_mb_total = 0.0
    link_mb_total    = 0.0
    hop_count_total  = 0
    per_stage: List[Dict] = []

    # 1) compute times per module
    per_module_compute = []
    for m, host in zip(modules, chosen):
        ct = compute_time_ms(m, host, rng, task_profile_name)
        per_module_compute.append(ct)
        compute_ms += ct

    # 2) network per edge (between consecutive modules)
    for i in range(len(modules) - 1):
        src_node = chosen[i]
        dst_node = chosen[i + 1]
        payload  = module_output_mb(modules[i])
        payload_mb_total += payload

        hops = route_hops_nodes(topo, src_node.nid, dst_node.nid)  # list of edge objects
        seg_hops = len(hops)
        hop_count_total += seg_hops

        if not hops:
            net_latency_ms = float("inf")
            # record stage with inf net; compute still valid
            per_stage.append({
                "m_src": modules[i],
                "m_dst": modules[i+1],
                "compute_ms": per_module_compute[i],
                "net_ms": float("inf"),
                "seg_hops": 0,
                "payload_mb": payload,
            })
            break

        seg_net_ms = 0.0
        for e in hops:
            hop_ms = _hop_latency_ms(e, payload, rng)
            seg_net_ms += hop_ms
            link_mb_total += payload  # payload counted per hop

        net_latency_ms += seg_net_ms
        per_stage.append({
            "m_src": modules[i],
            "m_dst": modules[i+1],
            "compute_ms": per_module_compute[i],
            "net_ms": seg_net_ms,
            "seg_hops": seg_hops,
            "payload_mb": payload,
        })

    total_latency_ms = compute_ms + net_latency_ms

    # E2E SLO (workflow-level)
    slo_wf = getattr(config, "SLO_MS_WORKFLOW", None)
    met, exc_ms, exc_pct = _slo_fields(total_latency_ms, slo_wf)

    return {
        "latency_ms":     net_latency_ms,   # total = compute + net
        "compute_ms":     net_latency_ms,
        "net_latency_ms": net_latency_ms,
        "payload_mb":     payload_mb_total,
        "link_mb":        link_mb_total,
        "hop_count":      hop_count_total,
        "acc":            acc_val,
        "per_stage":      per_stage,          # for per-stage SLO analysis upstream
        "met_slo":        met,
        "slo_ms":         slo_wf,
        "slo_excess_ms":  exc_ms,
        "slo_excess_pct": exc_pct,
    }
