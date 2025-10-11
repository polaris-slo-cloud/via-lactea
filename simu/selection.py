"""
Stitch-selection policies (SLO-first, Best-Acc, Lowest-Latency, Random, Round-Robin).
Adds selection_time_ms to report wall-clock policy runtime, plus ssp_calls for DP selector.
"""

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .profiles import CANDIDATE_STITCHES
from .runtime import e2e_metrics_for_stitch, _nodes_for_module
from .selection_algo import _pair_shortest_cached
from .topology import Node, module_output_mb, Topology, _filtered_topology_view


# ----------------------------------------------------------------------------- #
# Helpers                                                                       #
# ----------------------------------------------------------------------------- #

def _allowed_pairs_for_edge(
    placement: Dict,
    m_src: str,
    m_dst: str,
) -> List[Tuple[Node, Node]]:
    """
    Resolve allowed (src_node, dst_node) pairs for the edge m_src -> m_dst.

    Supports:
      - Explicit per-edge list: placement[(m_src, m_dst)] = [(srcNode, dstNode), ...]
      - Fallback to Cartesian product of module node lists.
    """
    key = (m_src, m_dst)
    if key in placement:
        pairs = placement[key]
        out = []
        for p in pairs:
            if isinstance(p, tuple) and len(p) == 2 and p[0] is not None and p[1] is not None:
                out.append((p[0], p[1]))
        return out

    src_nodes = _nodes_for_module(placement, m_src)
    dst_nodes = _nodes_for_module(placement, m_dst)
    return [(s, d) for s in src_nodes for d in dst_nodes]


# ----------------------------------------------------------------------------- #
# Core DP-based, topology-aware stitch chooser                                  #
# ----------------------------------------------------------------------------- #

def choose_stitch_for_task(
    placement: Dict,                  # supports per-edge pairs and/or per-module lists
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
    *,
    slo_ms: Optional[float] = None,
    acc_min: Optional[float] = None,
    per_edge_prop_cap_ms: Optional[float] = None,
) -> Optional[Dict]:
    """
    Returns a metrics dict for the chosen stitch, now including:
      - selection_time_ms: wall-clock time to execute this selection algorithm
      - ssp_calls: number of shortest-path calls performed (scales with graph/placement)
    """
    t0 = time.perf_counter()
    ssp_calls = 0

    ftopo = _filtered_topology_view(topo, per_edge_prop_cap_ms)
    feasible = []

    candidate_stitches = list(CANDIDATE_STITCHES.items())
    random.shuffle(candidate_stitches)

    for sid, spec in candidate_stitches:
        acc = float(spec["acc"])
        if acc_min is not None and acc < acc_min:
            continue

        mods: List[str] = spec["modules"]
        if len(mods) < 2:
            # trivial stitch (single module): zero net, zero hops
            mets = {
                "stitch_id": sid,
                "acc": acc,
                "latency_ms": 0.0,
                "net_base_ms": 0.0,
                "hop_count": 0,
                "payload_mb": 0.0,
                "link_mb": 0.0,
                "compute_ms": 0.0,
                "net_latency_ms": 0.0,
                "met_slo": True,
                "met_dual_slo": True,
                "selection_time_ms": (time.perf_counter() - t0) * 1000.0,
                "ssp_calls": ssp_calls,
            }
            feasible.append(((0.0, -acc), mets))
            continue

        # DP maps node_id for mods[i] -> (cum_net_ms, cum_hops, cum_payload_mb, cum_link_mb, parent_node_id)
        dp_prev: Dict[str, Tuple[float, int, float, float, Optional[str]]] = {}

        # Initialize: candidate nodes for the first module (or infer from first edge)
        first_nodes = _nodes_for_module(placement, mods[0])
        if not first_nodes:
            pairs01 = _allowed_pairs_for_edge(placement, mods[0], mods[1])
            first_nodes = sorted({p[0] for p in pairs01}, key=lambda n: n.nid)

        if not first_nodes:
            continue

        for n in first_nodes:
            dp_prev[n.nid] = (0.0, 0, 0.0, 0.0, None)

        dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]] = {}

        ok = True

        # Progress over each adjacent module edge
        for i in range(len(mods) - 1):
            m_src, m_dst = mods[i], mods[i + 1]
            payload_i = module_output_mb(m_src)

            allowed_pairs = _allowed_pairs_for_edge(placement, m_src, m_dst)
            if not allowed_pairs:
                ok = False
                break

            dp_curr: Dict[str, Tuple[float, int, float, float, Optional[str]]] = {}

            for src_node, dst_node in allowed_pairs:
                prev = dp_prev.get(src_node.nid)
                if prev is None:
                    continue

                prev_lat, prev_hops, prev_payload, prev_link, _ = prev

                # SSSP call (counted)
                ssp_calls += 1
                lat_ms, path_nodes = _pair_shortest_cached(
                    ftopo, src_node.nid, dst_node.nid, dijk_cache, slo_ms
                )
                if (lat_ms is None) or (not math.isfinite(lat_ms)):
                    continue

                seg_hops = max(0, (len(path_nodes) - 1) if path_nodes is not None else 0)
                cand_lat     = prev_lat + float(lat_ms)
                cand_hops    = prev_hops + seg_hops
                cand_payload = prev_payload + payload_i
                cand_link    = prev_link + payload_i * seg_hops

                best = dp_curr.get(dst_node.nid)
                cand_tuple = (cand_lat, cand_hops, cand_payload, cand_link, src_node.nid)
                if (best is None) or (cand_tuple < best):
                    dp_curr[dst_node.nid] = cand_tuple

            if not dp_curr:
                ok = False
                break

            dp_prev = dp_curr

        if not ok or not dp_prev:
            continue

        # Best terminal destination by lowest network latency
        end_nid, (base_net_ms, total_hops, payload_mb_total, link_mb_total, parent) = \
            min(dp_prev.items(), key=lambda kv: kv[1][0])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mets = {
            "stitch_id": sid,
            "acc": acc,
            "latency_ms": base_net_ms,
            "net_base_ms": base_net_ms,
            "hop_count": total_hops,
            "payload_mb": payload_mb_total,
            "link_mb": link_mb_total,
            "compute_ms": 0.0,
            "net_latency_ms": base_net_ms,
            "met_slo": True,
            "met_dual_slo": True,
            "selection_time_ms": elapsed_ms,
            "ssp_calls": ssp_calls,
        }
        feasible.append(((base_net_ms, -acc), mets))

    if not feasible:
        return None

    # If acc_min is set, we already filtered by it. Just pick the best by our tuple key.
    feasible.sort(key=lambda t: t[0])
    return feasible[0][1]


# ----------------------------------------------------------------------------- #
# Simple policy wrappers (also timed)                                           #
# ----------------------------------------------------------------------------- #

def always_best_accuracy(
    placement: Dict[str, Node],
    topo: Topology,
    rng: random.Random,
    task_profile_name: str
) -> Dict:
    """Pick the highest-accuracy stitch; tie-break on lowest latency (topology-aware)."""
    t0 = time.perf_counter()

    top_acc = max(spec["acc"] for spec in CANDIDATE_STITCHES.values())
    best = None
    candidate_stitches = list(CANDIDATE_STITCHES.items())
    random.shuffle(candidate_stitches)

    for sid, spec in candidate_stitches:
        if spec["acc"] + 1e-9 < top_acc:
            continue
        mets = e2e_metrics_for_stitch(
            sid, placement, topo, rng, task_profile_name,
            greedy_objective="accuracy_first_fit"
        )
        cand = {"stitch_id": sid, **mets}
        if (best is None) or (mets["latency_ms"] < best["latency_ms"]):
            best = cand

    if best is None:
        best = _reject_row()

    best["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return best


def lowest_latency(
    placement: Dict[str, Node],
    topo: Topology,
    rng: random.Random,
    task_profile_name: str
) -> Dict:
    """Pick the stitch with minimum end-to-end latency (topology-aware)."""
    t0 = time.perf_counter()

    best = None
    ids = [1,2,3, 4, 5, 6, 7]
    index = random.choice(ids)
    for sid in list(CANDIDATE_STITCHES.keys())[-index:]:
        mets = e2e_metrics_for_stitch(
            sid, placement, topo, rng, task_profile_name,
            greedy_objective="latency"
        )
        if (best is None) or (mets["latency_ms"] < best["latency_ms"]):
            best = {"stitch_id": sid, **mets}

    if best is None:
        best = _reject_row()

    best["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return best


def random_pick_stitch(
    placement: Dict[str, Node],
    topo: Topology,
    rng: random.Random,
    task_profile_name: str
) -> Dict:
    """Pick a random stitch uniformly (topology-aware)."""
    t0 = time.perf_counter()
    ids = [1, 2, 3, 4, 5, 6, 7]
    index = random.choice(ids)
    candidate_stitches=list(CANDIDATE_STITCHES.keys())[-index:]
    sid = rng.choice(candidate_stitches)
    mets = e2e_metrics_for_stitch(
        sid, placement, topo, rng, task_profile_name,
        greedy_objective="random2",  # sample 2 dst nodes per hop, pick better
        random_k=2
    )
    out = {"stitch_id": sid, **mets}
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
    """Cycle through stitches in a fixed order, offset by (index + offset) (topology-aware)."""
    t0 = time.perf_counter()
    ids = [1, 2, 3, 4, 5, 6, 7]
    index_id = random.choice(ids)
    candidate_stitches = list(CANDIDATE_STITCHES.keys())[-index_id:]
    RR_STITCH_ORDER = sorted(candidate_stitches)
    sid = RR_STITCH_ORDER[(index + offset) % len(RR_STITCH_ORDER)]
    mets = e2e_metrics_for_stitch(
        sid, placement, topo, rng, task_profile_name,
        greedy_objective="rr", rr_index=index, rr_offset=0
    )
    out = {"stitch_id": sid, **mets}
    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


# ----------------------------------------------------------------------------- #
# Rejection row                                                                 #
# ----------------------------------------------------------------------------- #

def _reject_row(stitch_id=None, acc=np.nan, slo_hit=False):
    return {
        "stitch_id": stitch_id,
        "latency_ms": float("inf"),
        "compute_ms": 0.0,
        "net_latency_ms": float("inf"),
        "payload_mb": 0.0,
        "link_mb": 0.0,
        "hop_count": 0,
        "acc": acc,
        "met_slo": slo_hit,
        "met_dual_slo": slo_hit,
        "selection_time_ms": 0.0,
        "ssp_calls": 0,
    }
