"""
Stitch-selection policies (SLO-first, Best-Acc, Lowest-Latency, Random, Round-Robin).
Adds selection_time_ms to report wall-clock policy runtime, plus ssp_calls for DP selector.

Selector SLO policy:
- Stage SLO = config.SLO_MS_STAGE (net-only, used to prune & report inside DP).
- Per-edge propagation cap is controlled separately via per_edge_prop_cap_ms.

Caching (ONLY for SLO-first):
- Reports payload_mb_cached and link_mb_cached assuming per-layer caching policy:
  * config.CACHEABLE_LAYER_PATTERNS (glob-style, e.g., "swin_stage1_b*")
  * config.CACHE_FIRST_RUN (True -> first-run miss; False -> steady-state hit)
For all other strategies, cached fields are present but identical to uncached fields.
"""
import math
import random
import time
import fnmatch
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from .profiles import CANDIDATE_STITCHES
from .runtime import e2e_metrics_for_stitch, _nodes_for_module
from .selection_algo import _pair_shortest_cached
from .topology import Node, module_output_mb, Topology, _filtered_topology_view

# ----------------------------------------------------------------------------- #
# SLO helper (stage SLO only)                                                   #
# ----------------------------------------------------------------------------- #

def _stage_slo_fields(total_net_ms: float):
    """
    Stage/task SLO check (NET-ONLY) against config.SLO_MS_STAGE.
    Returns (met_slo, excess_ms, excess_pct, slo_ms_used).
    """
    slo_ms = getattr(config, "SLO_MS_STAGE", None)
    if slo_ms is None or not math.isfinite(slo_ms) or slo_ms <= 0:
        return True, 0.0, float("nan"), slo_ms
    excess = max(0.0, float(total_net_ms) - float(slo_ms))
    met = (excess <= 1e-9)
    pct = 100.0 * excess / float(slo_ms) if slo_ms > 0 else float("nan")
    return met, excess, pct, slo_ms

# ----------------------------------------------------------------------------- #
# Caching helpers (used ONLY by SLO-first)                                      #
# ----------------------------------------------------------------------------- #

def _is_cacheable_layer(layer_name: str) -> bool:
    """
    Match a module/part (e.g., 'resnet_layer1', 'swin_stage1_b3') against patterns in
    config.CACHEABLE_LAYER_PATTERNS. Simple glob-style patterns (fnmatch) are supported.
    """
    pats = getattr(config, "CACHEABLE_LAYER_PATTERNS", [])
    if not pats:
        return False
    for pat in pats:
        if fnmatch.fnmatch(layer_name, pat):
            return True
    return False

def _cached_payload_mb_for_layer(layer_name: str, uncached_mb: float) -> float:
    """
    Return MB to account for this layer in the 'cached' totals.

    First-run (config.CACHE_FIRST_RUN=True): pay full miss (same as uncached).
    Steady-state (False): cacheable layers contribute 0 MB; others unchanged.
    """
    first_run = bool(getattr(config, "CACHE_FIRST_RUN", True))
    if first_run:
        return float(uncached_mb)
    return 0.0 if _is_cacheable_layer(layer_name) else float(uncached_mb)

# ----------------------------------------------------------------------------- #
# Pair filtering                                                                #
# ----------------------------------------------------------------------------- #

def _allowed_pairs_for_edge(placement: Dict, m_src: str, m_dst: str) -> List[Tuple[Node, Node]]:
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
# SLO-first (ONLY path that applies caching)                                    #
# ----------------------------------------------------------------------------- #

def choose_stitch_for_task(
    placement: Dict,
    topo: Topology,
    rng: random.Random,
    task_profile_name: str,
    *,
    slo_ms: Optional[float] = None,          # IGNORED; we always use config.SLO_MS_STAGE
    acc_min: Optional[float] = None,
    per_edge_prop_cap_ms: Optional[float] = None,
) -> Optional[Dict]:
    """
    Returns a metrics dict for the chosen stitch, including:
      - selection_time_ms (selector runtime)
      - ssp_calls (SSSP invocation count)

    IMPORTANT:
      - Stage SLO = config.SLO_MS_STAGE is used (net-only). Any passed slo_ms is ignored.
      - Per-edge propagation caps are applied via per_edge_prop_cap_ms and the filtered topology.
      - CACHED FIELDS are computed ONLY here (SLO-first). Other strategies mirror uncached.
    """
    stage_slo_ms = getattr(config, "SLO_MS_STAGE", None)

    t0 = time.perf_counter()
    ssp_calls = 0

    ftopo = _filtered_topology_view(topo, per_edge_prop_cap_ms)
    feasible: List[Tuple[Tuple[float, float], Dict]] = []

    candidate_stitches = list(CANDIDATE_STITCHES.items())
    random.shuffle(candidate_stitches)

    for sid, spec in candidate_stitches:
        acc = float(spec["acc"])
        if acc_min is not None and acc < acc_min:
            continue

        mods: List[str] = spec["modules"]
        if len(mods) < 2:
            met_slo, exc_ms, exc_pct, slo_used = _stage_slo_fields(0.0)
            mets = {
                "stitch_id": sid,
                "acc": acc,
                "latency_ms": 0.0,           # selection-time = net-only
                "net_base_ms": 0.0,
                "hop_count": 0,
                # Uncached baseline
                "payload_mb": 0.0,
                "link_mb": 0.0,
                # Cached view (first-run == same as uncached; steady-state may differ)
                "payload_mb_cached": 0.0,
                "link_mb_cached": 0.0,
                "compute_ms": 0.0,
                "net_latency_ms": 0.0,
                "met_slo": met_slo,
                "slo_ms": slo_used,
                "slo_excess_ms": exc_ms,
                "slo_excess_pct": exc_pct,
                "met_dual_slo": met_slo and (acc_min is None or acc >= acc_min),
                "selection_time_ms": (time.perf_counter() - t0) * 1000.0,
                "ssp_calls": ssp_calls,
            }
            feasible.append(((0.0, -acc), mets))
            continue

        # DP state:
        # nid -> (cum_net_ms, cum_hops,
        #         cum_payload_mb, cum_link_mb,
        #         cum_payload_mb_cached, cum_link_mb_cached,
        #         parent_nid)
        dp_prev: Dict[str, Tuple[float, int, float, float, float, float, Optional[str]]] = {}

        first_nodes = _nodes_for_module(placement, mods[0])
        if not first_nodes:
            pairs01 = _allowed_pairs_for_edge(placement, mods[0], mods[1])
            first_nodes = sorted({p[0] for p in pairs01}, key=lambda n: n.nid)
        if not first_nodes:
            continue

        for n in first_nodes:
            dp_prev[n.nid] = (0.0, 0, 0.0, 0.0, 0.0, 0.0, None)

        dijk_cache: Dict[str, Dict[str, Tuple[Optional[float], Optional[List[str]]]]] = {}
        ok = True

        if getattr(config, "CACHE_DEBUG", False):
            print(f"[cache] evaluating stitch {sid} modules={mods}")

        for i in range(len(mods) - 1):
            m_src, m_dst = mods[i], mods[i + 1]
            payload_i = module_output_mb(m_src)
            payload_i_cached = _cached_payload_mb_for_layer(m_src, payload_i)

            if getattr(config, "CACHE_DEBUG", False):
                first_run = bool(getattr(config, "CACHE_FIRST_RUN", True))
                hit = (not first_run) and _is_cacheable_layer(m_src)
                tag = "HIT " if hit else "MISS"
                print(f"[cache] layer={m_src:22s} uncached={payload_i:.3f}MB  "
                      f"cached_contrib={payload_i_cached:.3f}MB  [{tag}]")

            allowed_pairs = _allowed_pairs_for_edge(placement, m_src, m_dst)
            if not allowed_pairs:
                ok = False
                break

            dp_curr: Dict[str, Tuple[float, int, float, float, float, float, Optional[str]]] = {}

            for src_node, dst_node in allowed_pairs:
                prev = dp_prev.get(src_node.nid)
                if prev is None:
                    continue

                (prev_lat, prev_hops,
                 prev_payload, prev_link,
                 prev_payload_cached, prev_link_cached,
                 _parent) = prev

                # Use stage SLO for pruning inside SSSP
                ssp_calls += 1
                lat_ms, path_nodes = _pair_shortest_cached(
                    ftopo, src_node.nid, dst_node.nid, dijk_cache, stage_slo_ms
                )
                if (lat_ms is None) or (not math.isfinite(lat_ms)):
                    continue

                seg_hops = max(0, (len(path_nodes) - 1) if path_nodes is not None else 0)

                cand_lat   = prev_lat + float(lat_ms)
                cand_hops  = prev_hops + seg_hops

                # Uncached totals (original behavior)
                cand_payload = prev_payload + payload_i
                cand_link    = prev_link + payload_i * seg_hops

                # Cached totals (respect config.CACHE_* policy)
                cand_payload_cached = prev_payload_cached + payload_i_cached
                cand_link_cached    = prev_link_cached + payload_i_cached * seg_hops

                best = dp_curr.get(dst_node.nid)
                cand_tuple = (cand_lat, cand_hops,
                              cand_payload, cand_link,
                              cand_payload_cached, cand_link_cached,
                              src_node.nid)
                if (best is None) or (cand_tuple < best):
                    dp_curr[dst_node.nid] = cand_tuple

            if not dp_curr:
                ok = False
                break

            dp_prev = dp_curr

        if not ok or not dp_prev:
            continue

        # Best terminal destination by lowest network latency
        end_nid, (base_net_ms, total_hops,
                  payload_mb_total, link_mb_total,
                  payload_mb_cached_total, link_mb_cached_total,
                  parent) = min(dp_prev.items(), key=lambda kv: kv[1][0])

        met_slo, exc_ms, exc_pct, slo_used = _stage_slo_fields(base_net_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mets = {
            "stitch_id": sid,
            "acc": acc,
            "latency_ms": base_net_ms,      # net-only at selection time
            "net_base_ms": base_net_ms,
            "hop_count": total_hops,

            # Uncached (original)
            "payload_mb": payload_mb_total,
            "link_mb": link_mb_total,

            # Cached (first-run == same as uncached; steady-state may be smaller)
            "payload_mb_cached": payload_mb_cached_total,
            "link_mb_cached": link_mb_cached_total,

            "compute_ms": 0.0,              # unknown here
            "net_latency_ms": base_net_ms,
            "met_slo": met_slo,             # stage SLO
            "slo_ms": slo_used,
            "slo_excess_ms": exc_ms,
            "slo_excess_pct": exc_pct,
            "met_dual_slo": met_slo and (acc_min is None or acc >= acc_min),
            "selection_time_ms": elapsed_ms,
            "ssp_calls": ssp_calls,
        }
        feasible.append(((base_net_ms, -acc), mets))

    if not feasible:
        return None

    feasible.sort(key=lambda t: t[0])
    return feasible[0][1]

# ----------------------------------------------------------------------------- #
# Other strategies (NO caching effect; cached fields mirror uncached)           #
# ----------------------------------------------------------------------------- #

def _mirror_cached_fields(mets: Dict) -> Dict:
    """Ensure cached fields exist and mirror uncached values."""
    out = dict(mets)
    out["payload_mb_cached"] = float(out.get("payload_mb", 0.0))
    out["link_mb_cached"]    = float(out.get("link_mb", 0.0))
    return out

def always_best_accuracy(placement: Dict[str, Node], topo: Topology, rng: random.Random, task_profile_name: str) -> Dict:
    t0 = time.perf_counter()
    top_acc = max(spec["acc"] for spec in CANDIDATE_STITCHES.values())
    best = None
    candidate_stitches = list(CANDIDATE_STITCHES.items())
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
    for sid in CANDIDATE_STITCHES.keys():
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
    sid = rng.choice(list(CANDIDATE_STITCHES.keys()))
    mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="random2", random_k=2)
    out = {"stitch_id": sid, **mets}
    out = _mirror_cached_fields(out)
    out["selection_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return out

def round_robin_pick_stitch(index: int, placement: Dict[str, Node], topo: Topology, rng: random.Random, task_profile_name: str, offset: int = config.RR_START_OFFSET) -> Dict:
    t0 = time.perf_counter()
    RR_STITCH_ORDER = sorted(CANDIDATE_STITCHES.keys())
    sid = RR_STITCH_ORDER[(index + offset) % len(RR_STITCH_ORDER)]
    mets = e2e_metrics_for_stitch(sid, placement, topo, rng, task_profile_name, greedy_objective="rr", rr_index=index, rr_offset=offset)
    out = {"stitch_id": sid, **mets}
    out = _mirror_cached_fields(out)
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
        "payload_mb_cached": 0.0,
        "link_mb_cached": 0.0,
        "hop_count": 0,
        "acc": acc,
        "met_slo": slo_hit,
        # Keep workflow SLO metadata here; selector uses stage SLO internally.
        "slo_ms": getattr(config, "SLO_MS_WORKFLOW", None),
        "slo_excess_ms": float("inf") if not slo_hit else 0.0,
        "slo_excess_pct": float("nan"),
        "met_dual_slo": slo_hit,
        "selection_time_ms": 0.0,
        "ssp_calls": 0,
    }
