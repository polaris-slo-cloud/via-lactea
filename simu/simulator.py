"""
Simulation entry points: single-task and multi-stage workflow (graph-based).
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Callable

import pandas as pd

from . import config
from .selection import (
    always_best_accuracy,
    lowest_latency,
    random_pick_stitch,
    choose_stitch_for_task,
    round_robin_pick_stitch,
    _reject_row,
)
from .placement import assign_modules_to_nodes, build_local_allowed_pairs_all_stitches
from .profiles import CANDIDATE_STITCHES
from .topology import Topology


# ---------------------------
# Shared helpers
# ---------------------------

def _all_modules() -> List[str]:
    """All modules that can appear in any stitch."""
    return sorted({m for s in CANDIDATE_STITCHES.values() for m in s["modules"]})


def _slo_fields(total_ms: float, slo_ms: Optional[float]):
    """Return (met_slo, excess_ms, excess_pct). If slo_ms is None/invalid, treat as met."""
    if slo_ms is None or not math.isfinite(slo_ms) or slo_ms <= 0:
        return True, 0.0, float("nan")
    excess = max(0.0, float(total_ms) - float(slo_ms))
    met = (excess <= 1e-9)
    pct = (100.0 * excess / float(slo_ms))
    return met, excess, pct


def _build_run_placement(
    topo: Topology,
    all_modules: List[str],
    rng: random.Random,
    run_idx: int,
) -> Dict:
    """
    Build a per-run placement:
      1) global, balanced module -> [Node] mapping
      2) per-edge randomized neighborhoods computed once across all stitches
    """
    base = assign_modules_to_nodes(
        modules=all_modules,
        topo=topo,
        run_idx=run_idx,
        shuffle_once_seed=config.SEED,
    )

    # Tunables chosen to increase variance while staying fast.
    # Feel free to tweak centrally here without touching call sites.
    return build_local_allowed_pairs_all_stitches(
        topo=topo,
        base_placement=base,
        rng=rng,
        # neighborhood size
        k_neighbors=3,
        jitter_neighbors=1,
        # variance knobs (match placement.py parameters)
        edge_keep_prob=0.55,
        buckets=16,
        local_ratio=0.60,
        wild_prob=0.12,
        # pool size
        max_per_module=80,
        # safety
        check_reachability=True,
    )


# Strategy registry (names → callable or sentinel for RR)
_Strategy = Tuple[str, Optional[Callable[..., Dict]]]

STRATEGIES: List[_Strategy] = [
    ("SLO-first", lambda pl, t, r, prof, slo: choose_stitch_for_task(
        placement=pl, topo=t, rng=r, task_profile_name=prof,
        slo_ms=slo, acc_min=config.SLO_ACC_MIN
    )),
    ("Best-Acc",       always_best_accuracy),
    ("Lowest-Latency", lowest_latency),
    ("Random",         random_pick_stitch),
    ("Round-Robin",    None),  # handled specially
]


def _eval_strategy_once(
    strategy_name: str,
    picker: Optional[Callable[..., Dict]],
    placement: Dict,
    topo: Topology,
    rng: random.Random,
    profile: str,
    *,
    slo_ms: float,
    rr_index: int = 0,
) -> Dict:
    """
    Run one strategy and return a metrics dict with a **normalized E2E SLO**:
    - `latency_ms` is assumed to be total (compute + net) from the picker (wrappers do this).
    - We recompute `met_slo`, `slo_ms`, `slo_excess_ms`, `slo_excess_pct` here so all strategies
      are compared against the same simulator-provided SLO, even if the picker used a different notion.
    """
    if strategy_name == "Round-Robin":
        res = round_robin_pick_stitch(rr_index, placement, topo, rng, profile)
    elif strategy_name == "SLO-first":
        # picker here is the lambda defined in STRATEGIES that requires slo_ms
        res = picker(placement, topo, rng, profile, slo_ms)  # type: ignore[arg-type]
    else:
        res = picker(placement, topo, rng, profile)  # type: ignore[misc]

    if res is None:
        res = _reject_row()

    out = dict(res)  # copy before annotating
    total = float(out.get("latency_ms", math.inf))
    met, exc_ms, exc_pct = _slo_fields(total, slo_ms)
    out["met_slo"] = met
    out["slo_ms"] = slo_ms
    out["slo_excess_ms"] = exc_ms
    out["slo_excess_pct"] = exc_pct
    return out


# ---------------------------
# Single-task simulation
# ---------------------------

def simulate_task(
    topo: Topology,
    num_runs: int,
    slo_ms: float,
    seed: int,
    task_profile_name: str,
) -> pd.DataFrame:
    """
    For a fixed profile, run multiple independent placements and evaluate
    the strategies; return a long DataFrame of per-run results.
    """
    base_rng = random.Random(seed)
    records: List[Dict] = []

    all_modules = _all_modules()

    for run in range(num_runs):
        rng = random.Random(base_rng.getrandbits(64))

        # one placement per run (shared across strategies)
        placement = _build_run_placement(topo, all_modules, rng, run_idx=run)

        for name, picker in STRATEGIES:
            rr_idx = run  # RR index policy: one step per run
            mets = _eval_strategy_once(
                name, picker, placement, topo, rng, task_profile_name,
                slo_ms=slo_ms, rr_index=rr_idx
            )
            records.append({"run": run, "strategy": name, "profile": task_profile_name, **mets})

    return pd.DataFrame(records)


# ---------------------------
# Workflow simulation
# ---------------------------

def simulate_workflow(
    topo: Topology,
    num_runs: int,
    stages: int,
    slo_ms_stage: float,
    seed: int,
    stage_profiles: List[str],
) -> pd.DataFrame:
    """
    Simulate a K-stage workflow. For each stage, create a fresh placement and
    apply each selection strategy; accumulate totals over stages.

    We aggregate:
      - latency_ms:            sum over stages
      - payload_mb / link_mb:  sum over stages
      - hop_count:             sum over stages
      - selection_time_ms:     sum over stages (to reflect total selector runtime)
      - ssp_calls:             sum over stages
      - met_slo:               True only if *all* stages met the (E2E) stage SLO
    """
    assert len(stage_profiles) == stages
    base_rng = random.Random(seed)
    records: List[Dict] = []

    all_modules = _all_modules()

    for run in range(num_runs):
        rng = random.Random(base_rng.getrandbits(64))

        # Build and store placements for each stage ONCE per run
        stage_placements: List[Dict] = []
        for k in range(stages):
            # Make run+stage unique to vary placements
            placement = _build_run_placement(topo, all_modules, rng, run_idx=run * stages + k)
            stage_placements.append(placement)

        # Evaluate each strategy across stages, aggregating totals
        for name, picker in STRATEGIES:
            lat_sum = payload_sum = link_sum = 0.0
            hops_sum = 0
            sel_time_sum = 0.0
            ssp_calls_sum = 0
            acc_last: Optional[float] = None
            met_stage_count = 0
            met_task_slo_count = 0  # same as met_stage_count, but kept for clarity

            for k in range(stages):
                prof = stage_profiles[k]
                placement = stage_placements[k]

                rr_idx = run * stages + k  # RR index policy for workflows
                mets = _eval_strategy_once(
                    name, picker, placement, topo, rng, prof,
                    slo_ms=slo_ms_stage, rr_index=rr_idx
                )

                latency_ms = mets.get("latency_ms", math.inf)
                lat_sum     += latency_ms
                payload_sum += mets.get("payload_mb", 0.0)
                link_sum    += mets.get("link_mb", 0.0)
                hops_sum    += int(mets.get("hop_count", 0))
                acc_last     = mets.get("acc", acc_last)

                # accumulate selector instrumentation if present
                sel_time_sum += float(mets.get("selection_time_ms", 0.0))
                ssp_calls_sum += int(mets.get("ssp_calls", 0))

                # E2E stage SLO (already normalized in _eval_strategy_once)
                met = bool(mets.get("met_slo", False))
                met_stage_count += int(met)
                met_task_slo_count += int(met)

            records.append({
                "run": run,
                "strategy": name,
                "latency_ms": lat_sum,
                "acc": (0.0 if acc_last is None else acc_last),
                "payload_mb": payload_sum,
                "link_mb": link_sum,
                "hop_count": hops_sum,
                "selection_time_ms": sel_time_sum,
                "ssp_calls": ssp_calls_sum,
                "met_slo": (met_stage_count == stages),  # all stages met the (E2E) stage SLO
                "stages_met": met_stage_count,
                "met_task_slo_all": (met_task_slo_count == stages),
                "stages_met_task_slo": met_task_slo_count,
                "stages": stages,
                "profiles": "|".join(stage_profiles),
            })

    return pd.DataFrame(records)
