"""
Simulation entry points: single-task and multi-stage workflow (graph-based).
"""

import math
import os
import random
from typing import List, Dict, Tuple, Optional, Callable, Iterable

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

# Canonical order (used for output ordering too)
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

# Name → impl for fast lookup
_STRATEGY_MAP: Dict[str, Optional[Callable[..., Dict]]] = {k: v for k, v in STRATEGIES}


def _parse_enabled_list(raw: Optional[Iterable[str]]) -> List[str]:
    if raw is None:
        return []
    out: List[str] = []
    for item in raw:
        if item is None:
            continue
        name = str(item).strip()
        if not name:
            continue
        out.append(name)
    return out


def _resolve_enabled_strategies(explicit: Optional[Iterable[str]] = None) -> List[_Strategy]:
    """
    Priority:
      1) explicit (function arg `enabled_strategies`)
      2) env var VL_BASELINES (comma or pipe separated)
      3) config.ENABLED_BASELINES (list or 'all')
      4) default: all strategies
    Returns list preserving canonical STRATEGIES order.
    """
    # 1) function arg
    if explicit is not None:
        names = _parse_enabled_list(explicit)
        # allow "all"
        if len(names) == 1 and names[0].lower() == "all":
            names = [name for name, _ in STRATEGIES]
    else:
        # 2) env var
        env_val = os.getenv("VL_BASELINES", "").strip()
        if env_val:
            # split by comma or pipe
            parts = [p.strip() for chunk in env_val.split("|") for p in chunk.split(",")]
            names = _parse_enabled_list(parts)
            if len(names) == 1 and names[0].lower() == "all":
                names = [name for name, _ in STRATEGIES]
        else:
            # 3) config
            cfg_val = getattr(config, "ENABLED_BASELINES", "all")
            if isinstance(cfg_val, str):
                if cfg_val.strip().lower() == "all" or not cfg_val.strip():
                    names = [name for name, _ in STRATEGIES]
                else:
                    parts = [p.strip() for chunk in cfg_val.split("|") for p in chunk.split(",")]
                    names = _parse_enabled_list(parts)
            elif isinstance(cfg_val, (list, tuple, set)):
                names = _parse_enabled_list(cfg_val)
                if not names:
                    names = [name for name, _ in STRATEGIES]
            else:
                names = [name for name, _ in STRATEGIES]

    # Filter to valid and preserve canonical order
    valid = set(_STRATEGY_MAP.keys())
    selected = [name for name, _ in STRATEGIES if name in names and name in valid]

    # Warn about unknowns (print once)
    unknown = [n for n in names if n not in valid]
    if unknown:
        print(f"[warn] Unknown baseline(s) ignored: {', '.join(unknown)}")

    # Fallback to all if user filtered out everything
    if not selected:
        selected = [name for name, _ in STRATEGIES]

    return [(name, _STRATEGY_MAP[name]) for name in selected]


def _normalize_metrics_for_csv(out: Dict) -> Dict:
    """
    Ensure all rows contain the same keys so CSVs are tidy.
    Strategies that don't populate the new cached fields will get zeros.
    """
    # numeric defaults
    out.setdefault("stitch_id", None)
    out.setdefault("payload_mb", 0.0)
    out.setdefault("link_mb", 0.0)
    out.setdefault("payload_mb_cached", 0.0)
    out.setdefault("link_mb_cached", 0.0)
    out.setdefault("selection_time_ms", 0.0)
    out.setdefault("ssp_calls", 0)
    out.setdefault("hop_count", 0)
    out.setdefault("acc", float("nan"))
    out.setdefault("latency_ms", float("inf"))
    out.setdefault("net_latency_ms", float("inf"))
    # SLO fields may be overwritten by caller; make sure they exist
    out.setdefault("met_slo", False)
    out.setdefault("slo_ms", None)
    out.setdefault("slo_excess_ms", 0.0)
    out.setdefault("slo_excess_pct", float("nan"))
    return out


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
    out = _normalize_metrics_for_csv(out)

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
    csv_path: Optional[str] = None,
    enabled_strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    For a fixed profile, run multiple independent placements and evaluate
    the strategies; return a long DataFrame of per-run results.

    If csv_path is provided, results are also written to that CSV.
    Limit which baselines run via:
      • enabled_strategies=["SLO-first","Round-Robin"]
      • or env VL_BASELINES, or config.ENABLED_BASELINES
    """
    base_rng = random.Random(seed)
    records: List[Dict] = []

    all_modules = _all_modules()
    active_strategies = _resolve_enabled_strategies(enabled_strategies)

    for run in range(num_runs):
        rng = random.Random(base_rng.getrandbits(64))

        # one placement per run (shared across strategies)
        placement = _build_run_placement(topo, all_modules, rng, run_idx=run)

        for name, picker in active_strategies:
            rr_idx = run  # RR index policy: one step per run
            mets = _eval_strategy_once(
                name, picker, placement, topo, rng, task_profile_name,
                slo_ms=slo_ms, rr_index=rr_idx
            )
            records.append({"run": run, "strategy": name, "profile": task_profile_name, **mets})

    df = pd.DataFrame(records)

    if csv_path:
        df.to_csv(csv_path, index=False)

    return df


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
    csv_path: Optional[str] = None,
    enabled_strategies: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Simulate a K-stage workflow. For each stage, create a fresh placement and
    apply each selection strategy; accumulate totals over stages.

    We aggregate:
      - latency_ms:                sum over stages
      - payload_mb / link_mb:      sum over stages
      - payload_mb_cached / link_mb_cached: sum over stages
      - hop_count:                 sum over stages
      - selection_time_ms:         sum over stages (reflect total selector runtime)
      - ssp_calls:                 sum over stages
      - met_slo:                   True only if *all* stages met the (E2E) stage SLO
    """
    assert len(stage_profiles) == stages
    base_rng = random.Random(seed)
    records: List[Dict] = []

    all_modules = _all_modules()
    active_strategies = _resolve_enabled_strategies(enabled_strategies)

    for run in range(num_runs):
        rng = random.Random(base_rng.getrandbits(64))
        stitch_run_idx = 0
        # Build and store placements for each stage ONCE per run
        stage_placements: List[Dict] = []
        for k in range(stages):
            # Make run+stage unique to vary placements
            placement = _build_run_placement(topo, all_modules, rng, run_idx=run * stages + k)
            stage_placements.append(placement)

        # Evaluate each strategy across stages, aggregating totals
        for name, picker in active_strategies:
            lat_sum = 0.0
            payload_sum = 0.0
            link_sum = 0.0
            payload_cached_sum = 0.0
            link_cached_sum = 0.0
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
                lat_sum            += latency_ms
                payload_sum        += float(mets.get("payload_mb", 0.0))
                link_sum           += float(mets.get("link_mb", 0.0))
                payload_cached_sum += float(mets.get("payload_mb_cached", 0.0))
                link_cached_sum    += float(mets.get("link_mb_cached", 0.0))
                hops_sum           += int(mets.get("hop_count", 0))
                acc_last            = mets.get("acc", acc_last)

                # accumulate selector instrumentation if present
                sel_time_sum += float(mets.get("selection_time_ms", 0.0))
                ssp_calls_sum += int(mets.get("ssp_calls", 0))

                # E2E stage SLO (already normalized in _eval_strategy_once)
                met = bool(mets.get("met_slo", False))
                met_stage_count += int(met)
                met_task_slo_count += int(met)
                stitch_run_idx=mets.get("stitch_id", 0)

            records.append({
                "run": run,
                "stitch_id": stitch_run_idx,
                "strategy": name,
                "latency_ms": lat_sum,
                "acc": (0.0 if acc_last is None else acc_last),
                "payload_mb": payload_sum,
                "link_mb": link_sum,
                "payload_mb_cached": payload_cached_sum,
                "link_mb_cached": link_cached_sum,
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

    df = pd.DataFrame(records)

    if csv_path:
        df.to_csv(csv_path, index=False)

    return df
