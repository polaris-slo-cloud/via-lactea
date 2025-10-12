"""
Command-line entrypoint that reproduces the original script's outputs (graph-based),
now also reporting selector runtime (selection_time_ms) and SSSP call counts (ssp_calls).

New:
- RUN MODE switch via config.RUN_MODE / VL_RUN_MODE / --mode {task,workflow,both}
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
from typing import Optional  # 3.9-compatible Optional

from simu import config
from simu.topology import build_topology
from simu.simulator import simulate_task, simulate_workflow
from simu.stats_io import (
    agg_stats, agg_stats_by_profile, accuracy_stats, accuracy_stats_by_profile,
    slo_violation_rates_task, slo_violation_rates_task_by_profile, save_csv
)


# ---------- small helper to avoid merge suffix collisions ----------
def _prefixed(df: pd.DataFrame, keys, prefix: str) -> pd.DataFrame:
    if df is None:
        return None
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in keys}
    return df.rename(columns=rename_map)


def _resolve_run_mode(cli_mode: Optional[str]) -> str:
    """
    Priority: CLI (--mode) > env (VL_RUN_MODE) > config.RUN_MODE > default 'both'
    """
    if cli_mode:
        mode = cli_mode
    else:
        mode = os.getenv("VL_RUN_MODE", getattr(config, "RUN_MODE", "both"))
    mode = str(mode).strip().lower()
    if mode not in {"task", "workflow", "both"}:
        print(f"[warn] Unknown RUN_MODE='{mode}', falling back to 'both'.")
        mode = "both"
    return mode


def _run_task_section(topo, outdir: str):
    task_dfs = []
    for i, prof in enumerate(config.TASK_PROFILES_FOR_TASK):
        dfp = simulate_task(
            topo,
            config.NUM_RUNS_TASK,
            config.SLO_MS_TASK,
            seed=config.SEED + 1 + i,
            task_profile_name=prof,
        )
        task_dfs.append(dfp)
    task_df_all = pd.concat(task_dfs, ignore_index=True)

    base_cols = ["latency_ms", "payload_mb", "link_mb", "hop_count"]

    print("\n=== TASK (single-stage) — aggregated across profiles ===")
    for col in base_cols:
        print(f"{col}:")
        print(agg_stats(task_df_all, col).to_string(index=False))

    if "selection_time_ms" in task_df_all.columns:
        print("Selector runtime (ms):")
        print(agg_stats(task_df_all, "selection_time_ms").to_string(index=False))

    print("Accuracy (%):")
    print(accuracy_stats(task_df_all).to_string(index=False))
    print("TASK SLO violation rate (% of task requests) [based on SLO_MS_TASK]:")
    print(slo_violation_rates_task(task_df_all).to_string(index=False))

    print("\n=== TASK (single-stage) — by profile ===")
    for col in base_cols:
        print(f"{col} by profile:")
        print(agg_stats_by_profile(task_df_all, col).to_string(index=False))

    if "selection_time_ms" in task_df_all.columns:
        print("Selector runtime (ms) by profile:")
        print(agg_stats_by_profile(task_df_all, "selection_time_ms").to_string(index=False))

    print("Accuracy (%) by profile:")
    print(accuracy_stats_by_profile(task_df_all).to_string(index=False))
    print("SLO violation rate by profile (% of task requests):")
    print(slo_violation_rates_task_by_profile(task_df_all).to_string(index=False))

    # New: E2E SLO excess stats (ms and %), aggregated across profiles
    if {"slo_excess_ms", "slo_excess_pct"}.issubset(task_df_all.columns):
        def _agg_excess(df, col):
            g = df.groupby("strategy")[col]
            return (g.mean().rename("mean_" + col)
                    .to_frame()
                    .join(g.median().rename("p50_" + col))
                    .join(g.quantile(0.95).rename("p95_" + col)))
        print("TASK SLO excess (ms) across strategies:")
        print(_agg_excess(task_df_all, "slo_excess_ms").to_string())
        print("TASK SLO excess (%) across strategies:")
        print(_agg_excess(task_df_all, "slo_excess_pct").to_string())
        save_csv(_agg_excess(task_df_all, "slo_excess_ms"), outdir, "task_slo_excess_ms_by_strategy.csv")
        save_csv(_agg_excess(task_df_all, "slo_excess_pct"), outdir, "task_slo_excess_pct_by_strategy.csv")

    # Save task CSVs (existing)
    save_csv(task_df_all, outdir, "task_runs_all.csv")
    save_csv(agg_stats_by_profile(task_df_all, "latency_ms"), outdir, "task_summary_latency_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "payload_mb"), outdir, "task_summary_payload_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "link_mb"),    outdir, "task_summary_link_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "hop_count"),  outdir, "task_summary_hopcount_by_strategy_profile.csv")
    save_csv(accuracy_stats_by_profile(task_df_all), outdir, "task_accuracy_by_strategy_profile.csv")
    save_csv(slo_violation_rates_task_by_profile(task_df_all), outdir, "task_slo_violation_by_profile.csv")

    # Save new task CSVs
    if "selection_time_ms" in task_df_all.columns:
        save_csv(agg_stats_by_profile(task_df_all, "selection_time_ms"), outdir, "task_summary_selector_time_by_strategy_profile.csv")
    if "ssp_calls" in task_df_all.columns:
        save_csv(agg_stats_by_profile(task_df_all, "ssp_calls"), outdir, "task_summary_ssp_calls_by_strategy_profile.csv")

    # ---- TASK wide joined summary with safe prefixes ----
    keys = ["profile", "strategy"]
    frames = [
        _prefixed(agg_stats_by_profile(task_df_all, "latency_ms"), keys, "latency"),
        _prefixed(accuracy_stats_by_profile(task_df_all),          keys, "acc"),
        _prefixed(agg_stats_by_profile(task_df_all, "payload_mb"), keys, "payload"),
        _prefixed(agg_stats_by_profile(task_df_all, "link_mb"),    keys, "link"),
        _prefixed(agg_stats_by_profile(task_df_all, "hop_count"),  keys, "hops"),
    ]
    if "selection_time_ms" in task_df_all.columns:
        frames.append(_prefixed(agg_stats_by_profile(task_df_all, "selection_time_ms"), keys, "selector"))
    if "ssp_calls" in task_df_all.columns:
        frames.append(_prefixed(agg_stats_by_profile(task_df_all, "ssp_calls"), keys, "ssp"))

    frames = [f for f in frames if f is not None]
    _task_wide = frames[0]
    for f in frames[1:]:
        _task_wide = _task_wide.merge(f, on=keys)

    save_csv(_task_wide, outdir, "task_summary_ALL_by_strategy_profile.csv")


def _run_workflow_section(topo, outdir: str):
    wf_df = simulate_workflow(
        topo,
        config.NUM_RUNS_WORKFLOW,
        config.WORKFLOW_STAGES,
        config.SLO_MS_STAGE,
        seed=config.SEED + 10,
        stage_profiles=config.TASK_PROFILES_FOR_WORKFLOW,
    )

    base_cols = ["latency_ms", "payload_mb", "link_mb", "hop_count"]

    print(f"\n=== WORKFLOW ({config.WORKFLOW_STAGES} stages) ===")
    for col in base_cols:
        print(f"{col}:")
        print(agg_stats(wf_df, col).to_string(index=False))

    if "selection_time_ms" in wf_df.columns:
        print("Selector runtime (ms):")
        print(agg_stats(wf_df, "selection_time_ms").to_string(index=False))

    # Save workflow CSVs (existing)
    save_csv(wf_df, outdir, "workflow_runs.csv")
    save_csv(agg_stats(wf_df, "latency_ms"), outdir, "workflow_summary_latency_by_strategy.csv")
    save_csv(agg_stats(wf_df, "payload_mb"), outdir, "workflow_summary_payload_by_strategy.csv")
    save_csv(agg_stats(wf_df, "link_mb"),    outdir, "workflow_summary_link_by_strategy.csv")
    save_csv(agg_stats(wf_df, "hop_count"),  outdir, "workflow_summary_hopcount_by_strategy.csv")
    save_csv(accuracy_stats(wf_df), outdir, "workflow_accuracy_by_strategy.csv")

    # Save new workflow CSVs
    if "selection_time_ms" in wf_df.columns:
        save_csv(agg_stats(wf_df, "selection_time_ms"), outdir, "workflow_summary_selector_time_by_strategy.csv")
    if "ssp_calls" in wf_df.columns:
        save_csv(agg_stats(wf_df, "ssp_calls"), outdir, "workflow_summary_ssp_calls_by_strategy.csv")

    # ---- WORKFLOW wide joined summary with safe prefixes ----
    keys_wf = ["strategy"]
    frames_wf = [
        _prefixed(agg_stats(wf_df, "latency_ms"), keys_wf, "latency"),
        _prefixed(accuracy_stats(wf_df),          keys_wf, "acc"),
        _prefixed(agg_stats(wf_df, "payload_mb"), keys_wf, "payload"),
        _prefixed(agg_stats(wf_df, "link_mb"),    keys_wf, "link"),
        _prefixed(agg_stats(wf_df, "hop_count"),  keys_wf, "hops"),
    ]
    if "selection_time_ms" in wf_df.columns:
        frames_wf.append(_prefixed(agg_stats(wf_df, "selection_time_ms"), keys_wf, "selector"))
    if "ssp_calls" in wf_df.columns:
        frames_wf.append(_prefixed(agg_stats(wf_df, "ssp_calls"), keys_wf, "ssp"))

    frames_wf = [f for f in frames_wf if f is not None]
    _wf_wide = frames_wf[0]
    for f in frames_wf[1:]:
        _wf_wide = _wf_wide.merge(f, on=keys_wf)

    save_csv(_wf_wide, outdir, "workflow_summary_ALL_by_strategy.csv")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["task", "workflow", "both"], help="Override run mode")
    args, _ = parser.parse_known_args()

    run_mode = _resolve_run_mode(args.mode)

    rng = random.Random(config.SEED)

    # -------- BUILD THE GRAPH --------
    topo = build_topology(
        sats_per_ring     = config.SATS_PER_RING,
        num_rings         = config.NUM_RINGS,
        cloud_count       = config.NODE_COUNTS["cloud"],
        edge_count        = config.NODE_COUNTS["edge"],
        isl_neighbor_span = getattr(config, "ISL_NEIGHBOR_SPAN", 1),
        gateways_per_ring = getattr(config, "GATEWAYS_PER_RING", 2),
        inter_ring_links  = getattr(config, "INTER_RING_LINKS", True),
    )

    outdir = config.ensure_results_dir()
    print(f"[info] RUN_MODE = {run_mode}")
    print(f"[info] Saving all CSVs to: {outdir}")

    if run_mode in ("task", "both"):
        _run_task_section(topo, outdir)

    if run_mode in ("workflow", "both"):
        _run_workflow_section(topo, outdir)


if __name__ == "__main__":
    main()
