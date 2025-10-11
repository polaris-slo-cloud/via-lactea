"""
Command-line entrypoint that reproduces the original script's outputs (graph-based),
now also reporting selector runtime (selection_time_ms) and SSSP call counts (ssp_calls).
"""

import random
import numpy as np
import pandas as pd

from simu import config
from simu.topology import build_topology
from simu.simulate import simulate_task, simulate_workflow, pareto_front
from simu.stats_io import (
    agg_stats, agg_stats_by_profile, accuracy_stats, accuracy_stats_by_profile,
    slo_violation_rates_task, slo_violation_rates_task_by_profile,
    slo_violation_rates_workflow_task_slo, save_csv
)


# ---------- small helper to avoid merge suffix collisions ----------
def _prefixed(df: pd.DataFrame, keys, prefix: str) -> pd.DataFrame:
    if df is None:
        return None
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in keys}
    return df.rename(columns=rename_map)


def main():
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
    print(f"[info] Saving all CSVs to: {outdir}")

    # ---------------- TASK ----------------
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

    # New metrics
    if "selection_time_ms" in task_df_all.columns:
        print("Selector runtime (ms):")
        print(agg_stats(task_df_all, "selection_time_ms").to_string(index=False))
    if "ssp_calls" in task_df_all.columns:
        print("SSSP calls (count):")
        print(agg_stats(task_df_all, "ssp_calls").to_string(index=False))

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
    if "ssp_calls" in task_df_all.columns:
        print("SSSP calls by profile:")
        print(agg_stats_by_profile(task_df_all, "ssp_calls").to_string(index=False))

    print("Accuracy (%) by profile:")
    print(accuracy_stats_by_profile(task_df_all).to_string(index=False))
    print("SLO violation rate by profile (% of task requests):")
    print(slo_violation_rates_task_by_profile(task_df_all).to_string(index=False))

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

    # drop Nones and merge
    frames = [f for f in frames if f is not None]
    _task_wide = frames[0]
    for f in frames[1:]:
        _task_wide = _task_wide.merge(f, on=keys)

    save_csv(_task_wide, outdir, "task_summary_ALL_by_strategy_profile.csv")

    # ---------------- WORKFLOW ----------------
    wf_df = simulate_workflow(
        topo,
        config.NUM_RUNS_WORKFLOW,
        config.WORKFLOW_STAGES,
        config.SLO_MS_STAGE,
        seed=config.SEED + 10,
        stage_profiles=config.TASK_PROFILES_FOR_WORKFLOW,
    )

    print(f"\n=== WORKFLOW ({config.WORKFLOW_STAGES} stages) ===")
    for col in base_cols:
        print(f"{col}:")
        print(agg_stats(wf_df, col).to_string(index=False))

    # New metrics
    if "selection_time_ms" in wf_df.columns:
        print("Selector runtime (ms):")
        print(agg_stats(wf_df, "selection_time_ms").to_string(index=False))
    if "ssp_calls" in wf_df.columns:
        print("SSSP calls (count):")
        print(agg_stats(wf_df, "ssp_calls").to_string(index=False))

    print("Accuracy (%):")
    print(accuracy_stats(wf_df).to_string(index=False))

    runs_rate_task, stages_rate_task = slo_violation_rates_workflow_task_slo(wf_df)
    print("Workflow SLO violation (per-run; any stage violates) — based on SLO_MS_TASK:")
    print(runs_rate_task.to_string(index=False))
    print("Workflow SLO violation (per-stage across all runs) — based on SLO_MS_TASK:")
    print(stages_rate_task.to_string(index=False))

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
    save_csv(runs_rate_task, outdir, "workflow_slo_violation_task_slo_runs.csv")
    save_csv(stages_rate_task, outdir, "workflow_slo_violation_task_slo_stages.csv")

    # Optional Pareto counts
    # for name, df in [("Task (all profiles)", task_df_all), ("Workflow", wf_df)]:
    #     pts = np.c_[df["latency_ms"].to_numpy(), df["acc"].to_numpy()]
    #     mask = pareto_front(pts)
    #     print(f"\n{name}: {mask.sum()} non-dominated points out of {len(df)} total.")


if __name__ == "__main__":
    main()
