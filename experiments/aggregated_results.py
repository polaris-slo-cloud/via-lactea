#!/usr/bin/env python3
"""
Aggregate experiment results across node-count folders and plot mean latency.

It reads (per node-count folder, e.g. results/100):
- task_summary_latency_by_strategy_profile.csv
- task_summary_hopcount_by_strategy_profile.csv
- task_summary_link_by_strategy_profile.csv
- task_summary_selector_time_by_strategy_profile.csv   [TASK-ONLY]
- workflow_summary_latency_by_strategy.csv
- workflow_summary_hopcount_by_strategy.csv
- workflow_summary_link_by_strategy.csv

Outputs (to OUTPUT_DIR):
- aggregated_latency.csv
- aggregated_hopcount.csv
- aggregated_link_mb.csv
- aggregated_selector_ms.csv                            [TASK-ONLY]
- aggregated_merged.csv
- latency_nodes_<strategy>.png    (one plot per strategy; lines = profiles incl. 'workflow')
"""

import os
import re
import argparse
from typing import List, Optional
import pandas as pd

# -------------------------
# Defaults (override via CLI)
# -------------------------
BASE_DIR_DEFAULT = os.path.expanduser("results/")
OUTPUT_DIR_DEFAULT = os.path.expanduser("_plots")

# If None, include ALL strategies found in the CSVs.
STRATEGIES_DEFAULT: Optional[List[str]] = None

TASK_FILES = {
    "latency":   "task_summary_latency_by_strategy_profile.csv",
    "hopcount":  "task_summary_hopcount_by_strategy_profile.csv",
    "link":      "task_summary_link_by_strategy_profile.csv",
    "selector":  "task_summary_selector_time_by_strategy_profile.csv",
}

WF_FILES = {
    "latency":   "workflow_summary_latency_by_strategy.csv",
    "hopcount":  "workflow_summary_hopcount_by_strategy.csv",
    "link":      "workflow_summary_link_by_strategy.csv",
    # NOTE: no workflow-level selector file
}

# -------------------------
# Helpers
# -------------------------

def is_node_dir(name: str) -> bool:
    """True if folder name is an integer (e.g., '100', '1000', ...)."""
    return bool(re.fullmatch(r"\d+", name))

def read_task_metric(dirpath: str, filename: str, nodes: int, strategies: Optional[List[str]]) -> pd.DataFrame:
    """
    Reads a task metric summary CSV with columns:
      profile,strategy,mean,median,p95,p99
    Returns columns: nodes, profile, strategy, mean
    """
    path = os.path.join(dirpath, filename)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])
    df = pd.read_csv(path)
    if not {"profile","strategy","mean"}.issubset(df.columns):
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    df = df[["profile","strategy","mean"]].copy()
    df.insert(0, "nodes", nodes)
    return df

def read_workflow_metric(dirpath: str, filename: str, nodes: int, strategies: Optional[List[str]]) -> pd.DataFrame:
    """
    Reads a workflow metric summary CSV with columns:
      strategy,mean,median,p95,p99
    Returns columns: nodes, profile='workflow', strategy, mean
    """
    path = os.path.join(dirpath, filename)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])
    df = pd.read_csv(path)
    if not {"strategy","mean"}.issubset(df.columns):
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])
    if strategies:
        df = df[df["strategy"].isin(strategies)]
    df = df[["strategy","mean"]].copy()
    df.insert(0, "profile", "workflow")
    df.insert(0, "nodes", nodes)
    return df

def build_metric_across_nodes(base_dir: str, metric_key: str, strategies: Optional[List[str]]) -> pd.DataFrame:
    """
    Builds a metric across nodes for both task-level and (if available) workflow-level.
    metric_key in {"latency","hopcount","link","selector"}
    Returns DataFrame with columns: nodes, profile, strategy, mean
    """
    rows = []
    for name in sorted(os.listdir(base_dir), key=lambda s: (len(s), s)):
        if not is_node_dir(name):
            continue
        nodes = int(name)
        dirpath = os.path.join(base_dir, name)

        # Task metric (always try)
        task_file = TASK_FILES.get(metric_key)
        if task_file:
            rows.append(read_task_metric(dirpath, task_file, nodes, strategies))

        # Workflow metric (only if defined for this metric)
        wf_file = WF_FILES.get(metric_key)
        if wf_file:
            rows.append(read_workflow_metric(dirpath, wf_file, nodes, strategies))

    if not rows:
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])

    out = pd.concat(rows, ignore_index=True)
    out["nodes"] = pd.to_numeric(out["nodes"], errors="coerce")
    out = out.dropna(subset=["nodes", "mean"])
    out = out.sort_values(["profile","strategy","nodes"]).reset_index(drop=True)
    return out

def save_csv(df: pd.DataFrame, outdir: str, filename: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    df.to_csv(path, index=False)
    return path

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Aggregate experiment CSVs and plot latency by nodes (all baselines by default).")
    ap.add_argument("--base-dir", default=BASE_DIR_DEFAULT,
                    help="Root containing numeric node-count subfolders (default: %(default)s)")
    ap.add_argument("--out-dir", default=OUTPUT_DIR_DEFAULT,
                    help="Where to write aggregated CSVs and plots (default: %(default)s)")
    ap.add_argument("--strategy", action="append",
                    help="Strategy to include (can pass multiple). Omit to include ALL found strategies.")
    args = ap.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    strategies = args.strategy if args.strategy else STRATEGIES_DEFAULT  # None => include all

    print(f"[info] Base dir: {base_dir}")
    print(f"[info] Output  : {out_dir}")
    print(f"[info] Strategies: {'ALL' if strategies is None else strategies}")

    # Build metric dataframes (possibly unfiltered if strategies is None)
    lat_df      = build_metric_across_nodes(base_dir, "latency", strategies)
    hop_df      = build_metric_across_nodes(base_dir, "hopcount", strategies)
    lnk_df      = build_metric_across_nodes(base_dir, "link", strategies)
    selector_df = build_metric_across_nodes(base_dir, "selector", strategies)  # task-only

    if lat_df.empty and hop_df.empty and lnk_df.empty and selector_df.empty:
        print("[warn] No data found. Check paths and files.")
        return

    # Save individual metric CSVs
    lat_path   = save_csv(lat_df, out_dir, "aggregated_latency.csv")
    hop_path   = save_csv(hop_df, out_dir, "aggregated_hopcount.csv")
    lnk_path   = save_csv(lnk_df, out_dir, "aggregated_link_mb.csv")
    sel_path   = save_csv(selector_df, out_dir, "aggregated_selector_ms.csv")

    print(f"[ok] Wrote: {lat_path}")
    print(f"[ok] Wrote: {hop_path}")
    print(f"[ok] Wrote: {lnk_path}")
    print(f"[ok] Wrote: {sel_path}")

    # Merge (left join on nodes+profile+strategy; latency as base)
    merged = lat_df.rename(columns={"mean": "mean_latency_ms"})
    if not hop_df.empty:
        merged = merged.merge(hop_df.rename(columns={"mean":"mean_hop_count"}),
                              on=["nodes","profile","strategy"], how="left")
    if not lnk_df.empty:
        merged = merged.merge(lnk_df.rename(columns={"mean":"mean_link_mb"}),
                              on=["nodes","profile","strategy"], how="left")
    if not selector_df.empty:
        merged = merged.merge(selector_df.rename(columns={"mean":"mean_selector_ms"}),
                              on=["nodes","profile","strategy"], how="left")

    merged_path = save_csv(merged, out_dir, "aggregated_merged.csv")
    print(f"[ok] Wrote: {merged_path}")

if __name__ == "__main__":
    main()
