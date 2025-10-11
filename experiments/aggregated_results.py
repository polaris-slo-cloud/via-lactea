#!/usr/bin/env python3
"""
Aggregate experiment results across node-count folders and plot mean latency.

It reads (per node-count folder, e.g. experiments/via-lactea/100):
- task_summary_latency_by_strategy_profile.csv
- task_summary_hopcount_by_strategy_profile.csv
- task_summary_link_by_strategy_profile.csv
- workflow_summary_latency_by_strategy.csv
- workflow_summary_hopcount_by_strategy.csv
- workflow_summary_link_by_strategy.csv

Outputs (to OUTPUT_DIR):
- aggregated_latency.csv
- aggregated_hopcount.csv
- aggregated_link_mb.csv
- aggregated_merged.csv
- latency_nodes_<strategy>.png    (one plot per strategy; lines = profiles incl. 'workflow')
"""

import os
import re
import argparse
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Defaults (override via CLI)
# -------------------------
BASE_DIR_DEFAULT = os.path.expanduser("~/workspace/via-lactea/experiments/via-lactea")
OUTPUT_DIR_DEFAULT = os.path.expanduser("~/workspace/via-lactea/experiments/_plots")

# If None, include ALL strategies found in the CSVs.
STRATEGIES_DEFAULT: Optional[List[str]] = None

TASK_FILES = {
    "latency":   "task_summary_latency_by_strategy_profile.csv",
    "hopcount":  "task_summary_hopcount_by_strategy_profile.csv",
    "link":      "task_summary_link_by_strategy_profile.csv",
}

WF_FILES = {
    "latency":   "workflow_summary_latency_by_strategy.csv",
    "hopcount":  "workflow_summary_hopcount_by_strategy.csv",
    "link":      "workflow_summary_link_by_strategy.csv",
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
    metric_key in {"latency","hopcount","link"}
    Returns DataFrame with columns: nodes, profile, strategy, mean
    """
    rows = []
    for name in sorted(os.listdir(base_dir), key=lambda s: (len(s), s)):
        if not is_node_dir(name):
            continue
        nodes = int(name)
        dirpath = os.path.join(base_dir, name)
        # task metric
        rows.append(read_task_metric(dirpath, TASK_FILES[metric_key], nodes, strategies))
        # workflow metric
        rows.append(read_workflow_metric(dirpath, WF_FILES[metric_key], nodes, strategies))

    if not rows:
        return pd.DataFrame(columns=["nodes","profile","strategy","mean"])

    out = pd.concat(rows, ignore_index=True)
    out["nodes"] = pd.to_numeric(out["nodes"], errors="coerce")
    out = out.dropna(subset=["nodes", "mean"])
    out = out.sort_values(["profile","strategy","nodes"]).reset_index(drop=True)
    return out

def plot_latency(df_lat: pd.DataFrame, outdir: str, strategy_list: Optional[List[str]]):
    """
    For each strategy, create a line plot:
      x = nodes, y = mean (latency_ms)
      separate line per profile (including 'workflow')
    If strategy_list is None, plot ALL strategies present.
    """
    os.makedirs(outdir, exist_ok=True)
    if strategy_list is None:
        strategy_list = sorted(df_lat["strategy"].unique())
    for strat in strategy_list:
        d = df_lat[df_lat["strategy"] == strat]
        if d.empty:
            continue
        plt.figure(figsize=(9, 5))
        # Sort so 'workflow' appears last in legend ordering tweakable by key
        for profile in sorted(d["profile"].unique(), key=lambda p: (p!="workflow", p)):
            sub = d[d["profile"] == profile].sort_values("nodes")
            if sub.empty:
                continue
            plt.plot(sub["nodes"], sub["mean"], marker="o", label=profile)
        plt.xlabel("Nodes")
        plt.ylabel("Mean latency (ms)")
        plt.title(f"Mean Latency vs Nodes — Strategy: {strat}")
        plt.grid(True, linewidth=0.3)
        plt.legend(title="Profile", ncol=2)
        outpath = os.path.join(outdir, f"latency_nodes_{strat.replace(' ','_')}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()

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
    lat_df = build_metric_across_nodes(base_dir, "latency", strategies)
    hop_df = build_metric_across_nodes(base_dir, "hopcount", strategies)
    lnk_df = build_metric_across_nodes(base_dir, "link", strategies)

    if lat_df.empty and hop_df.empty and lnk_df.empty:
        print("[warn] No data found. Check paths and files.")
        return

    # Save individual metric CSVs
    lat_path = save_csv(lat_df, out_dir, "aggregated_latency.csv")
    hop_path = save_csv(hop_df, out_dir, "aggregated_hopcount.csv")
    lnk_path = save_csv(lnk_df, out_dir, "aggregated_link_mb.csv")

    print(f"[ok] Wrote: {lat_path}")
    print(f"[ok] Wrote: {hop_path}")
    print(f"[ok] Wrote: {lnk_path}")

    # Merge (left join on nodes+profile+strategy; latency as base)
    merged = lat_df.rename(columns={"mean": "mean_latency_ms"})
    if not hop_df.empty:
        merged = merged.merge(hop_df.rename(columns={"mean":"mean_hop_count"}),
                              on=["nodes","profile","strategy"], how="left")
    if not lnk_df.empty:
        merged = merged.merge(lnk_df.rename(columns={"mean":"mean_link_mb"}),
                              on=["nodes","profile","strategy"], how="left")

    merged_path = save_csv(merged, out_dir, "aggregated_merged.csv")
    print(f"[ok] Wrote: {merged_path}")

    # Plots (one per strategy, lines per profile incl. workflow)
    if not lat_df.empty:
        # If strategies None, plot all present
        plot_latency(lat_df, out_dir, strategies)
        print("[ok] Latency plots saved.")

if __name__ == "__main__":
    main()
