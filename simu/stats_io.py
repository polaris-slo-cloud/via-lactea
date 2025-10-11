"""
Aggregation utilities and CSV saving.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

def agg_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by strategy."""
    return (
        df.groupby("strategy")[col]
          .agg(mean="mean", median="median",
               p95=lambda s: np.percentile(s, 95),
               p99=lambda s: np.percentile(s, 99))
          .reset_index()
    )

def agg_stats_by_profile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by (profile, strategy)."""
    return (
        df.groupby(["profile", "strategy"])[col]
          .agg(mean="mean", median="median",
               p95=lambda s: np.percentile(s, 95),
               p99=lambda s: np.percentile(s, 99))
          .reset_index()
    )

def accuracy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by strategy."""
    return df.groupby("strategy")["acc"].mean().reset_index(name="acc_mean")

def accuracy_stats_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by (profile, strategy)."""
    return df.groupby(["profile","strategy"])["acc"].mean().reset_index(name="acc_mean")

def slo_violation_rates_task(df: pd.DataFrame) -> pd.DataFrame:
    """Task-level SLO violation percentage per strategy."""
    rates = df.groupby("strategy")["met_slo"].mean().reset_index(name="met_rate")
    rates["slo_violation_pct_task_slo"] = (1.0 - rates["met_rate"]) * 100.0
    return rates[["strategy", "slo_violation_pct_task_slo"]]

def slo_violation_rates_task_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Task-level SLO violation percentage per (profile, strategy)."""
    rates = df.groupby(["profile","strategy"])["met_slo"].mean().reset_index(name="met_rate")
    rates["slo_violation_pct_task_slo"] = (1.0 - rates["met_rate"]) * 100.0
    return rates[["profile","strategy","slo_violation_pct_task_slo"]]

def slo_violation_rates_workflow_task_slo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Workflow-level SLO (based on config.SLO_MS_TASK):
    - per-run: % runs where all stages met the task SLO
    - per-stage: % of stages (aggregated) that violated the task SLO
    """
    runs = df.groupby("strategy")["met_task_slo_all"].mean().reset_index(name="runs_met_rate")
    runs["slo_violation_pct_per_run_task_slo"] = (1.0 - runs["runs_met_rate"]) * 100.0
    runs = runs[["strategy", "slo_violation_pct_per_run_task_slo"]]

    stage_totals = df.groupby("strategy")[["stages_met_task_slo", "stages"]].sum().reset_index()
    stage_totals["per_stage_violation_pct_task_slo"] = (
        1.0 - (stage_totals["stages_met_task_slo"] / stage_totals["stages"])
    ) * 100.0
    stages = stage_totals[["strategy", "per_stage_violation_pct_task_slo"]]
    return runs, stages

def save_csv(df: pd.DataFrame, outdir: str, filename: str) -> str:
    """Save a DataFrame to CSV under outdir/filename and return the path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    df.to_csv(path, index=False)
    return path
