"""
Aggregation utilities and CSV saving.
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


# ---------------------------
# Generic aggregations
# ---------------------------

def agg_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by strategy."""

    def p95(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 95)

    def p99(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 99)

    return (
        df.groupby("strategy")[col]
          .agg(
              mean="mean",
              median="median",
              p95=p95,
              p99=p99,
          )
          .reset_index()
    )




def agg_stats_by_profile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by (profile, strategy)."""
    return (
        df.groupby(["profile", "strategy"])[col]
          .agg(
              mean="mean",
              median="median",
              p95=lambda s: np.percentile(s.dropna(), 95) if len(s.dropna()) else np.nan,
              p99=lambda s: np.percentile(s.dropna(), 99) if len(s.dropna()) else np.nan,
          )
          .reset_index()
    )


def accuracy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by strategy."""
    return df.groupby("strategy")["acc"].mean().reset_index(name="acc_mean")

def stitch_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["run", "strategy", "stitch_id"]
    return df[cols].drop_duplicates().sort_values(cols)


def accuracy_stats_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by (profile, strategy)."""
    return df.groupby(["profile", "strategy"])["acc"].mean().reset_index(name="acc_mean")


# ---------------------------
# E2E SLO violation (per-profile) helpers
# ---------------------------

def _get_per_profile_slo_map():
    """
    Resolve per-profile **E2E** SLO (ms) from config.

    Priority:
      1) config.SLO_MS_TASK_PER_PROFILE  (dict: profile -> E2E ms)
      2) fallback E2E value: config.SLO_MS_TASK (scalar)
    """
    slo_map = getattr(config, "SLO_MS_TASK_PER_PROFILE", None)
    fallback = getattr(config, "SLO_MS_TASK", np.nan)
    try:
        fallback = float(fallback)
    except Exception:
        fallback = np.nan
    if not np.isfinite(fallback):
        fallback = np.nan
    return slo_map, fallback


def _pick_latency_column(df: pd.DataFrame) -> pd.Series:
    """
    Choose which latency column to use for E2E comparisons.
    Prefer 'latency_ms' (E2E). If missing, fall back to 'net_latency_ms'.
    """
    if "latency_ms" in df.columns:
        return pd.to_numeric(df["latency_ms"], errors="coerce")
    if "net_latency_ms" in df.columns:
        return pd.to_numeric(df["net_latency_ms"], errors="coerce")
    # if nothing exists, create NaNs to keep shapes consistent
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _per_profile_e2e_exceed_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per (profile, strategy):
      mean_latency_ms = mean(E2E latency_ms)
      allowed_ms      = per-profile E2E SLO from config (ms)
      exceed %        = max(0, (mean_latency_ms - allowed_ms) / allowed_ms * 100)

    Returns columns:
      profile, strategy, rows, mean_latency_ms, allowed_e2e_ms, slo_violation_pct_task_slo
    """
    tmp = df.copy()

    # Ensure required cols exist
    if "profile" not in tmp.columns:
        tmp["profile"] = "__all__"  # handle workflow totals gracefully

    lat = _pick_latency_column(tmp)
    tmp["__lat"] = lat

    g = tmp.groupby(["profile", "strategy"], dropna=False)
    mean_latency = g["__lat"].mean()
    rows         = g.size()

    # Per-profile E2E SLO lookup
    slo_map, fallback = _get_per_profile_slo_map()
    profiles = mean_latency.index.get_level_values(0)
    if isinstance(slo_map, dict):
        allowed_series = profiles.map(lambda p: slo_map.get(p, fallback)).astype(float)
    else:
        allowed_series = pd.Series(fallback, index=profiles, dtype=float)

    # Compute exceed %
    allowed_ok = (allowed_series > 0) & np.isfinite(allowed_series)
    exceed_pct = pd.Series(np.nan, index=mean_latency.index, dtype=float)
    valid = allowed_ok & mean_latency.notna()
    exceed_pct.loc[valid] = ((mean_latency[valid] - allowed_series[valid]) / allowed_series[valid]) * 100.0
    exceed_pct = exceed_pct.clip(lower=0)  # non-violations -> 0%

    out = pd.DataFrame({
        "profile": profiles,
        "strategy": mean_latency.index.get_level_values(1),
        "rows": rows.to_numpy(dtype=float),
        "mean_latency_ms": mean_latency.to_numpy(dtype=float),
        "allowed_e2e_ms": allowed_series.to_numpy(dtype=float),
        "slo_violation_pct_task_slo": exceed_pct.to_numpy(dtype=float),
    })

    # Sanitize infs to NaN
    for c in ("rows", "mean_latency_ms", "allowed_e2e_ms", "slo_violation_pct_task_slo"):
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


# ---------------------------
# SLO "violation" reporting (as average % exceed from per-profile summaries)
# ---------------------------

def slo_violation_rates_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level SLO violation percentage per strategy (rows-weighted over profiles).

    Steps:
      1) Per (profile,strategy) compute:
         mean_latency_ms = mean(E2E latency)
         allowed_ms      = E2E SLO per profile from config
         exceed %        = max(0, (mean_latency_ms - allowed_ms)/allowed_ms * 100)
      2) Weighted average across profiles with weights = row counts per (profile,strategy).

    Output:
      strategy, slo_violation_pct_task_slo
    """
    per_profile = _per_profile_e2e_exceed_pct(df)

    pp = per_profile.dropna(subset=["slo_violation_pct_task_slo", "rows"]).copy()
    pp = pp[pp["rows"] > 0]
    if pp.empty:
        return pd.DataFrame({"strategy": [], "slo_violation_pct_task_slo": []})

    pp["wx"] = pp["rows"] * pp["slo_violation_pct_task_slo"]
    agg = (
        pp.groupby("strategy", as_index=False, dropna=False)
          .agg(total_w=("rows", "sum"), total_wx=("wx", "sum"))
    )
    agg["slo_violation_pct_task_slo"] = agg["total_wx"] / agg["total_w"]
    out = agg[["strategy", "slo_violation_pct_task_slo"]]
    return out


def slo_violation_rates_task_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level SLO violation percentage per (profile, strategy),
    computed from **summary E2E latency per profile** against **per-profile E2E SLO**.
    """
    per_profile = _per_profile_e2e_exceed_pct(df)
    return per_profile[["profile", "strategy", "slo_violation_pct_task_slo"]]


def slo_violation_rates_workflow_task_slo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Workflow-level SLO reporting (signature kept for compatibility).

    Reuses the same E2E logic:
      - runs:   rows-weighted average pct across profiles per strategy
      - stages: same weighting (kept for compatibility)
    """
    task_rates = slo_violation_rates_task(df).rename(
        columns={"slo_violation_pct_task_slo": "slo_violation_pct_per_run_task_slo"}
    )
    stages = task_rates.rename(
        columns={"slo_violation_pct_per_run_task_slo": "per_stage_violation_pct_task_slo"}
    )
    return task_rates, stages


# ---------------------------
# CSV I/O
# ---------------------------

def save_csv(df: pd.DataFrame, outdir: str, filename: str) -> str:
    """Save a DataFrame to CSV under outdir/filename and return the path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    df.to_csv(path, index=False)
    return path
