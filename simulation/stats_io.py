"""
Aggregation utilities and CSV saving.
"""
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from . import config, profiles


# ---------------------------
# Generic aggregations
# ---------------------------

def agg_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by strategy (sanitized: no inf/-inf)."""

    tmp = df.copy()
    tmp[col] = (
        pd.to_numeric(tmp[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    def p95(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 95)

    def p99(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 99)

    return (
        tmp.groupby("strategy", dropna=False)[col]
           .agg(
               mean=lambda s: s.dropna().mean(),
               median=lambda s: s.dropna().median(),
               p95=p95,
               p99=p99,
           )
           .reset_index()
    )


def combined_slo_violation_by_strategy(
    df: pd.DataFrame,
    *,
    slo_l_ms: Optional[float],
    slo_a_min: Optional[float],
    wL: float = 1.0,
    wA: float = 1.0,
    pL: float = 2.0,
    pA: float = 2.0,
    include_overall: bool = True,
    overall_label: str = "Overall",
) -> pd.DataFrame:
    if "strategy" not in df.columns:
        raise ValueError("combined_slo_violation_by_strategy: missing column 'strategy'")
    if "latency_ms" not in df.columns:
        raise ValueError("combined_slo_violation_by_strategy: missing column 'latency_ms'")
    if "acc" not in df.columns:
        raise ValueError("combined_slo_violation_by_strategy: missing column 'acc'")

    def _to_num(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        return x.replace([np.inf, -np.inf], np.nan)

    def _pctl(s: pd.Series, q: float) -> float:
        x = _to_num(s).dropna()
        if x.empty:
            return float("nan")
        return float(np.percentile(x.to_numpy(), q))

    def _hinge_pow(x: pd.Series, p: float) -> pd.Series:
        x = _to_num(x).clip(lower=0)
        return x if p == 1.0 else x.pow(p)

    lat = _to_num(df["latency_ms"])
    acc = _to_num(df["acc"])

    slo_l_ok = slo_l_ms is not None and np.isfinite(float(slo_l_ms)) and float(slo_l_ms) > 0
    slo_l_val = float(slo_l_ms) if slo_l_ok else float("nan")

    slo_a_ok = slo_a_min is not None and np.isfinite(float(slo_a_min))
    slo_a_val = float(slo_a_min) if slo_a_ok else float("nan")

    if slo_l_ok:
        vL = (lat - slo_l_val).clip(lower=0)
        lat_ok = vL <= 1e-12
        vL_norm = vL / slo_l_val
    else:
        vL = pd.Series(np.nan, index=df.index, dtype="float64")
        lat_ok = pd.Series(True, index=df.index, dtype="boolean")
        vL_norm = pd.Series(0.0, index=df.index, dtype="float64")

    if slo_a_ok:
        vA = (slo_a_val - acc).clip(lower=0)
        acc_ok = vA <= 1e-12
        denom = slo_a_val if slo_a_val != 0 else 1.0
        vA_norm = vA / denom
    else:
        vA = pd.Series(np.nan, index=df.index, dtype="float64")
        acc_ok = pd.Series(True, index=df.index, dtype="boolean")
        vA_norm = pd.Series(0.0, index=df.index, dtype="float64")

    any_viol = (~lat_ok) | (~acc_ok)
    both_viol = (~lat_ok) & (~acc_ok)

    tmp = df.copy()
    tmp["v_L_ms"] = vL
    tmp["v_A_abs"] = vA
    tmp["lat_ok"] = lat_ok
    tmp["acc_ok"] = acc_ok
    tmp["any_viol"] = any_viol
    tmp["both_viol"] = both_viol
    tmp["viol_mag"] = (float(wL) * _hinge_pow(vL_norm, float(pL))) + (float(wA) * _hinge_pow(vA_norm, float(pA)))
    tmp["soft_score"] = tmp["viol_mag"]

    def _agg_block(g: pd.DataFrame) -> dict:
        n = len(g)

        def pct(mask: pd.Series) -> float:
            if n == 0:
                return float("nan")
            m = mask.astype("boolean").fillna(False)
            return float(m.mean() * 100.0)

        return {
            "runs": n,

            "pct_any_viol": pct(g["any_viol"]),
            "pct_both_viol": pct(g["both_viol"]),
            "pct_latency_viol": pct(~g["lat_ok"]),
            "pct_acc_viol": pct(~g["acc_ok"]),

            "v_L_ms_mean": float(_to_num(g["v_L_ms"]).mean()),
            "v_L_ms_median": float(_to_num(g["v_L_ms"]).median()),
            "v_L_ms_p95": _pctl(g["v_L_ms"], 95),

            "v_A_mean": float(_to_num(g["v_A_abs"]).mean()),
            "v_A_median": float(_to_num(g["v_A_abs"]).median()),
            "v_A_p95": _pctl(g["v_A_abs"], 95),

            "viol_mag_mean": float(_to_num(g["viol_mag"]).mean()),
            "viol_mag_median": float(_to_num(g["viol_mag"]).median()),
            "viol_mag_p95": _pctl(g["viol_mag"], 95),

            "soft_score_mean": float(_to_num(g["soft_score"]).mean()),
            "soft_score_median": float(_to_num(g["soft_score"]).median()),
            "soft_score_p95": _pctl(g["soft_score"], 95),

            "latency_ms_mean": float(_to_num(g["latency_ms"]).mean()),
            "latency_ms_median": float(_to_num(g["latency_ms"]).median()),
            "acc_mean": float(_to_num(g["acc"]).mean()),
            "acc_median": float(_to_num(g["acc"]).median()),
        }

    per_rows = []
    for strategy, g in tmp.groupby("strategy", dropna=False):
        row = {"strategy": strategy}
        row.update(_agg_block(g))
        per_rows.append(row)
    per = pd.DataFrame(per_rows)

    per["SLO_L_ms"] = slo_l_val if slo_l_ok else np.nan
    per["SLO_A_min"] = slo_a_val if slo_a_ok else np.nan
    per["wL"] = float(wL)
    per["wA"] = float(wA)
    per["pL"] = float(pL)
    per["pA"] = float(pA)

    order = ["SLO-first", "Best-Acc", "Lowest-Latency", "Random", "Full-model", "Round-Robin"]
    per["__ord"] = per["strategy"].map({k: i for i, k in enumerate(order)}).fillna(9999)
    per = per.sort_values(["__ord", "strategy"]).drop(columns="__ord").reset_index(drop=True)

    if include_overall:
        overall = pd.DataFrame([{
            "strategy": overall_label,
            **_agg_block(tmp),
            "SLO_L_ms": slo_l_val if slo_l_ok else np.nan,
            "SLO_A_min": slo_a_val if slo_a_ok else np.nan,
            "wL": float(wL),
            "wA": float(wA),
            "pL": float(pL),
            "pA": float(pA),
        }])
        out = pd.concat([overall, per], ignore_index=True)
    else:
        out = per

    cols = [
        "strategy",
        "pct_any_viol", "pct_both_viol", "pct_latency_viol", "pct_acc_viol",
        "SLO_L_ms", "SLO_A_min", "wL", "wA", "pL", "pA", "runs",
        "v_L_ms_mean", "v_L_ms_median", "v_L_ms_p95",
        "v_A_mean", "v_A_median", "v_A_p95",
        "viol_mag_mean", "viol_mag_median", "viol_mag_p95",
        "soft_score_mean", "soft_score_median", "soft_score_p95",
        "latency_ms_mean", "latency_ms_median", "acc_mean", "acc_median",
    ]
    return out[[c for c in cols if c in out.columns]]


def agg_stats_by_profile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by (profile, strategy)."""
    tmp = df.copy()
    tmp[col] = (
        pd.to_numeric(tmp[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    def p95(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 95)

    def p99(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return np.percentile(s.to_numpy(), 99)

    return (
        tmp.groupby(["profile", "strategy"], dropna=False)[col]
           .agg(
               mean=lambda s: s.dropna().mean(),
               median=lambda s: s.dropna().median(),
               p95=p95,
               p99=p99,
           )
           .reset_index()
    )


def accuracy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by strategy."""
    tmp = df.copy()
    tmp["acc"] = pd.to_numeric(tmp["acc"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return tmp.groupby("strategy", dropna=False)["acc"].mean().reset_index(name="acc_mean")


def stitch_stats(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["run", "strategy", "stitch_id"]
    return df[cols].drop_duplicates().sort_values(cols)


def accuracy_stats_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean accuracy grouped by (profile, strategy)."""
    tmp = df.copy()
    tmp["acc"] = pd.to_numeric(tmp["acc"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return tmp.groupby(["profile", "strategy"], dropna=False)["acc"].mean().reset_index(name="acc_mean")


# ---------------------------
# E2E SLO violation helpers
# ---------------------------

def _get_per_profile_slo_map():
    """
    Resolve per-profile E2E SLO (ms) from config.

    Priority:
      1) config.SLO_MS_TASK_PER_PROFILE  (dict: profile -> E2E ms)
      2) fallback: config.SLO_MS_TASK    (scalar)
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
    Prefer latency_ms. If missing, fall back to net_latency_ms.
    """
    if "latency_ms" in df.columns:
        return pd.to_numeric(df["latency_ms"], errors="coerce")
    if "net_latency_ms" in df.columns:
        return pd.to_numeric(df["net_latency_ms"], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _per_profile_e2e_exceed_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (profile, strategy), compare mean latency against the profile task SLO.

    Output columns:
      - profile
      - strategy
      - rows
      - allowed_ms
      - mean_latency_ms
      - slo_violation_pct_task_slo

    The violation metric is:
        max(0, (mean_latency_ms - allowed_ms) / allowed_ms * 100)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "profile", "strategy", "rows", "allowed_ms",
            "mean_latency_ms", "slo_violation_pct_task_slo"
        ])

    tmp = df.copy()
    tmp["lat_cmp_ms"] = _pick_latency_column(tmp).replace([np.inf, -np.inf], np.nan)

    agg = (
        tmp.groupby(["profile", "strategy"], as_index=False, dropna=False)
           .agg(
               rows=("lat_cmp_ms", "size"),
               mean_latency_ms=("lat_cmp_ms", "mean"),
           )
    )

    per_profile_map, fallback = _get_per_profile_slo_map()

    if isinstance(per_profile_map, dict) and len(per_profile_map) > 0:
        agg["allowed_ms"] = agg["profile"].map(per_profile_map)
    else:
        agg["allowed_ms"] = np.nan

    agg["allowed_ms"] = pd.to_numeric(agg["allowed_ms"], errors="coerce")

    if np.isfinite(fallback):
        agg["allowed_ms"] = agg["allowed_ms"].fillna(float(fallback))

    agg["mean_latency_ms"] = pd.to_numeric(agg["mean_latency_ms"], errors="coerce")

    valid = (
        agg["allowed_ms"].notna()
        & np.isfinite(agg["allowed_ms"])
        & (agg["allowed_ms"] > 0)
        & agg["mean_latency_ms"].notna()
        & np.isfinite(agg["mean_latency_ms"])
    )

    agg["slo_violation_pct_task_slo"] = np.nan
    agg.loc[valid, "slo_violation_pct_task_slo"] = (
        ((agg.loc[valid, "mean_latency_ms"] - agg.loc[valid, "allowed_ms"]).clip(lower=0.0)
         / agg.loc[valid, "allowed_ms"]) * 100.0
    )

    return agg[[
        "profile", "strategy", "rows", "allowed_ms",
        "mean_latency_ms", "slo_violation_pct_task_slo"
    ]]


# ---------------------------
# SLO violation reporting
# ---------------------------

def slo_violation_rates_task(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level SLO violation percentage per strategy, weighted by number of rows
    in each profile/strategy block.
    """
    per_profile = _per_profile_e2e_exceed_pct(df)

    pp = per_profile.dropna(subset=["slo_violation_pct_task_slo", "rows"]).copy()
    pp = pp[pp["rows"] > 0]

    if pp.empty:
        return pd.DataFrame(columns=["strategy", "slo_violation_pct_task_slo"])

    pp["wx"] = pp["rows"] * pp["slo_violation_pct_task_slo"]

    agg = (
        pp.groupby("strategy", as_index=False, dropna=False)
          .agg(total_w=("rows", "sum"), total_wx=("wx", "sum"))
    )
    agg["slo_violation_pct_task_slo"] = agg["total_wx"] / agg["total_w"]

    return agg[["strategy", "slo_violation_pct_task_slo"]]


def slo_violation_rates_task_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task-level SLO violation percentage per (profile, strategy),
    based on mean E2E latency against per-profile task SLO.
    """
    per_profile = _per_profile_e2e_exceed_pct(df)
    return per_profile[["profile", "strategy", "slo_violation_pct_task_slo"]]


def slo_violation_rates_workflow_task_slo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Workflow-level compatibility wrapper.

    Reuses the same E2E logic:
      - runs: rows-weighted average across profiles per strategy
      - stages: same values, kept only for compatibility with existing callers
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