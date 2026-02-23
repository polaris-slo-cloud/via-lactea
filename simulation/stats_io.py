"""
Aggregation utilities and CSV saving.
"""
import math
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from . import config


# ---------------------------
# Generic aggregations
# ---------------------------

def agg_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean/median/p95/p99 grouped by strategy (sanitized: no inf/-inf)."""

    # Make a sanitized copy of the column
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
        tmp.groupby("strategy")[col]
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
    import math
    import numpy as np
    import pandas as pd

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

    slo_l_ok = slo_l_ms is not None and math.isfinite(float(slo_l_ms)) and float(slo_l_ms) > 0
    slo_l_val = float(slo_l_ms) if slo_l_ok else float("nan")

    slo_a_ok = slo_a_min is not None and math.isfinite(float(slo_a_min))
    slo_a_val = float(slo_a_min) if slo_a_ok else float("nan")

    # violations
    if slo_l_ok:
        vL = (lat - slo_l_val).clip(lower=0)
        lat_ok = vL <= 1e-12
        vL_norm = vL / slo_l_val
    else:
        vL = pd.Series(np.nan, index=df.index, dtype="float64")
        lat_ok = pd.Series(True, index=df.index)
        vL_norm = pd.Series(0.0, index=df.index, dtype="float64")

    if slo_a_ok:
        vA = (slo_a_val - acc).clip(lower=0)
        acc_ok = vA <= 1e-12
        denom = slo_a_val if slo_a_val != 0 else 1.0
        vA_norm = vA / denom
    else:
        vA = pd.Series(np.nan, index=df.index, dtype="float64")
        acc_ok = pd.Series(True, index=df.index)
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

    # combined magnitude (normalized, weighted, hinge-powered)
    tmp["viol_mag"] = (float(wL) * _hinge_pow(vL_norm, float(pL))) + (float(wA) * _hinge_pow(vA_norm, float(pA)))

    # keep your previous soft_score too (optional)
    tmp["soft_score"] = tmp["viol_mag"]

    def _agg_block(g: pd.DataFrame) -> pd.Series:
        n = len(g)

        def pct(mask: pd.Series) -> float:
            if n == 0:
                return float("nan")
            m = mask.astype("boolean").fillna(False)
            return float(m.mean() * 100.0)

        return pd.Series({
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
        })

    per = tmp.groupby("strategy", dropna=False).apply(_agg_block).reset_index()

    # add config cols (after pct columns, per your request)
    per["SLO_L_ms"] = slo_l_val if slo_l_ok else np.nan
    per["SLO_A_min"] = slo_a_val if slo_a_ok else np.nan
    per["wL"] = float(wL)
    per["wA"] = float(wA)
    per["pL"] = float(pL)
    per["pA"] = float(pA)

    # stable strategy order
    order = ["SLO-first", "Best-Acc", "Lowest-Latency", "Random", "Full-model", "Round-Robin"]
    per["__ord"] = per["strategy"].map({k: i for i, k in enumerate(order)}).fillna(9999)
    per = per.sort_values(["__ord", "strategy"]).drop(columns="__ord").reset_index(drop=True)

    if include_overall:
        overall_stats = _agg_block(tmp)
        overall = pd.DataFrame([{
            "strategy": overall_label,
            **overall_stats.to_dict(),
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

    # column order: pct first
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
