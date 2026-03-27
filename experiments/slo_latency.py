#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_FILE = "results/sat250_edge85_cloud1_seed42/task_runs_all.csv"
OUTPUT_FILE = "latency_slo_compact_table.csv"

LATENCY_SLO_MS = 600.0

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_FILE)

df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
df = df.dropna(subset=["strategy", "profile", "latency_ms"]).copy()

# -------------------------
# Compute SLO metrics
# -------------------------
df["slo_excess_ms"] = np.maximum(0.0, df["latency_ms"] - LATENCY_SLO_MS)

df["slo_excess_pct"] = np.maximum(
    0.0,
    (df["latency_ms"] - LATENCY_SLO_MS) / LATENCY_SLO_MS * 100.0
)

# in how many times the latency compares to the SLO
# 1.0x = exactly at SLO, 2.0x = twice the SLO, 0.7x = below SLO
df["slo_violation_times"] = df["latency_ms"] / LATENCY_SLO_MS

df["met_slo"] = df["latency_ms"] <= LATENCY_SLO_MS

# -------------------------
# Compact aggregation
# -------------------------
table = (
    df.groupby(["profile", "strategy"], as_index=False)
      .agg(
          mean_latency_ms=("latency_ms", "mean"),
          mean_slo_excess_ms=("slo_excess_ms", "mean"),
          mean_slo_excess_pct=("slo_excess_pct", "mean"),
          mean_slo_violation_times=("slo_violation_times", "mean"),
          met_slo_rate=("met_slo", "mean"),
          count=("latency_ms", "count"),
      )
)

table["met_slo_rate"] *= 100.0

# optional ordering for readability
profile_order = ["extract-frames", "object-det", "prepare-ds"]
strategy_order = [
    "Best-Acc",
    "Full-model",
    "SLO-first",
    "Round-Robin",
    "Random",
    "Lowest-Latency",
]

table["profile"] = pd.Categorical(table["profile"], categories=profile_order, ordered=True)
table["strategy"] = pd.Categorical(table["strategy"], categories=strategy_order, ordered=True)
table = table.sort_values(["profile", "strategy"]).reset_index(drop=True)

# round for readability
table = table.round({
    "mean_latency_ms": 2,
    "mean_slo_excess_ms": 2,
    "mean_slo_excess_pct": 2,
    "mean_slo_violation_times": 2,
    "met_slo_rate": 2
})

table.to_csv(OUTPUT_FILE, index=False)

print("\nCompact latency SLO table (SLO = 600 ms):\n")
print(table.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)