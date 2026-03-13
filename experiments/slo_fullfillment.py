#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_FILE = "results/wildfire/sat250_edge85_cloud1_seed42/workflow_runs.csv"
OUTPUT_FILE = "overall_slo_fulfillment.csv"

LATENCY_SLO_MS = 2000.0
ACCURACY_SLO = 89.0

W_LAT = 0.5
W_ACC = 0.5

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_FILE)

# ensure numeric
df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
df["acc"] = pd.to_numeric(df["acc"], errors="coerce")

df = df.dropna(subset=["strategy", "latency_ms", "acc"]).copy()

# -------------------------
# Compute fulfillment scores
# -------------------------

# latency: lower is better
df["lat_score"] = np.minimum(1.0, LATENCY_SLO_MS / df["latency_ms"])

# accuracy: higher is better
df["acc_score"] = np.minimum(1.0, df["acc"] / ACCURACY_SLO)

# combined fulfillment
df["overall_slo_fulfillment"] = (
    W_LAT * df["lat_score"] +
    W_ACC * df["acc_score"]
)

# -------------------------
# Aggregate statistics
# -------------------------
result = (
    df.groupby("strategy")["overall_slo_fulfillment"]
      .agg(["mean", "std", "count"])
      .reset_index()
)

# standard error
result["sem"] = result["std"] / np.sqrt(result["count"])

# sort best → worst
result = result.sort_values("mean", ascending=False)

# -------------------------
# Save output
# -------------------------
result.to_csv(OUTPUT_FILE, index=False)

# -------------------------
# Print summary
# -------------------------
print("\nOverall SLO fulfillment per strategy:\n")
print(result)

print("\nSaved:", OUTPUT_FILE)