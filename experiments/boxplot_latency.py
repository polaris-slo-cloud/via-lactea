#!/usr/bin/env python3

import pandas as pd
import numpy as np

INPUT_FILE = "workflow_runs.csv"

ORDER = [
    "Best-Acc",
    "Full-model",
    "Lowest-Latency",
    "Random",
    "Round-Robin",
    "SLO-first",
]

def whiskers_boxplot(values):
    values = np.sort(np.asarray(values, dtype=float))

    q1 = np.percentile(values, 25)
    median = np.percentile(values, 50)
    q3 = np.percentile(values, 75)

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    lower_whisker = values[values >= lower_bound].min()
    upper_whisker = values[values <= upper_bound].max()

    outliers = values[(values < lower_whisker) | (values > upper_whisker)]
    return q1, median, q3, lower_whisker, upper_whisker, outliers

def main():
    df = pd.read_csv(INPUT_FILE)

    required = {"strategy", "latency_ms"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # ms -> sec
    df["latency_s"] = pd.to_numeric(df["latency_ms"], errors="coerce") / 1000.0
    df = df.dropna(subset=["strategy", "latency_s"])

    pos = 1
    for strategy in ORDER:
        g = df[df["strategy"] == strategy]["latency_s"].to_numpy()
        if len(g) == 0:
            continue

        q1, med, q3, low, up, outliers = whiskers_boxplot(g)

        print(f"% {strategy} baseline")
        print(r"\addplot+[")
        print(r"  boxplot prepared={")
        print(f"    median={med:.6f}, upper quartile={q3:.6f}, lower quartile={q1:.6f},")
        print(f"    upper whisker={up:.6f}, lower whisker={low:.6f}, draw position={pos}")
        print(r"  },")
        print(r"  fill=orange!40, draw=orange!80,")
        print(r"] coordinates {};")

        if len(outliers) > 0:
            print(r"\addplot+[")
            print(r"  only marks,")
            print(r"  mark=*,")
            print(r"  mark size=1.2,")
            print(r"  draw=orange!80,")
            print(r"] coordinates {")
            for v in outliers:
                print(f"  ({pos},{v:.6f})")
            print(r"};")

        print()
        pos += 1

if __name__ == "__main__":
    main()