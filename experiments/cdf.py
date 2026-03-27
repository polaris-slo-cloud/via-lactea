#!/usr/bin/env python3

import glob
import os
import re
import numpy as np
import pandas as pd

STYLE_MAP = {
    "Best-Acc": {
        "color": "blue",
        "mark": "square*",
    },
    "Full-model": {
        "color": "black",
        "mark": "diamond*",
    },
    "Lowest-Latency": {
        "color": "red",
        "mark": "o",
    },
    "Random": {
        "color": "orange",
        "mark": "x",
    },
    "Round-Robin": {
        "color": "green!60!black",
        "mark": "triangle*",
    },
    "SLO-first": {
        "color": "purple",
        "mark": "star",
    },
}

STRATEGY_ORDER = [
    "Best-Acc",
    "Full-model",
    "Lowest-Latency",
    "Random",
    "Round-Robin",
    "SLO-first",
]

FILES_GLOB = "results/wildfire/*/task_runs_all.csv"
OUT_DIR = "results/wildfire/cdf_by_profile"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def compute_cdf(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.sort(values)
    n = len(values)
    if n == 0:
        return np.array([]), np.array([])
    cdf = np.arange(1, n + 1) / n * 100.0
    return values, cdf


def emit_addplot(strategy: str, lat_s: np.ndarray, cdf: np.ndarray) -> str:
    style = STYLE_MAP.get(strategy, {"color": "black", "mark": "*"})

    lines = []
    lines.append(f"% {strategy}")
    lines.append(r"\addplot[")
    lines.append(f"  color={style['color']},")
    lines.append(f"  mark={style['mark']},")
    lines.append(r"  mark size=1.2,")
    lines.append(r"  very thin")
    lines.append(r"] coordinates {")

    for x, y in zip(lat_s, cdf):
        lines.append(f"  ({x:.6f},{y:.2f})")

    lines.append(r"};")
    lines.append(f"\\addlegendentry{{{strategy}}}")
    lines.append("")
    return "\n".join(lines)


def main():
    files = sorted(glob.glob(FILES_GLOB))
    if not files:
        raise SystemExit(f"No files found for pattern: {FILES_GLOB}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    required = {"profile", "strategy", "latency_ms"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["profile", "strategy", "latency_ms"]).copy()
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["latency_ms"]).copy()

    # Keep all strategies, do not filter by valid
    df["profile"] = df["profile"].astype(str).str.strip()
    df["strategy"] = df["strategy"].astype(str).str.strip()
    df["latency_s"] = df["latency_ms"] / 1000.0

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\nCounts by profile/strategy:")
    print(df.groupby(["profile", "strategy"]).size().reset_index(name="n"))

    if "valid" in df.columns:
        print("\nCounts by profile/strategy/valid:")
        print(df.groupby(["profile", "strategy", "valid"]).size().reset_index(name="n"))

    profiles = sorted(df["profile"].unique())

    for profile in profiles:
        gp = df[df["profile"] == profile].copy()

        present = list(gp["strategy"].dropna().unique())
        ordered_present = [s for s in STRATEGY_ORDER if s in present]
        remaining = sorted([s for s in present if s not in STRATEGY_ORDER])
        strategies = ordered_present + remaining

        print(f"\nProfile: {profile}")
        print(f"Strategies found: {strategies}")

        out_lines = []
        out_lines.append(f"% Auto-generated CDF plots for profile: {profile}")
        out_lines.append("")

        for strategy in strategies:
            gs = gp[gp["strategy"] == strategy].copy()
            lat_s, cdf = compute_cdf(gs["latency_s"].to_numpy())

            print(f"  {strategy:<16} n={len(lat_s)}")

            if len(lat_s) == 0:
                continue

            out_lines.append(emit_addplot(strategy, lat_s, cdf))

        out_path = os.path.join(OUT_DIR, f"cdf_{sanitize_filename(profile)}.tex")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))

        print(f"  wrote: {out_path}")

    combined_path = os.path.join(OUT_DIR, "cdf_all_profiles.tex")
    combined_lines = []

    for profile in profiles:
        profile_path = os.path.join(OUT_DIR, f"cdf_{sanitize_filename(profile)}.tex")
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                combined_lines.append(f"% ===== profile: {profile} =====")
                combined_lines.append(f.read())
                combined_lines.append("")

    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines))

    print(f"\nWrote combined file: {combined_path}")


if __name__ == "__main__":
    main()