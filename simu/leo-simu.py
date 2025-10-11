#!/usr/bin/env python3
"""
Stitch Selection with Concrete Node Placement (Per-Module on Different Nodes)
+ Per-task Execution Profiles (different mean runtimes per task/stage)
+ Baselines: Random, Round-Robin, and Lowest-Latency
+ DATA TRAFFIC: payload_MB vs link_MB (multi-hop aware)
+ HOP COUNT: total number of physical network hops (scales with network size)
+ SLO VIOLATIONS (TASK-SLO-BASED): % crossing SLO_MS_TASK at task and workflow levels
+ CSV EXPORTS for task (by profile) and workflow summaries (incl. accuracy, payload, link, hops)
+ CONFIGURABLE RESULTS DIRECTORY
"""

import math, random, os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Tuple as Tup
import numpy as np
import pandas as pd

# ============================
# Parameters
# ============================

SEED               = 42
NUM_RUNS_TASK      = 400          # single-task decisions (per profile)
NUM_RUNS_WORKFLOW  = 200          # multi-stage workflow decisions
WORKFLOW_STAGES    = 3            # number of sequential tasks in the workflow
SLO_MS_TASK        = 50.0         # task SLO (also used to judge workflow violations)
SLO_MS_STAGE       = 76.0         # stage SLO used by the SLO-first picker in workflows
ALPHA, BETA        = 1.0, 0.7     # utility weights when SLO can't be met
REQUIRE_DISTINCT_NODES = True     # place every module on a different node per run

# ---- Results directory (configure me) ----
# If RESULT_DIR is None, a directory name is auto-generated from node counts + seed, under BASE_RESULTS_DIR.
BASE_RESULTS_DIR = "results"
RESULT_DIR = None  # e.g., "results/exp1" or None to auto-name

# Node pool sizes (concrete node IDs are created)
NODE_COUNTS = {
    "sat":   100,   # satellites
    "edge":  30,    # edge gateways
    "cloud": 10     # cloud/ground
}

# ============================
# Multi-hop scaling
# ============================
# Enable multi-hop expansion so hop_count/link_mb/latency grow with network size.
SCALE_HOPS = True

# Expected *intra-kind* hops ≈ base + k * growth(n_kind)
# Default growth is log2, but you can swap for n**0.5 etc. if you prefer.
INTRA_BASE = {"sat": 1.0, "edge": 1.0, "cloud": 1.0}    # minimum per intra segment
INTRA_K    = {"sat": 0.9, "edge": 0.7, "cloud": 0.4}    # sensitivity to size
INTRA_JITTER = 0.3   # 0 = deterministic; >0 adds modest variability

def _growth(n: int) -> float:
    return math.log2(max(n, 2))  # change to (n ** 0.5) if you want stronger scaling

# ============================
# Per-task execution profiles
# ============================
TASK_PROFILES = {
    "default": {
        "sat":   {"prefix": 22.0, "suffix": 28.0},
        "edge":  {"prefix": 40.0, "suffix": 60.0},
        "cloud": {"prefix": 12.0, "suffix": 18.0},
    },
    "fast": {
        "sat":   {"prefix": 15.0, "suffix": 20.0},
        "edge":  {"prefix": 28.0, "suffix": 42.0},
        "cloud": {"prefix":  9.0, "suffix": 14.0},
    },
    "heavy": {
        "sat":   {"prefix": 35.0, "suffix": 45.0},
        "edge":  {"prefix": 60.0, "suffix": 90.0},
        "cloud": {"prefix": 20.0, "suffix": 30.0},
    },
}

# Runtime variability (coefficient of variation).
CV_PREFIX = 0.15
CV_SUFFIX = 0.15

# Profiles to use
TASK_PROFILES_FOR_TASKS = ["fast", "default", "heavy"]     # run tasks for all three
TASK_PROFILES_FOR_WORKFLOW = ["fast", "default", "heavy"]  # per-stage profiles

# ============================
# Output sizes (MiB, fp16)
# ============================

def mb(tensors, bytes_per_elem=2, use_mib=True):
    denom = (1024*1024) if use_mib else 1_000_000
    return tensors * bytes_per_elem / denom

OUTPUT_SIZES_MB = {
    # ---------- ResNet18 ----------
    "resnet_stem":    mb(56*56*64),
    "resnet_layer1":  mb(56*56*64),
    "resnet_layer2":  mb(28*28*128),
    "resnet_layer3":  mb(14*14*256),
    "resnet_layer4":  mb( 7* 7*512),

    # ---------- Swin-T (approx; aligned to your cuts) ----------
    "swin_patch_embed":  mb(56*56*96),
    "swin_stage1_b0":    mb(28*28*192),
    "swin_stage1_b1":    mb(28*28*192),
    "swin_stage2_b0":    mb(14*14*384),
    "swin_stage2_b1":    mb(14*14*384),
    "swin_stage3":       mb( 7* 7*768),
    "swin_stage4":       mb( 7* 7*768),
    "swin_tail_extras":  mb(768),
    "head":              mb(10),
}

# ============================
# Modules & stitch candidates
# ============================

MODULE_LIST_SWIN = [
    "swin_patch_embed",
    "swin_stage1_b0",
    "swin_stage1_b1",
    "swin_stage2_b0",
    "swin_stage2_b1",
    "swin_stage3",
    "swin_stage4",
    "swin_tail_extras",
    "head",
]

PREFIX_LAYER1 = ["resnet_stem", "resnet_layer1"]
PREFIX_LAYER2 = ["resnet_stem", "resnet_layer1", "resnet_layer2"]

CANDIDATE_STITCHES: Dict[int, Dict] = {
    1: {"acc": 90.47, "modules": MODULE_LIST_SWIN},

    # Layer-1 prefix (C2)
    2: {"acc": 89.99, "modules": PREFIX_LAYER1 + ["swin_stage1_b0", "swin_stage1_b1", "swin_stage2_b0",
                                                  "swin_stage2_b1", "swin_stage3", "swin_stage4",
                                                  "swin_tail_extras", "head"]},
    3: {"acc": 89.01, "modules": PREFIX_LAYER1 + ["swin_stage2_b0", "swin_stage2_b1", "swin_stage3",
                                                  "swin_stage4", "swin_tail_extras", "head"]},
    4: {"acc": 88.07, "modules": PREFIX_LAYER1 + ["swin_stage2_b1", "swin_stage3",
                                                  "swin_stage4", "swin_tail_extras", "head"]},

    # Layer-2 prefix (C3)
    5: {"acc": 86.76, "modules": PREFIX_LAYER2 + ["swin_stage2_b0", "swin_stage2_b1",
                                                  "swin_stage3", "swin_stage4", "swin_tail_extras", "head"]},
    6: {"acc": 84.74, "modules": PREFIX_LAYER2 + ["swin_stage2_b1", "swin_stage3",
                                                  "swin_stage4", "swin_tail_extras", "head"]},
    7: {"acc": 82.24, "modules": PREFIX_LAYER2 + ["swin_stage2_b1", "swin_stage3",
                                                  "swin_stage4", "swin_tail_extras", "head"]},
}

# ---- Round-robin config ----
RR_STITCH_ORDER = sorted(CANDIDATE_STITCHES.keys())
RR_START_OFFSET = 0

# ==================================
# Cluster & Links
# ==================================

@dataclass(frozen=True)
class Node:
    nid: str
    kind: str     # "sat" | "edge" | "cloud"

def build_nodes() -> List[Node]:
    nodes: List[Node] = []
    for k in ("sat", "edge", "cloud"):
        for i in range(NODE_COUNTS[k]):
            nodes.append(Node(f"{k}{i}", k))
    return nodes

def link_kind(a_kind: str, b_kind: str) -> str:
    if a_kind == b_kind:
        return "isl" if a_kind == "sat" else "edge_backhaul"
    if {"sat", "cloud"} == {a_kind, b_kind}:
        return "downlink"
    if {"edge", "cloud"} == {a_kind, b_kind}:
        return "edge_backhaul"
    return "edge_backhaul"

def sample_link_state(kind: str, rng: random.Random) -> Tup[float, float]:
    # (bandwidth_Mbps, propagation_ms)
    if kind == "downlink":
        return rng.uniform(30, 150), rng.uniform(25, 60)
    if kind == "isl":
        return rng.uniform(20, 80),  rng.uniform(30, 80)
    if kind == "edge_backhaul":
        return rng.uniform(100, 1000), rng.uniform(5, 25)
    return 10.0, 100.0

# ---- New: size-aware intra-kind hop estimator ----
def expected_intra_hops(kind: str, nodes: List[Node]) -> int:
    if not SCALE_HOPS:
        return 1
    n_kind = sum(1 for n in nodes if n.kind == kind)
    mean_hops = INTRA_BASE[kind] + INTRA_K[kind] * _growth(n_kind)
    mean_hops = max(1.0, mean_hops)
    if INTRA_JITTER > 0:
        sigma = INTRA_JITTER
        mu = math.log(mean_hops) - 0.5 * sigma * sigma
        sample = max(1.0, random.lognormvariate(mu, sigma))
        return max(1, int(round(sample)))
    return max(1, int(round(mean_hops)))

# ---- New: multi-hop routes that scale with size ----
def route_hops(src: Node, dst: Node, nodes: List[Node], rng: random.Random) -> List[Tuple[str, Node, Node]]:
    """
    Build a physical hop list:
      - same kind: intra(kind)
      - edge<->cloud or sat<->cloud: intra(src) + 1 cross + intra(dst)
      - edge<->sat via cloud: intra(edge) + 1 cross(e->c) + intra(cloud) + 1 cross(c->s) + intra(sat)
    We don't enumerate actual intermediate Node IDs (not needed for timing);
    we just emit per-hop link kinds so transfer time and link_mb/hop_count scale properly.
    """
    def expand(kind: str, count: int, a: Node, b: Node) -> List[Tuple[str, Node, Node]]:
        return [(kind, a, b)] * max(0, count)

    if src.nid == dst.nid:
        return []

    if src.kind == dst.kind:
        intra = expected_intra_hops(src.kind, nodes)
        return expand("isl" if src.kind == "sat" else "edge_backhaul", intra, src, dst)

    kinds = {src.kind, dst.kind}

    if kinds == {"edge", "sat"}:
        hops: List[Tuple[str, Node, Node]] = []
        # intra edge
        hops += expand("edge_backhaul", expected_intra_hops("edge", nodes), src, src)
        # edge -> cloud cross
        hops += expand("edge_backhaul", 1, src, src)
        # intra cloud
        hops += expand("edge_backhaul", expected_intra_hops("cloud", nodes), src, dst)
        # cloud -> sat cross
        hops += expand("downlink", 1, src, dst)
        # intra sat
        hops += expand("isl", expected_intra_hops("sat", nodes), src, dst)
        return hops

    if kinds == {"sat", "cloud"}:
        hops  = expand("isl", expected_intra_hops("sat", nodes), src, src)
        hops += expand("downlink", 1, src, dst)
        hops += expand("edge_backhaul", expected_intra_hops("cloud", nodes), src, dst)
        return hops

    if kinds == {"edge", "cloud"}:
        hops  = expand("edge_backhaul", expected_intra_hops("edge", nodes), src, src)
        hops += expand("edge_backhaul", 1, src, dst)
        hops += expand("edge_backhaul", expected_intra_hops("cloud", nodes), src, dst)
        return hops

    # Fallback
    return [("edge_backhaul", src, dst)]

def hop_transfer_ms_and_bytes(kind: str, payload_mb: float, rng: random.Random) -> Tup[float, float]:
    bw_mbps, prop_ms = sample_link_state(kind, rng)
    tx_ms = float("inf") if bw_mbps <= 0 else (payload_mb / (bw_mbps / 8.0)) * 1000.0
    jitter = rng.uniform(0, 10)
    return tx_ms + prop_ms + jitter, payload_mb

# ==================================
# Placement (per run)
# ==================================

def assign_distinct_nodes(modules: List[str], nodes: List[Node], rng: random.Random) -> Dict[str, Node]:
    chosen = rng.sample(nodes, k=len(modules))
    return {m: n for m, n in zip(modules, chosen)}

# ==================================
# Per-task compute time (profiles)
# ==================================

def _sample_runtime_lognormal(mean_ms: float, cv: float, rng: random.Random) -> float:
    if mean_ms <= 0:
        return 0.0
    sigma2 = math.log(1.0 + cv*cv)
    sigma  = math.sqrt(sigma2)
    mu     = math.log(mean_ms) - 0.5 * sigma2
    return rng.lognormvariate(mu, sigma)

def compute_time_ms(module: str, node: Node, rng: random.Random, task_profile_name: str) -> float:
    prof = TASK_PROFILES.get(task_profile_name, TASK_PROFILES["default"])
    is_prefix = module.startswith("resnet_")
    bucket = "prefix" if is_prefix else "suffix"
    base_mean = prof[node.kind][bucket]
    cv = CV_PREFIX if is_prefix else CV_SUFFIX
    return _sample_runtime_lognormal(base_mean, cv, rng)

def module_output_mb(module: str) -> float:
    return OUTPUT_SIZES_MB.get(module, 0.0)

# ==================================
# Stitch evaluation on a concrete placement
# ==================================

def e2e_metrics_for_stitch(stitch_id: int,
                           placement: Dict[str, Node],
                           nodes: List[Node],
                           rng: random.Random,
                           task_profile_name: str) -> Dict[str, float]:
    spec = CANDIDATE_STITCHES[stitch_id]
    modules = spec["modules"]

    latency_ms = 0.0
    payload_mb_total = 0.0
    link_mb_total = 0.0
    hop_count_total = 0

    for i, m in enumerate(modules):
        node = placement[m]
        latency_ms += compute_time_ms(m, node, rng, task_profile_name=task_profile_name)

        if i < len(modules) - 1:
            src = node
            dst = placement[modules[i+1]]
            payload = module_output_mb(m)

            payload_mb_total += payload

            hops = route_hops(src, dst, nodes, rng)
            hop_count_total += len(hops)
            for kind, _, _ in hops:
                t_ms, hop_mb = hop_transfer_ms_and_bytes(kind, payload, rng)
                latency_ms += t_ms
                link_mb_total += hop_mb

    return {
        "latency_ms": latency_ms,
        "payload_mb": payload_mb_total,
        "link_mb":    link_mb_total,
        "hop_count":  hop_count_total,
        "acc":        spec["acc"]
    }

# ==================================
# Selection policies (5 strategies)
# ==================================

def choose_stitch_for_task(placement: Dict[str, Node],
                           nodes: List[Node],
                           rng: random.Random,
                           slo_ms: float,
                           task_profile_name: str,
                           alpha: float = ALPHA,
                           beta: float = BETA) -> Dict:
    best_meet = None
    for sid, spec in CANDIDATE_STITCHES.items():
        mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
        t = mets["latency_ms"]
        if t <= slo_ms:
            key = (spec["acc"], -t)
            cand = {"stitch_id": sid, "met_slo": True, **mets}
            if (best_meet is None) or (key > (best_meet["acc"], -best_meet["latency_ms"])):
                best_meet = cand
    if best_meet:
        return best_meet

    best_u = None
    for sid, spec in CANDIDATE_STITCHES.items():
        mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
        t = mets["latency_ms"]
        U = alpha * (spec["acc"]/100.0) - beta * (t/slo_ms)
        cand = {"stitch_id": sid, "met_slo": False, "utility": U, **mets}
        if (best_u is None) or (U > best_u["utility"]) or (math.isclose(U, best_u["utility"]) and t < best_u["latency_ms"]):
            best_u = cand
    return best_u

def always_best_accuracy(placement: Dict[str, Node],
                         nodes: List[Node],
                         rng: random.Random,
                         task_profile_name: str) -> Dict:
    top_acc = max(spec["acc"] for spec in CANDIDATE_STITCHES.values())
    best = None
    for sid, spec in CANDIDATE_STITCHES.items():
        if not math.isclose(spec["acc"], top_acc):
            continue
        mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
        cand = {"stitch_id": sid, **mets}
        if (best is None) or (mets["latency_ms"] < best["latency_ms"]):
            best = cand
    return best

def lowest_latency(placement: Dict[str, Node],
                   nodes: List[Node],
                   rng: random.Random,
                   task_profile_name: str) -> Dict:
    best = None
    for sid in CANDIDATE_STITCHES.keys():
        mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
        if (best is None) or (mets["latency_ms"] < best["latency_ms"]):
            best = {"stitch_id": sid, **mets}
    return best

def random_pick_stitch(placement: Dict[str, Node],
                       nodes: List[Node],
                       rng: random.Random,
                       task_profile_name: str) -> Dict:
    sid = rng.choice(list(CANDIDATE_STITCHES.keys()))
    mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
    return {"stitch_id": sid, **mets}

def round_robin_pick_stitch(index: int,
                            placement: Dict[str, Node],
                            nodes: List[Node],
                            rng: random.Random,
                            task_profile_name: str,
                            offset: int = RR_START_OFFSET) -> Dict:
    sid = RR_STITCH_ORDER[(index + offset) % len(RR_STITCH_ORDER)]
    mets = e2e_metrics_for_stitch(sid, placement, nodes, rng, task_profile_name)
    return {"stitch_id": sid, **mets}

# ==================================
# Simulations
# ==================================

def simulate_task(nodes: List[Node], num_runs: int, slo_ms: float, seed: int,
                  task_profile_name: str) -> pd.DataFrame:
    rng = random.Random(seed)
    records = []
    all_modules: List[str] = sorted({m for s in CANDIDATE_STITCHES.values() for m in s["modules"]})

    for run in range(num_runs):
        placement = assign_distinct_nodes(all_modules, nodes, rng) if REQUIRE_DISTINCT_NODES \
                    else {m: rng.choice(nodes) for m in all_modules}

        a = choose_stitch_for_task(placement, nodes, rng, slo_ms, task_profile_name)
        records.append({"run": run, "strategy": "SLO-first", **a, "profile": task_profile_name})

        b = always_best_accuracy(placement, nodes, rng, task_profile_name)
        b["met_slo"] = (b["latency_ms"] <= slo_ms)
        records.append({"run": run, "strategy": "Best-Acc", **b, "profile": task_profile_name})

        ll = lowest_latency(placement, nodes, rng, task_profile_name)
        ll["met_slo"] = (ll["latency_ms"] <= slo_ms)
        records.append({"run": run, "strategy": "Lowest-Latency", **ll, "profile": task_profile_name})

        r = random_pick_stitch(placement, nodes, rng, task_profile_name)
        r["met_slo"] = (r["latency_ms"] <= slo_ms)
        records.append({"run": run, "strategy": "Random", **r, "profile": task_profile_name})

        rr = round_robin_pick_stitch(run, placement, nodes, rng, task_profile_name)
        rr["met_slo"] = (rr["latency_ms"] <= slo_ms)
        records.append({"run": run, "strategy": "Round-Robin", **rr, "profile": task_profile_name})

    return pd.DataFrame(records)

def simulate_workflow(nodes: List[Node],
                      num_runs: int,
                      stages: int,
                      slo_ms_stage: float,
                      seed: int,
                      stage_profiles: List[str]) -> pd.DataFrame:
    assert len(stage_profiles) == stages
    rng = random.Random(seed)
    records = []
    all_modules: List[str] = sorted({m for s in CANDIDATE_STITCHES.values() for m in s["modules"]})

    for run in range(num_runs):
        for strat_name, picker in [
            ("SLO-first",     lambda pl,n,r,prof: choose_stitch_for_task(pl,n,r,slo_ms_stage,prof)),
            ("Best-Acc",      always_best_accuracy),
            ("Lowest-Latency",lowest_latency),
            ("Random",        random_pick_stitch),
            ("Round-Robin",   None),  # handled specially
        ]:
            lat_sum = acc_last = payload_sum = link_sum = 0.0
            hops_sum = 0
            stages_met_stage_slo = 0
            stages_met_task_slo  = 0

            for k in range(stages):
                prof = stage_profiles[k]
                placement = assign_distinct_nodes(all_modules, nodes, rng) if REQUIRE_DISTINCT_NODES \
                            else {m: rng.choice(nodes) for m in all_modules}
                if strat_name == "Round-Robin":
                    pick = round_robin_pick_stitch(run + k, placement, nodes, rng, prof)
                else:
                    pick = picker(placement, nodes, rng, prof)

                lat_sum     += pick["latency_ms"]
                payload_sum += pick["payload_mb"]
                link_sum    += pick["link_mb"]
                hops_sum    += pick["hop_count"]
                acc_last     = pick["acc"]
                stages_met_stage_slo += (pick["latency_ms"] <= slo_ms_stage)
                stages_met_task_slo  += (pick["latency_ms"] <= SLO_MS_TASK)

            records.append({
                "run": run, "strategy": strat_name,
                "latency_ms": lat_sum, "acc": acc_last,
                "payload_mb": payload_sum, "link_mb": link_sum, "hop_count": hops_sum,
                "met_slo": (stages_met_stage_slo == stages),
                "stages_met": stages_met_stage_slo,
                "met_task_slo_all": (stages_met_task_slo == stages),
                "stages_met_task_slo": stages_met_task_slo,
                "stages": stages, "profiles": "|".join(stage_profiles),
            })

    return pd.DataFrame(records)

# ==================================
# Stats & helpers
# ==================================

def agg_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby("strategy")[col]
          .agg(mean="mean", median="median",
               p95=lambda s: np.percentile(s, 95),
               p99=lambda s: np.percentile(s, 99))
          .reset_index()
    )

def agg_stats_by_profile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        df.groupby(["profile", "strategy"])[col]
          .agg(mean="mean", median="median",
               p95=lambda s: np.percentile(s, 95),
               p99=lambda s: np.percentile(s, 99))
          .reset_index()
    )

def accuracy_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("strategy")["acc"].mean().reset_index(name="acc_mean")

def accuracy_stats_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["profile","strategy"])["acc"].mean().reset_index(name="acc_mean")

def slo_violation_rates_task(df: pd.DataFrame) -> pd.DataFrame:
    rates = df.groupby("strategy")["met_slo"].mean().reset_index(name="met_rate")
    rates["slo_violation_pct_task_slo"] = (1.0 - rates["met_rate"]) * 100.0
    return rates[["strategy", "slo_violation_pct_task_slo"]]

def slo_violation_rates_task_by_profile(df: pd.DataFrame) -> pd.DataFrame:
    rates = df.groupby(["profile","strategy"])["met_slo"].mean().reset_index(name="met_rate")
    rates["slo_violation_pct_task_slo"] = (1.0 - rates["met_rate"]) * 100.0
    return rates[["profile","strategy","slo_violation_pct_task_slo"]]

def slo_violation_rates_workflow_task_slo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runs = df.groupby("strategy")["met_task_slo_all"].mean().reset_index(name="runs_met_rate")
    runs["slo_violation_pct_per_run_task_slo"] = (1.0 - runs["runs_met_rate"]) * 100.0
    runs = runs[["strategy", "slo_violation_pct_per_run_task_slo"]]

    stage_totals = df.groupby("strategy")[["stages_met_task_slo", "stages"]].sum().reset_index()
    stage_totals["per_stage_violation_pct_task_slo"] = (
        1.0 - (stage_totals["stages_met_task_slo"] / stage_totals["stages"])
    ) * 100.0
    stages = stage_totals[["strategy", "per_stage_violation_pct_task_slo"]]
    return runs, stages

def pareto_front(points: np.ndarray) -> np.ndarray:
    N = points.shape[0]
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]:
            continue
        Li, Ai = points[i]
        mask = (points[:,0] <= Li) & (points[:,1] >= Ai) & ((points[:,0] < Li) | (points[:,1] > Ai))
        if mask.any():
            dominated[i] = True
    return ~dominated

# ==================================
# Helpers: results directory & save
# ==================================

def ensure_results_dir() -> str:
    if RESULT_DIR:
        outdir = RESULT_DIR
    else:
        auto_name = f"sat{NODE_COUNTS['sat']}_edge{NODE_COUNTS['edge']}_cloud{NODE_COUNTS['cloud']}_seed{SEED}"
        outdir = os.path.join(BASE_RESULTS_DIR, auto_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def save_csv(df: pd.DataFrame, outdir: str, filename: str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    df.to_csv(path, index=False)
    return path

# ==================================
# Main
# ==================================

if __name__ == "__main__":
    rng = random.Random(SEED)
    nodes = build_nodes()
    outdir = ensure_results_dir()
    print(f"[info] Saving all CSVs to: {outdir}")

    # --- Single Task (run for ALL profiles) ---
    task_dfs = []
    for i, prof in enumerate(TASK_PROFILES_FOR_TASKS):
        dfp = simulate_task(
            nodes, NUM_RUNS_TASK, SLO_MS_TASK, seed=SEED+1+i,
            task_profile_name=prof
        )
        task_dfs.append(dfp)
    task_df_all = pd.concat(task_dfs, ignore_index=True)

    # Console summaries
    print("\n=== TASK (single-stage) — aggregated across profiles ===")
    for col in ["latency_ms","payload_mb","link_mb","hop_count"]:
        print(f"{col}:")
        print(agg_stats(task_df_all, col).to_string(index=False))
    print("Accuracy (%):")
    print(accuracy_stats(task_df_all).to_string(index=False))
    print("TASK SLO violation rate (% of task requests) [based on SLO_MS_TASK]:")
    print(slo_violation_rates_task(task_df_all).to_string(index=False))

    print("\n=== TASK (single-stage) — by profile ===")
    for col in ["latency_ms","payload_mb","link_mb","hop_count"]:
        print(f"{col} by profile:")
        print(agg_stats_by_profile(task_df_all, col).to_string(index=False))
    print("Accuracy (%) by profile:")
    print(accuracy_stats_by_profile(task_df_all).to_string(index=False))
    print("SLO violation rate by profile (% of task requests):")
    print(slo_violation_rates_task_by_profile(task_df_all).to_string(index=False))

    # --- Save TASK CSVs ---
    save_csv(task_df_all, outdir, "task_runs_all.csv")

    # Per-profile summaries for all metrics
    save_csv(agg_stats_by_profile(task_df_all, "latency_ms"), outdir, "task_summary_latency_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "payload_mb"), outdir, "task_summary_payload_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "link_mb"),    outdir, "task_summary_link_by_strategy_profile.csv")
    save_csv(agg_stats_by_profile(task_df_all, "hop_count"),  outdir, "task_summary_hopcount_by_strategy_profile.csv")

    # Accuracy & SLO per profile
    save_csv(accuracy_stats_by_profile(task_df_all), outdir, "task_accuracy_by_strategy_profile.csv")
    save_csv(slo_violation_rates_task_by_profile(task_df_all), outdir, "task_slo_violation_by_profile.csv")

    # One wide CSV (latency+accuracy+payload+link+hops) per profile+strategy
    _task_lat = agg_stats_by_profile(task_df_all, "latency_ms")
    _task_acc = accuracy_stats_by_profile(task_df_all)
    _task_payload = agg_stats_by_profile(task_df_all, "payload_mb")
    _task_link = agg_stats_by_profile(task_df_all, "link_mb")
    _task_hops = agg_stats_by_profile(task_df_all, "hop_count")
    _task_wide = _task_lat.merge(_task_acc, on=["profile","strategy"])\
                          .merge(_task_payload, on=["profile","strategy"], suffixes=("","_payload"))\
                          .merge(_task_link, on=["profile","strategy"], suffixes=("", "_link"))\
                          .merge(_task_hops, on=["profile","strategy"], suffixes=("", "_hops"))
    save_csv(_task_wide, outdir, "task_summary_ALL_by_strategy_profile.csv")

    # --- Workflow (per-stage profiles) ---
    wf_df = simulate_workflow(
        nodes, NUM_RUNS_WORKFLOW, WORKFLOW_STAGES, SLO_MS_STAGE,
        seed=SEED+10, stage_profiles=TASK_PROFILES_FOR_WORKFLOW
    )
    print(f"\n=== WORKFLOW ({WORKFLOW_STAGES} stages) ===")
    for col in ["latency_ms","payload_mb","link_mb","hop_count"]:
        print(f"{col}:")
        print(agg_stats(wf_df, col).to_string(index=False))
    print("Accuracy (%):")
    print(accuracy_stats(wf_df).to_string(index=False))

    # Workflow SLO (task-SLO based)
    runs_rate_task, stages_rate_task = slo_violation_rates_workflow_task_slo(wf_df)
    print("Workflow SLO violation (per-run; any stage violates) — based on SLO_MS_TASK:")
    print(runs_rate_task.to_string(index=False))
    print("Workflow SLO violation (per-stage across all runs) — based on SLO_MS_TASK:")
    print(stages_rate_task.to_string(index=False))

    # --- Save WORKFLOW CSVs ---
    save_csv(wf_df, outdir, "workflow_runs.csv")
    save_csv(agg_stats(wf_df, "latency_ms"), outdir, "workflow_summary_latency_by_strategy.csv")
    save_csv(agg_stats(wf_df, "payload_mb"), outdir, "workflow_summary_payload_by_strategy.csv")
    save_csv(agg_stats(wf_df, "link_mb"),    outdir, "workflow_summary_link_by_strategy.csv")
    save_csv(agg_stats(wf_df, "hop_count"),  outdir, "workflow_summary_hopcount_by_strategy.csv")
    save_csv(accuracy_stats(wf_df), outdir, "workflow_accuracy_by_strategy.csv")

    # Wide workflow summary
    _wf_lat   = agg_stats(wf_df, "latency_ms")
    _wf_acc   = accuracy_stats(wf_df)
    _wf_pl    = agg_stats(wf_df, "payload_mb")
    _wf_link  = agg_stats(wf_df, "link_mb")
    _wf_hops  = agg_stats(wf_df, "hop_count")
    _wf_wide = _wf_lat.merge(_wf_acc, on="strategy")\
                      .merge(_wf_pl, on="strategy", suffixes=("","_payload"))\
                      .merge(_wf_link, on="strategy", suffixes=("", "_link"))\
                      .merge(_wf_hops, on="strategy", suffixes=("", "_hops"))
    save_csv(_wf_wide, outdir, "workflow_summary_ALL_by_strategy.csv")

    save_csv(runs_rate_task, outdir, "workflow_slo_violation_task_slo_runs.csv")
    save_csv(stages_rate_task, outdir, "workflow_slo_violation_task_slo_stages.csv")

    # Quick look
    best_acc_task_mean = float(task_df_all.groupby("strategy")["latency_ms"].mean().loc["Best-Acc"])
    best_acc_wf_mean   = float(wf_df.groupby("strategy")["latency_ms"].mean().loc["Best-Acc"])
    print("\nLatency when ALWAYS picking best accuracy:")
    print(f"- Task mean E2E latency (across profiles): {best_acc_task_mean:.2f} ms")
    print(f"- Workflow mean E2E latency: {best_acc_wf_mean:.2f} ms")

    # Pareto counts
    for name, df in [("Task (all profiles)", task_df_all), ("Workflow", wf_df)]:
        pts = np.c_[df["latency_ms"].to_numpy(), df["acc"].to_numpy()]
        mask = pareto_front(pts)
        print(f"\n{name}: {mask.sum()} non-dominated points out of {len(df)} total.")
