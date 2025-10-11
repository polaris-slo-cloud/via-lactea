"""
Central configuration & defaults.
Change here (or override via your own launcher) to steer experiments.
"""

import os

# Determinism
SEED               = 42

# Simulation sizes
NUM_RUNS_TASK      = 5
NUM_RUNS_WORKFLOW  = 5
WORKFLOW_STAGES    = 3

# SLO knobs
SLO_MS_TASK        = 75.0    # task-level SLO (also used for workflow 'task-SLO' checks)
SLO_MS_STAGE       = 150.0    # per-stage SLO used by SLO-first in workflows
SLO_ACC_MIN = 89.0  # example minimum accuracy (%) required


# Utility weights for when SLO cannot be met
ALPHA, BETA        = 0.7, 1.5

# Placement constraint
REQUIRE_DISTINCT_NODES = True

# Results directory
BASE_RESULTS_DIR = "../experiments/results"
RESULT_DIR = None  # or set to a fixed dir string

SATS_PER_RING = 100
NUM_RINGS     = 10

# Node pool sizes
NODE_COUNTS = {
    "sat":   SATS_PER_RING * NUM_RINGS,
    "edge":  20,
    "cloud": 10,
}

# NODE_COUNTS already provides counts for 'edge' and 'cloud'
ISL_NEIGHBOR_SPAN = 1
GATEWAYS_PER_RING = 2
INTER_RING_LINKS  = True

# Multi-hop scaling flags/parameters
SCALE_HOPS    = True
INTRA_BASE    = {"sat": 1.0, "edge": 1.0, "cloud": 1.0}
INTRA_K       = {"sat": 0.9, "edge": 0.7, "cloud": 0.4}
INTRA_JITTER  = 0.3  # 0 => deterministic

# Runtime variability (coeff. of variation)
CV_PREFIX = 0.15
CV_SUFFIX = 0.15

# Profiles to evaluate
TASK_PROFILES_FOR_TASK     = ["extract-frames", "object-det", "prepare-ds"]
TASK_PROFILES_FOR_WORKFLOW = ["extract-frames", "object-det", "prepare-ds"]

# Round-robin
RR_START_OFFSET = 0

def ensure_results_dir() -> str:
    """Create and return the output directory."""
    if RESULT_DIR:
        outdir = RESULT_DIR
    else:
        auto = f"sat{NODE_COUNTS['sat']}_edge{NODE_COUNTS['edge']}_cloud{NODE_COUNTS['cloud']}_seed{SEED}"
        outdir = os.path.join(BASE_RESULTS_DIR, auto)
    os.makedirs(outdir, exist_ok=True)
    return outdir
