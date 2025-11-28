"""
Central configuration & defaults.
Change here (or override via your own launcher) to steer experiments.
"""

import os

# Determinism
SEED               = 42

# Simulation sizes
NUM_RUNS_TASK      = 10
NUM_RUNS_WORKFLOW  = 1
WORKFLOW_STAGES    = 3

# SLO knobs
SLO_MS_TASK        = 70.0    # task-level SLO (also used for workflow 'task-SLO' checks)
SLO_MS_STAGE       = 600.0    # per-stage SLO used by SLO-first in workflows
SLO_ACC_MIN = 89.0  # example minimum accuracy (%) required


RUN_MODE = "workflow"   # options: "task", "workflow", "both"


# Results directory
BASE_RESULTS_DIR = "../experiments/results"
RESULT_DIR = None  # or set to a fixed dir string

SLO_MS_TASK_PER_PROFILE = {
    "extract-frames": 1000.0,   # per-hop ms allowed for this profile
    "object-det": 1000.0,
    "prepare-ds": 1000.0,
}

SATS_PER_RING = 10
NUM_RINGS     = 2

# Node pool sizes
NODE_COUNTS = {
    "sat":   SATS_PER_RING * NUM_RINGS,
    "edge":  20,
    "cloud": 1,
}

# caching policy
CACHE_FIRST_RUN = False          # steady-state (allow cache hits)
CACHEABLE_LAYER_PATTERNS = [
    "swin_stage1_*",
    "resnet_layer1*",
    # add the REAL module names you want considered cacheable
]
CACHE_DEBUG = False

# Which baselines (strategies) to run by default.
# Accepted values:
#   "all"                             -> run everything
#   "SLO-first"                       -> run only SLO-first
#   "SLO-first,Round-Robin"           -> comma/pipe separated names
#   ["SLO-first", "Round-Robin"]      -> list is also fine
ENABLED_BASELINES = "all"


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
