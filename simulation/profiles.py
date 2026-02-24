"""
Task profiles (compute means), module output sizes, and candidate stitches.
"""

import math
from typing import Dict, List

# ---------------------------
# Output tensor size helpers
# ---------------------------

def mb(tensors: int, bytes_per_elem: int = 2, use_mib: bool = True) -> float:
    """Convert element count to MiB (default fp16 => 2 bytes/elem)."""
    denom = (1024 * 1024) if use_mib else 1_000_000
    return tensors * bytes_per_elem / denom

# ---------------------------
# Module output sizes (MiB)
# ---------------------------

# Swin module names in CANDIDATE_STITCHES are:
OUTPUT_SIZES_MB: Dict[str, float] = {
    # ResNet18 outputs at the cut points
    "resnet_stem":    mb(56 * 56 * 64),
    "resnet_layer1":  mb(56 * 56 * 64),
    "resnet_layer2":  mb(28 * 28 * 128),
    "resnet_layer3":  mb(14 * 14 * 256),
    "resnet_layer4":  mb( 7 *  7 * 512),

    # Swin-T aligned to your coarse stage modules
    "swin_patch":   mb(56 * 56 * 96),    # after patch embed
    "swin_stage1":  mb(28 * 28 * 192),   # stage1 output (regardless of internal blocks)
    "swin_stage2":  mb(14 * 14 * 384),   # stage2 output
    "swin_stage3":  mb( 7 *  7 * 768),   # stage3 output
    "swin_stage4":  mb( 7 *  7 * 768),   # stage4 output (often pooled later, keep as-is)
    "head":         mb(10),              # logits (your current setup)
}




# ---------------------------
# Candidate stitches (modules + accuracy)
# ---------------------------

MODULE_LIST_SWIN: List[str] = [
    "swin_patch",
    "swin_stage1",
    "swin_stage2",
    "swin_stage3",
    "swin_stage4",
    "head",
]

PREFIX_C1: List[str] = ["resnet_stem"]                       # C2
PREFIX_C2: List[str] = ["resnet_stem", "resnet_layer1"]                       # C2
PREFIX_C3: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2"]      # C3
PREFIX_C4: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2", "resnet_layer3"]  # C4
PREFIX_C5: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2", "resnet_layer3", "resnet_layer4"]

#  FloodNet
#  1 |      768 | acc@1 = 79.33%
#  2 |      768 | acc@1 = 85.11%
#  3 |      768 | acc@1 = 81.56%
#  4 |      768 | acc@1 = 86.00%
#  5 |      768 | acc@1 = 91.78%
#  6 |      768 | acc@1 = 91.11%
#  7 |      768 | acc@1 = 90.00%
#

######## wildfire #######
#        0 |      512 | SKIP (head expects 768)
#        1 |      768 | acc@1 = 91.62%
#        2 |      768 | acc@1 = 86.00%
#        3 |      768 | acc@1 = 84.33%
#        4 |      768 | acc@1 = 87.00%
#        5 |      768 | acc@1 = 84.38%
#        6 |      768 | acc@1 = 83.94%
#        7 |      768 | acc@1 = 84.60%


CANDIDATE_STITCHES: Dict[int, Dict] = {
    # sid=1 (all TinySwin)
    1: {"acc": 91.62, "modules": MODULE_LIST_SWIN},

    # sid=2: C2 -> S2
    2: {"acc": 86.00, "modules": PREFIX_C1 + ["swin_stage1", "swin_stage2", "swin_stage3", "swin_stage4", "head"]},

    # sid=3: C2 -> S3
    3: {"acc": 84.33, "modules": PREFIX_C2 + ["swin_stage2", "swin_stage3", "swin_stage4", "head"]},

    # sid=4: C3 -> S3
    4: {"acc": 87.00, "modules": PREFIX_C3 + ["swin_stage3", "swin_stage4", "head"]},

    # sid=5: C3 -> S4
    5: {"acc": 84.38, "modules": PREFIX_C4 + ["swin_stage4", "head"]},

    # sid=6: C4 -> S4
    6: {"acc": 83.94, "modules": PREFIX_C5 + ["head"]},

    # sid=7: C4 -> S5 (map S5 to head)
    7: {"acc": 84.60, "modules": PREFIX_C4 + ["swin_stage3","head"]},
}

# ---------------------------
# Per-task execution profiles
# ---------------------------

TASK_PROFILES = {
    "extract-frames": {
        "sat":   {"prefix": 22.0, "suffix": 28.0},
        "edge":  {"prefix": 40.0, "suffix": 60.0},
        "cloud": {"prefix": 12.0, "suffix": 18.0},
    },
    "object-det": {
        "sat":   {"prefix": 15.0, "suffix": 20.0},
        "edge":  {"prefix": 28.0, "suffix": 42.0},
        "cloud": {"prefix":  9.0, "suffix": 14.0},
    },
    "prepare-ds": {
        "sat":   {"prefix": 35.0, "suffix": 45.0},
        "edge":  {"prefix": 60.0, "suffix": 90.0},
        "cloud": {"prefix": 20.0, "suffix": 30.0},
    },
}
