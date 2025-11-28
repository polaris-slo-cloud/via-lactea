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

OUTPUT_SIZES_MB: Dict[str, float] = {
    # ResNet18 (prefix)
    "resnet_stem":    mb(56*56*64),
    "resnet_layer1":  mb(56*56*64),
    "resnet_layer2":  mb(28*28*128),
    "resnet_layer3":  mb(14*14*256),
    "resnet_layer4":  mb( 7* 7*512),

    # Swin-T (approx; aligned to cuts)
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




# ---------------------------
# Candidate stitches (modules + accuracy)
# ---------------------------

MODULE_LIST_SWIN: List[str] = [
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
    2: {"acc": 89.99, "modules": PREFIX_LAYER1 + [
        "swin_stage1_b0","swin_stage1_b1","swin_stage2_b0","swin_stage2_b1",
        "swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},
    3: {"acc": 89.01, "modules": PREFIX_LAYER1 + [
        "swin_stage2_b0","swin_stage2_b1","swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},
    4: {"acc": 88.07, "modules": PREFIX_LAYER1 + [
        "swin_stage2_b1","swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},

    # Layer-2 prefix (C3)
    5: {"acc": 86.76, "modules": PREFIX_LAYER2 + [
        "swin_stage2_b0","swin_stage2_b1","swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},
    6: {"acc": 84.74, "modules": PREFIX_LAYER2 + [
        "swin_stage2_b1","swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},
    7: {"acc": 82.24, "modules": PREFIX_LAYER2 + [
        "swin_stage2_b1","swin_stage3","swin_stage4","swin_tail_extras","head"
    ]},
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
