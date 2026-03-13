"""
Task profiles (compute means), module output sizes, and candidate stitches.

Adds dataset configurability.

Usage pattern:
- main.py sets config.DATASET (or env VIA_LACTEA_DATASET) before importing this module,
  or calls set_dataset("wildfire") early.
- This module exposes:
    - set_dataset(name)
    - get_dataset()
    - OUTPUT_SIZES_MB, CANDIDATE_STITCHES, TASK_PROFILES (built for the active dataset)
    - DATASET_NAME (string)

Results:
- The active dataset name can be written into CSVs by adding a "dataset" column
  in your stats writer (see snippet at bottom).
"""

import os
from typing import Dict, List, Optional

from . import config

# ---------------------------
# Output tensor size helpers
# ---------------------------

def mb(tensors: int, bytes_per_elem: int = 2, use_mib: bool = True) -> float:
    """Convert element count to MiB (default fp16 => 2 bytes/elem)."""
    denom = (1024 * 1024) if use_mib else 1_000_000
    return tensors * bytes_per_elem / denom

# ---------------------------
# Dataset selection
# ---------------------------

_SUPPORTED_DATASETS = {"wildfire", "floodnet", "c10"}

def _normalize_dataset(name: Optional[str]) -> str:
    if not name:
        return "wildfire"
    n = str(name).strip().lower()
    # accept some aliases
    if n in {"cifar10", "cifar-10"}:
        n = "c10"
    if n not in _SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {sorted(_SUPPORTED_DATASETS)}")
    return n

# Public: current dataset name
DATASET_NAME: str = _normalize_dataset(getattr(config, "DATASET", None))

def set_dataset(name: str) -> None:
    """
    Set dataset globally for this module and (optionally) config.
    Call this BEFORE running simulations if you want to switch datasets.
    """
    global DATASET_NAME, OUTPUT_SIZES_MB, CANDIDATE_STITCHES, TASK_PROFILES
    DATASET_NAME = _normalize_dataset(name)
    setattr(config, "DATASET", DATASET_NAME)
    OUTPUT_SIZES_MB = _build_output_sizes_mb()
    CANDIDATE_STITCHES = _build_candidate_stitches()
    TASK_PROFILES = _build_task_profiles()

def get_dataset() -> str:
    return DATASET_NAME

# ---------------------------
# Module output sizes (MiB)
# ---------------------------

def _build_output_sizes_mb() -> Dict[str, float]:
    # If you ever need dataset-dependent tensor sizes, branch on DATASET_NAME here.
    return {
        # ResNet18 outputs at the cut points
        "resnet_stem":    mb(56 * 56 * 64),
        "resnet_layer1":  mb(56 * 56 * 64),
        "resnet_layer2":  mb(28 * 28 * 128),
        "resnet_layer3":  mb(14 * 14 * 256),
        "resnet_layer4":  mb( 7 *  7 * 512),

        # Swin-T aligned to your coarse stage modules
        "swin_patch":   mb(56 * 56 * 96),
        "swin_stage1":  mb(28 * 28 * 192),
        "swin_stage2":  mb(14 * 14 * 384),
        "swin_stage3":  mb( 7 *  7 * 768),
        "swin_stage4":  mb( 7 *  7 * 768),
        "head":         mb(10),
    }

OUTPUT_SIZES_MB: Dict[str, float] = _build_output_sizes_mb()

# ---------------------------
# Candidate stitches
# ---------------------------

MODULE_LIST_SWIN: List[str] = [
    "swin_patch",
    "swin_stage1",
    "swin_stage2",
    "swin_stage3",
    "swin_stage4",
    "head",
]

PREFIX_C1: List[str] = ["resnet_stem"]
PREFIX_C2: List[str] = ["resnet_stem", "resnet_layer1"]
PREFIX_C3: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2"]
PREFIX_C4: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2", "resnet_layer3"]
PREFIX_C5: List[str] = ["resnet_stem", "resnet_layer1", "resnet_layer2", "resnet_layer3", "resnet_layer4"]

def _dataset_acc_table() -> Dict[int, float]:
    # Fill from your comments. Keep keys as stitch_id.
    if DATASET_NAME == "wildfire":
        return {
            1: 91.62,
            2: 86.00,
            3: 84.33,
            4: 87.00,
            5: 84.38,
            6: 83.94,
            7: 84.60,
        }

    if DATASET_NAME == "floodnet":
        return {
            1: 79.33,
            2: 85.11,
            3: 81.56,
            4: 86.00,
            5: 91.78,
            6: 91.11,
            7: 90.00,
        }

    # c10
    return {
        1: 90.47,
        2: 89.99,
        3: 89.01,
        4: 88.07,
        5: 86.76,
        6: 84.74,
        7: 82.24,
    }

def _build_candidate_stitches() -> Dict[int, Dict]:
    acc = _dataset_acc_table()
    print(f"Dataset {DATASET_NAME} has {len(acc)} entries.")

    # modules are dataset-invariant; only accuracy changes
    return {
        1: {"acc": float(acc[1]), "modules": MODULE_LIST_SWIN},

        2: {"acc": float(acc[2]), "modules": PREFIX_C1 + ["swin_stage1", "swin_stage2", "swin_stage3", "swin_stage4", "head"]},

        3: {"acc": float(acc[3]), "modules": PREFIX_C2 + ["swin_stage2", "swin_stage3", "swin_stage4", "head"]},

        4: {"acc": float(acc[4]), "modules": PREFIX_C3 + ["swin_stage3", "swin_stage4", "head"]},

        5: {"acc": float(acc[5]), "modules": PREFIX_C4 + ["swin_stage4", "head"]},

        6: {"acc": float(acc[6]), "modules": PREFIX_C5 + ["head"]},

        7: {"acc": float(acc[7]), "modules": PREFIX_C4 + ["swin_stage3", "head"]},
    }

CANDIDATE_STITCHES: Dict[int, Dict] = _build_candidate_stitches()

# ---------------------------
# Per-task execution profiles
# ---------------------------

def _build_task_profiles() -> Dict:
    # If task profiles differ by dataset, branch here. Otherwise keep shared.
    return {
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

TASK_PROFILES = _build_task_profiles()