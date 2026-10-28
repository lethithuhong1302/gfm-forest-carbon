#!/usr/bin/env python3
"""
Step 00: Dataset Audit and Train-Fitted Normalization Calculation.

Audits patch availability, channel dimensions, and computes per-channel
mean and standard deviation strictly on the training split to guarantee
zero test leakage.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, EXPERIMENT_OUTPUT_DIR, SCENARIOS
from src.dataset import get_patch_files, compute_train_norm_stats

def audit_dataset(data_dir=None):
    data_dir = data_dir or DATA_DIR
    print("=" * 80)
    print(f"AUDITING DATASET AT: {data_dir}")
    print("=" * 80)

    splits = ["train", "val", "test"]
    split_files = {}
    for s in splits:
        files = get_patch_files(s, data_dir)
        split_files[s] = files
        print(f"  - Split '{s}': {len(files)} patches found.")

    if not split_files["train"]:
        raise FileNotFoundError(f"No training patches found in {data_dir}/train.")

    # Inspect first patch
    first_patch = split_files["train"][0]
    with np.load(first_patch) as d:
        print("\nPatch Key Schema:")
        for k in d.files:
            print(f"  - {k:10s}: shape {d[k].shape}, dtype {d[k].dtype}")

    # Compute train-fitted normalization statistics
    print("\nComputing train-fitted per-channel normalization statistics...")
    stats = compute_train_norm_stats(data_dir, scenarios=list(SCENARIOS.keys()))
    serializable_stats = {
        scen: {k: v.tolist() for k, v in stat.items()}
        for scen, stat in stats.items()
    }

    out_file = os.path.join(EXPERIMENT_OUTPUT_DIR, "train_norm_stats.json")
    with open(out_file, "w") as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"[+] Normalization statistics cached successfully to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    audit_dataset()
