#!/usr/bin/env python3
"""
Step 02: Spatial Grid Splitting.

Partitions study area observations into training, validation, and test splits
grouped by spatial 5km grid tiles to eliminate spatial autocorrelation leakage.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DEFAULT_SEED, EXPERIMENT_OUTPUT_DIR
from src.splits import make_grouped_spatial_split

def run_spatial_splits(seed=DEFAULT_SEED):
    print("=" * 80)
    print("RUNNING SPATIAL GROUPED SPLITS (LEAKAGE PREVENTION)")
    print("=" * 80)

    # Synthetic or loaded metadata demonstration
    rng = pd.Series(range(300))
    grid_ids = [f"GRID_{i // 10:02d}" for i in range(300)]
    df = pd.DataFrame({
        "plot_id": [f"PLOT_{i:03d}" for i in range(300)],
        "grid_id": grid_ids,
        "biomass": rng * 1.1 + 35.0
    })

    split_df, summary = make_grouped_spatial_split(df, grid_col="grid_id", seed=seed)
    print(json.dumps(summary, indent=2))

    out_csv = os.path.join(EXPERIMENT_OUTPUT_DIR, "spatial_splits_summary.json")
    with open(out_csv, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Spatial split summary saved to: {out_csv}")

if __name__ == "__main__":
    run_spatial_splits()
