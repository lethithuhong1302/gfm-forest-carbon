#!/usr/bin/env python3
"""
Step 07: Sensor Modality Ablation Study across Scenarios S0 to S7.

Evaluates progressive sensor contribution:
  - S0: Constant Predictor
  - S1: Optical only (6 bands) or Optical + DEM (7 bands)
  - S2: Optical + Sentinel-1 C-band (10-11 bands)
  - S3: Optical + ALOS-2 L-band (8-9 bands)
  - S4: Optical + Full SAR (12-13 bands)
  - S5: Optical + SAR + GLCM Texture (20-21 bands)
  - S6: Spectral Indices only (20 bands)
  - S7: Full Multimodal Fusion (41 bands: Optical + SAR + GLCM + Indices + DEM)
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RESULTS_DIR, SCENARIOS
from scripts.06_train_multimodal_adapter import run_multimodal_adapter

def run_ablation_suite(epochs=10, batch_size=4, scenarios=None):
    print("=" * 80)
    print("RUNNING MULTIMODAL SENSOR ABLATION SUITE")
    print("=" * 80)

    target_scenarios = scenarios or [
        "S1_Optical",
        "S1_Opt_DEM",
        "S2_Opt_S1_DEM",
        "S4_Opt_SAR_DEM",
        "S5_Opt_SAR_GLCM_DEM",
        "S6_Indices",
        "S7_Full_Fusion"
    ]

    all_dfs = []
    for scen in target_scenarios:
        if scen not in SCENARIOS:
            continue
        print(f"\n---> Running Ablation for: {scen} ({SCENARIOS[scen]['desc']})")
        _, df = run_multimodal_adapter(scenario=scen, epochs=epochs, batch_size=batch_size)
        df["Scenario_Code"] = scen
        all_dfs.append(df)

    summary_df = pd.concat(all_dfs, ignore_index=True)
    out_csv = os.path.join(RESULTS_DIR, "sensor_ablation_summary.csv")
    summary_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("COMPLETE ABLATION SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df[["Scenario_Code", "Target", "R2", "RMSE", "nRMSE_%"]].to_string())
    print(f"\n[+] Saved full ablation summary to: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_ablation_suite(epochs=args.epochs, batch_size=args.batch_size)
