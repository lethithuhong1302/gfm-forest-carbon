#!/usr/bin/env python3
"""
Step 03: Baseline Model Training and Evaluation.

Trains non-foundation model baselines:
  - S0: Constant Predictor (mean/median)
  - Pixel Tabular Machine Learning: Random Forest & XGBoost
  - Deep Learning: SimpleUNet (trained from scratch)
Evaluates test metrics (R², RMSE, MAE, nRMSE %) for independent ecological targets.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    CHECKPOINT_DIR, RESULTS_DIR, DEFAULT_SEED, set_seed, get_device, SCENARIOS
)
from src.dataset import MultimodalPatchDataset, get_patch_dir, compute_train_norm_stats
from src.models import ConstantBaseline, TabularMLBaseline, SimpleUNet
from src.losses import MaskedHuberLoss
from src.metrics import calculate_regression_metrics, evaluate_multi_target

def run_baselines(scenario="S7_Full_Fusion", patch_dir=None, epochs_unet=50, batch_size=4):
    set_seed(DEFAULT_SEED)
    device = get_device()
    patch_dir = get_patch_dir(patch_dir)

    print("=" * 80)
    print(f"RUNNING BASELINE BENCHMARKS ON SCENARIO: {scenario}")
    print("=" * 80)

    norm_stats = compute_train_norm_stats(patch_dir, scenarios=[scenario])
    datasets = {
        s: MultimodalPatchDataset(split=s, patch_dir=patch_dir, scenario=scenario, norm_stats=norm_stats)
        for s in ("train", "val", "test")
    }

    # Extract tabular pixel data from masked locations for RF/XGB
    def extract_tabular_data(ds):
        X_list, y_list = [], []
        for x, y, mask, _ in ds:
            valid = (mask == 1.0)
            if np.any(valid.numpy()):
                # x: (C, H, W) -> (N, C)
                # y: (3, H, W) -> (N, 2) [AGB, GSV]
                x_val = x[:, valid].T.numpy()
                y_val = y[:2, valid].T.numpy()
                X_list.append(x_val)
                y_list.append(y_val)
        return np.vstack(X_list), np.vstack(y_list)

    X_train, y_train = extract_tabular_data(datasets["train"])
    X_test, y_test = extract_tabular_data(datasets["test"])

    results = []

    # 1. Constant Baseline
    const_model = ConstantBaseline(strategy="mean").fit(y_train)
    y_pred_const = const_model.predict(X_test)
    carbon_const = y_pred_const[:, 0:1] * 0.47
    y_pred_const_full = np.hstack([y_pred_const, carbon_const])
    y_test_full = np.hstack([y_test, y_test[:, 0:1] * 0.47])
    df_const = evaluate_multi_target(y_test_full, y_pred_const_full)
    df_const["Model"] = "Constant_Mean"
    results.append(df_const)
    print("  [+] Constant Baseline evaluated.")

    # 2. Random Forest Baseline
    rf_model = TabularMLBaseline(model_type="rf", n_estimators=100, max_depth=8).fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_rf_full = np.hstack([y_pred_rf, y_pred_rf[:, 0:1] * 0.47])
    df_rf = evaluate_multi_target(y_test_full, y_pred_rf_full)
    df_rf["Model"] = "RandomForest"
    results.append(df_rf)
    print("  [+] Random Forest evaluated.")

    # 3. SimpleUNet
    in_channels = SCENARIOS[scenario]["channels"]
    unet = SimpleUNet(in_channels=in_channels, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = MaskedHuberLoss(delta=1.0)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)

    print(f"  [*] Training SimpleUNet for {epochs_unet} epochs...")
    unet.train()
    for ep in range(epochs_unet):
        for x_b, y_b, m_b, _ in train_loader:
            x_b, y_b, m_b = x_b.to(device), y_b.to(device), m_b.to(device)
            optimizer.zero_grad()
            loss = criterion(unet(x_b), y_b[:, :2], m_b)
            loss.backward()
            optimizer.step()

    unet.eval()
    unet_preds, unet_truth = [], []
    with torch.no_grad():
        for x_b, y_b, m_b, _ in DataLoader(datasets["test"], batch_size=1):
            x_b = x_b.to(device)
            pred = unet(x_b).cpu().numpy()[0] # (2, H, W)
            mask_np = (m_b[0].numpy() == 1.0)
            if np.any(mask_np):
                unet_preds.append(pred[:, mask_np].T)
                unet_truth.append(y_b[0, :2].numpy()[:, mask_np].T)

    if unet_preds:
        up = np.vstack(unet_preds)
        ut = np.vstack(unet_truth)
        up_full = np.hstack([up, up[:, 0:1] * 0.47])
        ut_full = np.hstack([ut, ut[:, 0:1] * 0.47])
        df_unet = evaluate_multi_target(ut_full, up_full)
        df_unet["Model"] = "SimpleUNet_Scratch"
        results.append(df_unet)
        print("  [+] SimpleUNet evaluated.")

    summary_df = pd.concat(results, ignore_index=True)
    out_csv = os.path.join(RESULTS_DIR, f"baselines_{scenario}_results.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[+] Baseline results saved to: {out_csv}")
    print(summary_df.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="S7_Full_Fusion")
    parser.add_argument("--epochs-unet", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run_baselines(scenario=args.scenario, epochs_unet=args.epochs_unet, batch_size=args.batch_size)
