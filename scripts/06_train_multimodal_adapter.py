#!/usr/bin/env python3
"""
Step 06: Two-Stage Multimodal Adapter Training with Optical Warm-Start.

Pipeline:
  1. Loads optical foundation model checkpoint (trained via LoRA or partial blocks in Step 05).
  2. Freezes optical backbone ViT weights completely.
  3. Stage 1 (Fusion-only epochs): Trains auxiliary CNN encoder (SAR, GLCM, Indices, DEM)
     and Gated Cross-Modal Fusion module while holding decoder frozen.
  4. Stage 2 (Joint adaptation): Fine-tunes adapter and decoder jointly with conservative learning rates.
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
    CHECKPOINT_DIR, RESULTS_DIR, DEFAULT_SEED, USE_TTA, SCENARIOS, set_seed, get_device
)
from src.dataset import MultimodalPatchDataset, get_patch_dir, compute_train_norm_stats
from src.models import PrithviBackboneWrapper, PrithviMultimodalAdapterModel
from src.losses import MaskedHuberLoss
from src.metrics import evaluate_multi_target
from src.experiment_protocol import (
    target_scales, make_protocol, collect_predictions,
    validation_metrics, save_checkpoint, load_checkpoint,
    export_predictions
)

def run_multimodal_adapter(scenario="S7_Full_Fusion", epochs=25, fusion_only_epochs=5,
                           batch_size=4, lr=4e-4, lr_decoder=2e-5, optical_ckpt=None,
                           patch_dir=None, seed=DEFAULT_SEED, use_tta=USE_TTA):
    set_seed(seed)
    device = get_device()
    patch_dir = get_patch_dir(patch_dir)

    print("=" * 80)
    print(f"MULTIMODAL ADAPTER TRAINING | SCENARIO: {scenario}")
    print("=" * 80)

    # Locate optical checkpoint
    if optical_ckpt is None:
        candidates = [
            os.path.join(CHECKPOINT_DIR, "prithvi_optical_lora_best.pth"),
            os.path.join(CHECKPOINT_DIR, "prithvi_optical_last_blocks_best.pth"),
            os.path.join(CHECKPOINT_DIR, "prithvi_optical_frozen_best.pth"),
            os.path.join(CHECKPOINT_DIR, "prithvi_optical_best.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                optical_ckpt = c
                break

    norm_stats = compute_train_norm_stats(patch_dir, scenarios=[scenario])
    datasets = {
        s: MultimodalPatchDataset(split=s, patch_dir=patch_dir, scenario=scenario, norm_stats=norm_stats)
        for s in ("train", "val", "test")
    }
    loaders = {
        s: DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"))
        for s, ds in datasets.items()
    }
    scales = target_scales(datasets["train"].files, target_indices=[0, 1])

    # Backbone is completely frozen for multimodal adapter training
    backbone = PrithviBackboneWrapper(depth=12, lora_r=0).to(device)
    backbone.freeze_backbone()

    in_channels = SCENARIOS[scenario]["channels"]
    model = PrithviMultimodalAdapterModel(
        backbone=backbone, total_channels=in_channels, optical_channels=6,
        embed_dim=768, out_channels=2, use_adapter=True,
        target_names=["AGB", "GSV"]
    ).to(device)

    # Load optical warm-start weights if available
    if optical_ckpt and os.path.isfile(optical_ckpt):
        model.load_optical_checkpoint(optical_ckpt, device=device)
    else:
        print("[!] Note: Optical warm-start checkpoint not found. Initializing adapter from scratch.")

    criterion = MaskedHuberLoss(delta=1.0)
    ckpt_name = f"prithvi_adapter_{scenario}_best.pth"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    results_csv = os.path.join(RESULTS_DIR, f"prithvi_adapter_{scenario}_results.csv")

    protocol = make_protocol(
        scenario, norm_stats, scales, use_tta, datasets,
        optical_warm_start=str(optical_ckpt)
    )

    # Epoch 0 Baseline
    truth_val, pred_val, _ = collect_predictions(model, loaders["val"], device, use_tta, [0, 1])
    init_m = validation_metrics(truth_val, pred_val, scales, ["AGB", "GSV"])
    best_score = init_m["val_score"]
    save_checkpoint(ckpt_path, model, protocol, 0, init_m, phase="epoch0_initialization")
    print(f"[*] Epoch 0 Initial val_nRMSE = {best_score:.6f}")

    # Two-Stage Optimizer
    def build_optimizer(stage):
        if stage == 1:
            for p in model.decoder.parameters(): p.requires_grad = False
            return torch.optim.AdamW(
                list(model.aux_encoder.parameters()) + list(model.fusion.parameters()),
                lr=lr, weight_decay=1e-4
            )
        else:
            for p in model.decoder.parameters(): p.requires_grad = True
            return torch.optim.AdamW([
                {'params': list(model.aux_encoder.parameters()) + list(model.fusion.parameters()), 'lr': lr * 0.5},
                {'params': model.decoder.parameters(), 'lr': lr_decoder}
            ], weight_decay=1e-4)

    optimizer = build_optimizer(stage=1)

    for epoch in range(1, epochs + 1):
        if epoch == fusion_only_epochs + 1:
            print("[+] Transitioning to Stage 2: Joint Adapter & Decoder Fine-Tuning.")
            optimizer = build_optimizer(stage=2)

        model.train()
        total_loss, count = 0.0, 0
        for x, y, mask, _ in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y[:, :2], mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)

        truth_val, pred_val, _ = collect_predictions(model, loaders["val"], device, use_tta, [0, 1])
        m = validation_metrics(truth_val, pred_val, scales, ["AGB", "GSV"])
        is_best = m["val_score"] < best_score
        if is_best:
            best_score = m["val_score"]
            save_checkpoint(ckpt_path, model, protocol, epoch, m, phase="best_validation")

        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss={total_loss/max(count, 1):.4f} | "
              f"Val nRMSE={m['val_score']:.6f} {'[BEST]' if is_best else ''}")

    # Evaluate best model on test split
    best_data = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(best_data["model_state_dict"])
    truth_t, pred_t, names_t = collect_predictions(model, loaders["test"], device, use_tta, [0, 1])
    export_predictions(results_csv, truth_t, pred_t, names_t, ["AGB", "GSV"])

    carbon_t = truth_t[:, 0:1] * 0.47
    carbon_p = pred_t[:, 0:1] * 0.47
    truth_full = np.hstack([truth_t, carbon_t])
    pred_full = np.hstack([pred_t, carbon_p])

    df = evaluate_multi_target(truth_full, pred_full, ["AGB", "GSV", "Carbon"], compute_ci=True)
    df["Scenario"] = scenario
    metrics_path = results_csv.replace(".csv", "_metrics.csv")
    df.to_csv(metrics_path, index=False)

    print("\nFINAL MULTIMODAL TEST METRICS:")
    print(df.to_string())
    print("=" * 80)
    return model, df

def main():
    parser = argparse.ArgumentParser(description="Prithvi Multimodal Adapter Training.")
    parser.add_argument("--scenario", type=str, default="S7_Full_Fusion")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fusion-only-epochs", type=int, default=3)
    parser.add_argument("--optical-ckpt", type=str, default=None)
    args = parser.parse_args()
    run_multimodal_adapter(
        scenario=args.scenario,
        epochs=args.epochs,
        fusion_only_epochs=args.fusion_only_epochs,
        optical_ckpt=args.optical_ckpt
    )

if __name__ == "__main__":
    main()
