#!/usr/bin/env python3
"""
Step 05b: Prithvi Optical Naive Bilinear Baseline.

Ablation baseline pairing a frozen Prithvi ViT backbone with a direct
bilinear upsampling head (no progressive transposed-conv decoder) to measure
the architectural benefit of the progressive regression decoder.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CHECKPOINT_DIR, RESULTS_DIR, DEFAULT_SEED, USE_TTA, set_seed, get_device
from src.dataset import MultimodalPatchDataset, get_patch_dir
from src.models import PrithviBackboneWrapper
from src.models.decoder import NaiveBilinearDecoder
from src.losses import MaskedHuberLoss
from src.metrics import evaluate_multi_target
from src.experiment_protocol import (
    target_scales, make_protocol, collect_predictions,
    validation_metrics, save_checkpoint, load_checkpoint,
    export_predictions
)

class PrithviNaiveBaselineModel(nn.Module):
    def __init__(self, backbone, out_channels=2):
        super().__init__()
        self.backbone = backbone
        self.decoder = NaiveBilinearDecoder(in_dim=768, out_channels=out_channels)

    def forward(self, x):
        _, feats = self.backbone(x[:, :6])
        return self.decoder(feats)

def run_naive_baseline(epochs=15, batch_size=4, lr=1e-3):
    set_seed(DEFAULT_SEED)
    device = get_device()
    print("=" * 80)
    print("RUNNING PRITHVI NAIVE BILINEAR BASELINE")
    print("=" * 80)

    datasets = {s: MultimodalPatchDataset(split=s, scenario="S1_Optical") for s in ("train", "val", "test")}
    loaders = {s: DataLoader(ds, batch_size=batch_size, shuffle=(s == "train")) for s, ds in datasets.items()}
    scales = target_scales(datasets["train"].files, target_indices=[0, 1])

    backbone = PrithviBackboneWrapper(depth=12, lora_r=0).to(device)
    backbone.freeze_backbone()

    model = PrithviNaiveBaselineModel(backbone, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MaskedHuberLoss(delta=1.0)

    best_score = float("inf")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "prithvi_naive_baseline_best.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, mask, _ in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y[:, :2], mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)

        truth_val, pred_val, _ = collect_predictions(model, loaders["val"], device)
        m = validation_metrics(truth_val, pred_val, scales, ["AGB", "GSV"])
        if m["val_score"] < best_score:
            best_score = m["val_score"]
            torch.save({"model_state_dict": model.state_dict(), "metrics": m}, ckpt_path)
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss={total_loss/max(count, 1):.4f} | Val nRMSE={m['val_score']:.6f}")

    # Evaluate test
    best_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_data["model_state_dict"])
    truth_t, pred_t, names_t = collect_predictions(model, loaders["test"], device)
    carbon_t = truth_t[:, 0:1] * 0.47
    carbon_p = pred_t[:, 0:1] * 0.47
    df = evaluate_multi_target(np.hstack([truth_t, carbon_t]), np.hstack([pred_t, carbon_p]), compute_ci=True)
    df["Model"] = "Prithvi_Naive_Bilinear"
    out_csv = os.path.join(RESULTS_DIR, "prithvi_naive_baseline_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[+] Saved naive baseline results to: {out_csv}")
    print(df.to_string())

if __name__ == "__main__":
    run_naive_baseline()
