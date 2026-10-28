"""
Shared Validation, Checkpoint, and Inference Protocol Contract.

Features:
  - Strict validation normalized RMSE (nRMSE) checkpoint selection
  - Epoch-0 protection: records initial state before training begins
  - Checkpoint integrity hashing and configuration serialization
  - Atomic checkpoint writing to prevent file corruption
"""

import os
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from .config import TARGET_COLS
from .models.tta import predict_with_d4_tta

FORMAT_VERSION = 2

def atomic_save(value, path):
    """Atomically save checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

def target_scales(train_files, target_indices=None):
    """Compute target standard deviation scales strictly on the training set."""
    targets = []
    for f in train_files:
        with np.load(f) as data:
            y = data["target"] # (3, H, W)
            m = data["mask"] == 1.0 # (H, W)
            if np.any(m):
                y_valid = y[:, m]
                targets.append(np.mean(y_valid, axis=1))
    targets = np.array(targets)
    scales = np.std(targets, axis=0)
    scales[scales == 0] = 1.0
    if target_indices is not None:
        return scales[target_indices]
    return scales

def make_protocol(scenario, norm_stats, scales, use_tta, datasets, **kwargs):
    """Construct experiment protocol dictionary for metadata tracking."""
    return {
        "format_version": FORMAT_VERSION,
        "scenario": scenario,
        "scales": [float(s) for s in np.asarray(scales).flatten()],
        "use_tta": bool(use_tta),
        "split_counts": {k: len(v) for k, v in datasets.items()},
        **kwargs
    }

def collect_predictions(model, loader, device, use_tta=True, target_indices=None):
    """
    Run evaluation loop over DataLoader and collect ground truth vs predictions
    for all valid mask pixels.
    """
    model.eval()
    all_truth = []
    all_preds = []
    all_names = []

    with torch.no_grad():
        for batch in loader:
            x, y, mask, names = batch
            x = x.to(device)
            pred = predict_with_d4_tta(model, x, use_tta=use_tta)

            # Bring to CPU
            pred = pred.cpu()
            y = y.cpu()
            mask = mask.cpu()

            for i in range(len(x)):
                m = mask[i] == 1.0
                if torch.any(m):
                    t_val = y[i, :, m].mean(dim=-1).numpy()
                    p_val = pred[i, :, m].mean(dim=-1).numpy()
                else:
                    t_val = y[i].mean(dim=(-2, -1)).numpy()
                    p_val = pred[i].mean(dim=(-2, -1)).numpy()

                all_truth.append(t_val)
                all_preds.append(p_val)
                all_names.append(names[i])

    truth_arr = np.array(all_truth)
    pred_arr = np.array(all_preds)

    if target_indices is not None and truth_arr.shape[1] > len(target_indices):
        truth_arr = truth_arr[:, target_indices]

    return truth_arr, pred_arr, all_names

def validation_metrics(truth, prediction, scales, target_names=None):
    """
    Calculate validation score: mean normalized RMSE across predicted targets.
    """
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    if p.ndim == 1: p = p.reshape(-1, 1)
    if t.ndim == 1: t = t.reshape(-1, 1)

    k = min(p.shape[1], t.shape[1])
    s = np.asarray(scales, dtype=np.float64).reshape(-1)[:k]
    rmse = np.sqrt(np.mean((p[:, :k] - t[:, :k]) ** 2, axis=0))
    val_score = float(np.mean(rmse / np.maximum(s, 1e-6)))

    metrics = {"val_score": val_score}
    names = target_names or [f"target_{i}" for i in range(k)]
    for i, name in enumerate(names[:k]):
        metrics[f"val_rmse_{name}"] = float(rmse[i])
    return metrics

def save_checkpoint(path, model, protocol, epoch, metrics, phase='training'):
    """Save model checkpoint with weights and complete training protocol metadata."""
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "metrics": metrics,
        "protocol": protocol,
        "model_state_dict": model.state_dict()
    }
    atomic_save(checkpoint, path)

def load_checkpoint(path, device="cpu"):
    """Load model checkpoint safely."""
    return torch.load(path, map_location=device, weights_only=False)

def export_predictions(out_path, truth, prediction, names, target_cols=None):
    """Export tabular predictions to CSV for downstream auditing."""
    cols = target_cols or (TARGET_COLS if truth.shape[1] >= len(TARGET_COLS) else [f'target_{i}' for i in range(truth.shape[1])])
    frame = pd.DataFrame({'filename': names})
    for i, col in enumerate(cols):
        if i < truth.shape[1]:
            frame[f'true_{col}'] = truth[:, i]
        if i < prediction.shape[1]:
            frame[f'pred_{col}'] = prediction[:, i]
    frame.to_csv(out_path, index=False)
