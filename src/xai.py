"""
Explainable AI (XAI) and Uncertainty Quantification.

Features:
  - Group Permutation Feature Importance across remote sensing modalities
  - Stratified Residual Analysis across Forest Carbon and Biomass tiers
"""

import numpy as np
import torch
from .config import IPCC_CARBON_FACTOR

def compute_group_permutation_importance(model, dataset, device, groups=None,
                                         target_idx=2, n_repeats=5, seed=42, use_tta=True):
    """
    Computes permutation sensitivity across feature groups (Optical, SAR, GLCM, Indices, DEM).
    """
    from torch.utils.data import DataLoader, TensorDataset
    from .experiment_protocol import collect_predictions

    if groups is None:
        groups = {
            "Optical": list(range(0, 6)),
            "Sentinel-1": list(range(6, 10)),
            "ALOS-2": [10, 11],
            "GLCM Texture": list(range(12, 20)),
            "Indices": list(range(20, 40)),
            "DEM Topography": [40]
        }

    samples = list(dataset)
    if not samples:
        raise ValueError('Dataset is empty.')

    x = torch.stack([item[0] for item in samples])
    y = torch.stack([item[1] for item in samples])
    mask = torch.stack([item[2] for item in samples])
    ids = torch.arange(len(samples))

    def score(inputs):
        loader = DataLoader(TensorDataset(inputs, y, mask, ids), batch_size=8)
        truth, prediction, _ = collect_predictions(model, loader, device, use_tta)
        actual = truth[:, target_idx] if truth.shape[1] > target_idx else truth[:, 0] * IPCC_CARBON_FACTOR
        estimate = prediction[:, 0] * IPCC_CARBON_FACTOR if target_idx == 2 else prediction[:, target_idx]
        return float(np.sqrt(np.mean((actual - estimate) ** 2)))

    base_rmse = score(x)
    rng = np.random.default_rng(seed)
    importance = {}

    for name, channels in groups.items():
        channels = [i for i in channels if i < x.shape[1]]
        if not channels:
            continue
        changes = []
        for _ in range(n_repeats):
            permuted = x.clone()
            order = torch.as_tensor(rng.permutation(len(x)))
            permuted[:, channels] = x[order][:, channels]
            changes.append(score(permuted) - base_rmse)
        importance[name] = {
            'mean_rmse_increase': float(np.mean(changes)),
            'std': float(np.std(changes))
        }

    return importance, base_rmse

def compute_residual_analysis(y_true, y_pred):
    """
    Computes residual errors (Observed - Predicted) stratified by biomass tiers:
      - Low (<50 Mg/ha)
      - Medium (50-150 Mg/ha)
      - High (>150 Mg/ha)
    """
    residuals = y_true - y_pred
    tiers = {
        "Low (<50)": np.where(y_true < 50)[0],
        "Medium (50-150)": np.where((y_true >= 50) & (y_true <= 150))[0],
        "High (>150)": np.where(y_true > 150)[0]
    }
    tier_stats = {}
    for tier_name, idxs in tiers.items():
        if len(idxs) > 0:
            tier_stats[tier_name] = {
                "count": int(len(idxs)),
                "mean_bias": float(np.mean(residuals[idxs])),
                "rmse": float(np.sqrt(np.mean(residuals[idxs] ** 2)))
            }
        else:
            tier_stats[tier_name] = {"count": 0, "mean_bias": 0.0, "rmse": 0.0}
    return residuals, tier_stats
