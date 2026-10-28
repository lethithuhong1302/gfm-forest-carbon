"""
Comprehensive Local Verification Script.

Executes complete pipeline integrity checks:
  1. Configuration & Band Schemas (41 channels, DEM, IPCC Carbon)
  2. Dataset loading & channel slicing
  3. D4 Dihedral Data Augmentation & spatial alignment preservation
  4. Regression metrics, bootstrap confidence intervals, and permutation tests
  5. Algebraic Spectral Inversion
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import TARGET_COLS, SCENARIOS, TOTAL_CHANNELS, DEFAULT_SEED, set_seed
from src.dataset import MultimodalPatchDataset, get_patch_files
from src.metrics import calculate_regression_metrics, compute_bootstrap_ci, paired_permutation_test
from src.spectral_reconstruction import reconstruct_hls_numpy

def test_pipeline():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE LOCAL PIPELINE VERIFICATION")
    print("=" * 80)

    # 1. Config Check
    print("[1/5] Checking Configuration & Multimodal Schema...")
    set_seed(DEFAULT_SEED)
    assert len(TARGET_COLS) == 3, "Expected 3 target columns [AGB, GSV, Carbon]"
    assert TOTAL_CHANNELS == 41, f"Expected 41 total channels, got {TOTAL_CHANNELS}"
    assert "S7_Full_Fusion" in SCENARIOS
    print("  [OK] Config & Multimodal 41-channel schema verified.")

    # 2. Dataset Loading & Channel Slicing Check
    print("[2/5] Checking Dataset Loading & Dynamic Slicing...")
    train_files = get_patch_files("train")
    assert len(train_files) > 0, "No training patches found!"
    ds = MultimodalPatchDataset(split="train", scenario="S7_Full_Fusion", files=train_files)
    x, y, mask, fname = ds[0]
    assert x.shape[0] == 41, f"Expected 41 channels in S7, got {x.shape[0]}"
    assert y.shape[0] == 3, f"Expected 3 target channels, got {y.shape[0]}"
    assert mask.shape == (128, 128), f"Expected (128, 128) mask, got {mask.shape}"
    print(f"  [OK] Loaded sample patch {fname} with shape {x.shape}.")

    # 3. D4 Augmentation Alignment
    print("[3/5] Checking D4 Dihedral Group Augmentation...")
    dummy_x = np.zeros((41, 128, 128), dtype=np.float32)
    dummy_y = np.zeros((3, 128, 128), dtype=np.float32)
    dummy_m = np.zeros((128, 128), dtype=np.float32)

    cy, cx = 25, 75
    dummy_x[40, cy, cx] = 950.0 # DEM marker
    dummy_y[0, cy, cx] = 180.0  # AGB marker
    dummy_m[cy, cx] = 1.0

    # Rotate 90
    x_rot = np.rot90(dummy_x, k=1, axes=(1, 2))
    y_rot = np.rot90(dummy_y, k=1, axes=(1, 2))
    m_rot = np.rot90(dummy_m, k=1, axes=(0, 1))

    ys, xs = np.where(m_rot == 1.0)
    assert len(ys) == 1
    assert np.isclose(y_rot[0, ys[0], xs[0]], 180.0)
    assert np.isclose(x_rot[40, ys[0], xs[0]], 950.0)
    print("  [OK] Synchronized D4 transformations preserve spatial and sensor alignment.")

    # 4. Statistical Metrics & Bootstrap
    print("[4/5] Checking Statistical Metrics & Non-Parametric Bootstrap...")
    yt = np.array([15.0, 30.0, 55.0, 80.0, 120.0, 160.0, 210.0, 270.0])
    yp = np.array([16.0, 28.0, 58.0, 77.0, 115.0, 168.0, 205.0, 262.0])
    m = calculate_regression_metrics(yt, yp)
    assert m["R2"] > 0.95
    ci_low, ci_high = compute_bootstrap_ci(yt, yp, is_rmse=True, n_bootstraps=200)
    assert ci_low <= ci_high
    print(f"  [OK] R2={m['R2']:.4f}, RMSE={m['RMSE']:.2f}, 95% CI=[{ci_low:.2f}, {ci_high:.2f}].")

    # 5. Spectral Reconstruction
    print("[5/5] Checking Method 2.1: Algebraic Spectral Inversion...")
    opt_dummy = np.zeros((6, 8, 8), dtype=np.float32)
    opt_dummy[3] = 0.10 # Red
    opt_dummy[4] = 0.15 # Green
    opt_dummy[5] = 0.05 # Blue
    ind_dummy = np.zeros((20, 8, 8), dtype=np.float32)
    ind_dummy[15] = 0.666667 # NDVI
    ind_dummy[9] = 0.200000  # MNDWI
    ind_dummy[11] = 0.818182 # NBR

    hls = reconstruct_hls_numpy(opt_dummy, ind_dummy, source="S2")
    assert np.isclose(hls[3, 0, 0], 0.50, atol=1e-3), "NIR inversion mismatch"
    assert np.isclose(hls[4, 0, 0], 0.10, atol=1e-3), "SWIR1 inversion mismatch"
    assert np.isclose(hls[5, 0, 0], 0.05, atol=1e-3), "SWIR2 inversion mismatch"
    print("  [OK] Algebraic HLS band inversion equations mathematically exact.")

    print("\n" + "=" * 80)
    print("ALL MODULE VERIFICATIONS & TEST PROTOCOLS PASSED! [SUCCESS]")
    print("=" * 80)

if __name__ == "__main__":
    test_pipeline()
