#!/usr/bin/env python3
"""
Synthetic Anonymized Toy Dataset Generator for GFM Forest Carbon Pipeline.

Generates 128x128 multi-modal patches reflecting realistic Earth Observation
sensors (41 total channels):
  - Optical (6 channels): Landsat-8 RGB + Sentinel-2 RGB
  - SAR (6 channels): Sentinel-1 C-band (VH, VH/VV, VV, VV/VH) & ALOS-2 L-band (HH, HV)
  - GLCM Texture (8 channels): Haralick texture features
  - Spectral Indices (20 channels): NDVI, EVI, MNDWI, NBR, NDMI, NDWI, BSI, AWEI
  - DEM Topography (1 channel): SRTM Digital Elevation Model (meters)
Ground truth targets (3 channels):
  - Channel 0: Aboveground Biomass (AGB, Mg/ha)
  - Channel 1: Growing Stock Volume (GSV, m³/ha)
  - Channel 2: Forest Carbon Stock (Mg C/ha = 0.47 * AGB, IPCC Guidelines)
Observation mask:
  - Binary mask indicating sparse ground inventory sample plots.

Usage:
    python tools/generate_toy_dataset.py --output-dir data/toy_dataset
"""

import os
import argparse
import numpy as np

def generate_synthetic_patch(patch_id: int, seed: int = 42, H: int = 128, W: int = 128):
    """
    Generate a single physically-plausible 41-channel multi-modal patch.
    """
    rng = np.random.RandomState(seed + patch_id * 100)

    # 1. Base terrain elevation (DEM): 200m to 1200m with realistic topographic slope
    x_coords = np.linspace(0, 1, W)
    y_coords = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x_coords, y_coords)
    dem_base = (
        500.0
        + 300.0 * np.sin(2.0 * np.pi * xx)
        + 200.0 * np.cos(2.0 * np.pi * yy)
        + rng.normal(0, 15.0, (H, W))
    ).astype(np.float32)
    dem = np.clip(dem_base, 100.0, 1500.0)[np.newaxis, :, :]  # (1, 128, 128)

    # 2. Forest canopy density proxy (0.05 to 0.98) correlated with elevation & slope
    canopy_proxy = np.clip(
        0.72
        - 0.28 * (dem[0] - 500.0) / 700.0
        + rng.normal(0, 0.08, (H, W)),
        0.05, 0.98
    ).astype(np.float32)

    # 3. Optical Reflectance (6 bands: [LS_Red, LS_Green, LS_Blue, S2_Red, S2_Green, S2_Blue])
    s2_red = np.clip(0.04 + 0.12 * (1.0 - canopy_proxy) + rng.normal(0, 0.01, (H, W)), 0.01, 0.40)
    s2_green = np.clip(0.06 + 0.08 * canopy_proxy + rng.normal(0, 0.01, (H, W)), 0.02, 0.35)
    s2_blue = np.clip(0.03 + 0.10 * (1.0 - canopy_proxy) + rng.normal(0, 0.01, (H, W)), 0.01, 0.35)

    # Landsat-8 cross-sensor calibration variation
    ls_red = np.clip(s2_red + rng.normal(0.005, 0.008, (H, W)), 0.01, 0.45)
    ls_green = np.clip(s2_green + rng.normal(0.003, 0.008, (H, W)), 0.02, 0.40)
    ls_blue = np.clip(s2_blue + rng.normal(0.004, 0.008, (H, W)), 0.01, 0.40)

    optical = np.stack([ls_red, ls_green, ls_blue, s2_red, s2_green, s2_blue], axis=0).astype(np.float32)

    # 4. Proxy NIR and SWIR channels for synthetic spectral index equations
    nir_est = np.clip(0.20 + 0.35 * canopy_proxy + rng.normal(0, 0.02, (H, W)), 0.05, 0.75)
    swir1_est = np.clip(0.12 + 0.15 * (1.0 - canopy_proxy) + rng.normal(0, 0.015, (H, W)), 0.02, 0.50)
    swir2_est = np.clip(0.06 + 0.10 * (1.0 - canopy_proxy) + rng.normal(0, 0.01, (H, W)), 0.01, 0.35)

    # 5. SAR Backscatter (6 bands: [S1_VH, S1_VH_VV, S1_VV, S1_VV_VH, ALOS2_HH, ALOS2_HV])
    s1_vh = np.clip(-18.0 + 8.0 * canopy_proxy + rng.normal(0, 0.8, (H, W)), -25.0, -8.0)
    s1_vv = np.clip(-12.0 + 5.0 * canopy_proxy + rng.normal(0, 0.8, (H, W)), -18.0, -4.0)
    s1_vh_vv = s1_vh - s1_vv
    s1_vv_vh = s1_vv - s1_vh

    alos2_hh = np.clip(-14.0 + 6.0 * canopy_proxy + rng.normal(0, 1.0, (H, W)), -22.0, -6.0)
    alos2_hv = np.clip(-20.0 + 8.5 * canopy_proxy + rng.normal(0, 1.0, (H, W)), -28.0, -10.0)

    sar = np.stack([s1_vh, s1_vh_vv, s1_vv, s1_vv_vh, alos2_hh, alos2_hv], axis=0).astype(np.float32)

    # 6. GLCM Texture features (8 bands)
    glcm_mean = np.clip(canopy_proxy * 180.0 + rng.normal(0, 5.0, (H, W)), 10.0, 255.0)
    glcm_variance = np.clip(canopy_proxy * 40.0 + rng.normal(0, 3.0, (H, W)), 1.0, 100.0)
    glcm_homogeneity = np.clip(0.6 - 0.3 * canopy_proxy + rng.normal(0, 0.03, (H, W)), 0.1, 0.9)
    glcm_contrast = np.clip(20.0 + 35.0 * canopy_proxy + rng.normal(0, 4.0, (H, W)), 2.0, 120.0)
    glcm_dissimilarity = np.clip(3.0 + 5.0 * canopy_proxy + rng.normal(0, 0.5, (H, W)), 0.5, 15.0)
    glcm_entropy = np.clip(1.5 + 1.2 * canopy_proxy + rng.normal(0, 0.1, (H, W)), 0.5, 3.5)
    glcm_second_moment = np.clip(0.05 + 0.10 * (1.0 - canopy_proxy) + rng.normal(0, 0.01, (H, W)), 0.01, 0.30)
    glcm_correlation = np.clip(0.4 + 0.4 * canopy_proxy + rng.normal(0, 0.05, (H, W)), 0.0, 0.95)

    glcm = np.stack([
        glcm_mean, glcm_variance, glcm_homogeneity, glcm_contrast,
        glcm_dissimilarity, glcm_entropy, glcm_second_moment, glcm_correlation
    ], axis=0).astype(np.float32)

    # 7. Spectral Indices (20 bands, sorted alphabetically according to pipeline schema)
    eps = 1e-6
    ndvi_s2 = (nir_est - s2_red) / (nir_est + s2_red + eps)
    ndvi_ls = (nir_est - ls_red) / (nir_est + ls_red + eps)
    evi_s2 = 2.5 * (nir_est - s2_red) / (nir_est + 6.0 * s2_red - 7.5 * s2_blue + 1.0 + eps)
    evi_ls = 2.5 * (nir_est - ls_red) / (nir_est + 6.0 * ls_red - 7.5 * ls_blue + 1.0 + eps)
    mndwi_s2 = (s2_green - swir1_est) / (s2_green + swir1_est + eps)
    mndwi_ls = (ls_green - swir1_est) / (ls_green + swir1_est + eps)
    nbr_s2 = (nir_est - swir2_est) / (nir_est + swir2_est + eps)
    nbr_ls = (nir_est - swir2_est) / (nir_est + swir2_est + eps)
    ndmi_s2 = (nir_est - swir1_est) / (nir_est + swir1_est + eps)
    ndmi_ls = (nir_est - swir1_est) / (nir_est + swir1_est + eps)
    ndwi_s2 = (s2_green - nir_est) / (s2_green + nir_est + eps)
    ndwi_ls = (ls_green - nir_est) / (ls_green + nir_est + eps)
    bsi_s2 = ((swir1_est + s2_red) - (nir_est + s2_blue)) / ((swir1_est + s2_red) + (nir_est + s2_blue) + eps)
    bsi_ls = ((swir1_est + ls_red) - (nir_est + ls_blue)) / ((swir1_est + ls_red) + (nir_est + ls_blue) + eps)
    aweinsh_s2 = 4.0 * (s2_green - swir1_est) - (0.25 * nir_est + 2.75 * swir2_est)
    aweinsh_ls = 4.0 * (ls_green - swir1_est) - (0.25 * nir_est + 2.75 * swir2_est)
    aweish_s2 = s2_blue + 2.5 * s2_green - 1.5 * (nir_est + swir1_est) - 0.25 * swir2_est
    aweish_ls = ls_blue + 2.5 * ls_green - 1.5 * (nir_est + swir1_est) - 0.25 * swir2_est
    ndwf_s2 = ndwi_s2 * 0.9
    ndwf_ls = ndwi_ls * 0.9

    indices = np.stack([
        aweinsh_ls, aweinsh_s2, aweish_ls, aweish_s2,
        bsi_ls, bsi_s2, evi_ls, evi_s2,
        mndwi_ls, mndwi_s2, nbr_ls, nbr_s2,
        ndmi_ls, ndmi_s2, ndvi_ls, ndvi_s2,
        ndwf_ls, ndwf_s2, ndwi_ls, ndwi_s2
    ], axis=0).astype(np.float32)

    # 8. Forest Inventory Targets (AGB, Growing Stock Volume, Carbon Stock)
    # AGB (Mg/ha): typical range 50 - 320 Mg/ha
    agb = np.clip(
        30.0 + 260.0 * canopy_proxy + 0.15 * (s1_vh + 20.0) * 10.0 + rng.normal(0, 12.0, (H, W)),
        10.0, 380.0
    ).astype(np.float32)

    # GSV / Volume (m3/ha): allometrically proportional to AGB (expansion factor ~ 1.22)
    volume = np.clip(
        agb * 1.22 + rng.normal(0, 8.0, (H, W)),
        12.0, 480.0
    ).astype(np.float32)

    # Forest Carbon Stock (Mg C/ha): IPCC carbon fraction = 0.47 * AGB
    carbon = (agb * 0.47).astype(np.float32)

    target = np.stack([agb, volume, carbon], axis=0).astype(np.float32) # (3, 128, 128)

    # 9. Ground Truth Mask: Sparse inventory sample plots (e.g. 2 to 5 circular plot clusters)
    mask = np.zeros((H, W), dtype=np.float32)
    num_plots = rng.randint(2, 6)
    for _ in range(num_plots):
        py, px = rng.randint(15, H - 15), rng.randint(15, W - 15)
        radius = rng.randint(2, 4)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy * dy + dx * dx <= radius * radius:
                    if 0 <= py + dy < H and 0 <= px + dx < W:
                        mask[py + dy, px + dx] = 1.0

    return {
        "optical": optical,
        "sar": sar,
        "glcm": glcm,
        "indices": indices,
        "dem": dem,
        "target": target,
        "mask": mask
    }

def generate_split_dataset(output_dir: str, num_train: int = 12, num_val: int = 4, num_test: int = 4, seed: int = 42):
    """Generate train, val, and test splits with synthetic NPZ patches."""
    splits = {
        "train": (num_train, 0),
        "val": (num_val, 100),
        "test": (num_test, 200)
    }

    total_created = 0
    for split_name, (count, offset) in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"[i] Generating {count} patches for split '{split_name}' in {split_dir}...")

        for i in range(count):
            patch_data = generate_synthetic_patch(patch_id=offset + i, seed=seed)
            filename = f"patch_{split_name}_{i+1:03d}.npz"
            filepath = os.path.join(split_dir, filename)
            np.savez_compressed(
                filepath,
                optical=patch_data["optical"],
                sar=patch_data["sar"],
                glcm=patch_data["glcm"],
                indices=patch_data["indices"],
                dem=patch_data["dem"],
                target=patch_data["target"],
                mask=patch_data["mask"]
            )
            total_created += 1

    print(f"[+] Successfully generated {total_created} synthetic patches across all splits at: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic toy dataset for GFM Forest Carbon pipeline.")
    parser.add_argument("--output-dir", type=str, default="data/toy_dataset", help="Output directory path")
    parser.add_argument("--train-count", type=int, default=12, help="Number of training patches")
    parser.add_argument("--val-count", type=int, default=4, help="Number of validation patches")
    parser.add_argument("--test-count", type=int, default=4, help="Number of test patches")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation")
    args = parser.parse_args()

    generate_split_dataset(
        output_dir=args.output_dir,
        num_train=args.train_count,
        num_val=args.val_count,
        num_test=args.test_count,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
