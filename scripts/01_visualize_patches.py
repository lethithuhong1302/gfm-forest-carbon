#!/usr/bin/env python3
"""
Step 01: Multi-Modal Patch Visualization.

Generates multi-sensor preview figures (Optical RGB, SAR VH/VV, GLCM, Indices, DEM,
and Sparse Target Plot Footprints) for verification.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA_DIR, FIGURES_DIR
from src.dataset import get_patch_files

def visualize_sample_patch(patch_path=None, out_dir=None):
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    if patch_path is None:
        train_files = get_patch_files("train", DATA_DIR)
        if not train_files:
            print("[!] No patches found to visualize.")
            return
        patch_path = train_files[0]

    with np.load(patch_path) as d:
        opt = d["optical"]
        sar = d["sar"]
        glcm = d["glcm"]
        ind = d["indices"]
        dem = d["dem"][0] if "dem" in d else np.zeros((128, 128))
        target = d["target"]
        mask = d["mask"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 1. Optical RGB (Sentinel-2 bands 3, 4, 5 -> Red, Green, Blue)
    rgb = np.stack([opt[3], opt[4], opt[5]], axis=-1)
    rgb = np.clip(rgb / np.percentile(rgb, 98), 0.0, 1.0)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Optical RGB (Sentinel-2)")

    # 2. SAR C-band False Color (VH, VV, VH/VV)
    sar_rgb = np.stack([sar[0], sar[2], sar[1]], axis=-1)
    sar_norm = (sar_rgb - np.min(sar_rgb)) / (np.max(sar_rgb) - np.min(sar_rgb) + 1e-6)
    axes[0, 1].imshow(sar_norm)
    axes[0, 1].set_title("SAR C-band (VH, VV, VH/VV)")

    # 3. GLCM Texture Contrast
    axes[0, 2].imshow(glcm[3], cmap="magma")
    axes[0, 2].set_title("GLCM Texture (Contrast)")

    # 4. NDVI Vegetation Index
    axes[0, 3].imshow(ind[15], cmap="RdYlGn", vmin=-0.2, vmax=0.9)
    axes[0, 3].set_title("NDVI Spectral Index")

    # 5. DEM Topography Elevation
    axes[1, 0].imshow(dem, cmap="terrain")
    axes[1, 0].set_title("SRTM DEM Elevation (m)")

    # 6. Aboveground Biomass Ground Truth (AGB)
    axes[1, 1].imshow(target[0], cmap="viridis")
    axes[1, 1].set_title("Aboveground Biomass (AGB)")

    # 7. Sparse Sample Plot Footprint Mask
    axes[1, 2].imshow(mask, cmap="gray")
    axes[1, 2].set_title("Sparse Plot Mask (1.0 = Observed)")

    # 8. Derived Carbon Stock (0.47 * AGB)
    axes[1, 3].imshow(target[2], cmap="YlGn")
    axes[1, 3].set_title("Derived Carbon Stock (Mg C/ha)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "multimodal_patch_preview.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"[+] Saved multimodal preview figure to: {out_file}")

if __name__ == "__main__":
    visualize_sample_patch()
