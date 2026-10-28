# Dataset Documentation

This directory contains the data layout and multimodal feature specifications for the `gfm-forest-carbon` benchmark.

## 1. Anonymized Toy Dataset

To ensure immediate, out-of-the-box algorithmic reproducibility without requiring multi-gigabyte satellite imagery downloads, an anonymized toy dataset is bundled under `data/toy_dataset/`:

- `data/toy_dataset/train/`: 12 multi-modal training patches (`.npz`)
- `data/toy_dataset/val/`: 4 validation patches (`.npz`)
- `data/toy_dataset/test/`: 4 test patches (`.npz`)

Each `.npz` archive contains 128x128 spatial arrays across all 41 remote sensing channels and sparse in-situ inventory ground truth:

| Key | Shape | Data Type | Physical Modality / Description |
|---|---|---|---|
| `optical` | `(6, 128, 128)` | `float32` | 6 surface reflectance bands: Landsat-8 RGB + Sentinel-2 RGB |
| `sar` | `(6, 128, 128)` | `float32` | SAR backscatter: Sentinel-1 C-band (VH, VH/VV, VV, VV/VH) & ALOS-2 PALSAR-2 L-band (HH, HV) |
| `glcm` | `(8, 128, 128)` | `float32` | Grey-Level Co-occurrence Matrix texture features (Mean, Variance, Homogeneity, Contrast, Dissimilarity, Entropy, Second Moment, Correlation) |
| `indices` | `(20, 128, 128)` | `float32` | 20 spectral vegetation, moisture, and water indices (NDVI, EVI, MNDWI, NBR, NDMI, NDWI, BSI, AWEI) |
| `dem` | `(1, 128, 128)` | `float32` | Shuttle Radar Topography Mission (SRTM) digital elevation model (meters) |
| `target` | `(3, 128, 128)` | `float32` | Ground truth targets: Channel 0: Aboveground Biomass (AGB, Mg/ha), Channel 1: Growing Stock Volume (GSV, m³/ha), Channel 2: Carbon Stock (Mg C/ha = 0.47 * AGB) |
| `mask` | `(128, 128)` | `float32` | Binary observation mask (1.0 = measured in-situ inventory plot; 0.0 = unobserved background pixel) |

## 2. Regenerating or Scaling the Toy Dataset

You can deterministically regenerate or scale the toy dataset using the included generator:

```bash
python tools/generate_toy_dataset.py --output-dir data/toy_dataset --train-count 20 --val-count 5 --test-count 5 --seed 42
```

## 3. Integrating Real Earth Observation Data

To run experiments with your own satellite rasters and field plots:
1. Export 128x128 geographic patches matching the channel schema into `data/<custom_dataset_name>/{train, val, test}/`.
2. Set the environment variable:
   ```bash
   export GFM_DATA_DIR=data/<custom_dataset_name>
   ```
   Or pass `--patch-dir data/<custom_dataset_name>` to any script.
