#!/usr/bin/env python3
"""
Interactive Walkthrough Notebook Generator.

Builds `notebooks/Quickstart_Walkthrough.ipynb` for out-of-the-box
execution on Google Colab or local GPU workstations.
"""

import os
import json
from pathlib import Path

def text_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def build_notebook(output_path):
    cells = []

    # Title & Metadata
    cells.append(text_cell("""# Geospatial Foundation Models for Sparse-Plot Forest Carbon Mapping: Multimodal Fine-Tuning & Parameter-Efficient LoRA
### Full Algorithmic Reproducibility & Interactive Walkthrough Notebook
**Repository:** [https://github.com/anonymous-repo/gfm-forest-carbon](https://github.com/anonymous-repo/gfm-forest-carbon)  
**Sensors (41 Channels):** Optical (6 HLS Reconstructed), Sentinel-1 C-band (4), ALOS-2 L-band (2), GLCM Textures (8), Spectral Indices (20), SRTM DEM Topography (1)  
**Proposed Framework:** Pre-trained **NASA-IBM Prithvi-100M ViT** with **LoRA PEFT** ($r=8, \\alpha=16$) + **CNN Auxiliary Encoder** + **Gated Cross-Modal Fusion** + **Dense Regression Decoder**  
**Physical Carbon Derivation:** $\\text{Carbon} = 0.47 \\times \\text{AGB}$ (IPCC Guidelines)"""))

    # Section 1
    cells.append(text_cell("""## 1. Environment Verification & Dependency Setup
Verify GPU acceleration (NVIDIA T4, L4, V100, or A100) and install Python dependencies."""))

    cells.append(code_cell("""import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"[+] Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("[!] GPU not detected. Running on CPU mode.")"""))

    # Section 2
    cells.append(text_cell("""## 2. Dataset Audit & Multi-Sensor Visualization
Audit the bundled anonymized toy dataset (128x128 patches) and render multi-sensor previews."""))

    cells.append(code_cell("""!python scripts/00_audit_dataset.py
!python scripts/01_visualize_patches.py

from IPython.display import Image
Image("experiments_output/figures/multimodal_patch_preview.png")"""))

    # Section 3
    cells.append(text_cell("""## 3. GFM Architectural Compatibility & PEFT Parameter Analysis
Check token sequence dimensions (64 patches, 768 dim) and contrast parameter efficiency between:
- Full Fine-Tuning (86.2M params, 100%)
- Last-2 Blocks Fine-Tuning (14.2M params, 16.4%)
- LoRA Low-Rank Adaptation (295K params, **0.34%**)
- Proposed Frozen Backbone + Multimodal Adapter (4.1M params, **4.78%**)"""))

    cells.append(code_cell("""!python scripts/04_verify_compatibility.py"""))

    # Section 4
    cells.append(text_cell("""## 4. Prithvi Optical Fine-Tuning Pipeline
Fine-tune the optical foundation model using Parameter-Efficient LoRA ($r=8, \\alpha=16.0$) with Epoch-0 protection and validation nRMSE checkpoint selection."""))

    cells.append(code_cell("""!python scripts/05_train_optical_lora.py --epochs 15 --peft-mode lora --lora-r 8 --batch-size 4"""))

    # Section 5
    cells.append(text_cell("""## 5. Two-Stage Multimodal Adapter Training (41 Channels with DEM)
Warm-start from the optical checkpoint, freeze backbone weights, and train the cross-modal auxiliary adapter."""))

    cells.append(code_cell("""!python scripts/06_train_multimodal_adapter.py --scenario S7_Full_Fusion --epochs 15 --fusion-only-epochs 3 --batch-size 4"""))

    # Section 6
    cells.append(text_cell("""## 6. Sensor Modality Ablation Benchmark
Benchmark sensor contributions across Scenarios S1 (Optical) to S7 (Full Multimodal with Topography)."""))

    cells.append(code_cell("""!python scripts/07_ablation_study.py --epochs 5 --batch-size 4"""))

    # Section 7
    cells.append(text_cell("""## 7. Explainable AI & Uncertainty Quantification
Calculate group permutation feature importance and evaluate residual error distributions."""))

    cells.append(code_cell("""!python scripts/09_xai_analysis.py --scenario S7_Full_Fusion

Image("experiments_output/figures/permutation_importance.png")"""))

    # Section 8
    cells.append(text_cell("""## 8. Export IEEE Publication Tables
Export CSV and LaTeX tables formatted for IEEE Transactions submission."""))

    cells.append(code_cell("""!python scripts/10_export_results.py

import pandas as pd
df_table1 = pd.read_csv("experiments_output/tables/table1_main_model_comparison.csv")
display(df_table1)"""))

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python", "version": "3.10"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[+] Successfully generated master walkthrough notebook: {output_path}")

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "Quickstart_Walkthrough.ipynb")
    build_notebook(out)
