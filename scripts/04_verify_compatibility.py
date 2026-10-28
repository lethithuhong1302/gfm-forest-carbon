#!/usr/bin/env python3
"""
Step 04: Geospatial Foundation Model (GFM) Compatibility & Parameter-Efficiency Analysis.

Verifies:
  - Input patch dimensions (128x128) -> Patch projection (16x16) -> Token sequence (64 tokens, 768 dim)
  - Parameter-Efficient Fine-Tuning (PEFT) ratios:
      * Full Fine-Tuning (~100% trainable)
      * Last-2 Blocks Fine-Tuning (~16% trainable)
      * LoRA Low-Rank Adaptation (r=8, alpha=16) (~0.34% trainable)
      * Frozen Backbone + Multimodal Adapter (~4.8% trainable)
  - End-to-end forward pass and IPCC Carbon conservation (Carbon = 0.47 * AGB).
"""

import os
import sys
import json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EXPERIMENT_OUTPUT_DIR, get_device, set_seed, DEFAULT_SEED
from src.models import (
    PrithviBackboneWrapper,
    PrithviMultimodalAdapterModel,
    DenseRegressionDecoder
)

def verify_gfm_compatibility():
    set_seed(DEFAULT_SEED)
    device = get_device()
    print("=" * 80)
    print("GFM COMPATIBILITY & PARAMETER-EFFICIENCY GATEWAY")
    print("=" * 80)

    # 1. Optical ViT Backbone Architecture
    print("[1/4] Initializing Prithvi Optical ViT Backbone...")
    backbone = PrithviBackboneWrapper(
        in_channels=6, patch_size=16, embed_dim=768, depth=12,
        num_heads=12, img_size=128, lora_r=8, lora_alpha=16.0
    ).to(device)

    dummy_opt = torch.randn(2, 6, 128, 128, device=device)
    tokens, spatial_feats = backbone(dummy_opt)
    print(f"  - Optical input shape:        {dummy_opt.shape}")
    print(f"  - ViT Token sequence shape:    {tokens.shape} (Expected: [2, 64, 768])")
    print(f"  - Spatial feature map shape:  {spatial_feats.shape} (Expected: [2, 768, 8, 8])")
    assert tokens.shape == (2, 64, 768)
    assert spatial_feats.shape == (2, 768, 8, 8)

    # 2. Parameter Counts Analysis Across Adaptation Strategies
    total_backbone = sum(p.numel() for p in backbone.parameters())
    base_backbone = sum(p.numel() for n, p in backbone.named_parameters() if "lora_" not in n)
    lora_params = sum(p.numel() for n, p in backbone.named_parameters() if "lora_" in n)

    # Strategy A: LoRA PEFT
    backbone.unfreeze_lora()
    lora_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    # Strategy B: Last-2 Blocks
    backbone.unfreeze_last_blocks(count=2)
    last2_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    # Strategy C: Full Freeze
    backbone.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)

    print("\n[2/4] Adaptation Strategy Parameter Breakdown:")
    print(f"  - Total Base Backbone Parameters:      {base_backbone:,}")
    print(f"  - LoRA Adapter Parameters (r=8):       {lora_params:,} ({lora_params/base_backbone*100:.2f}%)")
    print(f"  - Last-2 Blocks Fine-Tuning Parameters: {last2_trainable:,} ({last2_trainable/base_backbone*100:.2f}%)")
    print(f"  - Full Freeze Backbone Parameters:     {frozen_trainable:,} (0.00%)")

    # 3. Multimodal Adapter Forward Pass
    print("\n[3/4] Testing End-to-End Multimodal Adapter (41 Channels)...")
    backbone.freeze_backbone()
    adapter_model = PrithviMultimodalAdapterModel(
        backbone=backbone,
        total_channels=41,
        optical_channels=6,
        embed_dim=768,
        out_channels=2,
        use_adapter=True
    ).to(device)

    dummy_full = torch.randn(2, 41, 128, 128, device=device)
    pred_direct = adapter_model(dummy_full)
    print(f"  - Multimodal input shape:     {dummy_full.shape}")
    print(f"  - Direct predicted output:    {pred_direct.shape} (Expected: [2, 2, 128, 128] for [AGB, GSV])")
    assert pred_direct.shape == (2, 2, 128, 128)

    # 4. IPCC Carbon Stock Conservation
    pred_full = adapter_model.predict_full_targets(dummy_full)
    print(f"  - Full output with Carbon:    {pred_full.shape} (Expected: [2, 3, 128, 128])")
    assert pred_full.shape == (2, 3, 128, 128)
    diff = torch.max(torch.abs(pred_full[:, 2] - pred_full[:, 0] * 0.47)).item()
    print(f"  - IPCC Conservation residual: {diff:.8f} (Exactly Carbon = 0.47 * AGB)")
    assert torch.allclose(pred_full[:, 2], pred_full[:, 0] * 0.47, atol=1e-6)

    trainable_adapter, total_adapter = adapter_model.get_trainable_params_count()
    peft_ratio = (trainable_adapter / total_adapter) * 100.0
    print("\n[4/4] End-to-End Multimodal Efficiency:")
    print(f"  - Total Parameters:     {total_adapter:,}")
    print(f"  - Trainable Parameters: {trainable_adapter:,}")
    print(f"  - PEFT Trainable Ratio: {peft_ratio:.2f}% (Frozen Optical ViT + Multimodal Adapter)")

    report = {
        "status": "PASSED",
        "device": str(device),
        "backbone_total_params": base_backbone,
        "lora_params_r8": lora_params,
        "lora_peft_ratio_percent": round(lora_params / base_backbone * 100.0, 3),
        "last2_blocks_trainable_params": last2_trainable,
        "last2_blocks_ratio_percent": round(last2_trainable / base_backbone * 100.0, 2),
        "multimodal_total_params": total_adapter,
        "multimodal_trainable_params": trainable_adapter,
        "multimodal_peft_ratio_percent": round(peft_ratio, 2),
        "ipcc_carbon_conservation_verified": True
    }

    out_file = os.path.join(EXPERIMENT_OUTPUT_DIR, "gfm_compatibility_report.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Compatibility report saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    verify_gfm_compatibility()
