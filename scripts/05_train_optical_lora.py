#!/usr/bin/env python3
"""
Step 05: Prithvi Optical Fine-Tuning Pipeline with LoRA and Partial Layer Unfreezing.

Supports three distinct adaptation regimes:
  1. LoRA PEFT (--peft-mode lora):
     Injects trainable low-rank decomposition matrices (r=8, alpha=16.0) into Query
     and Value self-attention projections while freezing 100% of base ViT weights.
  2. Last-N Blocks Fine-Tuning (--peft-mode last_blocks):
     Unfreezes the last N Transformer encoder blocks (e.g., last 2 blocks = ~14M params).
  3. Full Freeze (--peft-mode frozen):
     Completely freezes the backbone and trains only the regression decoder.

Enforces Epoch-0 protection and validation normalized RMSE checkpoint selection.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    CHECKPOINT_DIR, RESULTS_DIR, DEFAULT_SEED, USE_TTA, set_seed, get_device,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT, DEFAULT_FINETUNE_BLOCKS
)
from src.dataset import MultimodalPatchDataset, get_patch_dir
from src.models import PrithviBackboneWrapper, PrithviMultimodalAdapterModel
from src.losses import MaskedHuberLoss
from src.metrics import evaluate_multi_target
from src.experiment_protocol import (
    target_scales, make_protocol, collect_predictions,
    validation_metrics, save_checkpoint, load_checkpoint,
    export_predictions
)

def run_prithvi_optical(epochs=25, batch_size=4, lr=5e-4, backbone_lr=None,
                        peft_mode="lora", lora_r=DEFAULT_LORA_R, lora_alpha=DEFAULT_LORA_ALPHA,
                        lora_dropout=DEFAULT_LORA_DROPOUT, finetune_blocks=DEFAULT_FINETUNE_BLOCKS,
                        load_pretrained=False, prithvi_checkpoint=None,
                        target="all", seed=DEFAULT_SEED, patch_dir=None, use_tta=USE_TTA):
    set_seed(seed)
    device = get_device()
    patch_dir = get_patch_dir(patch_dir)

    print("=" * 80)
    print(f"PRITHVI OPTICAL TRAINING | PEFT MODE: {peft_mode.upper()}")
    print("=" * 80)

    # Resolve target channels
    target_clean = str(target).strip().lower()
    if target_clean in ("agb",):
        target_name = "AGB"
        target_indices = [0]
        target_cols = ["AGB"]
        out_channels = 1
    elif target_clean in ("gsv", "volume", "tru_luong", "truluong"):
        target_name = "GSV"
        target_indices = [1]
        target_cols = ["GSV"]
        out_channels = 1
    else:
        target_name = "all"
        target_indices = [0, 1]
        target_cols = ["AGB", "GSV"]
        out_channels = 2

    # Datasets
    datasets = {
        split: MultimodalPatchDataset(split=split, patch_dir=patch_dir, scenario="S1_Optical")
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"))
        for split, ds in datasets.items()
    }
    scales = target_scales(datasets["train"].files, target_indices=target_indices)

    # Initialize Prithvi Backbone
    use_lora = (peft_mode == "lora")
    r_val = lora_r if use_lora else 0

    backbone = PrithviBackboneWrapper(
        in_channels=6, patch_size=16, embed_dim=768, depth=12, img_size=128,
        checkpoint_path=prithvi_checkpoint, load_pretrained=load_pretrained,
        lora_r=r_val, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    ).to(device)

    # Apply requested PEFT regime
    if peft_mode == "lora":
        trainable_count = backbone.unfreeze_lora()
        backbone_lr = backbone_lr or 1e-4
    elif peft_mode == "last_blocks":
        backbone.unfreeze_last_blocks(count=finetune_blocks)
        trainable_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        backbone_lr = backbone_lr or 1e-5
    else: # frozen
        backbone.freeze_backbone()
        trainable_count = 0
        backbone_lr = 1e-5

    model = PrithviMultimodalAdapterModel(
        backbone=backbone, total_channels=6, optical_channels=6,
        embed_dim=768, out_channels=out_channels, use_adapter=False,
        target_names=target_cols
    ).to(device)

    # Optimizer with differential learning rates
    param_groups = [{'params': model.decoder.parameters(), 'lr': lr}]
    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    if trainable_backbone:
        param_groups.append({'params': trainable_backbone, 'lr': backbone_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MaskedHuberLoss(delta=1.0)

    # Experiment Protocol and File Names
    suffix = f"_{target_name}" if target_name != "all" else ""
    ckpt_name = f"prithvi_optical_{peft_mode}{suffix}_best.pth"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    results_csv = os.path.join(RESULTS_DIR, f"prithvi_optical_{peft_mode}{suffix}_results.csv")

    protocol = make_protocol(
        "S1_Optical", {}, scales, use_tta, datasets,
        peft_mode=peft_mode, lora_r=r_val, finetune_blocks=finetune_blocks,
        batch_size=batch_size, predicted_targets=target_cols
    )

    # Epoch 0 Protection: record initial baseline before gradient steps
    truth_val, pred_val, names_val = collect_predictions(model, loaders["val"], device, use_tta, target_indices)
    init_metrics = validation_metrics(truth_val, pred_val, scales, target_cols)
    best_score = init_metrics["val_score"]
    save_checkpoint(ckpt_path, model, protocol, 0, init_metrics, phase="epoch0_initialization")
    print(f"[*] Epoch 0 Initial val_nRMSE = {best_score:.6f}")

    # Training Loop
    history = [{"epoch": 0, "train_loss": None, **init_metrics}]
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0.0, 0
        for x, y, mask, _ in loaders["train"]:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y[:, target_indices], mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)

        scheduler.step()
        train_loss = total_loss / max(count, 1)

        # Validation
        truth_val, pred_val, names_val = collect_predictions(model, loaders["val"], device, use_tta, target_indices)
        metrics = validation_metrics(truth_val, pred_val, scales, target_cols)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

        is_best = metrics["val_score"] < best_score
        if is_best:
            best_score = metrics["val_score"]
            save_checkpoint(ckpt_path, model, protocol, epoch, metrics, phase="best_validation")

        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss={train_loss:.4f} | "
              f"Val nRMSE={metrics['val_score']:.6f} {'[BEST]' if is_best else ''}")

    # Load best checkpoint and evaluate on Test Split
    print(f"\n[+] Loading best checkpoint from: {ckpt_path}")
    best_ckpt = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    truth_test, pred_test, names_test = collect_predictions(model, loaders["test"], device, use_tta, target_indices)
    export_predictions(results_csv, truth_test, pred_test, names_test, target_cols)

    # Compute multi-target reporting with derived Carbon stock
    if "AGB" in target_cols:
        carbon_truth = truth_test[:, 0:1] * 0.47
        carbon_pred = pred_test[:, 0:1] * 0.47
        truth_report = np.hstack([truth_test, carbon_truth])
        pred_report = np.hstack([pred_test, carbon_pred])
        cols_report = target_cols + ["Carbon"]
    else:
        truth_report, pred_report, cols_report = truth_test, pred_test, target_cols

    df_test = evaluate_multi_target(truth_report, pred_report, cols_report, compute_ci=True)
    df_test["PEFT_Mode"] = peft_mode
    df_test.to_csv(results_csv.replace(".csv", "_metrics.csv"), index=False)

    print("\nFINAL TEST METRICS:")
    print(df_test.to_string())
    print("=" * 80)
    return model, df_test

def main():
    parser = argparse.ArgumentParser(description="Prithvi Optical GFM Fine-Tuning with LoRA / Partial Blocks.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--peft-mode", type=str, default="lora", choices=["lora", "last_blocks", "frozen"])
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--finetune-blocks", type=int, default=DEFAULT_FINETUNE_BLOCKS)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--target", type=str, default="all")
    args = parser.parse_args()

    run_prithvi_optical(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        peft_mode=args.peft_mode,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        finetune_blocks=args.finetune_blocks,
        load_pretrained=args.load_pretrained,
        target=args.target
    )

if __name__ == "__main__":
    main()
