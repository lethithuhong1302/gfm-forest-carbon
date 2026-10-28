"""
Multimodal Dataset Module for 128x128 Multi-Sensor Patches.

Features:
  - Loads 41-channel Earth Observation NPZ patches (Optical, SAR, GLCM, Indices, DEM)
  - Supports dynamic channel slicing across Scenarios S1 to S7
  - Strict train-only normalization statistics fitting to prevent test leakage
  - Synchronized D4 Dihedral Group Augmentation (8 transformations) for training
  - Algebraic HLS Spectral Inversion for Prithvi Foundation Model compatibility
"""

import os
import glob
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from .config import (
    DATA_DIR, SCENARIOS,
    USE_SPECTRAL_RECONSTRUCTION, SPECTRAL_RECONSTRUCTION_SOURCE,
    USE_D4_AUGMENTATION
)
from .spectral_reconstruction import reconstruct_hls_numpy

def get_patch_dir(base_dir=None):
    if base_dir and os.path.exists(base_dir):
        return base_dir
    return DATA_DIR

def get_patch_files(split_name, patch_dir=None):
    pdir = get_patch_dir(patch_dir)
    split_dir = os.path.join(pdir, split_name)
    files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    return files

class MultimodalPatchDataset(Dataset):
    """
    PyTorch Dataset loading 128x128 multi-modal patches from NPZ files.
    """
    def __init__(self, split="train", patch_dir=None, files=None, scenario="S7_Full_Fusion",
                 norm_stats=None, transform=None, augment=None,
                 normalize_from_channel=0,
                 prepend_zero_channels=0,
                 use_spectral_recon=USE_SPECTRAL_RECONSTRUCTION,
                 spectral_source=SPECTRAL_RECONSTRUCTION_SOURCE):
        self.split = split
        self.scenario = scenario
        self.transform = transform
        self.files = files if files is not None else get_patch_files(split, patch_dir)
        self.norm_stats = norm_stats
        self.normalize_from_channel = int(normalize_from_channel)
        self.prepend_zero_channels = int(prepend_zero_channels)
        self.use_spectral_recon = use_spectral_recon
        self.spectral_source = spectral_source
        if augment is None:
            self.augment = (split == "train" and USE_D4_AUGMENTATION)
        else:
            self.augment = augment

    def __len__(self):
        return len(self.files)

    @staticmethod
    def slice_channels(data_dict, scenario, use_spectral_recon=USE_SPECTRAL_RECONSTRUCTION,
                       spectral_source=SPECTRAL_RECONSTRUCTION_SOURCE):
        """Extract requested channels based on scenario with optional HLS reconstruction and DEM."""
        opt = data_dict["optical"]      # (6, 128, 128) raw LS+S2 RGB
        sar = data_dict["sar"]          # (6, 128, 128) -> S1: 0..3, ALOS-2: 4..5
        glcm = data_dict["glcm"]        # (8, 128, 128)
        ind = data_dict["indices"]      # (20, 128, 128)

        # DEM channel fallback if missing in legacy NPZ
        if "dem" in data_dict and data_dict["dem"] is not None and data_dict["dem"].size > 0:
            dem = data_dict["dem"]
            if dem.ndim == 2:
                dem = np.expand_dims(dem, axis=0)
        else:
            dem = np.zeros((1, opt.shape[1], opt.shape[2]), dtype=np.float32)

        if use_spectral_recon and opt.shape[0] == 6 and ind.shape[0] == 20:
            # Reconstruct 6 standard HLS bands: [Blue, Green, Red, NIR_recon, SWIR1_recon, SWIR2_recon]
            opt = reconstruct_hls_numpy(opt, ind, source=spectral_source, clip_max=1.0)

        if scenario == "S1_Optical":
            x = opt
        elif scenario == "S1_Opt_DEM":
            x = np.concatenate([opt, dem], axis=0)
        elif scenario == "S2_Opt_S1":
            s1 = sar[0:4] # VH, VH/VV, VV, VV/VH
            x = np.concatenate([opt, s1], axis=0)
        elif scenario == "S2_Opt_S1_DEM":
            s1 = sar[0:4]
            x = np.concatenate([opt, s1, dem], axis=0)
        elif scenario == "S3_Opt_ALOS2":
            alos2 = sar[4:6] # HH, HV
            x = np.concatenate([opt, alos2], axis=0)
        elif scenario == "S3_Opt_ALOS2_DEM":
            alos2 = sar[4:6]
            x = np.concatenate([opt, alos2, dem], axis=0)
        elif scenario == "S4_Opt_SAR":
            x = np.concatenate([opt, sar], axis=0)
        elif scenario == "S4_Opt_SAR_DEM":
            x = np.concatenate([opt, sar, dem], axis=0)
        elif scenario == "S5_Opt_SAR_GLCM":
            x = np.concatenate([opt, sar, glcm], axis=0)
        elif scenario == "S5_Opt_SAR_GLCM_DEM":
            x = np.concatenate([opt, sar, glcm, dem], axis=0)
        elif scenario == "S6_Indices":
            x = ind
        elif scenario == "S7_Full_Fusion":
            x = np.concatenate([opt, sar, glcm, ind, dem], axis=0)
        else:
            x = np.concatenate([opt, sar, glcm, ind, dem], axis=0)
        return x

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with np.load(file_path) as data:
            opt = data["optical"]
            sar = data["sar"]
            glcm = data["glcm"]
            ind = data["indices"]
            dem = data["dem"] if "dem" in data else np.zeros((1, opt.shape[1], opt.shape[2]), dtype=np.float32)
            target = data["target"].astype(np.float32) # (3, 128, 128)
            mask = data["mask"].astype(np.float32)     # (128, 128)

        data_dict = {"optical": opt, "sar": sar, "glcm": glcm, "indices": ind, "dem": dem}
        x = self.slice_channels(
            data_dict, self.scenario,
            use_spectral_recon=self.use_spectral_recon,
            spectral_source=self.spectral_source
        ).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply train-fitted normalization if provided
        if self.norm_stats is not None:
            mean = self.norm_stats.get(self.scenario, {}).get("mean", None)
            std = self.norm_stats.get(self.scenario, {}).get("std", None)
            if mean is not None and std is not None:
                start = self.normalize_from_channel
                x[start:] = (x[start:] - mean[start:]) / np.maximum(std[start:], 1e-6)

        # Synchronized D4 Dihedral Data Augmentation (Rotation + Flips) for training
        if self.augment:
            # 1. Random 90-degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                target = np.rot90(target, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

            # 2. Random Horizontal Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                target = np.flip(target, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            # 3. Random Vertical Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                target = np.flip(target, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

        if self.prepend_zero_channels:
            zeros = np.zeros((self.prepend_zero_channels, x.shape[1], x.shape[2]), dtype=x.dtype)
            x = np.concatenate([zeros, x], axis=0)

        if torch is not None:
            x_tensor = torch.from_numpy(x)
            y_tensor = torch.from_numpy(target)
            mask_tensor = torch.from_numpy(mask)
            return x_tensor, y_tensor, mask_tensor, os.path.basename(file_path)

        return x, target, mask, os.path.basename(file_path)

def compute_train_norm_stats(patch_dir=None, scenarios=None):
    """
    Computes per-channel mean and std strictly on the TRAIN split for all scenarios.
    Adheres strictly to scientific leakage prevention (fitted strictly on training).
    """
    train_files = get_patch_files("train", patch_dir)
    if not train_files:
        print("[!] Warning: No training patches found to compute normalization statistics.")
        return {}

    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    stats = {}
    for scen in scenarios:
        if scen == "S0_Constant":
            continue
        all_channels = []
        for fp in train_files:
            with np.load(fp) as data:
                opt = data["optical"]
                sar = data["sar"]
                glcm = data["glcm"]
                ind = data["indices"]
                dem = data["dem"] if "dem" in data else np.zeros((1, opt.shape[1], opt.shape[2]), dtype=np.float32)
                data_dict = {"optical": opt, "sar": sar, "glcm": glcm, "indices": ind, "dem": dem}
                x = MultimodalPatchDataset.slice_channels(data_dict, scen)
                all_channels.append(x)
        all_channels = np.stack(all_channels, axis=0) # (N, C, H, W)
        mean = np.nanmean(all_channels, axis=(0, 2, 3), keepdims=True)
        std = np.nanstd(all_channels, axis=(0, 2, 3), keepdims=True)
        std[std == 0] = 1.0 # Avoid div-by-zero
        stats[scen] = {"mean": mean, "std": std}
    return stats
