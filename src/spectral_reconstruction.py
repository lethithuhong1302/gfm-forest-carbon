"""
Spectral Reconstruction Module for Geospatial Foundation Models (NASA-IBM Prithvi-100M).

Implements Method 2.1: Algebraic Spectral Inversion of missing infrared bands
from precomputed Normalized Difference Spectral Indices (NDVI, MNDWI, NBR, NDMI).

Resolves the Input Modality Mismatch:
  Original Prithvi-100M HLS Bands:
    0: Blue (~0.49 um)
    1: Green (~0.56 um)
    2: Red (~0.66 um)
    3: Narrow NIR (~0.86 um)
    4: SWIR 1 (~1.61 um)
    5: SWIR 2 (~2.19 um)

  Algebraic Inversion Equations:
    NIR_recon   = Red   * (1 + NDVI)  / (1 - NDVI  + eps)
    SWIR1_recon = Green * (1 - MNDWI) / (1 + MNDWI + eps)
    SWIR2_recon = NIR   * (1 - NBR)   / (1 + NBR   + eps)
"""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

OPTICAL_CHANNEL_MAP = {
    "LS": {"Red": 0, "Green": 1, "Blue": 2},
    "S2": {"Red": 3, "Green": 4, "Blue": 5}
}

INDICES_CHANNEL_MAP = {
    "LS": {"MNDWI": 8, "NBR": 10, "NDMI": 12, "NDVI": 14, "NDWI": 18},
    "S2": {"MNDWI": 9, "NBR": 11, "NDMI": 13, "NDVI": 15, "NDWI": 19}
}

def reconstruct_hls_numpy(optical, indices, source="S2", eps=1e-5, clip_max=1.0):
    """
    Reconstructs 6 HLS-compatible bands from optical and indices numpy arrays.
    
    Args:
        optical: np.ndarray of shape (6, H, W) or (6,)
        indices: np.ndarray of shape (20, H, W) or (20,)
        source: 'S2' (Sentinel-2, 10m) or 'LS' (Landsat-8)
        eps: Small constant to avoid division by zero
        clip_max: Maximum reflectance ceiling (physical surface reflectance <= 1.0)
        
    Returns:
        hls_bands: np.ndarray of shape (6, H, W) or (6,) with bands:
                   [Blue, Green, Red, NIR_recon, SWIR1_recon, SWIR2_recon]
    """
    max_val = clip_max if clip_max is not None else 1.0
    optical = np.nan_to_num(optical, nan=0.0, posinf=max_val, neginf=0.0)
    indices = np.nan_to_num(indices, nan=0.0, posinf=1.0, neginf=-1.0)

    opt_idx = OPTICAL_CHANNEL_MAP[source]
    ind_idx = INDICES_CHANNEL_MAP[source]

    red = optical[opt_idx["Red"]]
    green = optical[opt_idx["Green"]]
    blue = optical[opt_idx["Blue"]]

    ndvi = indices[ind_idx["NDVI"]]
    mndwi = indices[ind_idx["MNDWI"]]
    nbr = indices[ind_idx["NBR"]]

    # 1. Algebraic Inversion for NIR
    denom_nir = np.maximum(1.0 - ndvi, eps)
    nir_recon = red * (1.0 + ndvi) / denom_nir

    # 2. Algebraic Inversion for SWIR1
    denom_swir1 = np.maximum(1.0 + mndwi, eps)
    swir1_recon = green * (1.0 - mndwi) / denom_swir1

    # 3. Algebraic Inversion for SWIR2
    denom_swir2 = np.maximum(1.0 + nbr, eps)
    swir2_recon = nir_recon * (1.0 - nbr) / denom_swir2

    # Physical clamping to valid reflectance ceiling
    if clip_max is not None:
        blue = np.clip(blue, 0.0, max_val)
        green = np.clip(green, 0.0, max_val)
        red = np.clip(red, 0.0, max_val)
        nir_recon = np.clip(nir_recon, 0.0, max_val)
        swir1_recon = np.clip(swir1_recon, 0.0, max_val)
        swir2_recon = np.clip(swir2_recon, 0.0, max_val)

    return np.stack([blue, green, red, nir_recon, swir1_recon, swir2_recon], axis=0).astype(np.float32)

def reconstruct_hls_torch(optical, indices, source="S2", eps=1e-5, clip_max=1.0):
    """
    PyTorch tensor implementation supporting batched inputs.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for reconstruct_hls_torch.")

    max_val = clip_max if clip_max is not None else 1.0
    optical = torch.nan_to_num(optical, nan=0.0, posinf=max_val, neginf=0.0)
    indices = torch.nan_to_num(indices, nan=0.0, posinf=1.0, neginf=-1.0)

    opt_idx = OPTICAL_CHANNEL_MAP[source]
    ind_idx = INDICES_CHANNEL_MAP[source]

    c_dim = 1 if optical.dim() == 4 else 0
    red = optical.select(c_dim, opt_idx["Red"])
    green = optical.select(c_dim, opt_idx["Green"])
    blue = optical.select(c_dim, opt_idx["Blue"])

    ndvi = indices.select(c_dim, ind_idx["NDVI"])
    mndwi = indices.select(c_dim, ind_idx["MNDWI"])
    nbr = indices.select(c_dim, ind_idx["NBR"])

    denom_nir = torch.clamp(1.0 - ndvi, min=eps)
    nir_recon = red * (1.0 + ndvi) / denom_nir

    denom_swir1 = torch.clamp(1.0 + mndwi, min=eps)
    swir1_recon = green * (1.0 - mndwi) / denom_swir1

    denom_swir2 = torch.clamp(1.0 + nbr, min=eps)
    swir2_recon = nir_recon * (1.0 - nbr) / denom_swir2

    if clip_max is not None:
        blue = torch.clamp(blue, 0.0, max_val)
        green = torch.clamp(green, 0.0, max_val)
        red = torch.clamp(red, 0.0, max_val)
        nir_recon = torch.clamp(nir_recon, 0.0, max_val)
        swir1_recon = torch.clamp(swir1_recon, 0.0, max_val)
        swir2_recon = torch.clamp(swir2_recon, 0.0, max_val)

    return torch.stack([blue, green, red, nir_recon, swir1_recon, swir2_recon], dim=c_dim)

def reconstruct_hls_from_40ch_tensor(x, source="S2", eps=1e-5, clip_max=1.0):
    """
    Extracts optical (channels 0:6) and indices (channels 20:40) from a full tensor
    and reconstructs 6 standard Prithvi HLS bands.
    """
    if x.dim() == 4:
        optical = x[:, 0:6, :, :]
        indices = x[:, 20:40, :, :]
    else:
        optical = x[0:6, :, :]
        indices = x[20:40, :, :]
    return reconstruct_hls_torch(optical, indices, source=source, eps=eps, clip_max=clip_max)
