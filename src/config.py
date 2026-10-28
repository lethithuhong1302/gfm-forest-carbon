"""
Configuration and Hyperparameter Settings for GFM Forest Carbon Pipeline.

Defines:
  - Repository-relative and environment-configurable directory paths
  - Multimodal 41-channel Earth Observation schema
  - Experimental scenarios S0 to S7
  - LoRA (Low-Rank Adaptation) and Partial Fine-Tuning hyperparameters
  - Deterministic random seed and device utilities
"""

import os
import random
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# -------------------------------------------------------------------------
# 1. Directory Paths Configuration
# -------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def resolve_data_dir(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return custom_path
    env_path = os.environ.get("GFM_DATA_DIR") or os.environ.get("CAOBANG_DATA_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Check default repository locations
    candidates = [
        os.path.join(REPO_ROOT, "data", "toy_dataset"),
        os.path.join(REPO_ROOT, "data"),
        os.path.join(REPO_ROOT, "dataset", "patches_128_06102026"),
        os.path.join(REPO_ROOT, "dataset", "patches_128_MultiModa1"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(REPO_ROOT, "data", "toy_dataset")

DATA_DIR = resolve_data_dir()
PATCH_DATASET_DIR = DATA_DIR

# Experiment output directories
EXPERIMENT_OUTPUT_DIR = os.environ.get(
    "EXPERIMENT_OUTPUT_DIR",
    os.path.join(REPO_ROOT, "experiments_output")
)
CHECKPOINT_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "results")
FIGURES_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(EXPERIMENT_OUTPUT_DIR, "tables")

for d in [EXPERIMENT_OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    try:
        os.makedirs(d, exist_ok=True)
    except (OSError, PermissionError):
        pass

# -------------------------------------------------------------------------
# 2. Augmentation & Test-Time Augmentation (D4 Dihedral Group)
# -------------------------------------------------------------------------
USE_D4_AUGMENTATION = os.environ.get("USE_D4_AUGMENTATION", "1").lower() in ("1", "true", "yes")
USE_TTA = os.environ.get("USE_TTA", "1").lower() in ("1", "true", "yes")

# -------------------------------------------------------------------------
# 3. LoRA & Fine-Tuning Hyperparameters for Prithvi Optical GFM
# -------------------------------------------------------------------------
DEFAULT_LORA_R = int(os.environ.get("LORA_R", "8"))
DEFAULT_LORA_ALPHA = float(os.environ.get("LORA_ALPHA", "16.0"))
DEFAULT_LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
DEFAULT_FINETUNE_BLOCKS = int(os.environ.get("FINETUNE_BLOCKS", "2"))

PRITHVI_MODEL_ID = "ibm-nasa-geospatial/Prithvi-100M"
PRITHVI_CHECKPOINT_PATH = os.environ.get(
    "PRITHVI_CHECKPOINT_PATH",
    os.path.join(REPO_ROOT, "models", "Prithvi_100M.pt")
)
PRITHVI_MIN_WEIGHT_COVERAGE = 0.90

# -------------------------------------------------------------------------
# 4. Target Variables & Units (2 Learned Channels + 1 Derived Channel)
# -------------------------------------------------------------------------
# Neural network explicitly predicts 2 independent ecological targets:
PREDICTED_TARGETS = ["AGB", "GSV"]
NUM_PREDICTED_TARGETS = len(PREDICTED_TARGETS) # 2

# Target name aliases mapping
TARGET_ALIASES = {
    "tru_luong": "GSV",
    "truluong": "GSV",
    "volume": "GSV",
    "gsv": "GSV",
    "agb": "AGB",
    "carbon": "Carbon"
}

# Forest carbon stock is deterministically derived: Carbon = 0.47 * AGB (IPCC Guidelines)
IPCC_CARBON_FACTOR = 0.47
DERIVED_TARGETS = {"Carbon": IPCC_CARBON_FACTOR}

# Full reported target columns:
TARGET_COLS = ["AGB", "GSV", "Carbon"]
NUM_TOTAL_TARGETS = len(TARGET_COLS) # 3

TARGET_UNITS = {
    "AGB": "Mg/ha",
    "GSV": "m³/ha",
    "Carbon": "Mg C/ha"
}

# -------------------------------------------------------------------------
# 5. Multimodal Band Schema (Total = 41 channels)
# -------------------------------------------------------------------------
RAW_OPTICAL_BANDS = ["LS_Red", "LS_Green", "LS_Blue", "S2_Red", "S2_Green", "S2_Blue"]
PRITHVI_HLS_BANDS = ["Blue", "Green", "Red", "NIR_recon", "SWIR1_recon", "SWIR2_recon"]
OPTICAL_BANDS = PRITHVI_HLS_BANDS

USE_SPECTRAL_RECONSTRUCTION = True
SPECTRAL_RECONSTRUCTION_SOURCE = "S2" # "S2" (10m) or "LS" (Landsat-8)

SAR_BANDS = ["S1_VH", "S1_VH_VV", "S1_VV", "S1_VV_VH", "ALOS2_HH", "ALOS2_HV"] # 6 bands
GLCM_BANDS = [
    "Mean", "Variance", "Homogeneity", "Contrast", 
    "Dissimilarity", "Entropy", "Second_Moment", "Correlation"
] # 8 bands
INDICES_BANDS = [
    "AWEInsh_LS", "AWEInsh_S2", "AWEIsh_LS", "AWEIsh_S2",
    "BSI_LS", "BSI_S2", "EVI_LS", "EVI_S2",
    "MNDWI_LS", "MNDWI_S2", "NBR_LS", "NBR_S2",
    "NDMI_LS", "NDMI_S2", "NDVI_LS", "NDVI_S2",
    "NDWF_LS", "NDWF_S2", "NDWI_LS", "NDWI_S2"
] # 20 bands
DEM_BANDS = ["DEM_SRTM"] # 1 band

BAND_SCHEMA = {
    "optical_raw": {"count": 6, "bands": RAW_OPTICAL_BANDS, "indices": list(range(0, 6))},
    "optical": {"count": 6, "bands": PRITHVI_HLS_BANDS, "indices": list(range(0, 6))},
    "sar": {"count": 6, "bands": SAR_BANDS, "indices": list(range(6, 12))},
    "glcm": {"count": 8, "bands": GLCM_BANDS, "indices": list(range(12, 20))},
    "indices": {"count": 20, "bands": INDICES_BANDS, "indices": list(range(20, 40))},
    "dem": {"count": 1, "bands": DEM_BANDS, "indices": [40]}
}
TOTAL_CHANNELS = 41
PATCH_SIZE = 128

# -------------------------------------------------------------------------
# 6. Experimental Scenarios (S0 to S7) - Controlled Dynamic Slicing Protocol
# -------------------------------------------------------------------------
# Slicing Protocol: X^(S) = S_S(X), C_S = C_opt^(S) + C_aux^(S)
# Optical ViT engaged when C_opt=6, bypassed when C_opt=0.
# Auxiliary CNN encoder matches C_aux in {1, 3, 5, 7, 15, 20, 35}.
SCENARIOS = {
    "S0_Constant": {"desc": "Constant Predictor (Training mean/median baseline)", "channels": 0, "c_opt": 0, "c_aux": 0, "keys": []},
    "S1_Optical": {"desc": "Optical surface reflectance only (6 bands)", "channels": 6, "c_opt": 6, "c_aux": 0, "keys": ["optical"]},
    "S1_Opt_DEM": {"desc": "S1: Optical (6) + DEM (1)", "channels": 7, "c_opt": 6, "c_aux": 1, "keys": ["optical", "dem"]},
    "S2_Opt_S1": {"desc": "Optical (6) + Sentinel-1 (4 bands)", "channels": 10, "c_opt": 6, "c_aux": 4, "keys": ["optical", "s1_only"]},
    "S2_Opt_S1_DEM": {"desc": "S2: Optical (6) + Sentinel-1 (4) + DEM (1)", "channels": 11, "c_opt": 6, "c_aux": 5, "keys": ["optical", "s1_only", "dem"]},
    "S3_Opt_ALOS2": {"desc": "Optical (6) + ALOS-2 (2 bands)", "channels": 8, "c_opt": 6, "c_aux": 2, "keys": ["optical", "alos2_only"]},
    "S3_Opt_ALOS2_DEM": {"desc": "S3: Optical (6) + ALOS-2 (2) + DEM (1)", "channels": 9, "c_opt": 6, "c_aux": 3, "keys": ["optical", "alos2_only", "dem"]},
    "S4_Opt_SAR": {"desc": "Optical (6) + Full SAR (6 bands: S1 + ALOS-2)", "channels": 12, "c_opt": 6, "c_aux": 6, "keys": ["optical", "sar"]},
    "S4_Opt_SAR_DEM": {"desc": "S4: Optical (6) + Full SAR (6) + DEM (1)", "channels": 13, "c_opt": 6, "c_aux": 7, "keys": ["optical", "sar", "dem"]},
    "S5_Opt_SAR_GLCM": {"desc": "Optical (6) + SAR (6) + GLCM Texture (8)", "channels": 20, "c_opt": 6, "c_aux": 14, "keys": ["optical", "sar", "glcm"]},
    "S5_Opt_SAR_GLCM_DEM": {"desc": "S5: Optical (6) + SAR (6) + GLCM (8) + DEM (1)", "channels": 21, "c_opt": 6, "c_aux": 15, "keys": ["optical", "sar", "glcm", "dem"]},
    "S6_Indices": {"desc": "S6: Spectral & Moisture Indices only (20 bands, ViT bypassed)", "channels": 20, "c_opt": 0, "c_aux": 20, "keys": ["indices"]},
    "S7_Full_Fusion": {"desc": "S7: Full Multimodal Fusion with DEM (Optical + SAR + GLCM + Indices + DEM = 41 bands)", "channels": 41, "c_opt": 6, "c_aux": 35, "keys": ["optical", "sar", "glcm", "indices", "dem"]}
}

# -------------------------------------------------------------------------
# 7. Reproducibility & Device Utilities
# -------------------------------------------------------------------------
DEFAULT_SEED = 42

def set_seed(seed=DEFAULT_SEED):
    """Ensure strict deterministic execution across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Retrieve PyTorch compute device."""
    if torch is not None and torch.cuda.is_available():
        return torch.device("cuda")
    return "cpu"
