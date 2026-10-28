"""
Checkpoint Utilities for State Dict Compatibility and Target Mapping.
"""

import torch

def extract_state_dict(checkpoint):
    """Safely extract model weights dict regardless of nesting."""
    if isinstance(checkpoint, dict):
        for k in ("model_state_dict", "state_dict", "model", "network"):
            if k in checkpoint and isinstance(checkpoint[k], dict):
                return checkpoint[k]
    return checkpoint

def checkpoint_targets(checkpoint):
    """Retrieve target column names saved in protocol metadata."""
    if isinstance(checkpoint, dict):
        proto = checkpoint.get("protocol", {})
        if "predicted_targets" in proto:
            return proto["predicted_targets"]
        if "target" in proto:
            t = proto["target"]
            return ["AGB", "GSV"] if t in ("all", "both") else [t]
    return ["AGB", "GSV"]
