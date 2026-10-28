"""
Spatial Grid and Clustered Group Splitting to Prevent Autocorrelation Leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DEFAULT_SEED

def make_grouped_spatial_split(gdf_otc, grid_col="grid_id",
                               train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                               seed=DEFAULT_SEED):
    """
    Groups in-situ observations by spatial grid block IDs to avoid spatial leakage.
    Returns:
      gdf_otc with added 'split' column ('train', 'val', 'test').
      split_summary dictionary.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    unique_grids = gdf_otc[grid_col].dropna().unique()

    # 1. Train vs (Val + Test)
    train_grids, temp_grids = train_test_split(
        unique_grids,
        test_size=(val_ratio + test_ratio),
        random_state=seed
    )

    # 2. Val vs Test
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_grids, test_grids = train_test_split(
        temp_grids,
        test_size=rel_test_ratio,
        random_state=seed
    )

    # Spatial isolation assertion
    s_train = set(train_grids)
    s_val = set(val_grids)
    s_test = set(test_grids)
    assert s_train.isdisjoint(s_val), "Spatial leakage detected between train and val grids!"
    assert s_train.isdisjoint(s_test), "Spatial leakage detected between train and test grids!"
    assert s_val.isdisjoint(s_test), "Spatial leakage detected between val and test grids!"

    grid_to_split = {g: "train" for g in train_grids}
    grid_to_split.update({g: "val" for g in val_grids})
    grid_to_split.update({g: "test" for g in test_grids})

    gdf_otc["split"] = gdf_otc[grid_col].map(grid_to_split)

    summary = {
        "split_type": "Grouped Spatial Block Split",
        "seed": seed,
        "grid_counts": {
            "train": len(train_grids),
            "val": len(val_grids),
            "test": len(test_grids)
        },
        "observation_counts": gdf_otc["split"].value_counts().to_dict()
    }
    return gdf_otc, summary
