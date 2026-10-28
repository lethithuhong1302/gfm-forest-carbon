"""
Baseline Models for Benchmark Comparison.

Includes:
  1. ConstantBaseline: Training set mean/median predictor (Scenario S0).
  2. TabularMLBaseline: Pixel-level Random Forest & XGBoost.
  3. SimpleUNet: Convolutional U-Net baseline.
  4. SmallRandomCNN & SmallRandomViT: Control architectures trained from scratch.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = type("nn", (), {"Module": object})()

try:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
except ImportError:
    RandomForestRegressor = None
    XGBRegressor = None
    MultiOutputRegressor = None

from ..config import DEFAULT_SEED, NUM_PREDICTED_TARGETS

class ConstantBaseline:
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.constants = None

    def fit(self, y_train):
        if self.strategy == "median":
            self.constants = np.nanmedian(y_train, axis=0)
        else:
            self.constants = np.nanmean(y_train, axis=0)
        return self

    def predict(self, X):
        n_samples = len(X)
        return np.tile(self.constants, (n_samples, 1))

class TabularMLBaseline:
    def __init__(self, model_type="rf", multi_target=True, n_estimators=150,
                 max_depth=6, seed=DEFAULT_SEED):
        self.model_type = model_type
        self.multi_target = multi_target
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == "rf":
            if RandomForestRegressor is None:
                raise ImportError("scikit-learn is required for Random Forest.")
            base = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                random_state=self.seed, n_jobs=-1
            )
        elif self.model_type == "xgb":
            if XGBRegressor is None:
                raise ImportError("xgboost is required for XGBoost.")
            base = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, random_state=self.seed, n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown tabular model_type: {self.model_type}")

        if self.multi_target:
            return MultiOutputRegressor(base)
        return base

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(X), 0.0, None)

class SimpleUNet(nn.Module):
    """Compact 4-stage U-Net baseline for 128x128 dense regression."""
    def __init__(self, in_channels=41, out_channels=NUM_PREDICTED_TARGETS, base_dim=32):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.BatchNorm2d(base_dim), nn.ReLU(),
            nn.Conv2d(base_dim, base_dim, 3, padding=1),
            nn.BatchNorm2d(base_dim), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_dim, base_dim * 2, 3, padding=1),
            nn.BatchNorm2d(base_dim * 2), nn.ReLU(),
            nn.Conv2d(base_dim * 2, base_dim * 2, 3, padding=1),
            nn.BatchNorm2d(base_dim * 2), nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_dim * 2, base_dim * 4, 3, padding=1),
            nn.BatchNorm2d(base_dim * 4), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2),
            nn.BatchNorm2d(base_dim * 2), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim, 2, stride=2),
            nn.BatchNorm2d(base_dim), nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_dim, out_channels, 1),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(e3)
        cat2 = torch.cat([d2, e2], dim=1)
        d1 = self.dec1(cat2)
        cat1 = torch.cat([d1, e1], dim=1)
        return self.out(cat1)
