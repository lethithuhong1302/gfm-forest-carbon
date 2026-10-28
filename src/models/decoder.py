"""
Progressive Dense Regression Decoder for Geospatial Feature Upsampling.

Takes spatial token features (B, 768, 8, 8) from the ViT/Adapter and progressively
upsamples through four transposed convolutional stages to (B, out_channels, 128, 128).
Predicts non-negative continuous ecological targets (Aboveground Biomass, Growing Stock Volume).
"""

import torch
import torch.nn as nn
from ..config import NUM_PREDICTED_TARGETS

class DenseRegressionDecoder(nn.Module):
    def __init__(self, in_dim=768, out_channels=NUM_PREDICTED_TARGETS):
        super().__init__()
        # 8x8 -> 16x16
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        # 16x16 -> 32x32
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        # 32x32 -> 64x64
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        # 64x64 -> 128x128
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU()
        )
        # Output Head (128x128 -> out_channels)
        self.head = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, out_channels, kernel_size=1),
            nn.ReLU() # Strict non-negativity constraint for biomass
        )

    def forward(self, x):
        h = self.up1(x)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        return self.head(h)


class NaiveBilinearDecoder(nn.Module):
    """
    Naive baseline head for ablation: direct bilinear interpolation followed by 1x1 conv.
    Used in Script 05b to contrast against the progressive decoder.
    """
    def __init__(self, in_dim=768, out_channels=NUM_PREDICTED_TARGETS, target_size=(128, 128)):
        super().__init__()
        self.target_size = target_size
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        upsampled = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return self.head(upsampled)
