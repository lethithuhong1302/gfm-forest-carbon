"""
Loss Functions for Sparse Ground Truth Observation Supervision.

Calculates regression loss strictly over valid measured sample plots (mask == 1.0),
ignoring unobserved background pixels without treating them as zero biomass.
"""

import torch
import torch.nn as nn

class MaskedHuberLoss(nn.Module):
    """
    Masked Huber Loss for Sparse-Plot Regression.
    Formula:
      L = sum(mask * Huber(pred - target)) / max(sum(mask), 1.0)
    """
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, pred, target, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if target.shape[1] > pred.shape[1]:
            target = target[:, :pred.shape[1], :, :]
        elif pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1], :, :]

        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand_as(pred)

        element_loss = self.huber(pred, target)
        masked_loss = element_loss * mask
        valid_pixels = torch.clamp(mask.sum(), min=1.0)
        return masked_loss.sum() / valid_pixels

class MaskedMSELoss(nn.Module):
    """Masked Mean Squared Error Loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if target.shape[1] > pred.shape[1]:
            target = target[:, :pred.shape[1], :, :]
        elif pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1], :, :]

        if mask.shape[1] != pred.shape[1]:
            mask = mask.expand_as(pred)

        sq_diff = (pred - target) ** 2
        masked_loss = sq_diff * mask
        valid_pixels = torch.clamp(mask.sum(), min=1.0)
        return masked_loss.sum() / valid_pixels
