"""
Test-Time Augmentation (TTA) via D4 Dihedral Symmetry Group.

Computes predictions across all 8 spatial dihedral transformations
(rotations and reflections), inverts them to canonical orientation,
and computes the ensemble average for smoothed, boundary-consistent predictions.
"""

import torch

def predict_with_d4_tta(model, x, use_tta=True, return_derived_targets=False):
    """
    Args:
        model: Callable model or nn.Module
        x: Input tensor (B, C, H, W)
        use_tta: bool, if False performs standard single-pass inference
        return_derived_targets: bool, if True invokes predict_full_targets()
    Returns:
        Ensemble prediction tensor (B, K, H, W)
    """
    if not use_tta:
        if return_derived_targets and hasattr(model, "predict_full_targets"):
            return model.predict_full_targets(x)
        return model(x)

    def forward_fn(t_in):
        if return_derived_targets and hasattr(model, "predict_full_targets"):
            return model.predict_full_targets(t_in)
        return model(t_in)

    preds = []

    # 0. Identity
    preds.append(forward_fn(x))

    # 1. Rot90
    x_rot1 = torch.rot90(x, k=1, dims=(-2, -1))
    p_rot1 = forward_fn(x_rot1)
    preds.append(torch.rot90(p_rot1, k=-1, dims=(-2, -1)))

    # 2. Rot180
    x_rot2 = torch.rot90(x, k=2, dims=(-2, -1))
    p_rot2 = forward_fn(x_rot2)
    preds.append(torch.rot90(p_rot2, k=-2, dims=(-2, -1)))

    # 3. Rot270
    x_rot3 = torch.rot90(x, k=3, dims=(-2, -1))
    p_rot3 = forward_fn(x_rot3)
    preds.append(torch.rot90(p_rot3, k=-3, dims=(-2, -1)))

    # 4. Horizontal Flip
    x_flip_h = torch.flip(x, dims=[-1])
    p_flip_h = forward_fn(x_flip_h)
    preds.append(torch.flip(p_flip_h, dims=[-1]))

    # 5. Vertical Flip
    x_flip_v = torch.flip(x, dims=[-2])
    p_flip_v = forward_fn(x_flip_v)
    preds.append(torch.flip(p_flip_v, dims=[-2]))

    # 6. Horizontal Flip + Rot90
    x_fh_r1 = torch.rot90(x_flip_h, k=1, dims=(-2, -1))
    p_fh_r1 = forward_fn(x_fh_r1)
    p_fh_r1_inv = torch.flip(torch.rot90(p_fh_r1, k=-1, dims=(-2, -1)), dims=[-1])
    preds.append(p_fh_r1_inv)

    # 7. Vertical Flip + Rot90
    x_fv_r1 = torch.rot90(x_flip_v, k=1, dims=(-2, -1))
    p_fv_r1 = forward_fn(x_fv_r1)
    p_fv_r1_inv = torch.flip(torch.rot90(p_fv_r1, k=-1, dims=(-2, -1)), dims=[-2])
    preds.append(p_fv_r1_inv)

    return torch.stack(preds, dim=0).mean(dim=0)
