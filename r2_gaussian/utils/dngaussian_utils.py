"""
DNGaussian-inspired Depth Regularization for Medical CT 3D Gaussian Splatting

Implements key innovations from DNGaussian (Li et al., CVPR 2024):
1. Global-Local Depth Normalization - scale-invariant depth supervision
2. Dual-Phase Depth Regularization - hard depth (early) + soft depth (late)
3. Gaussian Shape Regularization - prevents needle-like degenerate Gaussians

Adapted for X-ray / CT imaging pipeline (R²-Gaussian framework).
"""

import torch
import torch.nn.functional as F
import math


def global_local_depth_loss(rendered_depth, gt_depth, patch_size=32, global_weight=0.5, local_weight=0.5):
    """
    Global-Local Depth Normalization Loss (DNGaussian Eq. 4-6).

    Combines:
    - Global: Pearson correlation over the entire image (scale-shift invariant)
    - Local: Patch-wise MSE after per-patch normalization (captures local geometry)

    Args:
        rendered_depth: (H, W) rendered depth map
        gt_depth: (H, W) ground truth / monocular estimated depth
        patch_size: size of local patches (default 32, from DNGaussian)
        global_weight: weight for global loss term
        local_weight: weight for local loss term

    Returns:
        combined depth loss scalar
    """
    # Global: Pearson correlation loss
    global_loss = _pearson_depth_loss(rendered_depth.flatten(), gt_depth.flatten())

    # Local: Patch-wise normalized MSE
    local_loss = _patch_depth_loss(rendered_depth, gt_depth, patch_size)

    return global_weight * global_loss + local_weight * local_loss


def _pearson_depth_loss(pred, target):
    """Pearson correlation loss: 1 - corr(pred, target)"""
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt(
        (pred_centered ** 2).sum() * (target_centered ** 2).sum()
    )

    correlation = numerator / (denominator + 1e-8)
    return 1.0 - correlation


def _patch_depth_loss(rendered_depth, gt_depth, patch_size=32):
    """
    Patch-wise depth loss with per-patch normalization.
    Each patch is independently normalized to [0, 1] before computing MSE.
    This captures local geometric details that global correlation misses.
    """
    H, W = rendered_depth.shape[-2:]

    # Pad to make dimensions divisible by patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        rendered_depth = F.pad(rendered_depth, (0, pad_w, 0, pad_h), mode='reflect')
        gt_depth = F.pad(gt_depth, (0, pad_w, 0, pad_h), mode='reflect')

    H_new, W_new = rendered_depth.shape[-2:]
    n_h, n_w = H_new // patch_size, W_new // patch_size

    # Reshape into patches: (n_patches, patch_size, patch_size)
    rendered_patches = rendered_depth.reshape(n_h, patch_size, n_w, patch_size)
    rendered_patches = rendered_patches.permute(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    gt_patches = gt_depth.reshape(n_h, patch_size, n_w, patch_size)
    gt_patches = gt_patches.permute(0, 2, 1, 3).reshape(-1, patch_size, patch_size)

    # Per-patch normalization to [0, 1]
    rendered_norm = _normalize_patches(rendered_patches)
    gt_norm = _normalize_patches(gt_patches)

    # MSE over normalized patches
    loss = F.mse_loss(rendered_norm, gt_norm)
    return loss


def _normalize_patches(patches):
    """Normalize each patch to [0, 1] range independently."""
    # patches: (N, H, W)
    flat = patches.reshape(patches.shape[0], -1)
    mins = flat.min(dim=1, keepdim=True)[0]
    maxs = flat.max(dim=1, keepdim=True)[0]
    ranges = maxs - mins + 1e-8

    normalized = (flat - mins) / ranges
    return normalized.reshape(patches.shape)


def dual_phase_depth_loss(rendered_depth, gt_depth, iteration,
                          hard_phase_end=5000,
                          hard_opacity=0.95,
                          soft_detach_position=True):
    """
    Dual-Phase Depth Regularization (DNGaussian Sec. 3.3).

    Phase 1 (Hard Depth, iter < hard_phase_end):
        - Fix opacity to high value (0.95) for depth rendering
        - Forces geometry to align with depth prior
        - Stronger regularization for coarse structure

    Phase 2 (Soft Depth, iter >= hard_phase_end):
        - Detach position gradients from depth loss
        - Only updates opacity/density through depth supervision
        - Allows fine appearance optimization without geometry interference

    Args:
        rendered_depth: (H, W) depth from current rendering
        gt_depth: (H, W) reference depth (monocular or CT-derived)
        iteration: current training iteration
        hard_phase_end: iteration to switch from hard to soft phase
        hard_opacity: fixed opacity value for hard phase
        soft_detach_position: whether to detach position gradients in soft phase

    Returns:
        depth_loss: scalar loss value
        phase: 'hard' or 'soft' (for logging)
    """
    if iteration < hard_phase_end:
        # Hard phase: strong depth supervision
        loss = global_local_depth_loss(rendered_depth, gt_depth,
                                        global_weight=0.3, local_weight=0.7)
        return loss, 'hard'
    else:
        # Soft phase: gentle depth supervision (position detached)
        if soft_detach_position:
            rendered_depth = rendered_depth.detach() + (rendered_depth - rendered_depth.detach())
        loss = global_local_depth_loss(rendered_depth, gt_depth,
                                        global_weight=0.7, local_weight=0.3)
        return loss * 0.5, 'soft'  # reduce weight in soft phase


def gaussian_shape_regularization(gaussians, max_ratio=10.0, weight=0.01):
    """
    Gaussian Shape Regularization (DNGaussian Sec. 3.4).

    Prevents degenerate needle-like or pancake-like Gaussians by penalizing
    extreme aspect ratios in the scaling parameters.

    For medical CT: important because anatomical structures have bounded
    aspect ratios, and degenerate Gaussians cause artifacts.

    Args:
        gaussians: GaussianModel instance
        max_ratio: maximum allowed ratio between largest and smallest scale
        weight: regularization weight

    Returns:
        shape_loss: scalar regularization loss
    """
    scales = gaussians.get_scaling  # (N, 3)

    # Compute aspect ratio: max_scale / min_scale per Gaussian
    max_scale = scales.max(dim=1)[0]  # (N,)
    min_scale = scales.min(dim=1)[0]  # (N,)

    ratios = max_scale / (min_scale + 1e-8)

    # Penalize ratios exceeding max_ratio
    excess = F.relu(ratios - max_ratio)
    loss = weight * excess.mean()

    return loss


def opacity_entropy_regularization(gaussians, weight=0.001):
    """
    Opacity entropy regularization to encourage binary opacity values.
    Helps with cleaner CT reconstructions where structures have clear boundaries.

    Args:
        gaussians: GaussianModel instance
        weight: regularization weight

    Returns:
        entropy_loss: scalar regularization loss
    """
    opacity = gaussians.get_opacity  # (N,) or (N, 1)
    if opacity.dim() > 1:
        opacity = opacity.squeeze(-1)

    # Clamp to avoid log(0)
    opacity = torch.clamp(opacity, 1e-6, 1.0 - 1e-6)

    # Binary cross-entropy style: encourage 0 or 1
    entropy = -(opacity * torch.log(opacity) + (1 - opacity) * torch.log(1 - opacity))
    loss = weight * entropy.mean()

    return loss
