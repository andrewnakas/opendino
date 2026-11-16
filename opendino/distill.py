"""Losses, masking and feature-consistent augmentation.

The teacher targets are cached for one fixed base view per image. At train
time one affine view (patch-aligned crop + optional flip) is sampled per
image and applied to both the image and the cached teacher patch grid, so the
student input and its targets stay spatially aligned.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_views(batch: int, grid: int, min_frac: float = 0.5, flip: bool = True,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Affine thetas [B,2,3] for patch-aligned square crops of a grid x grid base view."""
    g_min = max(1, int(round(min_frac * grid)))
    rand = lambda: torch.rand(batch, generator=generator)
    g = (g_min + torch.floor(rand() * (grid - g_min + 1))).clamp(max=grid)
    ox = torch.floor(rand() * (grid - g + 1))
    oy = torch.floor(rand() * (grid - g + 1))
    scale = g / grid
    cx = (ox + g / 2) / grid * 2 - 1
    cy = (oy + g / 2) / grid * 2 - 1
    sx = scale * (torch.where(rand() < 0.5, -1.0, 1.0) if flip else 1.0)
    theta = torch.zeros(batch, 2, 3)
    theta[:, 0, 0], theta[:, 0, 2] = sx, cx
    theta[:, 1, 1], theta[:, 1, 2] = scale, cy
    return theta


def apply_view(x: torch.Tensor, theta: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Resample NCHW tensor x under theta to out_hw."""
    grid = F.affine_grid(theta.to(x.device, x.dtype), (x.shape[0], x.shape[1], *out_hw), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def color_jitter(x: torch.Tensor, strength: float = 0.4, gray_p: float = 0.1,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Student-only photometric augmentation on float images in [0,1] (NCHW)."""
    b = x.shape[0]
    r = lambda: (torch.rand(b, 1, 1, 1, generator=generator).to(x.device) * 2 - 1) * strength
    x = x * (1 + r())                                   # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + r()) + mean                   # contrast
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * (1 + r()) + gray                   # saturation
    to_gray = (torch.rand(b, 1, 1, 1, generator=generator) < gray_p).to(x.device)
    x = torch.where(to_gray, x.mean(dim=1, keepdim=True).expand_as(x), x)
    return x.clamp(0, 1)


def sample_mask(batch: int, grid: int, sample_p: float = 0.5, ratio: tuple[float, float] = (0.1, 0.5),
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Bool [B,grid,grid]. A fraction sample_p of images get a random patch mask."""
    r = torch.rand(batch, 1, 1, generator=generator) * (ratio[1] - ratio[0]) + ratio[0]
    on = torch.rand(batch, 1, 1, generator=generator) < sample_p
    return (torch.rand(batch, grid, grid, generator=generator) < r) & on


def distill_loss(pred_cls, pred_patch, tgt_cls, tgt_patch) -> dict[str, torch.Tensor]:
    """Cosine + magnitude terms for the summary token and the patch grid (last dim = channels)."""
    pred_cls, pred_patch = pred_cls.float(), pred_patch.float()
    l_cls = (1 - F.cosine_similarity(pred_cls, tgt_cls, dim=-1)).mean() + F.smooth_l1_loss(pred_cls, tgt_cls)
    l_patch = (1 - F.cosine_similarity(pred_patch, tgt_patch, dim=-1)).mean() + F.mse_loss(pred_patch, tgt_patch)
    return {"cls": l_cls, "patch": l_patch}
