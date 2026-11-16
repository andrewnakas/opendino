"""Frozen pretrained backbones behind one interface.

Used both as distillation teachers (clean-room list in TEACHERS only) and as
evaluation backbones (any timm id, including baselines we never train on).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Teachers allowed in training. Every entry must be listed in PROVENANCE.md
# with a verified permissive licence. Do NOT add DINOv3 here.
TEACHERS = {
    "dinov2_s": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_b": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_l": "vit_large_patch14_reg4_dinov2.lvd142m",
    "siglip2_b": "vit_base_patch16_siglip_256.v2_webli",
    "siglip2_l": "vit_large_patch16_siglip_256.v2_webli",
    "siglip2_so400m": "vit_so400m_patch16_siglip_256.v2_webli",
}


class TimmBackbone(nn.Module):
    """encode(uint8 or float images in [0,1]) -> (pooled [B,D], patch [B,h,w,D])."""

    def __init__(self, timm_id: str, grid: int | None = None, pretrained: bool = True):
        super().__init__()
        kwargs = {}
        probe = timm.create_model(timm_id, pretrained=False, num_classes=0)
        self.patch_size = probe.patch_embed.patch_size[0]
        if grid is not None:
            kwargs["img_size"] = grid * self.patch_size
        del probe
        # dynamic_img_size lets one backbone run at any multiple of its patch size (dense evals use 512 px)
        self.model = timm.create_model(timm_id, pretrained=pretrained, num_classes=0, dynamic_img_size=True, **kwargs)
        cfg = self.model.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        self.embed_dim = self.model.num_features
        self.grid = grid

    def preprocess(self, x: torch.Tensor, grid: int | None = None) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        g = grid or self.grid
        if g is not None:
            res = g * self.patch_size
            if x.shape[-2:] != (res, res):
                x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def encode(self, x: torch.Tensor, grid: int | None = None):
        x = self.preprocess(x, grid)
        tokens = self.model.forward_features(x)
        pooled = self.model.forward_head(tokens, pre_logits=True)
        patch = tokens[:, self.model.num_prefix_tokens:]
        b, n, c = patch.shape
        h = x.shape[-2] // self.patch_size
        return pooled, patch.reshape(b, h, n // h, c)


def load_teacher(name: str, grid: int) -> TimmBackbone:
    if name not in TEACHERS:
        raise KeyError(f"{name!r} is not an approved teacher; see PROVENANCE.md. Known: {sorted(TEACHERS)}")
    return TimmBackbone(TEACHERS[name], grid=grid).eval()
