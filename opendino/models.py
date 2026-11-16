"""Student ViT and per-teacher adaptor heads."""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

ARCHS = {
    "vit_tiny": dict(embed_dim=192, depth=12, num_heads=3),
    "vit_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit_base": dict(embed_dim=768, depth=12, num_heads=12),
}


class Student(nn.Module):
    """ViT with register tokens that returns (cls [B,D], patch grid [B,h,w,D]).

    `mask` is an optional bool tensor [B,h,w]; True patches are replaced by a
    learned mask token before the transformer blocks.
    """

    def __init__(self, arch: str = "vit_small", patch_size: int = 16, img_size: int = 256,
                 reg_tokens: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.arch, self.patch_size = arch, patch_size
        self.vit = VisionTransformer(
            img_size=img_size, patch_size=patch_size, num_classes=0, class_token=True,
            reg_tokens=reg_tokens, no_embed_class=True, dynamic_img_size=True,
            init_values=1e-5, drop_path_rate=drop_path, **ARCHS[arch],
        )
        self.embed_dim = self.vit.embed_dim
        self.num_prefix = 1 + reg_tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        v = self.vit
        x = v.patch_embed(x)
        assert x.ndim == 4, "expected NHWC patch grid from dynamic_img_size patch embed"
        b, h, w, c = x.shape
        if mask is not None:
            x = torch.where(mask[..., None], self.mask_token.to(x.dtype), x)
        x = v._pos_embed(x)
        x = v.norm_pre(x)
        x = v.blocks(x)
        x = v.norm(x)
        return x[:, 0], x[:, self.num_prefix:].reshape(b, h, w, c)


def _mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))


class Adaptor(nn.Module):
    """Maps student features into one teacher's target space."""

    def __init__(self, dim: int, cls_dim: int, patch_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 2 * dim
        self.cls = _mlp(dim, cls_dim, hidden)
        self.patch = _mlp(dim, patch_dim, hidden)

    def forward(self, cls: torch.Tensor, patch: torch.Tensor):
        return self.cls(cls), self.patch(patch)
