"""Image shards and teacher-target caches.

Shard dir:   images.bin (concatenated JPEG bytes), offsets.npy [N+1], labels.npy [N], meta.json
Cache dir:   cls.npy float16 [N,D], patch.npy int8 [N,G,G,K], stats.npz, meta.json
Both are index-aligned: row i of a cache belongs to image i of the shard it was built from.

Large datasets are split into parts (Kaggle persists about 20 GB per notebook output).
Every reader accepts one directory or a list of part directories, concatenated in order.
A cache part also stores the start index of the shard rows it covers in meta.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import decode_jpeg

Q_SCALE = 24.0  # int8 = round(whitened * Q_SCALE); covers about +-5.3 sigma


def as_dirs(roots) -> list[Path]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    return [Path(r) for r in roots]


def _locate(starts: np.ndarray, i: int) -> tuple[int, int]:
    part = int(np.searchsorted(starts, i, side="right")) - 1
    return part, i - int(starts[part])


class ShardDataset(Dataset):
    """Returns (uint8 CHW image at the shard's fixed square resolution, label, global index)."""

    def __init__(self, roots):
        self.roots = as_dirs(roots)
        self.metas = [json.loads((r / "meta.json").read_text()) for r in self.roots]
        self.meta = self.metas[0]
        self.offsets = [np.load(r / "offsets.npy") for r in self.roots]
        self.labels = np.concatenate([np.load(r / "labels.npy") for r in self.roots])
        sizes = [len(o) - 1 for o in self.offsets]
        self.starts = np.cumsum([0] + sizes[:-1])
        self._bins = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        if self._bins is None:  # opened lazily so each DataLoader worker gets its own handles
            self._bins = [np.memmap(r / "images.bin", dtype=np.uint8, mode="r") for r in self.roots]
        p, j = _locate(self.starts, i)
        o = self.offsets[p]
        buf = np.array(self._bins[p][o[j]:o[j + 1]])
        return decode_jpeg(torch.from_numpy(buf)), int(self.labels[i]), i


class TeacherCache:
    """Read side of one teacher's cached targets, possibly split into parts.

    All parts must share one PCA fit. The summary-token mean/variance are pooled
    from per-part sums stored in each part's meta.json.
    """

    def __init__(self, roots, patch_dims: int | None = None):
        """patch_dims: use only the first k PCA components of the patch targets (default: all)."""
        self.roots = as_dirs(roots)
        metas = [json.loads((r / "meta.json").read_text()) for r in self.roots]
        self.meta = {**metas[0], "num_images": sum(m["num_images"] for m in metas)}
        fits = [np.load(r / "pca_fit.npz") for r in self.roots]
        for f in fits[1:]:
            if not np.array_equal(f["pca_components"], fits[0]["pca_components"]):
                raise ValueError("cache parts were built with different PCA fits")
        # Parts are concatenated in the order given. Alignment with the image shards is checked by
        # verify_labels (label fingerprints), so parts built independently on different machines
        # can be combined without knowing global row offsets in advance.
        self.starts = np.cumsum([0] + [m["num_images"] for m in metas[:-1]])
        n = self.meta["num_images"]
        c1 = sum(np.asarray(m["cls_sum"], dtype=np.float64) for m in metas)
        c2 = sum(np.asarray(m["cls_sq"], dtype=np.float64) for m in metas)
        mean = c1 / n
        self.cls_mean = torch.from_numpy(mean).float()
        self.cls_scale = float(np.sqrt(np.clip(c2 / n - mean ** 2, 0, None).mean()))
        k = min(patch_dims or self.meta["pca_dim"], self.meta["pca_dim"])
        self.pca_std = torch.from_numpy(fits[0]["pca_std"][:k]).float()
        self.cls_dim, self.patch_dim, self.grid = self.meta["cls_dim"], k, self.meta["grid"]
        self._cls = self._patch = None

    def verify_labels(self, labels: np.ndarray) -> None:
        """Check each part's labels_sha1 (if recorded) against the shard rows it is paired with."""
        for r, start in zip(self.roots, self.starts):
            m = json.loads((r / "meta.json").read_text())
            if "labels_sha1" not in m:
                continue
            got = hashlib.sha1(labels[start:start + m["num_images"]].astype("int32").tobytes()).hexdigest()
            if got != m["labels_sha1"]:
                raise ValueError(f"cache part {r} does not match its shard rows (label fingerprint differs)")

    def get(self, i: int):
        if self._cls is None:
            self._cls = [np.load(r / "cls.npy", mmap_mode="r") for r in self.roots]
            self._patch = [np.load(r / "patch.npy", mmap_mode="r") for r in self.roots]
        p, j = _locate(self.starts, i)
        patch = np.array(self._patch[p][j][..., :self.patch_dim])
        return torch.from_numpy(np.array(self._cls[p][j])), torch.from_numpy(patch)

    def decode(self, cls: torch.Tensor, patch: torch.Tensor, whiten: bool = False):
        """Raw cached tensors (on any device) -> float training targets with about unit scale."""
        cls = (cls.float() - self.cls_mean.to(cls.device)) / self.cls_scale
        patch = patch.float() / Q_SCALE
        if not whiten:  # restore relative component variances; one global scale keeps the geometry
            std = self.pca_std.to(patch.device)
            patch = patch * (std / std.pow(2).mean().sqrt())
        return cls, patch


class DistillDataset(Dataset):
    def __init__(self, shard_dirs, caches: dict[str, TeacherCache]):
        self.images = ShardDataset(shard_dirs)
        self.caches = caches
        for name, c in caches.items():
            if c.meta["num_images"] != len(self.images):
                raise ValueError(f"cache {name} has {c.meta['num_images']} rows, shard has {len(self.images)}")
            c.verify_labels(self.images.labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img, _, _ = self.images[i]
        return img, {name: c.get(i) for name, c in self.caches.items()}


class ResumableSampler(Sampler[int]):
    """Per-epoch seeded shuffle, sharded by DDP rank, that can skip already-consumed samples."""

    def __init__(self, n: int, seed: int = 0, rank: int = 0, world: int = 1):
        self.n, self.seed, self.rank, self.world = n, seed, rank, world
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch, self.skip = epoch, skip

    def __len__(self) -> int:
        return self.n // self.world - self.skip

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g)[: self.n - self.n % self.world]
        return iter(perm[self.rank::self.world][self.skip:].tolist())
