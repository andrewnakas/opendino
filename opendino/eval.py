"""Frozen-feature classification evals: weighted k-NN and linear probe.

    python -m opendino.eval --backbone opendino:runs/x/last.pt --train data/train --val data/val
    python -m opendino.eval --backbone timm:vit_small_patch14_reg4_dinov2.lvd142m --train ... --val ...

--train / --val are shard directories (scripts/make_shards.py). Every backbone is
evaluated through the same code path so numbers are comparable to each other.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from opendino.data import ShardDataset
from opendino.models import Student
from opendino.teachers import TEACHERS, TimmBackbone
from opendino.train import IMAGENET_MEAN, IMAGENET_STD


class StudentBackbone(nn.Module):
    def __init__(self, ckpt_path: str | None = None, cfg: dict | None = None):
        super().__init__()
        state = None
        if ckpt_path:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg, state = ck["cfg"], ck["student"]
        self.patch_size, self.grid = cfg["patch_size"], cfg["grid"]
        self.model = Student(cfg["arch"], cfg["patch_size"], cfg["grid"] * cfg["patch_size"], cfg["reg_tokens"])
        if state is not None:
            self.model.load_state_dict(state)
        self.embed_dim = self.model.embed_dim
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, grid: int | None = None):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        res = (grid or self.grid) * self.patch_size
        if x.shape[-2:] != (res, res):
            x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
        return self.model((x - self.mean) / self.std)


def load_backbone(spec: str, grid: int | None = None) -> nn.Module:
    kind, _, arg = spec.partition(":")
    if kind == "opendino":
        return StudentBackbone(arg).eval()
    if kind == "random":  # untrained student with the config of a checkpoint, as a floor
        return StudentBackbone(cfg=torch.load(arg, map_location="cpu", weights_only=False)["cfg"]).eval()
    if kind == "timm":
        return TimmBackbone(arg, grid=grid).eval()
    if kind == "teacher":
        return TimmBackbone(TEACHERS[arg], grid=grid).eval()
    raise ValueError(f"unknown backbone spec {spec!r}")


@torch.no_grad()
def extract(backbone: nn.Module, shard_dir, device, grid: int | None, pool: str = "cls",
            batch: int = 128, workers: int = 4):
    feats, labels = [], []
    for img, y, _ in DataLoader(ShardDataset(shard_dir), batch_size=batch, num_workers=workers):
        cls, patch = backbone.encode(img.to(device), grid)
        f = cls if pool == "cls" else torch.cat([cls, patch.mean(dim=(1, 2))], dim=-1)
        feats.append(f.float().cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_accuracy(train_f, train_y, val_f, val_y, k: int = 20, temp: float = 0.07, chunk: int = 1024) -> float:
    """DINO-style weighted k-NN on L2-normalised features. Returns top-1 in percent."""
    train_f, val_f = F.normalize(train_f, dim=-1), F.normalize(val_f, dim=-1)
    num_classes = int(train_y.max()) + 1
    k = min(k, train_f.shape[0])
    correct = 0
    for i in range(0, val_f.shape[0], chunk):
        sim, idx = (val_f[i:i + chunk] @ train_f.T).topk(k, dim=-1)
        votes = torch.zeros(sim.shape[0], num_classes).scatter_add_(1, train_y[idx], (sim / temp).exp())
        correct += (votes.argmax(-1) == val_y[i:i + chunk]).sum().item()
    return 100.0 * correct / val_f.shape[0]


def linear_accuracy(train_f, train_y, val_f, val_y, device, lrs=(1e-3, 3e-3, 1e-2), epochs: int = 30,
                    batch: int = 1024, seed: int = 0) -> dict:
    """Linear probe on standardised features. The learning rate is chosen on a 10% hold-out
    of the training set, never on the validation set."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = train_f.mean(0, keepdim=True), train_f.std(0, keepdim=True).clamp_min(1e-6)
    train_f, val_f = ((train_f - mu) / sd).to(device), ((val_f - mu) / sd).to(device)
    train_y, val_y = train_y.to(device), val_y.to(device)
    n = train_f.shape[0]
    perm = torch.randperm(n, generator=g).to(device)
    held, fit = perm[: n // 10], perm[n // 10:]
    num_classes = int(train_y.max()) + 1

    def run(lr, idx):
        torch.manual_seed(seed)
        head = nn.Linear(train_f.shape[1], num_classes).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
        steps = epochs * max(1, len(idx) // batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(steps):
            b = idx[torch.randint(0, len(idx), (min(batch, len(idx)),), device=device)]
            loss = F.cross_entropy(head(train_f[b]), train_y[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return head

    @torch.no_grad()
    def acc(head, f, y):
        return 100.0 * (head(f).argmax(-1) == y).float().mean().item()

    held_acc = {lr: acc(run(lr, fit), train_f[held], train_y[held]) for lr in lrs}
    best = max(held_acc, key=held_acc.get)
    return {"linear": acc(run(best, perm), val_f, val_y), "linear_lr": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--train", nargs="+", required=True, help="shard dir(s)")
    ap.add_argument("--val", nargs="+", required=True, help="shard dir(s)")
    ap.add_argument("--grid", type=int, default=None, help="token grid side at eval (default: backbone's own)")
    ap.add_argument("--pool", default="cls", choices=["cls", "cls+avg"])
    ap.add_argument("--no-linear", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bb = load_backbone(a.backbone, a.grid).to(dev)
    tf, ty = extract(bb, a.train, dev, a.grid, a.pool, workers=a.workers)
    vf, vy = extract(bb, a.val, dev, a.grid, a.pool, workers=a.workers)
    res = {"backbone": a.backbone, "grid": a.grid, "pool": a.pool, "n_train": len(ty), "n_val": len(vy),
           "knn": knn_accuracy(tf, ty, vf, vy)}
    if not a.no_linear:
        res.update(linear_accuracy(tf, ty, vf, vy, dev))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
