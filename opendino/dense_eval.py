"""Linear dense probes on frozen patch features: ADE20k segmentation and NYUv2 depth.

    python -m opendino.dense_eval ade20k --root ADEChallengeData2016 --backbone opendino:runs/x/final.pt
    python -m opendino.dense_eval nyu --mat nyu_depth_v2_labeled.mat --splits splits.mat --backbone timm:...

Protocol (follows the DINOv2/DINOv3 "linear" setting in spirit; absolute numbers are only
comparable between backbones run through this same harness):
  * the backbone is frozen; a BatchNorm + 1x1 linear layer is trained on last-layer patch tokens
  * logits/depth are bilinearly upsampled to the label resolution
  * ADE20k: 150 classes, random scale 0.5-2.0 + 512 crop + flip, mIoU on the 2000 val images
    with sliding-window inference at short side 512
  * NYUv2: official 795/654 split of the labelled set, depth in [1e-3, 10] m, scale-invariant log loss
    for training, RMSE on the Eigen crop at test time
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from opendino.eval import load_backbone

# ----------------------------------------------------------------------------- datasets


class ADE20k(Dataset):
    """ADEChallengeData2016 layout. Labels: 0 = ignore, 1..150 -> 0..149."""

    def __init__(self, root: str, split: str, crop: int = 512, train: bool = True, limit: int = 0):
        sub = "training" if split == "train" else "validation"
        self.imgs = sorted((Path(root) / "images" / sub).glob("*.jpg"))
        if limit:
            self.imgs = self.imgs[:limit]
        self.ann = Path(root) / "annotations" / sub
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        lab = Image.open(self.ann / (self.imgs[i].stem + ".png"))
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(lab).astype(np.int64)) - 1
        y[y < 0] = 255
        if not self.train:
            return x, y
        return random_scale_crop(x, y, self.crop, ignore=255)


class NYUv2(Dataset):
    """Labelled NYU Depth v2 (.mat, HDF5) with the official train/test split."""

    def __init__(self, mat: str, splits: str, split: str, crop: int = 448, train: bool = True, limit: int = 0):
        import h5py
        from scipy.io import loadmat

        idx = loadmat(splits)["trainNdxs" if split == "train" else "testNdxs"].ravel() - 1
        with h5py.File(mat, "r") as f:   # stored transposed: images [N,3,W,H], depths [N,W,H]
            self.images = np.stack([f["images"][j].transpose(0, 2, 1) for j in idx])
            self.depths = np.stack([f["depths"][j].T for j in idx]).astype(np.float32)
        if limit:
            self.images, self.depths = self.images[:limit], self.depths[:limit]
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i])
        d = torch.from_numpy(self.depths[i])
        if not self.train:
            return x, d
        x, d = random_scale_crop(x, d, self.crop, ignore=0.0, scale=(1.0, 1.5), depth=True)
        return x, d


def random_scale_crop(x, y, crop: int, ignore, scale=(0.5, 2.0), depth: bool = False):
    """Random rescale (relative to short side = crop), random crop with padding, random flip."""
    h, w = x.shape[-2:]
    s = random.uniform(*scale) * crop / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    x = F.interpolate(x[None].float(), (nh, nw), mode="bilinear", antialias=True, align_corners=False)[0]
    y = F.interpolate(y[None, None].float(), (nh, nw), mode="nearest")[0, 0]
    ph, pw = max(0, crop - nh), max(0, crop - nw)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=0.0)
        y = F.pad(y, (0, pw, 0, ph), value=float(ignore))
    top, left = random.randint(0, x.shape[1] - crop), random.randint(0, x.shape[2] - crop)
    x, y = x[:, top:top + crop, left:left + crop], y[top:top + crop, left:left + crop]
    if random.random() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    return x.round().clamp(0, 255).to(torch.uint8), (y if depth else y.long())


# ----------------------------------------------------------------------------- probe


class LinearHead(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, out, 1)

    def forward(self, patch):                      # [B,h,w,D] -> [B,out,h,w]
        return self.conv(self.bn(patch.permute(0, 3, 1, 2).float()))


def features(backbone, x, patch_size: int):
    """uint8 [B,3,H,W] with H,W multiples of the patch size -> patch grid [B,h,w,D]."""
    grid_h, grid_w = x.shape[-2] // patch_size, x.shape[-1] // patch_size
    if grid_h == grid_w:
        return backbone.encode(x, grid_h)[1]
    raise ValueError("square inputs only")


def train_probe(backbone, head, loader, loss_fn, device, iters: int, lr: float, patch_size: int, log_every=200):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    warm = max(1, iters // 20)   # linear warm-up for 5% of steps, then cosine to zero; safe for tiny runs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, iters - warm))))
    head.train()
    it = 0
    while it < iters:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f = features(backbone, x, patch_size)
            out = F.interpolate(head(f), size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            it += 1
            if it % log_every == 0:
                print(json.dumps({"iter": it, "loss": float(loss)}), flush=True)
            if it >= iters:
                break
    head.eval()


@torch.no_grad()
def sliding_predict(backbone, head, x, out_ch: int, crop: int, patch_size: int, device):
    """Resize short side to `crop`, slide square windows (2/3 overlap), average, resize back."""
    h, w = x.shape[-2:]
    s = crop / min(h, w)
    nh, nw = max(crop, round(h * s)), max(crop, round(w * s))
    xs = F.interpolate(x[None].float(), (nh, nw), mode="bilinear", antialias=True, align_corners=False)
    acc = torch.zeros(1, out_ch, nh, nw, device=device)
    cnt = torch.zeros(1, 1, nh, nw, device=device)
    stride = max(1, crop * 2 // 3)
    ys = sorted({*range(0, nh - crop + 1, stride), nh - crop})
    xs_ = sorted({*range(0, nw - crop + 1, stride), nw - crop})
    for t in ys:
        for l in xs_:
            win = xs[:, :, t:t + crop, l:l + crop].round().clamp(0, 255).to(torch.uint8).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f = features(backbone, win, patch_size)
            acc[:, :, t:t + crop, l:l + crop] += F.interpolate(head(f), (crop, crop), mode="bilinear", align_corners=False)
            cnt[:, :, t:t + crop, l:l + crop] += 1
    return F.interpolate(acc / cnt, (h, w), mode="bilinear", align_corners=False)[0]


# ----------------------------------------------------------------------------- tasks


def run_ade20k(a, backbone, device, ps):
    crop = a.crop - a.crop % ps
    train = DataLoader(ADE20k(a.root, "train", crop, True, a.limit), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    head = LinearHead(backbone.embed_dim, 150).to(device)
    train_probe(backbone, head, train, lambda o, y: F.cross_entropy(o, y, ignore_index=255), device, a.iters, a.lr, ps)
    inter = torch.zeros(150, dtype=torch.float64)
    union = torch.zeros(150, dtype=torch.float64)
    val = ADE20k(a.root, "val", crop, False, a.val_limit)
    for i in range(len(val)):
        x, y = val[i]
        pred = sliding_predict(backbone, head, x, 150, crop, ps, device).argmax(0).cpu()
        valid = y != 255
        p, t = pred[valid], y[valid]
        inter += torch.bincount(t[p == t], minlength=150).double()
        area_p = torch.bincount(p, minlength=150).double()
        area_t = torch.bincount(t, minlength=150).double()
        union += area_p + area_t - torch.bincount(t[p == t], minlength=150).double()
    present = union > 0
    return {"ade20k_miou": float((inter[present] / union[present]).mean() * 100), "n_val": len(val)}


def silog(pred, y, lam: float = 0.85):
    """pred: [B,1,H,W] log-depth; y: [B,H,W] metres (0 = invalid)."""
    valid = y > 1e-3
    d = pred[:, 0][valid] - torch.log(y[valid])
    return torch.sqrt((d ** 2).mean() - lam * d.mean() ** 2 + 1e-8) * 10


def run_nyu(a, backbone, device, ps):
    crop = a.crop - a.crop % ps
    train = DataLoader(NYUv2(a.mat, a.splits, "train", crop, True, a.limit), batch_size=a.batch, shuffle=True,
                       num_workers=a.workers, drop_last=True)
    head = LinearHead(backbone.embed_dim, 1).to(device)
    train_probe(backbone, head, train, silog, device, a.iters, a.lr, ps)
    val = NYUv2(a.mat, a.splits, "test", crop, False, a.val_limit)
    se, n = 0.0, 0
    for i in range(len(val)):
        x, d = val[i]
        pred = sliding_predict(backbone, head, x, 1, crop, ps, device)[0].exp().clamp(1e-3, 10).cpu()
        eigen = torch.zeros_like(d, dtype=torch.bool)
        eigen[45:471, 41:601] = True                     # Eigen crop on 480x640
        m = eigen & (d > 1e-3) & (d <= 10)
        se += ((pred[m] - d[m]) ** 2).mean().item()
        n += 1
    return {"nyu_rmse": math.sqrt(se / n), "n_val": n}  # mean of per-image MSE, then sqrt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["ade20k", "nyu"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--root", help="ADEChallengeData2016 directory")
    ap.add_argument("--mat", help="nyu_depth_v2_labeled.mat")
    ap.add_argument("--splits", help="NYU splits.mat")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (smoke tests)")
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    backbone = load_backbone(a.backbone).to(device).eval()
    ps = backbone.patch_size
    res = run_ade20k(a, backbone, device, ps) if a.task == "ade20k" else run_nyu(a, backbone, device, ps)
    print(json.dumps({"backbone": a.backbone, "task": a.task, "crop": a.crop, "iters": a.iters, **res}))


if __name__ == "__main__":
    main()
