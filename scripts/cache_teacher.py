"""Cache one teacher's targets for a shard, or for one part of it (see opendino/data.py).

    # fit PCA once (writes <out>/pca_fit.npz), then cache rows [start, end)
    python scripts/cache_teacher.py --shards S1 S2 --teacher dinov2_l --out cache_p0 --end 400000
    python scripts/cache_teacher.py --shards S1 S2 --teacher dinov2_l --out cache_p1 \
        --pca-fit cache_p0/pca_fit.npz --start 400000 --end 800000

Pass 1 fits a PCA of patch tokens on a random subset of the whole shard (skipped with --pca-fit).
Pass 2 writes the summary token (float16) and whitened top-K PCA patch tokens (int8).
Pass 2 is resumable: rerun the same command and it continues from the last flushed chunk.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from opendino.data import Q_SCALE, ShardDataset
from opendino.teachers import TEACHERS, load_teacher


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fit_pca(teacher, ds, dev, amp, loader_kw, n_images: int, k: int) -> dict:
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ds), generator=g)[: min(len(ds), n_images)].sort().values.tolist()
    s1 = s2 = None
    count = 0
    for img, _, _ in DataLoader(Subset(ds, idx), **loader_kw):
        with amp:
            _, patch = teacher.encode(img.to(dev))
        x = patch.reshape(-1, patch.shape[-1]).float().cpu().double()   # MPS has no float64
        s1 = x.sum(0) if s1 is None else s1 + x.sum(0)
        s2 = x.T @ x if s2 is None else s2 + x.T @ x
        count += x.shape[0]
    mean = s1 / count
    evals, evecs = torch.linalg.eigh(s2 / count - torch.outer(mean, mean))
    order = torch.argsort(evals, descending=True)[: min(k, evals.numel())]
    explained = float(evals[order].sum() / evals.sum())
    print(f"PCA fit on {count} tokens: top-{len(order)} explain {explained:.3f} of variance", flush=True)
    return {
        "pca_mean": mean.float().numpy(),
        "pca_components": evecs[:, order].T.float().numpy(),      # [K, D]
        "pca_std": evals[order].clamp_min(1e-12).sqrt().float().numpy(),
        "explained": np.float32(explained),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard dir(s), concatenated in order")
    ap.add_argument("--teacher", required=True, choices=sorted(TEACHERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=0, help="exclusive; 0 = end of shard")
    ap.add_argument("--pca-fit", default=None, help="reuse this pca_fit.npz (required for every part but the first)")
    ap.add_argument("--grid", type=int, default=16, help="token grid side; teacher input = grid * teacher patch size")
    ap.add_argument("--pca-dim", type=int, default=96)
    ap.add_argument("--pca-images", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long (0 = no limit)")
    a = ap.parse_args()

    t0 = time.time()
    dev = pick_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.json").exists():
        print(f"{out} is already complete")
        return
    ds = ShardDataset(a.shards)
    start, end = a.start, a.end or len(ds)
    n = end - start
    assert 0 <= start < end <= len(ds), (start, end, len(ds))
    teacher = load_teacher(a.teacher, a.grid).to(dev)
    amp = torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda")
    loader_kw = dict(batch_size=a.batch, num_workers=a.workers, shuffle=False)

    fit_path = out / "pca_fit.npz"
    if not fit_path.exists():
        if a.pca_fit:
            shutil.copy(a.pca_fit, fit_path)
        else:
            np.savez(fit_path, **fit_pca(teacher, ds, dev, amp, loader_kw, a.pca_images, a.pca_dim))
    st = dict(np.load(fit_path))
    pm = torch.from_numpy(st["pca_mean"]).to(dev)
    pc = torch.from_numpy(st["pca_components"]).to(dev)
    ps = torch.from_numpy(st["pca_std"]).to(dev)
    k, d = pc.shape

    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else None
    mode = "r+" if prog else "w+"
    cls_mm = np.lib.format.open_memmap(out / "cls.npy", mode=mode, dtype=np.float16,
                                       shape=None if prog else (n, teacher.embed_dim))
    patch_mm = np.lib.format.open_memmap(out / "patch.npy", mode=mode, dtype=np.int8,
                                         shape=None if prog else (n, a.grid, a.grid, k))
    done = prog["done"] if prog else 0
    zeros = torch.zeros(teacher.embed_dim, dtype=torch.float64)
    c1 = torch.tensor(prog["cls_sum"], dtype=torch.float64) if prog else zeros.clone()
    c2 = torch.tensor(prog["cls_sq"], dtype=torch.float64) if prog else zeros.clone()

    def flush():
        cls_mm.flush(); patch_mm.flush()
        prog_path.write_text(json.dumps({"done": done, "cls_sum": c1.tolist(), "cls_sq": c2.tolist()}))

    stopped = False
    for bi, (img, _, _) in enumerate(DataLoader(Subset(ds, range(start + done, end)), **loader_kw)):
        with amp:
            cls, patch = teacher.encode(img.to(dev))
        cls, patch = cls.float(), patch.float()
        q = (((patch - pm) @ pc.T) / ps * Q_SCALE).round().clamp(-127, 127).to(torch.int8)
        b = cls.shape[0]
        cls_mm[done:done + b] = cls.cpu().numpy().astype(np.float16)
        patch_mm[done:done + b] = q.cpu().numpy()
        cc = cls.cpu().double()
        c1 += cc.sum(0); c2 += cc.pow(2).sum(0)
        done += b
        if bi % 50 == 49:
            flush()
            rate = done / (time.time() - t0)
            print(f"{done}/{n}  {(time.time() - t0) / 60:.1f} min  ~{(n - done) / max(rate, 1e-9) / 60:.0f} min left", flush=True)
            if a.max_hours and time.time() - t0 > a.max_hours * 3600:
                stopped = True
                break
    flush()
    if stopped or done < n:
        print(f"stopped at {done}/{n}; rerun the same command to continue")
        return

    (out / "meta.json").write_text(json.dumps({
        "teacher": a.teacher, "timm_id": TEACHERS[a.teacher], "grid": a.grid, "start": start, "num_images": n,
        "cls_dim": int(teacher.embed_dim), "patch_token_dim": int(d), "pca_dim": int(k),
        "pca_explained": float(st["explained"]), "q_scale": Q_SCALE, "shards": [str(s) for s in a.shards],
        "cls_sum": c1.tolist(), "cls_sq": c2.tolist(),
    }))
    prog_path.unlink()
    print(f"done: rows [{start}, {end}) -> {out}")


if __name__ == "__main__":
    main()
