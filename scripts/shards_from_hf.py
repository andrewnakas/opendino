"""Build one shard part from the official Hugging Face ImageNet-1k Parquet files.

Worker i of W takes every W-th Parquet file (i, i+W, ...), keeps classes with label % stride == 0
(labels follow sorted WordNet ids, so stride 10 gives the same ImageNet-100 as make_shards.py),
re-indexes them to label // stride, and writes a fixed-resolution JPEG shard (see opendino/data.py).

    python scripts/shards_from_hf.py --split train --worker 3 --workers 20 --class-stride 10 --out out/part_003
"""
import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "ILSVRC/imagenet-1k"


def encode(args):
    data, res, quality = args
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:  # unreadable image: report, do not silently drop
        return None, str(e)
    w, h = im.size
    s = res / min(w, h)
    im = im.resize((max(res, round(w * s)), max(res, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - res) // 2, (h - res) // 2
    buf = io.BytesIO()
    im.crop((l, t, l + res, t + res)).save(buf, "JPEG", quality=quality)
    return buf.getvalue(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--class-stride", type=int, default=1)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None, help="where Parquet files are downloaded (deleted after use)")
    a = ap.parse_args()

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"data/{a.split}-") and f.endswith(".parquet"))
    mine = files[a.worker::a.workers]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    offsets, labels, failed, pos, t0 = [0], [], [], 0, time.time()
    with open(out / "images.bin", "wb") as fout, ProcessPoolExecutor(os.cpu_count()) as ex:
        for fname in mine:
            path = hf_hub_download(REPO, fname, repo_type="dataset", cache_dir=a.cache_dir)
            table = pq.read_table(path, columns=["image", "label"])
            lab = np.asarray(table.column("label").to_pylist())
            keep = np.nonzero(lab % a.class_stride == 0)[0]
            imgs = table.column("image").to_pylist()
            jobs = [(imgs[k]["bytes"], a.res, a.quality) for k in keep]
            for k, (data, err) in zip(keep, ex.map(encode, jobs, chunksize=32)):
                if err:
                    failed.append(f"{fname}#{k}: {err}")
                    continue
                fout.write(data)
                pos += len(data)
                offsets.append(pos)
                labels.append(int(lab[k]) // a.class_stride)
            del table, imgs
            Path(os.path.realpath(path)).unlink(missing_ok=True)   # keep runner disk usage low
            print(f"{fname}: kept {len(keep)}, total {len(labels)}, {(time.time() - t0) / 60:.1f} min", flush=True)

    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "labels.npy", np.asarray(labels, dtype=np.int32))
    (out / "meta.json").write_text(json.dumps({
        "res": a.res, "num_images": len(labels), "source": REPO, "split": a.split, "files": mine,
        "class_stride": a.class_stride, "num_classes": 1000 // a.class_stride,
        "worker": a.worker, "workers": a.workers, "failed": failed}))
    print(f"wrote {len(labels)} images, {pos / 1e6:.1f} MB, {len(failed)} failed -> {out}")


if __name__ == "__main__":
    main()
