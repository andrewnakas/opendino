"""Convert an image tree into a fixed-resolution JPEG shard (see opendino/data.py).

Each image is resized so its short side equals --res, centre-cropped to res x res,
and re-encoded as JPEG. Classes are sorted directory names of --src.

Layouts:
  train:  --src <root>/<class>/*.JPEG
  val:    --src <flat dir of images> --val-csv LOC_val_solution.csv --classes-from <train root>
          (the Kaggle ImageNet layout: ImageId,PredictionString with the wnid first)
Splitting: --num-parts N --part i writes the i-th contiguous slice of the sorted item list.
Subsets:   --class-stride 10 keeps every 10th class (ImageNet-1k -> ImageNet-100); labels are re-indexed.
"""
import argparse
import csv
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def encode(args):
    path, res, quality = args
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:  # unreadable file: report, do not silently drop
        return None, f"{path}: {e}"
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
    ap.add_argument("--src", required=True, help="ImageFolder root (class subdirectories)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--val-csv", default=None)
    ap.add_argument("--classes-from", default=None, help="dir whose sorted subdirectories define the class list")
    ap.add_argument("--class-stride", type=int, default=1)
    ap.add_argument("--num-parts", type=int, default=1)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    class_root = Path(a.classes_from) if a.classes_from else src
    classes = sorted(d.name for d in class_root.iterdir() if d.is_dir())[:: a.class_stride]
    index = {c: i for i, c in enumerate(classes)}
    items = []
    if a.val_csv:
        with open(a.val_csv) as f:
            for row in csv.DictReader(f):
                wnid = row["PredictionString"].split()[0]
                if wnid in index:
                    items.append((src / f"{row['ImageId']}.JPEG", index[wnid]))
        items.sort(key=lambda t: (t[1], t[0].name))
        if a.limit_per_class:
            kept, counts = [], {}
            for t in items:
                counts[t[1]] = counts.get(t[1], 0) + 1
                if counts[t[1]] <= a.limit_per_class:
                    kept.append(t)
            items = kept
    else:
        for c in classes:
            files = sorted(p for p in (src / c).rglob("*") if p.suffix.lower() in EXTS)
            if a.limit_per_class:
                files = files[: a.limit_per_class]
            items += [(p, index[c]) for p in files]
    total = len(items)
    lo, hi = total * a.part // a.num_parts, total * (a.part + 1) // a.num_parts
    items = items[lo:hi]
    out.mkdir(parents=True, exist_ok=True)

    offsets, labels, failed, pos = [0], [], [], 0
    with open(out / "images.bin", "wb") as f, ProcessPoolExecutor(a.workers) as ex:
        jobs = ((str(p), a.res, a.quality) for p, _ in items)
        for (path, label), (data, err) in zip(items, ex.map(encode, jobs, chunksize=64)):
            if err:
                failed.append(err)
                continue
            f.write(data)
            pos += len(data)
            offsets.append(pos)
            labels.append(label)

    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(out / "labels.npy", np.asarray(labels, dtype=np.int32))
    (out / "meta.json").write_text(json.dumps(
        {"res": a.res, "num_images": len(labels), "classes": classes, "source": str(src), "failed": failed,
         "part": a.part, "num_parts": a.num_parts, "item_range": [lo, hi], "total_items": total}))
    print(f"wrote {len(labels)} images, {pos / 1e6:.1f} MB, {len(failed)} failed -> {out}")
    for e in failed[:20]:
        print("FAILED", e)


if __name__ == "__main__":
    main()
