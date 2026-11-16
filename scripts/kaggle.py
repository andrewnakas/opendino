"""Build and push Kaggle kernels that run this repo's jobs on Kaggle hardware.

Each job is a private Kaggle "script" kernel. The repo code is embedded in the script as a
base64 tarball, so the code never has to be public. Job outputs land in /kaggle/working and
later jobs mount them read-only via `kernel_sources` (no Kaggle datasets needed).

    python scripts/kaggle.py shards --set in100 --split train            [--push]
    python scripts/kaggle.py shards --set in1k --split train --part 0 --num-parts 3
    python scripts/kaggle.py cache  --set in100 --teacher dinov2_l [--part i --num-parts n]
    python scripts/kaggle.py train  --set in100 --config configs/pilot_in100_vits16.yaml --run pilot --segment 0
    python scripts/kaggle.py eval   --set in100 --backbone teacher:dinov2_s
    python scripts/kaggle.py status <slug>       |   python scripts/kaggle.py logs <slug>

Without --push the kernel folder is only written to build/kaggle/<slug> for inspection.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "imagenet-object-localization-challenge"
SETS = {  # name -> (class stride over the 1000 ImageNet classes, default number of train parts)
    "in100": (10, 1),
    "in1k": (1, 3),
}
RES = 256


def username() -> str:
    for p in (Path.home() / ".kaggle" / "kaggle.json", Path.home() / ".config" / "kaggle" / "kaggle.json"):
        if p.exists():
            return json.loads(p.read_text())["username"]
    import os
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    sys.exit("no Kaggle username: set KAGGLE_USERNAME or create ~/.kaggle/kaggle.json")


def code_tarball() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in ["pyproject.toml", "opendino", "scripts", "configs"]:
            tar.add(ROOT / rel, arcname=rel, filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- slugs


def shard_slug(s: str, split: str, part: int = 0) -> str:
    return f"opendino-{s}-shards-{split}" + (f"-p{part}" if split == "train" else "")


def cache_slug(s: str, teacher: str, part: int) -> str:
    return f"opendino-{s}-cache-{teacher.replace('_', '-')}-p{part}"


def train_slug(run: str, segment: int) -> str:
    return f"opendino-train-{run}-s{segment}"


# --------------------------------------------------------------------------- the code that runs on Kaggle

PRELUDE = r'''
import base64, glob, io, json, os, subprocess, sys, tarfile, time
SRC = "/tmp/opendino"
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(SRC)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", SRC], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm>=1.0.15"], check=True)
os.chdir(SRC)
OUT = "/kaggle/working"

def find_input(name, kind="dir"):
    """Locate a mounted input by its final path component (mount layout differs between Kaggle versions)."""
    for depth in range(1, 6):
        hits = sorted(glob.glob("/kaggle/input/" + "/".join(["*"] * depth)))
        hits = [h for h in hits if os.path.basename(h) == name and (os.path.isdir(h) if kind == "dir" else os.path.isfile(h))]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"input {name!r} not found under /kaggle/input")

def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    t = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"[{(time.time() - t) / 60:.1f} min]", flush=True)

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "no GPU")
'''


def job_script(body: str) -> str:
    return f"CODE = {code_tarball()!r}\n" + PRELUDE + "\n" + body


def write_kernel(slug: str, body: str, gpu: bool, sources: list[str], competition: bool = True) -> Path:
    d = ROOT / "build" / "kaggle" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.py").write_text(job_script(body))
    user = username()
    meta = {
        "id": f"{user}/{slug}", "title": slug, "code_file": "job.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "kernel_sources": [f"{user}/{s}" for s in sources],
        "competition_sources": [COMPETITION] if competition else [], "model_sources": [],
    }
    if gpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return d


# --------------------------------------------------------------------------- jobs


def job_shards(a):
    stride, default_parts = SETS[a.set]
    num_parts = a.num_parts or (default_parts if a.split == "train" else 1)
    name = f"{a.set}_{a.split}_{RES}" + (f"_p{a.part}" if a.split == "train" else "")
    if a.split == "train":
        src = 'find_input("CLS-LOC") + "/train"'
        extra = ""
    else:
        src = 'find_input("CLS-LOC") + "/val"'
        extra = ', "--val-csv", find_input("LOC_val_solution.csv", "file"), "--classes-from", find_input("CLS-LOC") + "/train"'
    body = f'''
sh(sys.executable, "scripts/make_shards.py", "--src", {src}, "--out", OUT + "/{name}", "--res", {RES},
   "--quality", 90, "--workers", os.cpu_count(), "--class-stride", {stride},
   "--num-parts", {num_parts}, "--part", {a.part}{extra})
'''
    return write_kernel(shard_slug(a.set, a.split, a.part), body, gpu=False, sources=[])


def shard_inputs(s: str) -> tuple[list[str], list[str]]:
    """(kernel slugs, directory names) of all train shard parts of a set."""
    n = SETS[s][1]
    return [shard_slug(s, "train", i) for i in range(n)], [f"{s}_train_{RES}_p{i}" for i in range(n)]


def job_cache(a):
    slugs, dirs = shard_inputs(a.set)
    n = len(dirs)
    num_parts = a.num_parts or n
    if num_parts != n:
        sys.exit(f"cache parts must match shard parts ({n}) so each part fits in one Kaggle output")
    sources = list(slugs)
    pca = ""
    if a.part > 0:
        sources.append(cache_slug(a.set, a.teacher, 0))
        pca = f', "--pca-fit", find_input("cache_{a.teacher}_p0") + "/pca_fit.npz"'
    body = f'''
shards = [find_input(d) for d in {dirs!r}]
sizes = [json.load(open(s + "/meta.json"))["num_images"] for s in shards]
start = sum(sizes[:{a.part}]); end = start + sizes[{a.part}]
sh(sys.executable, "scripts/cache_teacher.py", "--shards", *shards, "--teacher", "{a.teacher}",
   "--out", OUT + "/cache_{a.teacher}_p{a.part}", "--start", start, "--end", end, "--grid", 16,
   "--pca-dim", {a.pca_dim}, "--batch", 128, "--workers", os.cpu_count(), "--max-hours", 11.0{pca})
'''
    return write_kernel(cache_slug(a.set, a.teacher, a.part), body, gpu=True, sources=sources, competition=False)


def job_train(a):
    cfg = yaml.safe_load((ROOT / a.config).read_text())
    slugs, dirs = shard_inputs(a.set)
    teachers = list(cfg["teachers"])
    n = len(dirs)
    sources = slugs + [cache_slug(a.set, t, i) for t in teachers for i in range(n)]
    if a.segment > 0:
        sources.append(train_slug(a.run, a.segment - 1))
    body = f'''
cfg_teachers = {json.dumps({t: cfg["teachers"][t].get("weight", 1.0) for t in teachers})}
shards = [find_input(d) for d in {dirs!r}]
teachers = {{t: {{"cache": [find_input(f"cache_{{t}}_p{{i}}") for i in range({n})], "weight": w}}
            for t, w in cfg_teachers.items()}}
run = OUT + "/run"
os.makedirs(run, exist_ok=True)
if {a.segment} > 0:
    prev = find_input("run")
    for f in ("last.pt", "log.jsonl"):
        if os.path.exists(prev + "/" + f):
            sh("cp", prev + "/" + f, run + "/" + f)
import torch, yaml
ngpu = max(1, torch.cuda.device_count())
over = ["out=" + run, "shards=" + json.dumps(shards), "teachers=" + json.dumps(teachers),
        "workers=" + str(max(1, os.cpu_count() // ngpu))] + {a.overrides!r}
launcher = ["torchrun", "--standalone", f"--nproc_per_node={{ngpu}}"] if ngpu > 1 else [sys.executable]
sh(*launcher, "-m", "opendino.train", "--config", "{a.config}", *over)
if os.path.exists(run + "/final.pt"):
    sh(sys.executable, "-m", "opendino.eval", "--backbone", "opendino:" + run + "/final.pt",
       "--train", *shards, "--val", find_input("{a.set}_val_{RES}"), "--no-linear")
'''
    sources.append(shard_slug(a.set, "val"))
    return write_kernel(train_slug(a.run, a.segment), body, gpu=True, sources=sources, competition=False)


def job_eval(a):
    slugs, dirs = shard_inputs(a.set)
    sources = slugs + [shard_slug(a.set, "val")]
    extra = []
    ckpt = ""
    if a.backbone.startswith("opendino:"):   # opendino:<train kernel slug>
        sources.append(a.backbone.split(":", 1)[1])
        ckpt = '"opendino:" + find_input("run") + "/final.pt"'
    spec = ckpt or repr(a.backbone)
    grid = f', "--grid", {a.grid}' if a.grid else ""
    body = f'''
sh(sys.executable, "-m", "opendino.eval", "--backbone", {spec}{grid}, "--pool", "{a.pool}",
   "--train", *[find_input(d) for d in {dirs!r}], "--val", find_input("{a.set}_val_{RES}"),
   "--workers", os.cpu_count())
'''
    safe = a.backbone.replace(":", "-").replace("_", "-").replace(".", "-").replace("/", "-")[:40]
    return write_kernel(f"opendino-eval-{a.set}-{safe}", body, gpu=True, sources=sources, competition=False)


ADE_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
NYU_MAT = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
NYU_SPLITS = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"


def job_dense(a):
    """ADE20k / NYUv2 linear probes. Downloads its own data, so no ImageNet access is needed."""
    sources, spec = [], repr(a.backbone)
    if a.backbone.startswith("opendino:"):   # opendino:<train kernel slug>
        sources.append(a.backbone.split(":", 1)[1])
        spec = '"opendino:" + find_input("run") + "/final.pt"'
    if a.task == "ade20k":
        data = (f'sh("curl", "-sSfL", "{ADE_URL}", "-o", "/tmp/ade.zip")\n'
                'sh("unzip", "-q", "/tmp/ade.zip", "-d", "/tmp")\n'
                'args = ["--root", "/tmp/ADEChallengeData2016"]\n')
    else:
        data = (f'sh("curl", "-sSfL", "{NYU_MAT}", "-o", "/tmp/nyu.mat")\n'
                f'sh("curl", "-sSfL", "{NYU_SPLITS}", "-o", "/tmp/splits.mat")\n'
                'args = ["--mat", "/tmp/nyu.mat", "--splits", "/tmp/splits.mat"]\n')
    body = data + (
        'sh(sys.executable, "-m", "pip", "install", "-q", "h5py", "scipy")\n'
        f'sh(sys.executable, "-m", "opendino.dense_eval", "{a.task}", "--backbone", {spec}, *args,\n'
        f'   "--crop", {a.crop}, "--iters", {a.iters}, "--batch", 16, "--workers", os.cpu_count())\n')
    safe = a.backbone.replace(":", "-").replace("_", "-").replace(".", "-").replace("/", "-")[:40]
    return write_kernel(f"opendino-dense-{a.task}-{safe}", body, gpu=True, sources=sources, competition=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="job", required=True)
    for name in ("shards", "cache", "train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=sorted(SETS), required=True)
        p.add_argument("--push", action="store_true")
        p.add_argument("--part", type=int, default=0)
        p.add_argument("--num-parts", type=int, default=0)
        if name == "shards":
            p.add_argument("--split", choices=["train", "val"], default="train")
        if name == "cache":
            p.add_argument("--teacher", required=True)
            p.add_argument("--pca-dim", type=int, default=96)
        if name == "train":
            p.add_argument("--config", required=True)
            p.add_argument("--run", required=True)
            p.add_argument("--segment", type=int, default=0)
            p.add_argument("overrides", nargs="*")
        if name == "eval":
            p.add_argument("--backbone", required=True)
            p.add_argument("--grid", type=int, default=0)
            p.add_argument("--pool", default="cls")
    p = sub.add_parser("dense")
    p.add_argument("--task", choices=["ade20k", "nyu"], required=True)
    p.add_argument("--backbone", required=True)
    p.add_argument("--crop", type=int, default=512, help="use 504 (= 36 x 14) for patch-14 backbones")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--push", action="store_true")
    for name in ("status", "logs"):
        p = sub.add_parser(name)
        p.add_argument("slug")
    a = ap.parse_args()

    if a.job == "status":
        subprocess.run(["kaggle", "kernels", "status", f"{username()}/{a.slug}"], check=True)
        return
    if a.job == "logs":
        out = ROOT / "build" / "kaggle-output" / a.slug
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", f"{username()}/{a.slug}", "-p", str(out),
                        "--file-pattern", r".*\.(log|jsonl|json)$"], check=True)
        for f in sorted(out.rglob("*")):
            if f.is_file():
                print(f"===== {f.relative_to(out)}\n{f.read_text()[-20000:]}")
        return

    d = {"shards": job_shards, "cache": job_cache, "train": job_train, "eval": job_eval, "dense": job_dense}[a.job](a)
    print(f"kernel folder: {d}")
    if a.push:
        subprocess.run(["kaggle", "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
