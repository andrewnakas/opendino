"""Multi-teacher distillation from cached targets.

    python -m opendino.train --config configs/smoke_imagenette.yaml [key=value ...]
    torchrun --nproc_per_node=2 -m opendino.train --config configs/in1k_vits16.yaml

Rerunning the same command resumes from <out>/last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from opendino.data import DistillDataset, ResumableSampler, TeacherCache
from opendino.distill import apply_view, color_jitter, distill_loss, sample_mask, sample_views
from opendino.models import Adaptor, Student

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

DEFAULTS = {
    "arch": "vit_small", "patch_size": 16, "reg_tokens": 4, "drop_path": 0.0,
    "grid": 16,                  # student input = grid * patch_size
    "teachers": {},              # name -> {"cache": dir, "weight": 1.0}
    "whiten": False, "cls_weight": 1.0, "patch_weight": 1.0,
    "min_crop_frac": 0.5, "color_jitter": 0.4, "mask_p": 0.5, "mask_ratio": [0.1, 0.5],
    "epochs": 100, "batch_size": 256, "lr": 1e-3, "min_lr": 1e-6, "weight_decay": 0.05,
    "warmup_steps": 2000, "clip_grad": 3.0, "workers": 4, "seed": 0,
    "log_every": 50, "ckpt_every": 1000, "max_hours": 0.0, "max_steps": 0,
}


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = {**DEFAULTS, **yaml.safe_load(Path(path).read_text())}
    for o in overrides:
        k, v = o.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown config key {k!r}")
        cfg[k] = yaml.safe_load(v)
    return cfg


def lr_at(step: int, total: int, cfg: dict) -> float:
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    t = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(1.0, t)))


def param_groups(modules: list[nn.Module], wd: float):
    decay, no_decay = [], []
    for m in modules:
        for n, p in m.named_parameters():
            (no_decay if p.ndim < 2 or "token" in n or "pos_embed" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


class DistillModel(nn.Module):
    """Student plus all adaptor heads in one module so DDP wraps a single forward."""

    def __init__(self, student: Student, heads: nn.ModuleDict):
        super().__init__()
        self.student, self.heads = student, heads

    def forward(self, x, mask=None):
        cls, patch = self.student(x, mask)
        return {name: head(cls, patch) for name, head in self.heads.items()}


def build_model(cfg: dict, caches: dict[str, TeacherCache]) -> DistillModel:
    student = Student(cfg["arch"], cfg["patch_size"], cfg["grid"] * cfg["patch_size"], cfg["reg_tokens"], cfg["drop_path"])
    heads = nn.ModuleDict({n: Adaptor(student.embed_dim, c.cls_dim, c.patch_dim) for n, c in caches.items()})
    return DistillModel(student, heads)


def distill_step(net, img, targets, caches, weights, cfg, gen, dev, use_amp: bool):
    """Augment one batch, run the student, and return (loss, per-teacher loss parts)."""
    grid = cfg["grid"]
    res = grid * cfg["patch_size"]
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    b = img.shape[0]
    theta = sample_views(b, grid, cfg["min_crop_frac"], generator=gen)
    x = apply_view(img.to(dev, non_blocking=True).float() / 255.0, theta, (res, res))
    if cfg["color_jitter"] > 0:
        x = color_jitter(x, cfg["color_jitter"], generator=gen)
    x = (x - mean) / std
    mask = sample_mask(b, grid, cfg["mask_p"], tuple(cfg["mask_ratio"]), generator=gen).to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        preds = net(x, mask)
    loss, parts = 0.0, {}
    for name, cache in caches.items():
        t_cls, t_patch = cache.decode(targets[name][0].to(dev), targets[name][1].to(dev), cfg["whiten"])
        t_patch = apply_view(t_patch.permute(0, 3, 1, 2), theta, (grid, grid)).permute(0, 2, 3, 1)
        l = distill_loss(*preds[name], t_cls, t_patch)
        loss = loss + weights[name] * (cfg["cls_weight"] * l["cls"] + cfg["patch_weight"] * l["patch"])
        parts[f"{name}/cls"], parts[f"{name}/patch"] = l["cls"].item(), l["patch"].item()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {parts}")
    return loss, parts


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    t_start = time.time()

    ddp = "LOCAL_RANK" in os.environ
    rank, world = 0, 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    main_proc = rank == 0
    out = Path(cfg["out"])
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"] + rank)

    caches = {n: TeacherCache(t["cache"]) for n, t in cfg["teachers"].items()}
    weights = {n: float(t.get("weight", 1.0)) for n, t in cfg["teachers"].items()}
    grid = cfg["grid"]
    for n, c in caches.items():
        if c.grid != grid:
            raise ValueError(f"teacher cache {n} has grid {c.grid}, config grid is {grid}")
    ds = DistillDataset(cfg["shards"], caches)
    sampler = ResumableSampler(len(ds), cfg["seed"], rank, world)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=cfg["workers"],
                        drop_last=True, pin_memory=dev.type == "cuda", persistent_workers=cfg["workers"] > 0)
    steps_per_epoch = (len(ds) // world) // cfg["batch_size"]
    total_steps = steps_per_epoch * cfg["epochs"]

    model = build_model(cfg, caches).to(dev)
    student, heads = model.student, model.heads
    opt = torch.optim.AdamW(param_groups([model], cfg["weight_decay"]), lr=cfg["lr"], betas=(0.9, 0.999))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch = step_in_epoch = step = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"]); heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        epoch, step_in_epoch, step = ck["epoch"], ck["step_in_epoch"], ck["step"]
        if main_proc:
            print(f"resumed from {ckpt_path} at epoch {epoch} step {step}", flush=True)

    net = nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if ddp else model

    def checkpoint():
        if main_proc:
            save_atomic({"student": student.state_dict(), "heads": heads.state_dict(), "opt": opt.state_dict(),
                         "scaler": scaler.state_dict(), "epoch": epoch, "step_in_epoch": step_in_epoch,
                         "step": step, "cfg": cfg}, ckpt_path)

    log_f = open(out / "log.jsonl", "a") if main_proc else None
    gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + rank + step)
    stop = False
    t_log, n_log = time.time(), 0
    net.train()

    while epoch < cfg["epochs"] and not stop:
        sampler.set_epoch(epoch, skip=step_in_epoch * cfg["batch_size"])
        for img, targets in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg)
            b = img.shape[0]
            loss, parts = distill_step(net, img, targets, caches, weights, cfg, gen, dev, use_amp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["clip_grad"]:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            scaler.step(opt); scaler.update()
            step += 1; step_in_epoch += 1; n_log += b * world

            if main_proc and step % cfg["log_every"] == 0:
                rec = {"step": step, "epoch": epoch, "loss": float(loss), "lr": opt.param_groups[0]["lr"],
                       "img_per_s": n_log / (time.time() - t_log), **parts}
                print(json.dumps(rec), flush=True); log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                t_log, n_log = time.time(), 0
            if step % cfg["ckpt_every"] == 0:
                checkpoint()
            over_time = cfg["max_hours"] and time.time() - t_start > cfg["max_hours"] * 3600
            if over_time or (cfg["max_steps"] and step >= cfg["max_steps"]):
                stop = True
                break
        else:
            epoch += 1; step_in_epoch = 0
        checkpoint()

    if main_proc:
        if epoch >= cfg["epochs"]:
            save_atomic({"student": student.state_dict(), "cfg": cfg}, out / "final.pt")
            print(f"finished {epoch} epochs, {step} steps -> {out / 'final.pt'}")
        else:
            print(f"stopped at epoch {epoch} step {step}; rerun to resume")


if __name__ == "__main__":
    main()
