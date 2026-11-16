"""DiLoCo-style distributed training over unreliable, unconnected machines, coordinated through a
Hugging Face repo.

Every worker repeats: download the latest global weights (round r) -> train H local steps on its
own data slice -> upload the pseudo-gradient (global - local). The coordinator averages the
pseudo-gradients that arrive for round r and applies an outer Nesterov-momentum step to produce
round r+1. Workers never talk to each other, so GitHub runners, Kaggle, Colab or volunteers can all
join the same run. Reference: Douillard et al., "DiLoCo: Distributed Low-Communication Training" (2023).

Repo layout (a public HF model repo):
    config.yaml                       training config (model + loss + inner/outer hyperparameters)
    global/round_00012.pt             global weights after 12 outer steps (fp32 state_dict)
    deltas/round_00012/<worker>.pt    pseudo-gradients from workers that started at round 12
    rounds.jsonl                      one line of metrics per outer step (written by the coordinator)

    python -m opendino.diloco init --config configs/diloco_in100_vits16.yaml --repo nakas/opendino-run-pilot
    python -m opendino.diloco work --repo nakas/opendino-run-pilot --worker gh-0 --shards ... \
        --cache dinov2_l=dir1,dir2 --cache siglip2_l=dir3,dir4 --hours 5 [--coordinator --expect 4]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import tempfile
import time
import zlib
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader, RandomSampler

from opendino.data import DistillDataset, TeacherCache
from opendino.train import DEFAULTS, build_model, distill_step, param_groups

DILOCO_DEFAULTS = {
    "inner_steps": 100,      # H: local steps per round
    "inner_lr": 1e-3, "inner_warmup": 10,
    "outer_lr": 0.7, "outer_momentum": 0.9,
    "total_rounds": 2000,    # sets the inner-lr cosine decay
    "round_timeout_min": 45, # coordinator waits this long for stragglers
    "keep_rounds": 3,        # global checkpoints kept besides every 25th
}


# ----------------------------------------------------------------------------- pure math (unit tested)


def pseudo_gradient(global_sd: dict, local_sd: dict) -> dict:
    return {k: (global_sd[k].float() - local_sd[k].float()).half() for k in global_sd if global_sd[k].is_floating_point()}


def outer_step(global_sd: dict, deltas: list[dict], momentum: dict | None, lr: float, mu: float):
    """Average pseudo-gradients and apply one Nesterov SGD step. Returns (new_sd, new_momentum)."""
    new_sd, new_m = {}, {}
    for k, v in global_sd.items():
        if not v.is_floating_point() or k not in deltas[0]:
            new_sd[k] = v
            continue
        g = torch.stack([d[k].float() for d in deltas]).mean(0)
        m = g if momentum is None else mu * momentum[k] + g
        new_m[k] = m
        new_sd[k] = v.float() - lr * (g + mu * m)
    return new_sd, new_m


def inner_lr(cfg: dict, rnd: int, step: int) -> float:
    """Cosine decay over the planned rounds, with a short warm-up at the start of every round
    (each round starts a fresh AdamW, because ephemeral workers cannot keep optimizer state)."""
    t = min(1.0, rnd / max(1, cfg["total_rounds"]))
    base = cfg["min_lr"] + 0.5 * (cfg["inner_lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * t))
    return base * min(1.0, (step + 1) / cfg["inner_warmup"])


# ----------------------------------------------------------------------------- hub helpers


class Hub:
    def __init__(self, repo: str):
        self.api, self.repo = HfApi(), repo

    def files(self, prefix: str = "") -> list[str]:
        return [f for f in self.api.list_repo_files(self.repo) if f.startswith(prefix)]

    def latest_round(self) -> int:
        rounds = [int(f.split("_")[-1].split(".")[0]) for f in self.files("global/round_")]
        return max(rounds) if rounds else -1

    def load(self, path: str):
        local = hf_hub_download(self.repo, path, force_download=True)
        return torch.load(local, map_location="cpu", weights_only=False)

    def save(self, obj, path: str, msg: str) -> None:
        buf = io.BytesIO()
        torch.save(obj, buf)
        for attempt in range(6):
            try:
                self.api.upload_file(path_or_fileobj=buf.getvalue(), path_in_repo=path, repo_id=self.repo, commit_message=msg)
                return
            except Exception as e:  # concurrent commits from many workers can collide; back off and retry
                if attempt == 5:
                    raise
                print(f"upload of {path} failed ({e}); retrying", flush=True)
                time.sleep(15 + random.random() * 45)


def load_cfg(hub: Hub) -> dict:
    cfg = yaml.safe_load(Path(hf_hub_download(hub.repo, "config.yaml", force_download=True)).read_text())
    return {**DEFAULTS, **DILOCO_DEFAULTS, **cfg}


# ----------------------------------------------------------------------------- commands


def cmd_init(a):
    cfg = {**DEFAULTS, **DILOCO_DEFAULTS, **yaml.safe_load(Path(a.config).read_text())}
    for o in a.overrides:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    hub = Hub(a.repo)
    hub.api.create_repo(a.repo, private=False, exist_ok=True)
    if hub.latest_round() >= 0 and not a.force:
        raise SystemExit(f"{a.repo} already has rounds; pass --force to overwrite config only")
    # head shapes depend on the teacher caches; a tiny stand-in reads them from cache metadata
    dims = {t: (v["cls_dim"], v["patch_dim"]) for t, v in cfg["teacher_dims"].items()}
    caches = {t: type("C", (), {"cls_dim": c, "patch_dim": p})() for t, (c, p) in dims.items()}
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg, caches)
    hub.api.upload_file(path_or_fileobj=yaml.safe_dump(cfg).encode(), path_in_repo="config.yaml", repo_id=a.repo,
                        commit_message="config")
    hub.save({"model": model.state_dict(), "round": 0}, "global/round_00000.pt", "round 0 (init)")
    print(f"initialised {a.repo} at round 0")


def cmd_work(a):
    hub = Hub(a.repo)
    cfg = load_cfg(hub)
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    caches = {}
    for spec in a.cache:
        name, dirs = spec.split("=", 1)
        caches[name] = TeacherCache(dirs.split(","))
    weights = {n: float(cfg["teachers"].get(n, {}).get("weight", 1.0)) for n in caches}
    ds = DistillDataset(a.shards, caches)
    model = build_model(cfg, caches).to(dev)
    round_times = []

    while True:
        rnd = hub.latest_round()
        state = hub.load(f"global/round_{rnd:05d}.pt")
        global_sd = state["model"]
        model.load_state_dict(global_sd)
        model.train()
        opt = torch.optim.AdamW(param_groups([model], cfg["weight_decay"]), lr=cfg["inner_lr"])
        gen = torch.Generator().manual_seed(zlib.crc32(f"{a.worker}/{rnd}".encode()))
        sampler = RandomSampler(ds, replacement=True, num_samples=cfg["inner_steps"] * cfg["batch_size"], generator=gen)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler, num_workers=a.loader_workers, drop_last=True)
        tr = time.time()
        losses = []
        for step, (img, targets) in enumerate(loader):
            for g in opt.param_groups:
                g["lr"] = inner_lr(cfg, rnd, step)
            loss, _ = distill_step(model, img, targets, caches, weights, cfg, gen, dev, use_amp)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
            opt.step()
            losses.append(float(loss))
            if step % 10 == 0:
                print(json.dumps({"round": rnd, "step": step, "loss": losses[-1],
                                  "img_per_s": (step + 1) * cfg["batch_size"] / (time.time() - tr)}), flush=True)
        delta = pseudo_gradient(global_sd, {k: v.cpu() for k, v in model.state_dict().items()})
        info = {"worker": a.worker, "round": rnd, "steps": len(losses), "loss": sum(losses) / len(losses),
                "seconds": time.time() - tr}
        hub.save({"delta": delta, "info": info}, f"deltas/round_{rnd:05d}/{a.worker}.pt", f"delta r{rnd} {a.worker}")
        print(json.dumps(info), flush=True)

        if a.coordinator:
            coordinate(hub, cfg, rnd, a.expect)
        else:
            wait_for_round(hub, rnd + 1, cfg["round_timeout_min"] * 60 + 600)
        round_times.append(time.time() - tr)
        budget = a.hours * 3600 - (time.time() - t0)
        if budget < 1.2 * max(round_times[-3:]):
            print(f"stopping: {budget / 60:.0f} min left, rounds take {round_times[-1] / 60:.0f} min")
            break


def wait_for_round(hub: Hub, target: int, timeout: float) -> None:
    t = time.time()
    while hub.latest_round() < target and time.time() - t < timeout:
        time.sleep(30)


def coordinate(hub: Hub, cfg: dict, rnd: int, expect: int) -> None:
    """Wait for `expect` deltas (or the timeout), then write round rnd+1."""
    t = time.time()
    prefix = f"deltas/round_{rnd:05d}/"
    while True:
        names = hub.files(prefix)
        if len(names) >= expect or time.time() - t > cfg["round_timeout_min"] * 60:
            break
        time.sleep(30)
    deltas, infos = [], []
    for n in names:
        d = hub.load(n)
        deltas.append(d["delta"]); infos.append(d["info"])
    state = hub.load(f"global/round_{rnd:05d}.pt")
    momentum = state.get("momentum")
    new_sd, new_m = outer_step(state["model"], deltas, momentum, cfg["outer_lr"], cfg["outer_momentum"])
    nxt = rnd + 1
    hub.save({"model": new_sd, "momentum": {k: v.half() for k, v in new_m.items()}, "round": nxt},
             f"global/round_{nxt:05d}.pt", f"round {nxt} from {len(deltas)} workers")
    rec = {"round": nxt, "workers": len(deltas), "loss": sum(i["loss"] for i in infos) / len(infos),
           "wait_s": time.time() - t, "time": time.time()}
    print(json.dumps(rec), flush=True)
    append_jsonl(hub, "rounds.jsonl", rec)
    prune(hub, nxt, cfg["keep_rounds"])


def append_jsonl(hub: Hub, path: str, rec: dict) -> None:
    try:
        old = Path(hf_hub_download(hub.repo, path, force_download=True)).read_text()
    except Exception:
        old = ""
    hub.api.upload_file(path_or_fileobj=(old + json.dumps(rec) + "\n").encode(), path_in_repo=path,
                        repo_id=hub.repo, commit_message=f"log round {rec['round']}")


def prune(hub: Hub, current: int, keep: int) -> None:
    """Delete old deltas and old global rounds (keeping every 25th); squash history now and then
    so deleted blobs stop counting against storage."""
    old = [f for f in hub.files("deltas/") if int(f.split("/")[1].split("_")[1]) < current - 1]
    old += [f for f in hub.files("global/round_")
            if (r := int(f.split("_")[-1].split(".")[0])) < current - keep and r % 25 != 0]
    if old:
        hub.api.delete_files(hub.repo, old, commit_message=f"prune before round {current}")
    if current % 20 == 0:
        hub.api.super_squash_history(hub.repo)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--config", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("overrides", nargs="*")
    p = sub.add_parser("work")
    p.add_argument("--repo", required=True)
    p.add_argument("--worker", required=True, help="unique id, e.g. gh-3 or kaggle-1")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--cache", action="append", required=True, help="teacher=dir1,dir2,... (repeat per teacher)")
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--coordinator", action="store_true")
    p.add_argument("--expect", type=int, default=1, help="coordinator: deltas to wait for per round")
    p.add_argument("--loader-workers", type=int, default=2)
    a = ap.parse_args()
    {"init": cmd_init, "work": cmd_work}[a.cmd](a)


if __name__ == "__main__":
    main()
