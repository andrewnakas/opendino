# OpenDINOv3 (working name)

A clean-room, Apache-2.0 vision encoder aiming to beat **DINOv3 ViT-S/16** (and later ViT-B/16)
on frozen-feature benchmarks, trained only from permissively licensed teachers.
DINOv3 is used strictly as an evaluation baseline. See [PROVENANCE.md](PROVENANCE.md).

**Method:** multi-teacher feature distillation (DINOv2-L reg4 + SigLIP 2-L) into a ViT-S/16
with registers. Teacher outputs are cached once (summary token in fp16, patch tokens as
whitened int8 PCA), so training only runs the student. Augmentations are patch-aligned crops
and flips applied to the image and the cached feature grid together, colour jitter on the
student only, and iBOT-style patch masking.

## Where things run

| What | Where |
|---|---|
| Unit tests, tiny end-to-end smoke run | GitHub Actions (`ci.yml`, every push, CPU) |
| Sharding, teacher caching, training, evaluation | Kaggle T4 GPUs, launched from GitHub Actions (`kaggle.yml`) |
| Editing code | your machine (no training locally) |

Launch a Kaggle job: Actions → **kaggle** → Run workflow, with e.g. `shards --set in100 --split train`,
or from a terminal:

```bash
gh workflow run kaggle.yml -f args="shards --set in100 --split train"
gh workflow run kaggle.yml -f args="status opendino-in100-shards-train-p0"
gh workflow run kaggle.yml -f args="logs opendino-in100-shards-train-p0"
```

## Pipeline (ImageNet-100 pilot; swap `in100` for `in1k`, which has 3 parts)

| Step | `args` | Hardware |
|---|---|---|
| 1. train shards | `shards --set in100 --split train` | CPU |
| 2. val shards | `shards --set in100 --split val` | CPU |
| 3. teacher caches | `cache --set in100 --teacher dinov2_l` and `... --teacher siglip2_l` | T4 |
| 4. baselines | `eval --set in100 --backbone teacher:dinov2_s` (any `timm:<id>` works) | T4 |
| 5. train, segment 0 | `train --set in100 --config configs/pilot_in100_vits16.yaml --run pilot --segment 0` | 2×T4 |
| 6. resume | same with `--segment 1`, `2`, ... until `final.pt` appears | 2×T4 |
| 7. evaluate | `eval --set in100 --backbone opendino:opendino-train-pilot-s<k>` | T4 |

For `in1k`, steps 1 and 3 run once per part (`--part 0/1/2`); cache part 0 must finish before parts 1 and 2,
which reuse its PCA fit.

Kaggle constraints this design works around: about 30 GPU hours per week, 12-hour sessions
(training stops cleanly at `max_hours` and resumes in the next segment), and about 20 GB of output
per job (hence parts).

## Code map

- `opendino/models.py`: student ViT (timm, registers, mask token) and per-teacher adaptor heads
- `opendino/teachers.py`: approved teachers (clean-room list) and a generic timm backbone wrapper
- `opendino/distill.py`: losses, patch-aligned views, masking, colour jitter
- `opendino/data.py`: JPEG shards and teacher caches, multi-part, resumable sampler
- `opendino/train.py`: training loop (DDP, AMP, checkpoint/resume, time limit)
- `opendino/eval.py`: k-NN and linear probe; `opendino/dense_eval.py`: ADE20k and NYUv2 linear probes
- `scripts/make_shards.py`, `scripts/cache_teacher.py`: data preparation; `scripts/kaggle.py`: job launcher

## Target scorecard (DINOv3 paper, Table 14)

| Model | IN-ReaL | IN-R | ObjectNet | Oxford-H | ADE20k | NYU RMSE ↓ | DAVIS | NAVI | SPair |
|---|---|---|---|---|---|---|---|---|---|
| DINOv3 ViT-S/16 | 87.0 | 60.4 | 50.9 | 49.5 | 47.0 | 0.403 | 72.7 | 56.3 | 50.4 |

Claims will be made per column, using numbers from this repo's harness for both models.

## License

Apache-2.0.
