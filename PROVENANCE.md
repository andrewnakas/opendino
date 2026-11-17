# Provenance ledger

Every model, dataset and dependency that influences released weights is listed here with its
licence, checked against the primary source (GitHub repo licence API or Hugging Face model card
metadata). Nothing derived from DINOv3 (weights, features, outputs) may be used in training.

Last checked: 2026-11-16.

## Teachers (allowed in `opendino/teachers.py::TEACHERS`)

| Teacher | Weights (timm id) | Weights licence | Code licence | Status |
|---|---|---|---|---|
| DINOv2 ViT-S/B/L reg4 | `vit_*_patch14_reg4_dinov2.lvd142m` | Apache-2.0 (HF `timm/...`, `facebook/dinov2-with-registers-large`) | Apache-2.0 (`facebookresearch/dinov2`) | in use |
| SigLIP 2 B/L/So400m | `vit_*_patch16_siglip_256.v2_webli` | Apache-2.0 (HF `google/siglip2-*`, `timm/...`) | Apache-2.0 (`google-research/big_vision`) | in use |
| Perception Encoder Spatial | `facebook/PE-Spatial-*` | Apache-2.0 (HF card) | Apache-2.0 (`facebookresearch/perception_models`) | candidate |
| Franca | see `valeoai/Franca` | check per-checkpoint before use | Apache-2.0 (`valeoai/Franca`) | candidate |

## Evaluation-only baselines (never used in training)

| Model | Licence | Use |
|---|---|---|
| DINOv3 ViT-S/16, ViT-B/16 | DINOv3 License (Meta, custom) | scorecard baseline only |

## Datasets

| Dataset | Terms | Use |
|---|---|---|
| Imagenette (fast.ai) | subset of ImageNet; ImageNet terms | Mac smoke tests only |
| ImageNet-1k | non-commercial research terms | Phase 3-5 training and eval |
| Openly licensed replacement (e.g. Open Images CC-BY subsets) | to be checked | Phase 6 |

Training on ImageNet-1k is how DINOv2-era Apache-2.0 weights were also produced, but it means the
images themselves are not openly licensed. A fully open release requires Phase 6 data.

## Dependencies

PyTorch (BSD-3), torchvision (BSD-3), timm (Apache-2.0), NumPy (BSD-3), Pillow (MIT-CMU), PyYAML (MIT).

## Storage and redistribution

| Artifact | Where | Visibility | Reason |
|---|---|---|---|
| ImageNet image shards | **not stored**: rebuilt on each runner from the gated `ILSVRC/imagenet-1k` files | none | ImageNet terms forbid redistributing the images; the free HF private quota is also too small |
| Teacher caches (PCA-compressed features) | HF dataset `nakas/opendino-caches` | public | features, not images; public ImageNet embedding sets are common practice, but this is a grey area and will be revisited before any release |
| Student checkpoints | HF model repo | public | Apache-2.0 release target |

ImageNet source for the GitHub swarm: the gated `ILSVRC/imagenet-1k` dataset on Hugging Face, accessed with the
maintainer's own token after accepting its terms.
