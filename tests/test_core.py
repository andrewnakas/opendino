import io
import json

import numpy as np
import torch
from PIL import Image

from opendino.data import Q_SCALE, DistillDataset, ResumableSampler, ShardDataset, TeacherCache
from opendino.distill import apply_view, color_jitter, distill_loss, sample_mask, sample_views
from opendino.eval import knn_accuracy, linear_accuracy
from opendino.models import Adaptor, Student
from opendino.train import DEFAULTS, lr_at


def test_student_shapes_and_dynamic_resolution():
    s = Student("vit_tiny", img_size=64).eval()
    for res, g in [(64, 4), (96, 6)]:
        cls, patch = s(torch.randn(2, 3, res, res))
        assert cls.shape == (2, 192) and patch.shape == (2, g, g, 192)


def test_grid_alignment_between_patch_sizes():
    # a /14 teacher at 224 px and a /16 student at 256 px must give the same 16x16 grid
    assert 224 // 14 == 256 // 16 == 16


def test_mask_changes_only_via_mask_token():
    s = Student("vit_tiny", img_size=64).eval()
    x = torch.randn(1, 3, 64, 64)
    none = s(x)[1]
    zero_mask = s(x, torch.zeros(1, 4, 4, dtype=torch.bool))[1]
    assert torch.allclose(none, zero_mask, atol=1e-6)
    torch.nn.init.normal_(s.mask_token)
    full = s(x, torch.ones(1, 4, 4, dtype=torch.bool))[1]
    assert not torch.allclose(none, full, atol=1e-3)


def test_identity_view_is_exact():
    x = torch.rand(2, 3, 32, 32)
    theta = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).repeat(2, 1, 1)
    assert torch.allclose(apply_view(x, theta, (32, 32)), x, atol=1e-5)


def test_flip_and_crop_consistent_between_image_and_features():
    grid, p = 8, 4
    feat = torch.rand(3, 5, grid, grid)
    img = feat.repeat_interleave(p, 2).repeat_interleave(p, 3)  # image whose patches equal the features
    flip = torch.tensor([[-1.0, 0, 0], [0, 1.0, 0]]).repeat(3, 1, 1)
    assert torch.allclose(apply_view(feat, flip, (grid, grid)), feat.flip(-1), atol=1e-5)
    g = torch.Generator().manual_seed(0)
    theta = sample_views(3, grid, min_frac=0.5, generator=g)
    side = (theta[:, 1, 1] * grid).round().int()
    assert ((side >= 4) & (side <= grid)).all()
    # crop exactly `side` patches to `side` tokens: image patches must still equal the feature tokens
    for i in range(3):
        s = int(side[i])
        f = apply_view(feat[i:i + 1], theta[i:i + 1], (s, s))
        im = apply_view(img[i:i + 1], theta[i:i + 1], (s * p, s * p))
        pooled = torch.nn.functional.avg_pool2d(im, p)
        assert torch.allclose(pooled, f, atol=1e-4)


def test_mask_and_jitter_ranges():
    g = torch.Generator().manual_seed(0)
    m = sample_mask(64, 16, sample_p=0.5, ratio=(0.1, 0.5), generator=g)
    per = m.float().mean(dim=(1, 2))
    assert m.dtype == torch.bool and (per == 0).any() and (per > 0).any() and per.max() < 0.7
    x = color_jitter(torch.rand(4, 3, 16, 16), generator=g)
    assert x.min() >= 0 and x.max() <= 1


def test_loss_zero_at_target_and_backprop():
    cls, patch = torch.randn(2, 16), torch.randn(2, 4, 4, 8)
    l = distill_loss(cls, patch, cls, patch)
    assert l["cls"] < 1e-5 and l["patch"] < 1e-5
    s, h = Student("vit_tiny", img_size=64), Adaptor(192, 16, 8)
    out = distill_loss(*h(*s(torch.randn(2, 3, 64, 64))), cls, patch)
    (out["cls"] + out["patch"]).backward()
    assert s.vit.patch_embed.proj.weight.grad.abs().sum() > 0


def _make_shard(root, n=12, res=32):
    root.mkdir()
    offsets, pos = [0], 0
    with open(root / "images.bin", "wb") as f:
        for i in range(n):
            buf = io.BytesIO()
            Image.fromarray(np.random.randint(0, 255, (res, res, 3), dtype=np.uint8)).save(buf, "JPEG")
            f.write(buf.getvalue()); pos += len(buf.getvalue()); offsets.append(pos)
    np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(root / "labels.npy", (np.arange(n) % 3).astype(np.int32))
    (root / "meta.json").write_text(json.dumps({"res": res, "num_images": n}))


def _make_cache(c, cls, white, std, start=0, components=None):
    c.mkdir()
    n, grid, _, k = white.shape
    np.save(c / "cls.npy", cls)
    np.save(c / "patch.npy", (white * Q_SCALE).round().astype(np.int8))
    comps = np.eye(k, 8, dtype=np.float32) if components is None else components
    np.savez(c / "pca_fit.npz", pca_mean=np.zeros(8, np.float32), pca_components=comps, pca_std=std,
             explained=np.float32(1.0))
    c64 = cls.astype(np.float64)
    (c / "meta.json").write_text(json.dumps({
        "num_images": n, "start": start, "cls_dim": cls.shape[1], "pca_dim": k, "grid": grid,
        "cls_sum": c64.sum(0).tolist(), "cls_sq": (c64 ** 2).sum(0).tolist()}))


def test_shard_and_cache_roundtrip(tmp_path):
    n, grid, k, d = 12, 2, 4, 6
    _make_shard(tmp_path / "shard", n)
    ds = ShardDataset(tmp_path / "shard")
    img, label, idx = ds[5]
    assert img.shape == (3, 32, 32) and img.dtype == torch.uint8 and label == 2 and idx == 5

    cls = np.random.randn(n, d).astype(np.float16)
    white = np.random.randn(n, grid, grid, k).clip(-4, 4)
    std = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)
    _make_cache(tmp_path / "cache", cls, white, std)
    cache = TeacherCache(tmp_path / "cache")
    img, tg = DistillDataset(tmp_path / "shard", {"t": cache})[3]
    t_cls, t_patch = cache.decode(*tg["t"], whiten=True)
    assert np.allclose(t_patch.numpy(), white[3], atol=0.5 / Q_SCALE + 1e-6)   # quantisation error bound
    _, unwhite = cache.decode(*tg["t"], whiten=False)
    scale = std / np.sqrt((std ** 2).mean())            # per-component rescale with one global scale
    assert np.allclose(unwhite.numpy(), t_patch.numpy() * scale, atol=1e-5)
    assert abs(float(t_cls.std()) - 1.0) < 0.5           # summary target has about unit scale


def test_multipart_shards_and_caches_match_single(tmp_path):
    n, grid, k, d = 12, 2, 4, 6
    for name, m in [("a", 5), ("b", 7)]:
        _make_shard(tmp_path / f"shard_{name}", m)
    parts = [tmp_path / "shard_a", tmp_path / "shard_b"]
    ds = ShardDataset(parts)
    assert len(ds) == 12
    assert torch.equal(ds[7][0], ShardDataset(parts[1])[2][0]) and ds[7][2] == 7

    cls = np.random.randn(n, d).astype(np.float16)
    white = np.random.randn(n, grid, grid, k).clip(-4, 4)
    std = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)
    _make_cache(tmp_path / "whole", cls, white, std)
    _make_cache(tmp_path / "p0", cls[:5], white[:5], std, start=0)
    _make_cache(tmp_path / "p1", cls[5:], white[5:], std, start=5)
    whole, split = TeacherCache(tmp_path / "whole"), TeacherCache([tmp_path / "p0", tmp_path / "p1"])
    assert split.meta["num_images"] == 12
    assert torch.allclose(whole.cls_mean, split.cls_mean, atol=1e-5) and abs(whole.cls_scale - split.cls_scale) < 1e-5
    for i in (0, 4, 5, 11):
        assert all(torch.equal(a, b) for a, b in zip(whole.get(i), split.get(i)))
    DistillDataset(parts, {"t": split})  # sizes agree
    import pytest
    with pytest.raises(ValueError):
        TeacherCache([tmp_path / "p1", tmp_path / "p0"])     # wrong order is rejected
    _make_cache(tmp_path / "other", cls[5:], white[5:], std, start=5, components=np.eye(k, 8, 1, dtype=np.float32))
    with pytest.raises(ValueError):
        TeacherCache([tmp_path / "p0", tmp_path / "other"])  # different PCA fits are rejected


def test_sampler_resume_and_rank_split():
    s = ResumableSampler(100, seed=1)
    s.set_epoch(3)
    full = list(s)
    s.set_epoch(3, skip=40)
    assert list(s) == full[40:] and sorted(full) == list(range(100))
    a, b = ResumableSampler(100, 1, 0, 2), ResumableSampler(100, 1, 1, 2)
    assert not set(a) & set(b) and len(set(a) | set(b)) == 100


def test_lr_schedule_endpoints():
    cfg = {**DEFAULTS, "lr": 1.0, "min_lr": 0.0, "warmup_steps": 10}
    assert lr_at(0, 100, cfg) == 0.1 and abs(lr_at(9, 100, cfg) - 1.0) < 1e-9
    assert lr_at(99, 100, cfg) < 0.01 and lr_at(10, 100, cfg) == 1.0


def test_knn_and_linear_on_separable_features():
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(5, 16, generator=g) * 5
    def draw(n):
        y = torch.randint(0, 5, (n,), generator=g)
        return centers[y] + torch.randn(n, 16, generator=g), y
    tf, ty = draw(500); vf, vy = draw(200)
    assert knn_accuracy(tf, ty, vf, vy) > 95
    assert linear_accuracy(tf, ty, vf, vy, "cpu", epochs=20, batch=128)["linear"] > 95
    assert knn_accuracy(tf, ty[torch.randperm(500, generator=g)], vf, vy) < 50  # shuffled labels -> near chance


def test_backbones_give_expected_grids_at_any_resolution():
    """Every architecture we train from or compare against must yield [B, g, g, D] patch grids."""
    import pytest
    from opendino.teachers import TEACHERS, TimmBackbone

    ids = [TEACHERS["dinov2_s"], TEACHERS["siglip2_b"], "vit_small_patch16_dinov3.lvd1689m"]
    for timm_id in ids:
        bb = TimmBackbone(timm_id, pretrained=False).eval()
        for grid in (8, 16, 32):
            pooled, patch = bb.encode(torch.randint(0, 255, (1, 3, 64, 64), dtype=torch.uint8), grid)
            assert patch.shape[:3] == (1, grid, grid), (timm_id, grid, patch.shape)
            assert pooled.shape == (1, bb.embed_dim) and patch.shape[-1] == bb.embed_dim, timm_id


def test_diloco_outer_step_matches_single_worker_sgd_and_averages():
    from opendino.diloco import outer_step, pseudo_gradient
    g = {"w": torch.tensor([1.0, 2.0]), "step": torch.tensor(3)}
    local_a = {"w": torch.tensor([0.5, 2.5]), "step": torch.tensor(3)}
    local_b = {"w": torch.tensor([1.5, 1.5]), "step": torch.tensor(3)}
    da, db = pseudo_gradient(g, local_a), pseudo_gradient(g, local_b)
    assert "step" not in da and torch.allclose(da["w"].float(), torch.tensor([0.5, -0.5]))
    # lr 1, no momentum, one worker: the outer step lands exactly on that worker's weights
    new, _ = outer_step(g, [da], None, lr=1.0, mu=0.0)
    assert torch.allclose(new["w"], local_a["w"]) and new["step"] == 3
    # two workers with opposite deltas average to no movement
    new, m = outer_step(g, [da, db], None, lr=0.7, mu=0.9)
    assert torch.allclose(new["w"], g["w"]) and torch.allclose(m["w"], torch.zeros(2))
    # Nesterov momentum: second identical step moves further than the first
    s1, m1 = outer_step(g, [da], None, lr=0.1, mu=0.9)
    s2, _ = outer_step(s1, [da], m1, lr=0.1, mu=0.9)
    assert ((s2["w"] - s1["w"]).abs() > (s1["w"] - g["w"]).abs()).all()


def test_cache_slice_not_starting_at_zero(tmp_path):
    n, grid, k, d = 6, 2, 4, 6
    cls = np.random.randn(n, d).astype(np.float16)
    white = np.random.randn(n, grid, grid, k).clip(-4, 4)
    std = np.ones(k, dtype=np.float32)
    _make_cache(tmp_path / "p2", cls[:3], white[:3], std, start=100)
    _make_cache(tmp_path / "p3", cls[3:], white[3:], std, start=103)
    c = TeacherCache([tmp_path / "p2", tmp_path / "p3"])
    assert c.base == 100 and c.meta["num_images"] == 6
    assert torch.equal(c.get(4)[0], torch.from_numpy(cls[4]))
