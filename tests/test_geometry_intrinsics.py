import ctypes
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poses.mini_ba import MiniBA, MiniBAInternal, project
from poses.triangulator import Triangulator, TriangulatorInternal, matches_to_points
from utils import depth2points, pts2px, sixD2mtx


def _scene(dtype=torch.float64, device="cpu"):
    generator = torch.Generator().manual_seed(61)
    xyz = torch.rand(24, 3, generator=generator, dtype=dtype)
    xyz[:, :2] = xyz[:, :2] * 3 - 1.5
    xyz[:, 2] = xyz[:, 2] * 4 + 3
    xyz = xyz.to(device)
    rotations = torch.eye(3, dtype=dtype, device=device)[None, :, :2].repeat(2, 1, 1)
    translations = xyz.new_tensor([[0.18, -0.08, 0.04], [-0.55, 0.12, -0.03]])
    focals = xyz.new_tensor([[800.0, 1100.0], [1200.0, 900.0]])
    centres = xyz.new_tensor([[949.57, 633.24], [927.1, 642.8]])
    uv = pts2px(xyz[:, None] + translations[None], focals, centres)
    return xyz, rotations, translations, focals, centres, uv


def _optimizer(npts, optimize_focal=False, iters=35):
    return MiniBAInternal(
        batch=1, n_opt_cams=1, n_fixed_cams=1, npts=npts,
        optimize_focal=optimize_focal, optimize_3Dpts=False,
        huber_delta=0, outlier_mad_scale=0, lm=1e-4, ep=1e-6,
        k=2, iters=iters,
    )


def test_projection_and_backprojection_keep_anisotropic_focal_and_true_principal_point():
    xyz = torch.tensor([1.2, -0.8, 4.0], dtype=torch.float64)
    focal = xyz.new_tensor([800.0, 1100.0])
    centre = xyz.new_tensor([949.57, 633.24])
    pose = torch.cat([torch.eye(3, dtype=xyz.dtype)[:, :2].reshape(-1), xyz.new_zeros(3)])
    expected = xyz.new_tensor([1189.57, 413.24])
    torch.testing.assert_close(project(xyz, pose, focal, centre), expected)
    torch.testing.assert_close(depth2points(expected, 4.0, focal, centre), xyz)


def test_ba_recovers_pose_with_per_camera_calibration_and_keeps_fixed_camera():
    xyz, rotations, translations, focals, centres, uv = _scene()
    initial_t = translations.clone()
    initial_t[0] += xyz.new_tensor([0.07, -0.04, 0.05])
    result = _optimizer(len(xyz))(rotations, initial_t, focals, xyz, centres, uv.reshape(-1))
    fitted_R, fitted_t, fitted_f, _, residual, _, _ = result
    torch.testing.assert_close(fitted_t, translations, atol=1e-5, rtol=0)
    torch.testing.assert_close(sixD2mtx(fitted_R), torch.eye(3, dtype=xyz.dtype)[None].expand(2, -1, -1), atol=1e-5, rtol=0)
    torch.testing.assert_close(fitted_R[1], rotations[1])
    torch.testing.assert_close(fitted_t[1], initial_t[1])
    torch.testing.assert_close(fitted_f, focals)
    assert residual.abs().max() < 1e-4


def test_ba_focal_optimization_preserves_every_camera_aspect_ratio():
    xyz, rotations, translations, focals, centres, uv = _scene()
    result = _optimizer(len(xyz), optimize_focal=True)(
        rotations, translations, focals * 0.9, xyz, centres, uv.reshape(-1),
    )
    fitted_f = result[2]
    torch.testing.assert_close(fitted_f / fitted_f[0, 0], focals / focals[0, 0])
    torch.testing.assert_close(fitted_f, focals, atol=1e-4, rtol=0)
    assert result[4].abs().max() < 1e-4


def test_triangulation_uses_each_views_own_intrinsics():
    xyz, _, _, focals, centres, _ = _scene()
    baseline = xyz.new_tensor([1.0, 0.08, 0.0])
    rotation = torch.eye(3, dtype=xyz.dtype)
    uv = pts2px(xyz, focals[0], centres[0])
    uv_other = pts2px(xyz - baseline, focals[1], centres[1])
    points, disambiguation, error = matches_to_points(
        uv, uv_other, rotation, baseline, focals[0], centres[0], focals[1], centres[1],
    )
    torch.testing.assert_close(points, xyz, atol=1e-10, rtol=0)
    assert error.max() < 1e-10
    assert (disambiguation > 0).all()


def test_triangulator_returns_world_points_for_different_camera_calibrations():
    xyz, _, _, focals, centres, _ = _scene()
    Rt = torch.eye(4, dtype=xyz.dtype)
    Rt[:3, 3] = xyz.new_tensor([0.2, 0.3, -0.1])
    other_Rt = Rt.clone()
    other_Rt[0, 3] -= 1
    ref_xyz = xyz + Rt[:3, 3]
    uv = pts2px(ref_xyz, focals[0], centres[0])
    uv_other = pts2px(xyz + other_Rt[:3, 3], focals[1], centres[1])
    result = TriangulatorInternal()(
        uv, uv_other[None], Rt, other_Rt[None], focals[0], centres[0],
        uv.new_tensor(1), uv.new_tensor(0), focals[1:2], centres[1:2],
    )
    torch.testing.assert_close(result[0], xyz, atol=1e-10, rtol=0)
    torch.testing.assert_close(result[1], ref_xyz[:, 2], atol=1e-10, rtol=0)
    assert result[3].all()


def _load_without_cupy(filename, monkeypatch):
    """Exercise Torch and packing paths even when optional CuPy is absent."""
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(float32=np.float32))
    spec = importlib.util.spec_from_file_location(
        f"_calibration_test_{filename}", Path(__file__).resolve().parents[1] / "poses" / f"{filename}.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pnp_inlier_scoring_uses_both_focal_lengths_and_calibrated_centre(monkeypatch):
    module = _load_without_cupy("ransac", monkeypatch)
    estimator = module.RANSACEstimator.__new__(module.RANSACEstimator)
    estimator.type = module.EstimatorType.P4P
    estimator.max_error = 0.1
    estimator.models = torch.eye(4, dtype=torch.float64)[None, :3]
    xyz, _, _, focals, centres, _ = _scene()
    uv = pts2px(xyz, focals[0], centres[0])
    assert estimator.get_inlier_mask(uv, xyz, focals[0], centres[0]).all()
    wrong_f = focals[0, 0].repeat(2)
    wrong_c = centres.new_tensor([959.5, 639.5])
    assert not estimator.get_inlier_mask(uv, xyz, wrong_f, wrong_c).any()


def test_guided_mvs_packs_reference_and_neighbor_intrinsics_in_cuda_order(monkeypatch):
    module = _load_without_cupy("guided_mvs", monkeypatch)
    mvs = module.GuidedMVS.__new__(module.GuidedMVS)
    mvs.n_cams = 3
    mvs.num_depth_candidates = 16
    mvs.idepth_range = 0.2
    captured = {}

    def capture_kernel(*, block, grid, args):
        pointer = args[4]
        n_cams = args[10]
        values = (ctypes.c_float * ((n_cams + 1) * 4)).from_address(pointer)
        captured["intrinsics"] = np.ctypeslib.as_array(values).copy().reshape(-1, 4)
        captured["n_cams"] = n_cams

    mvs.uvToDepth = capture_kernel

    def frame(f, c):
        return SimpleNamespace(
            f=torch.tensor(f, dtype=torch.float32), centre=torch.tensor(c, dtype=torch.float32),
            feat_map=torch.zeros(4, 5, 64, dtype=torch.float16),
            mono_idepth=torch.ones(4, 5), image_pyr=[torch.zeros(3, 1280, 1920)],
            get_Rt=lambda: torch.eye(4),
        )

    reference = frame([800, 1100], [949.57, 633.24])
    neighbor = frame([1200, 900], [927.1, 642.8])
    mvs(torch.tensor([[949.57, 633.24]]), reference, [neighbor])
    np.testing.assert_allclose(captured["intrinsics"], [[800, 1100, 949.57, 633.24], [1200, 900, 927.1, 642.8]], atol=1e-4)
    assert captured["n_cams"] == 1
    mvs(torch.tensor([[949.57, 633.24]]), reference, [])
    assert captured["intrinsics"].shape == (1, 4)
    assert captured["n_cams"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a CUDA GPU")
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("shared", [False, True])
def test_ba_cuda_graph_replays_shared_and_per_camera_intrinsics(batch, shared):
    xyz, rotations, translations, focals, centres, uv = _scene(torch.float32, "cuda")
    if shared:
        focals = focals[0]
        centres = centres[0]
        uv = pts2px(xyz[:, None] + translations[None], focals, centres)
    initial_t = translations.clone()
    initial_t[0] += xyz.new_tensor([0.07, -0.04, 0.05])
    optimizer = MiniBA(
        batch, 1, 1, len(xyz), optimize_focal=False, optimize_3Dpts=False,
        make_cuda_graph=True, huber_delta=0, outlier_mad_scale=0, ep=1e-5, iters=20,
    )
    if batch == 1:
        # Real bootstrap/incremental callers omit the singleton batch axis.
        result = optimizer(rotations, initial_t, focals, xyz, centres, uv.reshape(-1))
    else:
        result = optimizer(
            rotations[None].repeat(batch, 1, 1, 1), initial_t[None].repeat(batch, 1, 1),
            focals, xyz[None].repeat(batch, 1, 1), centres, uv.reshape(1, -1).repeat(batch, 1),
        )
    expected_f_shape = ((batch,) if batch > 1 else ()) + ((2,) if shared else (2, 2))
    assert result[2].shape == expected_f_shape
    assert result[4].abs().max() < 1e-3
    torch.testing.assert_close(result[1], translations.expand_as(result[1]), atol=2e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a CUDA GPU")
@pytest.mark.parametrize("n_cams", [2, 8])
def test_bootstrap_cuda_graph_replays_unbatched_poses_and_points(n_cams):
    xyz, rotations, translations, focals, centres, uv = _scene(torch.float32, "cuda")
    if n_cams == 8:
        rotations = rotations[:1].repeat(n_cams, 1, 1)
        offsets = torch.arange(n_cams, device=xyz.device, dtype=xyz.dtype)
        translations = translations[:1].repeat(n_cams, 1)
        translations[:, 0] -= offsets * 0.1
        focals = focals[:1].repeat(n_cams, 1)
        focals[:, 0] += offsets * 2
        focals[:, 1] += offsets * 3
        centres = centres[:1].repeat(n_cams, 1)
        centres[:, 0] += offsets * 0.2
        centres[:, 1] -= offsets * 0.1
        uv = pts2px(xyz[:, None] + translations[None], focals, centres)
    initial_t = translations.clone()
    initial_t[0] += xyz.new_tensor([0.02, -0.01, 0.01])
    optimizer = MiniBA(
        1, n_cams, 0, len(xyz), optimize_focal=False, optimize_3Dpts=True,
        make_cuda_graph=True, huber_delta=0, outlier_mad_scale=0, ep=1e-5, iters=30,
    )
    for replay in range(2):
        replay_focals = focals * (1 + replay * 0.05)
        replay_centres = centres + xyz.new_tensor([3.0, -2.0]) * replay
        replay_uv = pts2px(xyz[:, None] + translations[None], replay_focals, replay_centres)
        result = optimizer(rotations, initial_t, replay_focals, xyz, replay_centres, replay_uv.reshape(-1))
        assert result[2].shape == (n_cams, 2)
        torch.testing.assert_close(result[2], replay_focals)
        assert result[4].abs().max() < 2e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires a CUDA GPU")
def test_triangulator_cuda_graph_accepts_real_intrinsics_and_legacy_focal():
    xyz, _, _, focals, centres, _ = _scene(torch.float32, "cuda")
    Rt = torch.eye(4, device="cuda")
    other_Rt = Rt.clone()
    other_Rt[0, 3] = -1
    uv = pts2px(xyz, focals[0], centres[0])
    uv_other = pts2px(xyz + other_Rt[:3, 3], focals[1], centres[1])
    triangulator = Triangulator(len(xyz), 1, max_error=0.1)
    result = triangulator(
        uv, uv_other[None], Rt, other_Rt[None], focals[0], centres[0],
        other_focals=focals[1:2], other_centres=centres[1:2],
    )
    torch.testing.assert_close(result[0], xyz, atol=1e-4, rtol=0)
    assert result[3].all()
    # Legacy isotropic input is expanded before replaying the same graph.
    uv = pts2px(xyz, focals[0, :1], centres[0])
    uv_other = pts2px(xyz + other_Rt[:3, 3], focals[0, :1], centres[0])
    result = triangulator(uv, uv_other[None], Rt, other_Rt[None], focals[0, :1], centres[0])
    torch.testing.assert_close(result[0], xyz, atol=1e-4, rtol=0)
