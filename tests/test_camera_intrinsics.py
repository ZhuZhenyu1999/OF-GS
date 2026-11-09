from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slam import maybe_seed_from_dataset
from slam_frontend import SLAMFrontend
from utils import (
    depth2points, focal_pair, getProjectionMatrixFromIntrinsics,
    intrinsics_from_info, pts2px, resize_intrinsics,
)


def calibrated_K():
    return torch.tensor([[2051.2, 0.0, 949.57], [0.0, 2073.8, 633.24], [0.0, 0.0, 1.0]])


def test_projection_and_backprojection_use_both_focals_and_true_principal_point():
    K = calibrated_K()
    points = torch.tensor([[0.0, 0.0, 3.0], [0.2, -0.15, 2.0], [-0.9, 0.3, 5.0]])
    homogeneous = points @ K.T
    expected = homogeneous[:, :2] / homogeneous[:, 2:]
    pixels = pts2px(points, K.diagonal()[:2], K[:2, 2])
    torch.testing.assert_close(pixels, expected)
    torch.testing.assert_close(pixels[0], torch.tensor([949.57, 633.24]))
    restored = depth2points(pixels, points[:, 2:], K.diagonal()[:2], K[:2, 2])
    torch.testing.assert_close(restored, points)


def test_gaussian_projection_matches_pixel_intrinsics():
    K = calibrated_K()
    xyz = torch.tensor([[0.0, 0.0, 1.0], [-0.2, 0.1, 4.0], [0.7, -0.3, 3.0]])
    projection = getProjectionMatrixFromIntrinsics(0.01, 100.0, K, 1920, 1280)
    clip = torch.cat([xyz, torch.ones(len(xyz), 1)], dim=-1) @ projection.T
    ndc = clip[:, :2] / clip[:, 3:4]
    raster_pixels = ((ndc + 1) * torch.tensor([1920, 1280]) - 1) * 0.5
    torch.testing.assert_close(raster_pixels, pts2px(xyz, K.diagonal()[:2], K[:2, 2]))


def test_resize_intrinsics_matches_pixel_center_transform():
    K = calibrated_K()
    resized = resize_intrinsics(K, 0.61, 0.59)
    xyz = torch.tensor([[0.2, 0.1, 1.0], [0.0, 0.0, 2.0]])
    pixels = pts2px(xyz, K.diagonal()[:2], K[:2, 2])
    expected = (pixels + 0.5) * torch.tensor([0.61, 0.59]) - 0.5
    torch.testing.assert_close(pts2px(xyz, resized.diagonal()[:2], resized[:2, 2]), expected)


def test_dataset_calibration_overrides_focal_seed_without_losing_K():
    K = calibrated_K().numpy()
    for infos in ([{"K_np": K, "focal_np": np.array([2051.2, 2073.8])}], {"frame.png": {"K_np": K}}):
        args = SimpleNamespace(init_focal=100.0)
        maybe_seed_from_dataset(args, SimpleNamespace(infos=infos))
        np.testing.assert_allclose(args.init_intrinsics, K)
        assert args.init_focal == 100.0
    args = SimpleNamespace(init_focal=-1.0)
    maybe_seed_from_dataset(args, SimpleNamespace(infos=[{"K_np": K, "intrinsics_calibrated": False}]))
    assert args.init_intrinsics is None


def test_frame_calibration_updates_frontend_and_blocks_estimated_focal_override():
    frontend = object.__new__(SLAMFrontend)
    frontend.width, frontend.height = 1920, 1280

    class Initializer:
        calibrated = False
        K = torch.eye(3)
        f = torch.ones(2)
        centre = torch.zeros(2)

        def set_intrinsics(self, K):
            self.K = K
            self.f = K.diagonal()[:2].clone()
            self.centre = K[:2, 2].clone()

    frontend.pose_initializer = Initializer()
    frontend.set_frame_info({"K_np": calibrated_K().numpy()})
    frontend.set_focal(torch.tensor([1.0]))
    torch.testing.assert_close(frontend.pose_initializer.f, torch.tensor([2051.2, 2073.8]))
    torch.testing.assert_close(frontend.pose_initializer.centre, torch.tensor([949.57, 633.24]))
    second = calibrated_K()
    second[0, 2] = 940
    frontend.set_frame_info({"K": second})
    assert frontend.pose_initializer.centre[0] == 940


def test_fallback_and_invalid_calibration_are_explicit():
    K = intrinsics_from_info({}, 100, 6, 4)
    torch.testing.assert_close(K, torch.tensor([[100.0, 0, 2.5], [0, 100, 1.5], [0, 0, 1.0]]))
    torch.testing.assert_close(focal_pair(torch.tensor([100.0])), torch.tensor([100.0, 100.0]))
    K[0, 0] = -1
    with pytest.raises(ValueError, match="positive"):
        intrinsics_from_info({"K": K}, 100, 6, 4)


def test_known_to_uncalibrated_frame_keeps_focal_but_resets_principal_point():
    frontend = object.__new__(SLAMFrontend)
    frontend.width, frontend.height = 1920, 1280

    class Initializer:
        calibrated = False
        K = torch.eye(3)
        f = torch.ones(2)
        centre = torch.zeros(2)

        def set_intrinsics(self, K):
            self.K = K
            self.f = K.diagonal()[:2].clone()
            self.centre = K[:2, 2].clone()

    frontend.pose_initializer = Initializer()
    frontend.set_frame_info({"K_np": calibrated_K().numpy(), "intrinsics_calibrated": True})
    learned_focal = frontend.pose_initializer.f.clone()
    fallback = torch.tensor([[100.0, 0, 959.5], [0, 100.0, 639.5], [0, 0, 1.0]])
    frontend.set_frame_info({"K": fallback, "intrinsics_calibrated": False})
    assert not frontend.pose_initializer.calibrated
    torch.testing.assert_close(frontend.pose_initializer.f, learned_focal)
    torch.testing.assert_close(frontend.pose_initializer.centre, torch.tensor([959.5, 639.5]))
    torch.testing.assert_close(
        intrinsics_from_info({"K": fallback, "intrinsics_calibrated": False}, learned_focal, 1920, 1280),
        frontend.pose_initializer.K,
    )


@pytest.mark.parametrize("mode", ["known", "legacy", "mixed", "uncalibrated", "legacy_reboot"])
def test_bootstrap_passes_per_camera_intrinsics_and_preserves_legacy_focal(monkeypatch, mode):
    import importlib.util
    import sys
    from pathlib import Path

    # Isolate the bootstrap wiring from CUDA matchers, then execute its real
    # numeric setup and pose anchoring on CPU.
    monkeypatch.setitem(sys.modules, "scene.keyframe", SimpleNamespace(Keyframe=object))
    monkeypatch.setitem(sys.modules, "poses.ransac", SimpleNamespace(RANSACEstimator=object, EstimatorType=object))
    spec = importlib.util.spec_from_file_location(
        "_bootstrap_intrinsics_test", Path(__file__).resolve().parents[1] / "poses" / "pose_initializer.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def cpu_factory(factory):
        def call(*args, **kwargs):
            kwargs["device"] = "cpu"
            return factory(*args, **kwargs)
        return call

    monkeypatch.setattr(module, "torch", SimpleNamespace(
        eye=cpu_factory(torch.eye), zeros=cpu_factory(torch.zeros),
        stack=torch.stack, randn_like=torch.randn_like, cat=torch.cat,
    ))
    monkeypatch.setattr(module, "intrinsics_from_info", lambda info, focal, width, height, **kw:
                        intrinsics_from_info(info, focal, width, height, device="cpu"))
    initializer = module.PoseInitializer.__new__(module.PoseInitializer)
    initializer.width, initializer.height = 1920, 1280
    calibrated = mode in ("known", "mixed")
    initializer.calibrated = mode == "known"
    initializer.set_intrinsics(calibrated_K())
    initializer.f_init = initializer.f.clone()
    if mode == "legacy_reboot":
        reboot_K = initializer.K.clone()
        reboot_K[0, 0] *= 1.3
        reboot_K[1, 1] *= 1.3
        initializer.set_intrinsics(reboot_K)
    initializer.num_pts_miniba_bootstrap = 4
    initializer.matcher = lambda *args, **kwargs: None
    uvs = torch.tensor([[[950.0, 634.0], [947.0, 636.0]]] * 4)
    initializer.build_problem = lambda *args: (uvs, None)
    captured = {}

    def run_ba(rotations, translations, focals, xyz, centres, flat_uv):
        captured.update(focals=focals.clone(), centres=centres.clone())
        translations = translations.clone()
        translations[1, 0] = 0.1
        residual = torch.zeros_like(flat_uv)
        if mode == "uncalibrated":
            focals = focals * 1.2
        return rotations, translations, focals, xyz, residual, residual, torch.ones_like(residual)

    def unexpected_ba(*args):
        pytest.fail("Known K must use fixed intrinsics; unknown K must use the focal optimization path")

    fixed = calibrated or mode == "legacy_reboot"
    initializer.miniba_rebooting = run_ba if fixed else unexpected_ba
    initializer.miniba_bootstrap = unexpected_ba if fixed else run_ba
    second_K = calibrated_K()
    second_K[0, 0], second_K[1, 1] = 1900.0, 2200.0
    second_K[:2, 2] = torch.tensor([940.0, 620.0])
    infos = None
    expected_Ks = torch.stack([initializer.K] * 2)
    if mode not in ("legacy", "legacy_reboot"):
        infos = [{"K": K.clone(), "Rt": torch.full((4, 4), 99.0), "gt_c2w": torch.full((4, 4), -99.0),
                  "intrinsics_calibrated": mode == "known" or (mode == "mixed" and i == 1)}
                 for i, K in enumerate((calibrated_K(), second_K))]
        expected_Ks = torch.stack([calibrated_K(), second_K])
        for i, info in enumerate(infos):
            if not info["intrinsics_calibrated"]:
                expected_Ks[i] = intrinsics_from_info({}, initializer.f_init, 1920, 1280)
        source_Ks = [info["K"].clone() for info in infos]
    poses, focal, residual = initializer.initialize_bootstrap(
        [object(), object()], infos=infos, rebooting=mode == "legacy_reboot"
    )
    torch.testing.assert_close(captured["focals"], expected_Ks.diagonal(dim1=-2, dim2=-1)[:, :2])
    torch.testing.assert_close(captured["centres"], expected_Ks[:, :2, 2])
    if mode == "uncalibrated":
        expected_Ks[:, 0, 0] *= 1.2
        expected_Ks[:, 1, 1] *= 1.2
    if infos is not None:
        for i, info in enumerate(infos):
            torch.testing.assert_close(info["K"], source_Ks[i])
            if not info["intrinsics_calibrated"]:
                torch.testing.assert_close(torch.as_tensor(info["estimated_K"]), expected_Ks[i])
                torch.testing.assert_close(intrinsics_from_info(info, [1.0, 2.0], 1920, 1280), expected_Ks[i])
    torch.testing.assert_close(focal, expected_Ks[-1].diagonal()[:2])
    torch.testing.assert_close(initializer.K, expected_Ks[-1])
    torch.testing.assert_close(poses[0], torch.eye(4))
    torch.testing.assert_close(poses[1, :3, 3], torch.tensor([0.1, 0, 0]))
    assert residual == 0
