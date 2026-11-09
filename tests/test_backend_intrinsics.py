import importlib
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from slam_backend import SLAMBackend
from utils import getProjectionMatrixFromIntrinsics, pts2px


@pytest.fixture
def backend_modules(monkeypatch):
    """Exercise Python geometry without the optional compiled/model dependencies."""
    stubs = {
        "fused_ssim": {"fused_ssim": lambda *args, **kwargs: None},
        "diff_gaussian_rasterization": {
            "GaussianRasterizationSettings": object,
            "GaussianRasterizer": object,
        },
        "simple_knn": {},
        "simple_knn._C": {"distIndex2": lambda *args, **kwargs: None},
        "poses.feature_detector": {"DescribedKeypoints": object},
        "poses.matcher": {"Matcher": object},
        "poses.triangulator": {"Triangulator": object},
        "poses.guided_mvs": {"GuidedMVS": object},
        "scene.dense_extractor": {"DenseExtractor": object},
        "scene.mono_depth": {"MonoDepthEstimator": object, "align_depth": lambda *args: None},
        "scene.optimizers": {"BaseAdam": object, "SparseGaussianAdam": object},
        "scene.anchor": {"Anchor": object},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("scene.keyframe", "scene.scene_model"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    keyframe_module = importlib.import_module("scene.keyframe")
    scene_module = importlib.import_module("scene.scene_model")
    yield keyframe_module, scene_module
    for name in ("scene.keyframe", "scene.scene_model"):
        sys.modules.pop(name, None)


def _keyframe(keyframe_module, K, width=96, height=64, **kwargs):
    return keyframe_module.Keyframe(
        image=torch.zeros(3, height, width),
        info={"is_test": False, "K": K},
        desc_kpts=None,
        Rt=torch.eye(4),
        index=0,
        f=torch.tensor([999.0]),
        feat_extractor=None,
        depth_estimator=None,
        triangulator=None,
        args=None,
        inference_mode=True,
        **kwargs,
    )


def test_keyframe_keeps_calibrated_intrinsics(backend_modules):
    keyframe_module, _ = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)

    assert torch.equal(keyframe.f, torch.tensor([70.0, 50.0]))
    assert torch.equal(keyframe.centre, K[:2, 2])
    assert torch.equal(keyframe.get_intrinsics(), K)


@pytest.mark.parametrize("K_entry", [None, "absent"])
def test_keyframe_uses_numpy_calibration_when_torch_K_missing(backend_modules, K_entry):
    keyframe_module, _ = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    info = {"is_test": False, "K_np": K.numpy()}
    if K_entry is None:
        info["K"] = None
    keyframe = keyframe_module.Keyframe(
        torch.zeros(3, 64, 96), info, None, torch.eye(4), 0, torch.tensor([999.0]),
        None, None, None, None, inference_mode=True,
    )
    assert torch.equal(keyframe.get_intrinsics(), K)


def test_keyframe_json_and_colmap_keep_all_four_parameters(backend_modules):
    keyframe_module, _ = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)
    restored = keyframe_module.Keyframe.from_json(
        keyframe.to_json(), 0, height=64, width=96, device="cpu"
    )
    camera, _ = restored.to_colmap(1)

    assert (restored.width, restored.height) == (96, 64)
    assert torch.equal(restored.get_intrinsics(), K)
    assert camera.model == "PINHOLE"
    assert np.allclose(camera.params, [70.0, 50.0, 30.3, 21.4])


def test_keyframe_legacy_scalar_focal_loads_as_pair(backend_modules):
    keyframe_module, _ = backend_modules
    keyframe = keyframe_module.Keyframe.from_json(
        {"info": {"is_test": False}, "Rt": torch.eye(4).tolist(), "f": 70.0},
        0, height=64, width=96, device="cpu",
    )
    assert torch.equal(keyframe.f, torch.tensor([70.0, 70.0]))
    assert torch.equal(keyframe.centre, torch.tensor([47.5, 31.5]))


def test_render_from_id_uses_each_keyframes_pooling_intrinsics(backend_modules):
    keyframe_module, scene_module = backend_modules
    Ks = [
        torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]]),
        torch.tensor([[83.0, 0.0, 33.7], [0.0, 92.0, 25.8], [0.0, 0.0, 1.0]]),
    ]
    model = object.__new__(scene_module.SceneModel)
    model.keyframes = [_keyframe(keyframe_module, K, width=97, height=65) for K in Ks]
    calls = []

    def capture_render(width, height, *args, **kwargs):
        calls.append((width, height, kwargs["intrinsics"]))
        return {"render": torch.zeros(3, height, width)}

    model.render = capture_render
    for keyframe_id in range(2):
        model.render_from_id(keyframe_id, pyr_lvl=1)

    for (width, height, K_render), K in zip(calls, Ks):
        assert (width, height) == (48, 32)
        assert torch.equal(K_render[[0, 1], [0, 1]], K[[0, 1], [0, 1]] / 2)
        assert torch.allclose(K_render[:2, 2], (K[:2, 2] + 0.5) / 2 - 0.5)


def test_adding_keyframe_updates_default_K_without_changing_older_views(backend_modules):
    keyframe_module, scene_module = backend_modules
    Ks = [
        torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]]),
        torch.tensor([[83.0, 0.0, 33.7], [0.0, 92.0, 25.8], [0.0, 0.0, 1.0]]),
    ]
    model = object.__new__(scene_module.SceneModel)
    model.keyframes = []
    model.width, model.height = 96, 64
    model.f = torch.tensor([100.0, 100.0])
    model.centre = torch.tensor([47.5, 31.5])
    model.approx_cam_centres = None
    model.cached_Rts = torch.empty(0, 4, 4)
    model.valid_Rt_cache = torch.empty(0, dtype=torch.bool)
    model.gt_Rts = torch.empty(0, 4, 4)
    model.gt_Rts_mask = torch.empty(0, dtype=torch.bool)
    model.inference_mode = True
    model.join_optimization_thread = lambda: None
    for K in Ks:
        model.add_keyframe(_keyframe(keyframe_module, K), f=torch.tensor([999.0]))

    assert torch.equal(model.K, Ks[1])
    assert torch.equal(model.f, torch.tensor([83.0, 92.0]))
    assert torch.equal(model.keyframes[0].get_intrinsics(), Ks[0])


def test_render_passes_asymmetric_projection_to_rasterizer(backend_modules, monkeypatch):
    _, scene_module = backend_modules
    captured = []

    def settings(*args):
        captured.append(args)
        return args

    monkeypatch.setattr(scene_module, "GaussianRasterizationSettings", settings)
    monkeypatch.setattr(scene_module, "GaussianRasterizer", lambda settings: None)
    model = object.__new__(scene_module.SceneModel)
    model.gaussian_params = {
        "xyz": {"val": torch.empty(0, 3)},
        "f_dc": {"val": torch.empty(0, 1, 3)},
        "f_rest": {"val": torch.empty(0, 0, 3)},
        "scaling": {"val": torch.empty(0, 3)},
        "rotation": {"val": torch.empty(0, 4)},
        "opacity": {"val": torch.empty(0, 1)},
    }
    model.lock = threading.Lock()
    model.inference_mode = False
    model.active_sh_degree = 0
    width, height = 1920, 1280
    K = torch.tensor([[1520.0, 0.0, 949.57], [0.0, 1490.0, 633.24], [0.0, 0.0, 1.0]])
    model.render(width, height, torch.eye(4), 1.0, intrinsics=K)

    settings_args = captured[0]
    assert settings_args[2] == pytest.approx(width / (2 * 1520.0))
    assert settings_args[3] == pytest.approx(height / (2 * 1490.0))
    P = settings_args[6].T
    points = torch.tensor([[0.0, 0.0, 5.0], [1.0, -0.4, 5.0], [-0.8, 0.7, 3.0]])
    clip = torch.cat([points, torch.ones(3, 1)], dim=1) @ P.T
    ndc = clip[:, :2] / clip[:, 3:4]
    pixels = ((ndc + 1) * torch.tensor([width, height]) - 1) / 2
    assert torch.allclose(pixels, pts2px(points, K[[0, 1], [0, 1]], K[:2, 2]), atol=2e-4)
    assert torch.equal(P, getProjectionMatrixFromIntrinsics(0.01, 100.0, K, width, height))


def test_geometric_loss_projects_with_the_observed_keyframes_K(backend_modules):
    keyframe_module, scene_module = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)
    points = torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]])
    keyframe.desc_kpts = SimpleNamespace(
        kpts=pts2px(points, keyframe.f, keyframe.centre),
        pts3d=points,
        pts_conf=torch.ones(2),
        has_pt3d=torch.ones(2, dtype=torch.bool),
        to=lambda device: None,
    )
    model = object.__new__(scene_module.SceneModel)
    model.gaussian_params = {"xyz": {"val": points}}
    model.f = torch.tensor([999.0, 888.0])
    model.centre = torch.tensor([47.5, 31.5])
    invdepth = torch.full((1, 32, 48), 0.25)
    loss = model.compute_geometric_loss(keyframe, invdepth, lvl=1)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        keyframe.tW2C[0] = 0.2
    assert model.compute_geometric_loss(keyframe, invdepth, lvl=1).item() > 0


def test_geometric_loss_updates_pose_without_optimizing_landmarks_or_monocular_depth(backend_modules):
    keyframe_module, scene_module = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)
    points = torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]], requires_grad=True)
    pixels = pts2px(points.detach(), keyframe.f, keyframe.centre).requires_grad_()
    confidences = torch.tensor([1.0, 3.0], requires_grad=True)
    keyframe.desc_kpts = SimpleNamespace(
        kpts=pixels, pts3d=points, pts_conf=confidences,
        has_pt3d=torch.ones(2, dtype=torch.bool), to=lambda device: None,
    )
    model = object.__new__(scene_module.SceneModel)
    model.gaussian_params = {"xyz": {"val": points}}
    with torch.no_grad():
        keyframe.tW2C[0] = 0.2
    optimizer = torch.optim.SGD([keyframe.rW2C, keyframe.tW2C], lr=0.01)
    invdepth = torch.full((1, 64, 96), 0.25)
    loss = model.compute_geometric_loss(keyframe, invdepth)
    loss.backward()

    assert keyframe.rW2C.grad is not None and keyframe.rW2C.grad.abs().sum() > 0
    assert keyframe.tW2C.grad is not None and keyframe.tW2C.grad.abs().sum() > 0
    assert points.grad is None and pixels.grad is None and confidences.grad is None
    assert keyframe.depth_scale.grad is None and keyframe.depth_offset.grad is None
    optimizer.step()
    assert model.compute_geometric_loss(keyframe, invdepth) < loss.detach()
    assert not model.compute_geometric_loss(keyframe, invdepth, detach_pose=True).requires_grad


@pytest.mark.parametrize("detach_pose", [False, True])
def test_sparse_depth_consistency_preserves_gaussian_and_pose_gradients(backend_modules, detach_pose):
    keyframe_module, scene_module = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)
    landmarks = torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]], requires_grad=True)
    keyframe.desc_kpts = SimpleNamespace(
        kpts=pts2px(landmarks.detach(), keyframe.f, keyframe.centre),
        pts3d=landmarks, pts_conf=torch.ones(2), has_pt3d=torch.ones(2, dtype=torch.bool),
        to=lambda device: None,
    )
    model = object.__new__(scene_module.SceneModel)
    gaussian_points = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
    model.gaussian_params = {"xyz": {"val": gaussian_points}}
    aligned_depth = torch.full((1, 64, 96), 0.25)
    assert model.compute_geometric_loss(keyframe, aligned_depth).item() == pytest.approx(0.0)

    # Reprojection is exact. A differentiable inverse-depth render alone makes
    # the retained sparse geometric term nonzero and must still move the map.
    rendered_depth = gaussian_points[:, 2].reciprocal().expand(1, 64, 96)
    loss = model.compute_geometric_loss(keyframe, rendered_depth, detach_pose=detach_pose)
    assert loss.item() == pytest.approx(0.5 * (0.5 - 0.25) ** 2)
    loss.backward()

    assert gaussian_points.grad is not None and gaussian_points.grad[:, 2].abs().sum() > 0
    if detach_pose:
        assert keyframe.rW2C.grad is None and keyframe.tW2C.grad is None
    else:
        assert keyframe.tW2C.grad is not None and keyframe.tW2C.grad[2].abs() > 0
    assert landmarks.grad is None
    assert keyframe.depth_scale.grad is None and keyframe.depth_offset.grad is None


@pytest.mark.parametrize("include_valid", [False, True])
def test_geometric_loss_filters_invalid_features_and_nonpositive_depth(backend_modules, include_valid):
    keyframe_module, scene_module = backend_modules
    K = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K)
    valid_points = torch.tensor([[0.0, 0.0, 4.0], [0.2, 0.1, 4.0]])
    valid_pixels = pts2px(valid_points, keyframe.f, keyframe.centre) + torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    points = torch.tensor([
        [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1e-7],
        [float("nan"), 0.0, 2.0], [0.0, float("inf"), 2.0],
        [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0],
    ])
    pixels = torch.full((len(points), 2), 100.0)
    pixels[-1, 0] = float("nan")
    confidences = torch.ones(len(points))
    confidences[5:7] = torch.tensor([0.0, -1.0])
    has_pt3d = torch.ones(len(points), dtype=torch.bool)
    has_pt3d[7] = False
    if include_valid:
        points = torch.cat([valid_points, points])
        pixels = torch.cat([valid_pixels, pixels])
        confidences = torch.cat([torch.tensor([1.0, 3.0]), confidences])
        has_pt3d = torch.cat([torch.ones(2, dtype=torch.bool), has_pt3d])
    keyframe.desc_kpts = SimpleNamespace(
        kpts=pixels, pts3d=points, pts_conf=confidences, has_pt3d=has_pt3d,
        to=lambda device: None,
    )
    model = object.__new__(scene_module.SceneModel)
    model.gaussian_params = {"xyz": {"val": valid_points}}
    loss = model.compute_geometric_loss(keyframe, torch.full((1, 64, 96), 0.25))

    # These subpixel-normalized residuals are in the quadratic Smooth-L1 range.
    expected = ((0.5 * (2.0 / 96) ** 2 + 3 * 0.5 * (4.0 / 96) ** 2) / 4
                if include_valid else 0.0)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected)


@pytest.mark.parametrize("feature_state", ["missing", "empty", "untriangulated"])
def test_geometric_loss_without_valid_features_is_zero(backend_modules, feature_state):
    keyframe_module, scene_module = backend_modules
    keyframe = _keyframe(keyframe_module, torch.eye(3))
    if feature_state != "missing":
        count = int(feature_state == "untriangulated")
        keyframe.desc_kpts = SimpleNamespace(
            kpts=torch.zeros(count, 2), pts3d=torch.ones(count, 3),
            pts_conf=torch.ones(count), has_pt3d=torch.zeros(count, dtype=torch.bool),
            to=lambda device: None,
        )
    model = object.__new__(scene_module.SceneModel)
    model.gaussian_params = {"xyz": {"val": torch.ones(1, 3)}}

    loss = model.compute_geometric_loss(keyframe, torch.full((1, 64, 96), 0.25))

    assert loss.shape == ()
    assert loss.item() == 0.0


def test_keyframe_triangulation_passes_other_views_intrinsics(backend_modules):
    keyframe_module, _ = backend_modules
    K0 = torch.tensor([[70.0, 0.0, 30.3], [0.0, 50.0, 21.4], [0.0, 0.0, 1.0]])
    K1 = torch.tensor([[83.0, 0.0, 33.7], [0.0, 92.0, 25.8], [0.0, 0.0, 1.0]])
    keyframe = _keyframe(keyframe_module, K0)
    other = _keyframe(keyframe_module, K1)
    captured = {}
    keyframe.desc_kpts = SimpleNamespace(
        kpts=torch.zeros(1, 2), to=lambda device: None,
        update_3D_pts=lambda *args: None,
    )

    class Triangulator:
        n_cams = 2

        def prepare_matches(self, desc):
            return torch.zeros(1, 2), torch.zeros(1, 2, 2), [1]

        def __call__(self, uv, others, Rt, other_Rts, focal, centre, **kwargs):
            captured.update(kwargs)
            assert other_Rts.shape == (2, 4, 4)
            return torch.zeros(1, 3), torch.ones(1), torch.zeros(1), torch.ones(1, dtype=torch.bool)

    keyframe.triangulator = Triangulator()
    keyframe.update_3dpts([keyframe, other])

    assert torch.equal(captured["other_focals"], torch.stack([other.f, keyframe.f]))
    assert torch.equal(captured["other_centres"], torch.stack([other.centre, keyframe.centre]))


@pytest.mark.parametrize("levels", [1, 2, 3])
@pytest.mark.parametrize("cached_depth", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_keyframe_mapping_schedule_advances_without_an_optimizer(
    backend_modules, levels, cached_depth, masked
):
    keyframe_module, _ = backend_modules
    keyframe = object.__new__(keyframe_module.Keyframe)
    images = [torch.zeros(3, 8 // 2 ** level, 8 // 2 ** level) for level in range(levels)]
    keyframe.image_pyr = list(images)
    keyframe.mask_pyr = [image[:1].bool() for image in images] if masked else None
    keyframe.idepth_pyr = [image[:1].clone() for image in images] if cached_depth else None
    keyframe.pyr_lvl = levels - 1
    keyframe.num_steps = 0
    keyframe.depth_loss_weight = 2.0
    keyframe.depth_loss_weight_decay = 0.9

    # No parameter optimizer exists: the schedule must work independently.
    for count in range(1, 16):
        keyframe.advance_mapping_schedule()
        expected_level = max(0, levels - 1 - count // 5)
        assert keyframe.num_steps == count
        assert keyframe.depth_loss_weight == pytest.approx(2.0 * 0.9 ** count)
        assert keyframe.pyr_lvl == expected_level
        assert len(keyframe.image_pyr) == expected_level + 1
        assert keyframe.image_pyr[0] is images[0]
        if masked:
            assert len(keyframe.mask_pyr) == expected_level + 1
        else:
            assert keyframe.mask_pyr is None
        if cached_depth:
            assert len(keyframe.idepth_pyr) == expected_level + 1
        else:
            assert keyframe.idepth_pyr is None


def test_keyframe_parameter_step_does_not_advance_mapping_schedule(backend_modules):
    keyframe_module, _ = backend_modules
    keyframe = object.__new__(keyframe_module.Keyframe)
    calls = []
    keyframe.optimizer = SimpleNamespace(params={}, step=lambda: calls.append("step"))
    keyframe.num_steps = 4
    keyframe.pyr_lvl = 1
    keyframe.depth_loss_weight = 2.0

    keyframe.step()

    assert calls == ["step"]
    assert keyframe.num_steps == 4
    assert keyframe.pyr_lvl == 1
    assert keyframe.depth_loss_weight == 2.0


@pytest.mark.parametrize("focal,expected", [(70.0, [70.0, 70.0]), ([70.0, 50.0], [70.0, 50.0])])
def test_backend_focal_tensor_preserves_both_axes(focal, expected):
    backend = object.__new__(SLAMBackend)
    backend.args = SimpleNamespace(device="cpu")
    assert backend._make_focal_tensor(focal).tolist() == expected
