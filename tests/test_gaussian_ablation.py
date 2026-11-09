"""CPU behavior checks for Gaussian-generation and pose-gating ablations."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from slam_backend import (
    SLAMBackend,
    WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS,
    WINDOW_MAPPING_SAMPLE_ITERATIONS,
)
from utils import RGB2SH


@pytest.fixture
def scene_module(monkeypatch):
    # Load real scene/optimizer Python code, replacing only optional CUDA and
    # feature-extraction dependencies. Keep the installed renderer untouched.
    dependencies = {
        "fused_ssim": {"fused_ssim": lambda *a, **k: torch.tensor(0.5)},
        "diff_gaussian_rasterization": {
            "GaussianRasterizationSettings": object,
            "GaussianRasterizer": object,
            "adamUpdate": lambda *a, **k: None,
            "adamUpdateBasic": lambda *a, **k: None,
        },
        "simple_knn": {},
        "simple_knn._C": {"distIndex2": lambda *a, **k: None},
        "poses.feature_detector": {"DescribedKeypoints": object},
        "poses.matcher": {"Matcher": object},
        "poses.guided_mvs": {"GuidedMVS": object},
        "poses.triangulator": {"Triangulator": object},
        "scene.dense_extractor": {"DenseExtractor": object},
        "scene.mono_depth": {"MonoDepthEstimator": object, "align_depth": lambda *a, **k: None},
        "scene.anchor": {"Anchor": object},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    root = Path(__file__).resolve().parents[1]

    def load(name, relative_path):
        spec = importlib.util.spec_from_file_location(name, root / relative_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    optimizers = load("_ablation_optimizers", "scene/optimizers.py")
    monkeypatch.setitem(sys.modules, "scene.optimizers", optimizers)
    keyframes = load("_ablation_keyframe", "scene/keyframe.py")
    monkeypatch.setitem(sys.modules, "scene.keyframe", keyframes)
    return load("_ablation_scene_model", "scene/scene_model.py")


def make_model(module, n=4):
    model = object.__new__(module.SceneModel)
    model.direct_gaussian_sampling = False
    model.pose_gaussian_gate = True
    model.pose_gaussian_min_update_count = 5
    model.pose_gaussian_min_observed_keyframes = 3
    model.inference_mode = False
    model.densification_interval = 50
    model.densify_grad_threshold = 0.0002
    model.percent_dense = 0.01
    model.gaussian_mapping_update_count = 0
    model.lock = threading.RLock()
    model.approx_cam_centres = torch.tensor([[-1.0, 0, 0], [1.0, 0, 0]])
    values = {
        "xyz": torch.tensor([[0.0, 0.0, 2.0]]).repeat(n, 1),
        "scaling": torch.full((n, 3), 0.005).log(),
        "opacity": torch.zeros(n, 1),
        "rotation": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "f_dc": torch.zeros(n, 1, 3),
        "f_rest": torch.zeros(n, 0, 3),
    }
    model.gaussian_params = {
        name: {"val": value.requires_grad_(), "lr": torch.tensor(0.001)}
        for name, value in values.items()
    }
    model.optimizer = module.SparseGaussianAdam(model.gaussian_params)
    model.gaussian_optimization_counts = torch.arange(5, n + 5, dtype=torch.int32)
    model.gaussian_observed_keyframe_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32).repeat(n, 1)
    model.gaussian_observed_keyframe_counts = torch.full((n,), 3, dtype=torch.int16)
    model.gaussian_owner_keyframe_ids = torch.arange(1, n + 1, dtype=torch.int32) * 10
    model._fit_gaussian_densification_stats()
    return model


def gradient_package(gradients, visible=None):
    gradient = torch.as_tensor(gradients, dtype=torch.float32).reshape(-1, 3)
    points = torch.zeros_like(gradient, requires_grad=True)
    points.grad = gradient
    if visible is None:
        visible = torch.ones(gradient.shape[0], dtype=torch.bool)
    return {
        "screenspace_points": points,
        "visibility_filter": torch.as_tensor(visible, dtype=torch.bool),
    }


@pytest.mark.parametrize("gate", [False, True])
def test_gate_boundaries_and_observer_accounting_when_disabled(scene_module, gate):
    model = make_model(scene_module)
    model.pose_gaussian_gate = gate
    model.gaussian_optimization_counts[:] = torch.tensor([4, 5, 5, 0])
    model.gaussian_observed_keyframe_ids[:] = torch.tensor(
        [[0, 1, 2], [0, 1, -1], [0, 1, 2], [-1, -1, -1]]
    )
    model.gaussian_observed_keyframe_counts[:] = torch.tensor([3, 2, 3, 0])
    assert model.pose_optimization_gaussian_mask().tolist() == (
        [False, False, True, False] if gate else [True] * 4
    )

    visibility = torch.tensor([True, True, False, True])
    model._increment_gaussian_optimization_counts(visibility)
    model._mark_gaussians_observed_by_keyframe(visibility, 1)
    model._mark_gaussians_observed_by_keyframe(visibility, 1)
    assert model.gaussian_optimization_counts.tolist() == [5, 6, 5, 1]
    assert model.gaussian_observed_keyframe_counts.tolist() == [3, 2, 3, 1]
    model._mark_gaussians_observed_by_keyframe(visibility, 2)
    assert model.gaussian_observed_keyframe_counts.tolist() == [3, 3, 3, 2]
    assert model.pose_optimization_gaussian_mask().tolist() == (
        [True, True, True, False] if gate else [True] * 4
    )


def test_gradient_statistics_ignore_invisible_and_nonfinite_samples(scene_module):
    model = make_model(scene_module)
    model._accumulate_gaussian_densification_stats(gradient_package(
        [[3, 4, 100], [4, 3, 0], [float("nan"), 0, 0], [float("inf"), 0, 0]],
        [True, False, True, True],
    ))
    model._accumulate_gaussian_densification_stats(gradient_package(
        [[0, 1, 0], [0, 2, 0], [0, 0, 0], [0, 0, 0]],
        [True, True, False, False],
    ))
    assert model.gaussian_gradient_accum.tolist() == [6, 2, 0, 0]
    assert model.gaussian_gradient_counts.tolist() == [2, 1, 0, 0]
    model._accumulate_gaussian_densification_stats({})
    assert model.gaussian_gradient_counts.tolist() == [2, 1, 0, 0]


def test_gradient_threshold_uses_mean_over_visible_observations(scene_module):
    model = make_model(scene_module, n=2)
    model.gaussian_gradient_accum[:] = torch.tensor([0.0003, 0.0004])
    model.gaussian_gradient_counts[:] = 2
    result = model.densify_gaussians_from_gradients()
    assert result == {"cloned": 1, "split": 0, "added": 1}
    assert model.gaussian_owner_keyframe_ids.tolist() == [10, 20, 20]
    assert model.gaussian_gradient_accum.tolist() == [0, 0, 0]
    assert model.gaussian_gradient_counts.tolist() == [0, 0, 0]


def test_clone_and_split_preserve_parameters_owners_and_optimizer_state(scene_module):
    model = make_model(scene_module)
    with torch.no_grad():
        model.gaussian_params["scaling"]["val"][1] = torch.tensor([0.2, 0.1, 0.05]).log()
    for parameter in model.gaussian_params.values():
        parameter["exp_avg"].fill_(2.0)
        parameter["exp_avg_sq"].fill_(3.0)
    model.gaussian_gradient_accum[:] = torch.tensor([0.001, 0.001, 0.0001, 100.0])
    model.gaussian_gradient_counts[:] = torch.tensor([1, 1, 1, 0])

    result = model.densify_gaussians_from_gradients()

    assert result == {"cloned": 1, "split": 1, "added": 3}
    assert model.xyz.shape == (6, 3)  # Keep three originals; append clone plus two children.
    owners = model.gaussian_owner_keyframe_ids
    old = model.gaussian_optimization_counts > 0
    assert owners[old].tolist() == [10, 30, 40]
    assert model.gaussian_optimization_counts[old].tolist() == [5, 7, 8]
    assert sorted(owners[~old].tolist()) == [10, 20, 20]
    assert torch.allclose(model.scaling[owners == 10], torch.full((2, 3), 0.005))
    assert torch.allclose(
        model.scaling[owners == 20], torch.tensor([[0.2, 0.1, 0.05]]).repeat(2, 1) / 1.6
    )
    assert torch.all(model.gaussian_observed_keyframe_ids[~old] == -1)
    assert torch.all(model.gaussian_observed_keyframe_counts[~old] == 0)
    assert torch.all(model.gaussian_observed_keyframe_counts[old] == 3)
    assert torch.all(model.gaussian_gradient_accum == 0)
    assert torch.all(model.gaussian_gradient_counts == 0)
    for parameter in model.gaussian_params.values():
        assert parameter["val"].shape[0] == 6
        assert parameter["val"].is_leaf and parameter["val"].requires_grad
        assert torch.isfinite(parameter["val"]).all()
        assert torch.all(parameter["exp_avg"][old] == 2)
        assert torch.all(parameter["exp_avg_sq"][old] == 3)
        assert torch.all(parameter["exp_avg"][~old] == 0)
        assert torch.all(parameter["exp_avg_sq"][~old] == 0)


def make_rotated_split_model(scene_module):
    model = make_model(scene_module, n=1)
    with torch.no_grad():
        model.gaussian_params["xyz"]["val"][0] = torch.tensor([1.0, 2.0, 3.0])
        model.gaussian_params["scaling"]["val"][0] = torch.tensor([0.4, 0.2, 0.1]).log()
        # A 90-degree rotation around z: local +x becomes world +y.
        model.gaussian_params["rotation"]["val"][0] = torch.tensor([2 ** -0.5, 0, 0, 2 ** -0.5])
    model.gaussian_gradient_accum[:] = 0.001
    model.gaussian_gradient_counts[:] = 1
    return model


def test_split_rotates_anisotropic_offsets_into_world_coordinates(scene_module, monkeypatch):
    model = make_rotated_split_model(scene_module)

    def fixed_normal_samples(reference):
        assert reference.shape == (2, 3)
        return reference.new_tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])

    monkeypatch.setattr(torch, "randn_like", fixed_normal_samples)
    assert model.densify_gaussians_from_gradients() == {"cloned": 0, "split": 1, "added": 2}
    # Scale first, then rotate local (0.4, 0.4, 0.3) to (-0.4, 0.4, 0.3).
    assert torch.allclose(model.xyz, torch.tensor([[0.6, 2.4, 3.3], [1.4, 1.6, 2.7]]))
    assert torch.allclose(model.scaling, torch.tensor([[0.25, 0.125, 0.0625]]).repeat(2, 1))


def test_split_child_positions_are_reproducible_with_same_seed(scene_module):
    positions = []
    with torch.random.fork_rng(devices=[]):
        for seed in (123, 123, 124):
            model = make_rotated_split_model(scene_module)
            torch.random.default_generator.manual_seed(seed)
            model.densify_gaussians_from_gradients()
            positions.append(model.xyz.detach().clone())
    assert torch.equal(positions[0], positions[1])
    assert not torch.equal(positions[0], positions[2])


def test_add_and_prune_keeps_statistics_metadata_and_per_point_learning_rates_aligned(scene_module):
    model = make_model(scene_module, n=3)
    model.gaussian_gradient_accum[:] = torch.tensor([1.0, 2.0, 3.0])
    model.gaussian_gradient_counts[:] = torch.tensor([4, 5, 6])
    model.optimizer.lr_dict = {"xyz": {"lr_init": 0.01, "lr_decay": 0.99}}
    model.gaussian_params["xyz"]["lr"] = torch.tensor([[0.1], [0.2], [0.3]]).repeat(1, 3)
    extension = {name: parameter["val"][:1].detach().clone() for name, parameter in model.gaussian_params.items()}
    model._apply_gaussian_add_and_prune(
        extension, torch.tensor([False, True, True]),
        extension_owner_ids=torch.tensor([99], dtype=torch.int32),
    )
    assert model.gaussian_owner_keyframe_ids.tolist() == [20, 30, 99]
    assert model.gaussian_optimization_counts.tolist() == [6, 7, 0]
    assert model.gaussian_observed_keyframe_counts.tolist() == [3, 3, 0]
    assert model.gaussian_gradient_accum.tolist() == [2, 3, 0]
    assert model.gaussian_gradient_counts.tolist() == [5, 6, 0]
    assert torch.allclose(model.gaussian_params["xyz"]["lr"][:, 0], torch.tensor([0.2, 0.3, 0.01]))


@pytest.mark.parametrize("centres,expected", [
    (None, 1e-6), ([], 1e-6), ([[1, 2, 3]], 1e-6),
    ([[float("nan"), 0, 0]], 1e-6),
    ([[1, 0, 0], [3, 0, 0], [float("nan"), 0, 0]], 1.1),
])
def test_densification_extent_uses_camera_radius_and_finite_floor(scene_module, centres, expected):
    model = make_model(scene_module, n=0)
    model.approx_cam_centres = None if centres is None else torch.tensor(centres).reshape(-1, 3)
    assert model._gradient_densification_extent().item() == pytest.approx(expected)


def test_empty_map_and_nonfinite_gradients_do_not_create_gaussians(scene_module):
    model = make_model(scene_module, n=0)
    model._accumulate_gaussian_densification_stats(gradient_package([]))
    assert model.densify_gaussians_from_gradients() == {"cloned": 0, "split": 0, "added": 0}
    assert model.optimization_step(keyframe_id=0) is None
    assert model.gaussian_mapping_update_count == 0

    model = make_model(scene_module, n=2)
    model.gaussian_gradient_accum[:] = torch.tensor([float("nan"), float("inf")])
    model.gaussian_gradient_counts[:] = 1
    assert model.densify_gaussians_from_gradients()["added"] == 0
    assert model.xyz.shape[0] == 2
    assert model.gaussian_gradient_accum.tolist() == [0, 0]


def fail_if_called(*args, **kwargs):
    raise AssertionError("Direct pixel sampling/MVS/coarse replacement must not run")


@pytest.mark.parametrize("has_seeds", [False, True])
@pytest.mark.parametrize("has_existing_map", [False, True])
def test_gradient_mode_adds_only_triangulation_seeds_without_direct_sampling(
    scene_module, monkeypatch, has_seeds, has_existing_map
):
    model = make_model(scene_module, n=3 if has_existing_map else 0)
    if has_existing_map:
        with torch.no_grad():
            model.gaussian_params["opacity"]["val"][1] = torch.logit(torch.tensor(0.01))
            model.gaussian_params["scaling"]["val"][2] = torch.tensor(3.0).log()
    model.width = model.height = 20
    model.disc_kernel = torch.ones(1, 1, 1, 1)
    model.init_sampling_method = "scharr"
    model.init_proba_scaler = 1.0
    model.max_sh_degree = 0
    color = torch.tensor([0.4, 0.6, 0.8])
    seed_points = torch.tensor([[0.0, 0, 2.0], [0.2, 0, 2.0], [0.0, 0, 0.0]])
    has_pt3d = torch.tensor([has_seeds, has_seeds, False])
    keyframe = SimpleNamespace(
        index=7, info={"is_test": False},
        desc_kpts=SimpleNamespace(
            pts3d=seed_points, has_pt3d=has_pt3d,
            kpts=torch.tensor([[5.0, 5.0], [10.0, 10.0], [0.0, 0.0]]),
        ),
        image_pyr=[color[:, None, None].expand(3, 20, 20)], mask_pyr=None,
        approx_centre=torch.zeros(3), f=torch.tensor([10.0, 10.0]),
        centre=torch.tensor([10.0, 10.0]),
        get_t=lambda: torch.zeros(3), get_R=lambda: torch.eye(3),
        align_depth=lambda: None, update_3dpts=lambda keyframes: None,
    )
    model.keyframes = [keyframe]
    model.guided_mvs = fail_if_called
    model.get_prev_keyframes = fail_if_called
    model.render_from_id = fail_if_called
    monkeypatch.setattr(torch, "rand_like", fail_if_called)
    monkeypatch.setattr(scene_module, "get_sampling_response", lambda *a, **k: torch.full((20, 20), 0.25))

    model.add_new_gaussians(0)

    # The shared opacity/size pruning remains active in seed-only mode.
    retained = int(has_existing_map)
    assert model.xyz.shape[0] == retained + 2 * int(has_seeds)
    if has_seeds:
        assert torch.equal(model.xyz[retained:], seed_points[:2])
        assert torch.allclose(model.f_dc[retained:], RGB2SH(color[None, None]).repeat(2, 1, 1))
        assert torch.allclose(model.opacity[retained:], torch.full((2, 1), 0.2))
        expected_scales = 2.0 / 10.0 * torch.linalg.vector_norm(seed_points[:2], dim=-1)
        assert torch.allclose(model.scaling[retained:], expected_scales[:, None].repeat(1, 3))
        assert model.gaussian_owner_keyframe_ids.tolist() == [10] * retained + [7, 7]
        assert model.gaussian_optimization_counts.tolist() == [5] * retained + [0, 0]


def prepare_optimization_model(scene_module, monkeypatch):
    model = make_model(scene_module, n=1)
    model.pose_gaussian_gate = False
    model.lambda_dssim = model.geometry_loss_weight = 0.0
    model.valid_Rt_cache = torch.ones(1, dtype=torch.bool)
    model.last_trained_id = -1
    pose = torch.nn.Parameter(torch.tensor(0.25))

    keyframe = SimpleNamespace(
        index=0, device=SimpleNamespace(type="cuda"), pyr_lvl=0,
        image_pyr=[torch.zeros(3, 2, 2)], mask_pyr=None,
        desc_kpts=None, info={"is_test": False}, num_steps=0, idepth_pyr=None,
        depth_loss_weight=0.0, depth_loss_weight_decay=0.9,
        zero_grad=lambda: setattr(pose, "grad", None),
        get_mono_idepth=lambda level: torch.ones(1, 2, 2),
        step=lambda **kwargs: None,
    )
    keyframe.advance_mapping_schedule = MethodType(
        scene_module.Keyframe.advance_mapping_schedule, keyframe
    )
    model.keyframes = [keyframe]
    model.visible = True
    model.packages = []

    def render_from_id(keyframe_id, detach_pose=False, detach_gaussians=False, gaussian_mask=None, **kwargs):
        screen = torch.zeros_like(model.xyz, requires_grad=True)
        xyz = model.xyz if gaussian_mask is None else model.xyz[gaussian_mask]
        value = (xyz.detach() if detach_gaussians else xyz).sum()
        value = value + (pose.detach() if detach_pose else pose) + screen[:, :2].sum()
        package = {
            "render": value.expand(3, 2, 2),
            "invdepth": value.expand(1, 2, 2),
            "screenspace_points": screen,
            "visibility_filter": torch.full((len(model.xyz),), model.visible),
            "radii": torch.ones(len(model.xyz)),
        }
        model.packages.append(package)
        return package

    model.render_from_id = render_from_id
    rand = torch.rand
    monkeypatch.setattr(torch, "rand", lambda *a, **k: rand(*a, **{**k, "device": "cpu"}))
    return model


def prepare_online_mapping_model(scene_module, monkeypatch, num_keyframes, holdout, mature):
    model = prepare_optimization_model(scene_module, monkeypatch)
    model.direct_gaussian_sampling = True
    model.geometry_loss_weight = 1.0
    model.pose_gaussian_gate = True
    if not mature:
        # The map remains below this threshold throughout both phases.
        model.pose_gaussian_min_update_count = 1000
    template = model.keyframes[0]
    model.keyframes = []
    parameter_names = ("rW2C", "tW2C", "exposure", "depth_scale", "depth_offset")
    for index in range(num_keyframes):
        keyframe = SimpleNamespace(**vars(template))
        keyframe.index = index
        keyframe.info = {"is_test": holdout and index == num_keyframes - 1}
        keyframe.rW2C = torch.nn.Parameter(torch.eye(3)[:, :2].clone())
        keyframe.tW2C = torch.nn.Parameter(torch.tensor([index * 0.01, 0.0, 0.0]))
        keyframe.exposure = torch.nn.Parameter(torch.eye(3, 4))
        keyframe.depth_scale = torch.nn.Parameter(torch.ones(1))
        keyframe.depth_offset = torch.nn.Parameter(torch.zeros(1))
        keyframe.image_pyr = [torch.zeros(3, 4, 4), torch.zeros(3, 2, 2)]
        keyframe.mask_pyr = [torch.ones(1, 4, 4), torch.ones(1, 2, 2)]
        keyframe.idepth_pyr = [torch.ones(1, 4, 4), torch.ones(1, 2, 2)]
        keyframe.pyr_lvl = 1
        keyframe.depth_loss_weight = 2.0
        keyframe.advance_mapping_schedule = MethodType(
            scene_module.Keyframe.advance_mapping_schedule, keyframe
        )

        optimized_names = [name for name in parameter_names
                           if name != "exposure" or not keyframe.info["is_test"]]
        optimizer = torch.optim.SGD([getattr(keyframe, name) for name in optimized_names], lr=0.001)
        keyframe.optimizer = SimpleNamespace(
            params={name: {"val": getattr(keyframe, name)} for name in optimized_names},
            zero_grad=optimizer.zero_grad,
            step=optimizer.step,
        )
        keyframe.zero_grad = MethodType(scene_module.Keyframe.zero_grad, keyframe)
        keyframe.get_mono_idepth = lambda level, kf=keyframe: (
            kf.idepth_pyr[level] * kf.depth_scale + kf.depth_offset
        )
        keyframe.step = MethodType(scene_module.Keyframe.step, keyframe)
        model.keyframes.append(keyframe)
    model.valid_Rt_cache = torch.ones(num_keyframes, dtype=torch.bool)
    optimizer = torch.optim.SGD([param["val"] for param in model.gaussian_params.values()], lr=0.001)
    model.optimizer = SimpleNamespace(
        zero_grad=optimizer.zero_grad,
        step=lambda visibility, n_radii: optimizer.step(),
    )
    model.render_calls = []
    model.geometry_detach_flags = []

    def pose_value(keyframe, detach):
        rotation, translation = keyframe.rW2C, keyframe.tW2C
        if detach:
            rotation, translation = rotation.detach(), translation.detach()
        return rotation.sum() + translation.sum()

    def render(keyframe_id, pyr_lvl, detach_pose=False, detach_gaussians=False,
               gaussian_mask=None, **kwargs):
        model.render_calls.append((keyframe_id, pyr_lvl, detach_pose, detach_gaussians))
        keyframe = model.keyframes[keyframe_id]
        xyz = model.xyz if gaussian_mask is None else model.xyz[gaussian_mask]
        if detach_gaussians:
            xyz = xyz.detach()
        value = xyz.sum() + pose_value(keyframe, detach_pose) + keyframe.exposure.mean()
        height, width = keyframe.image_pyr[pyr_lvl].shape[-2:]
        return {
            "render": value.expand(3, height, width),
            "invdepth": value.expand(1, height, width),
            "visibility_filter": torch.tensor([True]),
            "radii": torch.ones(1),
        }

    def geometry_loss(keyframe, invdepth, level, detach_pose=False):
        model.geometry_detach_flags.append(detach_pose)
        return pose_value(keyframe, detach_pose).square() + invdepth.mean().square()

    model.render_from_id = render
    model.compute_geometric_loss = geometry_loss
    return model


@pytest.mark.parametrize("num_keyframes,holdout,mature", [
    (9, False, True), (21, False, True), (9, True, True), (9, True, False),
])
def test_online_mapping_updates_gaussians_then_jointly_refines_poses(
    scene_module, monkeypatch, num_keyframes, holdout, mature
):
    model = prepare_online_mapping_model(scene_module, monkeypatch, num_keyframes, holdout, mature)
    parameter_names = ("rW2C", "tW2C", "exposure", "depth_scale", "depth_offset")
    steps = []
    original_step = model.optimization_step

    def record_step(**kwargs):
        keyframe = model.keyframes[kwargs["keyframe_id"]]
        parameters_before = {name: getattr(keyframe, name).detach().clone() for name in parameter_names}
        xyz_before = model.xyz.detach().clone()
        schedule_before, level_before = keyframe.num_steps, keyframe.pyr_lvl
        renders_before = len(model.render_calls)
        result = original_step(**kwargs)
        joint = len(steps) >= WINDOW_MAPPING_SAMPLE_ITERATIONS
        pose_updated = joint and mature
        map_updated = not keyframe.info["is_test"]

        # Exercise the actual autograd and parameter updates at every boundary,
        # including held-out views and joint steps rejected by the pose gate.
        assert kwargs["optimize_pose"] is joint
        assert kwargs["optimize_gaussians"] is True
        assert result["optimize_pose"] is pose_updated
        assert result["pose_gaussians_used"] == int(pose_updated)
        for name in ("rW2C", "tW2C"):
            changed = not torch.equal(getattr(keyframe, name), parameters_before[name])
            assert changed is pose_updated
            if not pose_updated:
                assert getattr(keyframe, name).grad is None
        for name in ("depth_scale", "depth_offset"):
            parameter = getattr(keyframe, name)
            assert (not torch.equal(parameter, parameters_before[name])) is pose_updated
            if pose_updated:
                assert parameter.grad is not None and parameter.grad.abs().sum() > 0
        if not pose_updated:
            assert torch.equal(keyframe.exposure, parameters_before["exposure"])
        if keyframe.info["is_test"]:
            assert torch.equal(keyframe.exposure, parameters_before["exposure"])
        elif pose_updated:
            assert not torch.equal(keyframe.exposure, parameters_before["exposure"])
        assert (not torch.equal(model.xyz, xyz_before)) is map_updated
        assert keyframe.num_steps - schedule_before == int(map_updated or pose_updated)
        expected_renders = [(keyframe.index, level_before, True, False)]
        if pose_updated:
            expected_renders.append((keyframe.index, level_before, False, True))
        assert model.render_calls[renders_before:] == expected_renders
        steps.append((keyframe.index, level_before, map_updated, pose_updated))
        return result

    model.optimization_step = record_step
    backend = object.__new__(SLAMBackend)
    backend.args = SimpleNamespace(slam_latest_keyframe_sample_proba=0.25)
    backend.scene_model = model

    mapping, joint = backend.run_mapping_schedule()

    assert WINDOW_MAPPING_SAMPLE_ITERATIONS == 25
    assert WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS == 25
    assert len(steps) == 50
    for summary, name, count, optimize_pose in (
        (mapping, "mapping", 25, False), (joint, "joint_mapping", 25, True),
    ):
        assert summary["phase"] == name
        assert summary["max_iters"] == summary["iters"] == summary["sampled_iters"] == count
        assert summary["coverage_iters"] == 0
        assert summary["optimize_pose"] is optimize_pose
        assert summary["optimize_gaussians"] is True
        assert any(index < 8 for index in summary["sampled_keyframe_ids"])
        assert num_keyframes - 1 in summary["sampled_keyframe_ids"]
    assert model.geometry_detach_flags == [call[2] for call in model.render_calls]
    for keyframe in model.keyframes:
        frame_steps = [step for step in steps if step[0] == keyframe.index]
        schedule_count = sum(map_updated or pose_updated for _, _, map_updated, pose_updated in frame_steps)
        assert keyframe.num_steps == schedule_count
        assert keyframe.depth_loss_weight == pytest.approx(2.0 * 0.9 ** schedule_count)
        assert keyframe.pyr_lvl == (0 if schedule_count >= 5 else 1)
        assert len(keyframe.image_pyr) == len(keyframe.mask_pyr) == len(keyframe.idepth_pyr) == keyframe.pyr_lvl + 1
        prior_schedule_count = 0
        for _, level, map_updated, pose_updated in frame_steps:
            assert level == (0 if prior_schedule_count >= 5 else 1)
            prior_schedule_count += int(map_updated or pose_updated)
        assert bool(model.valid_Rt_cache[keyframe.index]) is not any(step[3] for step in frame_steps)
    assert torch.isfinite(model.xyz).all()
    assert model.gaussian_optimization_counts.tolist() == [5 + sum(step[2] for step in steps)]


@pytest.mark.parametrize("optimize_pose,optimize_gaussians", [(False, True), (True, True), (True, False)])
def test_monocular_depth_target_changes_loss_and_optimization_gradients(
    scene_module, monkeypatch, optimize_pose, optimize_gaussians
):
    models = [prepare_online_mapping_model(scene_module, monkeypatch, 1, False, True)
              for _ in range(2)]
    for model, value in zip(models, [1.0, 10.0]):
        keyframe = model.keyframes[0]
        keyframe.mono_idepth = torch.full((1, 1, 4, 4), value)
        keyframe.idepth_pyr = [torch.full_like(depth, value) for depth in keyframe.idepth_pyr]
    results = [model.optimization_step(
        keyframe_id=0, optimize_pose=optimize_pose, optimize_gaussians=optimize_gaussians,
        optimize_only_pose=not optimize_gaussians,
    ) for model in models]

    assert results[0]["depth"] != results[1]["depth"]
    for result in results:
        assert result["depth"] == result["render_depth"]
        assert torch.isfinite(torch.tensor(result["loss"]))
    for metric in ("photometric", "l1", "ssim", "geometry"):
        assert results[0][metric] == results[1][metric]
    assert results[1]["loss"] - results[0]["loss"] == pytest.approx(
        2.0 * (results[1]["depth"] - results[0]["depth"])
    )
    if optimize_gaussians:
        assert not torch.equal(models[0].xyz.grad, models[1].xyz.grad)
        assert not torch.equal(models[0].xyz, models[1].xyz)
    if optimize_pose:
        assert not torch.equal(models[0].keyframes[0].tW2C.grad, models[1].keyframes[0].tW2C.grad)
    for model in models:
        keyframe = model.keyframes[0]
        if optimize_pose and optimize_gaussians:
            assert keyframe.depth_scale.grad is not None and keyframe.depth_scale.grad.abs().sum() > 0
            assert keyframe.depth_offset.grad is not None and keyframe.depth_offset.grad.abs().sum() > 0
            assert keyframe.depth_scale.item() != 1.0 and keyframe.depth_offset.item() != 0.0
        else:
            assert keyframe.depth_scale.item() == 1.0 and keyframe.depth_offset.item() == 0.0


def test_monocular_depth_loss_masks_render_and_target(scene_module, monkeypatch):
    model = prepare_optimization_model(scene_module, monkeypatch)
    keyframe = model.keyframes[0]
    keyframe.depth_loss_weight = 2.0
    keyframe.mask_pyr = [torch.tensor([[[True, False], [False, False]]])]
    keyframe.get_mono_idepth = lambda level: torch.tensor([[[0.0, 20.0], [30.0, 40.0]]])
    render = model.render_from_id

    def render_with_depth(*args, **kwargs):
        package = render(*args, **kwargs)
        package["invdepth"] = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        return package

    model.render_from_id = render_with_depth
    result = model.optimization_step(keyframe_id=0, optimize_pose=False)

    assert result["depth"] == result["render_depth"] == pytest.approx(0.25)
    assert result["loss"] == pytest.approx(result["photometric"] + 2.0 * 0.25)


def test_densification_runs_on_50th_actual_mapping_update_only(scene_module, monkeypatch):
    model = prepare_optimization_model(scene_module, monkeypatch)
    calls = []
    original_densify = model.densify_gaussians_from_gradients

    def densify():
        calls.append(model.gaussian_mapping_update_count)
        return original_densify()

    model.densify_gaussians_from_gradients = densify
    for _ in range(49):
        model.optimization_step(keyframe_id=0, optimize_pose=False)
    assert model.gaussian_mapping_update_count == 49
    assert model.gaussian_gradient_counts.tolist() == [49]
    assert calls == []

    model.optimization_step(keyframe_id=0, optimize_gaussians=False, optimize_only_pose=True)
    model.keyframes[0].info["is_test"] = True
    model.optimization_step(keyframe_id=0, optimize_pose=False)
    model.keyframes[0].info["is_test"] = False
    model.visible = False
    model.optimization_step(keyframe_id=0, optimize_pose=False)
    model.visible = True
    model.inference_mode = True
    model.optimization_step(keyframe_id=0, optimize_pose=False)
    model.inference_mode = False
    assert model.gaussian_mapping_update_count == 49
    assert calls == []

    model.optimization_step(keyframe_id=0, optimize_pose=True)
    assert calls == [50]
    assert model.gaussian_mapping_update_count == 50
    assert model.xyz.shape[0] == 2
    assert model.gaussian_gradient_counts.tolist() == [0, 0]


def test_joint_optimization_collects_only_mapping_render_gradients(scene_module, monkeypatch):
    model = prepare_optimization_model(scene_module, monkeypatch)
    model.optimization_step(keyframe_id=0, optimize_pose=True, optimize_gaussians=True)
    assert len(model.packages) == 2
    assert model.gaussian_gradient_counts.tolist() == [1]
    expected = torch.linalg.vector_norm(model.packages[0]["screenspace_points"].grad[:, :2], dim=-1)
    assert torch.allclose(model.gaussian_gradient_accum, expected)
    assert torch.count_nonzero(model.packages[1]["screenspace_points"].grad) > 0
    assert model.keyframes[0].num_steps == 1


@pytest.mark.parametrize("mode", ["pose_only", "test_view", "disabled", "empty_map"])
def test_non_mapping_steps_do_not_advance_keyframe_schedule(scene_module, monkeypatch, mode):
    model = prepare_optimization_model(scene_module, monkeypatch)
    keyframe = model.keyframes[0]
    keyframe.depth_loss_weight = 2.0
    kwargs = {"keyframe_id": 0, "optimize_pose": False}
    if mode == "pose_only":
        kwargs.update(optimize_pose=True, optimize_gaussians=False, optimize_only_pose=True)
    elif mode == "test_view":
        keyframe.info["is_test"] = True
    elif mode == "disabled":
        kwargs["optimize_gaussians"] = False
    else:
        model.gaussian_params["xyz"]["val"] = torch.empty(0, 3, requires_grad=True)

    model.optimization_step(**kwargs)

    assert keyframe.num_steps == 0
    assert keyframe.depth_loss_weight == 2.0


@pytest.mark.parametrize("no_gaussian_gradient", [False, True])
def test_no_gaussian_update_does_not_advance_densification_clock(scene_module, monkeypatch, no_gaussian_gradient):
    model = prepare_optimization_model(scene_module, monkeypatch)
    model.densification_interval = 1
    model.densify_gaussians_from_gradients = fail_if_called
    if no_gaussian_gradient:
        render = model.render_from_id

        def detached_render(*args, **kwargs):
            return render(*args, **{**kwargs, "detach_gaussians": True})

        model.render_from_id = detached_render
    model.optimization_step(keyframe_id=0, optimize_pose=False, optimize_gaussians=no_gaussian_gradient)
    assert model.gaussian_mapping_update_count == 0
    assert model.gaussian_gradient_counts.tolist() == [0]


@pytest.mark.parametrize("explicit_default", [False, True])
def test_direct_sampling_default_never_accumulates_or_densifies(scene_module, monkeypatch, explicit_default):
    model = prepare_optimization_model(scene_module, monkeypatch)
    if explicit_default:
        model.direct_gaussian_sampling = True
    else:
        del model.direct_gaussian_sampling
    model.densification_interval = 1
    model.densify_gaussians_from_gradients = fail_if_called
    model.optimization_step(keyframe_id=0, optimize_pose=False)
    assert model.gaussian_mapping_update_count == 0
    assert model.gaussian_gradient_counts.tolist() == [0]
    assert model.gaussian_optimization_counts.tolist() == [6]
