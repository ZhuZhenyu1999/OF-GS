"""CPU checks for committing BA geometry without loading CUDA/model extensions."""

import importlib
import math
import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from utils import build_rotation


@pytest.fixture
def scene_class(monkeypatch):
    stubs = {
        "fused_ssim": {"fused_ssim": lambda *args, **kwargs: None},
        "diff_gaussian_rasterization": {
            "GaussianRasterizationSettings": object, "GaussianRasterizer": object,
        },
        "simple_knn": {},
        "simple_knn._C": {"distIndex2": lambda *args, **kwargs: None},
        "poses.feature_detector": {"DescribedKeypoints": object},
        "poses.matcher": {"Matcher": object},
        "poses.guided_mvs": {"GuidedMVS": object},
        "scene.keyframe": {"Keyframe": object},
        "scene.optimizers": {"SparseGaussianAdam": object},
        "scene.anchor": {"Anchor": object},
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "scene.scene_model", raising=False)
    module = importlib.import_module("scene.scene_model")
    yield module.SceneModel
    sys.modules.pop("scene.scene_model", None)


class _Keyframe:
    def __init__(self, keyframe_id, translation=0.0):
        self.index = keyframe_id
        self.Rt = torch.eye(4)
        self.Rt[0, 3] = translation
        self.approx_centre = -self.Rt[:3, :3].T @ self.Rt[:3, 3]
        self.latest_invdepth = torch.ones(1, 2, 2)
        self.desc_kpts = SimpleNamespace(
            pts3d=torch.tensor([[0.0, 0.0, 4.0], [2.0, 0.0, 5.0]]),
            has_pt3d=torch.tensor([True, True]),
            pts_conf=torch.tensor([0.8, 0.5]),
            depth=torch.tensor([4.0, 5.0]),
        )
        self.optimizer = SimpleNamespace(params={
            name: {"val": torch.ones(3), "exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3)}
            for name in ("rW2C", "tW2C", "exposure")
        })

    def get_Rt(self):
        return self.Rt.clone()

    def set_Rt(self, pose):
        self.Rt.copy_(pose)
        self.approx_centre = -pose[:3, :3].T @ pose[:3, 3]


def _params(points):
    values = {
        "xyz": torch.as_tensor(points, dtype=torch.float32).clone(),
        "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(len(points), 1),
    }
    return {
        name: {"val": value, "exp_avg": torch.ones_like(value), "exp_avg_sq": torch.ones_like(value)}
        for name, value in values.items()
    }


def _scene(scene_class):
    scene = object.__new__(scene_class)
    scene.join_optimization_thread = lambda: None
    scene.lock = threading.Lock()
    scene.keyframes = [_Keyframe(0), _Keyframe(1, 1.0)]
    scene.gaussian_params = _params([[1, 0, 4], [2, 1, 5], [3, 0, 6]])
    scene.gaussian_owner_keyframe_ids = torch.tensor([0, 1, 1], dtype=torch.int32)
    scene.approx_cam_centres = torch.stack([kf.approx_centre for kf in scene.keyframes])
    scene.valid_Rt_cache = torch.ones(2, dtype=torch.bool)
    scene.anchors = [SimpleNamespace(gaussian_params=scene.gaussian_params, keyframe_ids=[0, 1])]
    return scene


def test_commit_preserves_owner_camera_geometry_and_resets_only_moved_moments(scene_class):
    scene = _scene(scene_class)
    before = scene.gaussian_params["xyz"]["val"].clone()
    old_pose = scene.keyframes[1].get_Rt()
    new_pose = old_pose.clone()
    new_pose[:3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    new_pose[0, 3] = 2
    summary = scene.apply_local_ba_result({1: new_pose}, [])
    after = scene.gaussian_params["xyz"]["val"]
    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(after[1:] @ new_pose[:3, :3].T + new_pose[:3, 3],
                               before[1:] @ old_pose[:3, :3].T + old_pose[:3, 3])
    expected_rotation = new_pose[:3, :3].T @ old_pose[:3, :3]
    torch.testing.assert_close(build_rotation(scene.gaussian_params["rotation"]["val"])[1:],
                               expected_rotation.expand(2, -1, -1))
    for name in ("xyz", "rotation"):
        for moment in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(scene.gaussian_params[name][moment][0], torch.ones_like(scene.gaussian_params[name][moment][0]))
            assert torch.count_nonzero(scene.gaussian_params[name][moment][1:]) == 0
    assert torch.count_nonzero(scene.keyframes[1].optimizer.params["tW2C"]["exp_avg"]) == 0
    assert torch.count_nonzero(scene.keyframes[0].optimizer.params["tW2C"]["exp_avg"]) == 3
    assert torch.count_nonzero(scene.keyframes[1].optimizer.params["exposure"]["exp_avg"]) == 3
    assert summary == {"poses_updated": 1, "landmarks_updated": 0, "observations_updated": 0, "gaussians_warped": 2}


def test_commit_updates_shared_points_unoptimized_points_and_caches(scene_class):
    scene = _scene(scene_class)
    new_pose = scene.keyframes[1].get_Rt()
    new_pose[0, 3] += 0.5
    old_unoptimized = scene.keyframes[1].desc_kpts.pts3d[1].clone()
    point = torch.tensor([0.2, 0.3, 6.0])
    summary = scene.apply_local_ba_result({1: new_pose}, [([(0, 0), (1, 0)], point)])
    for keyframe in scene.keyframes:
        torch.testing.assert_close(keyframe.desc_kpts.pts3d[0], point)
        assert keyframe.desc_kpts.depth[0] == 6
        assert keyframe.desc_kpts.pts_conf[0] == pytest.approx(0.8)
        assert keyframe.latest_invdepth is None
    torch.testing.assert_close(scene.keyframes[1].desc_kpts.pts3d[1], old_unoptimized - torch.tensor([0.5, 0, 0]))
    assert scene.valid_Rt_cache.tolist() == [True, False]
    torch.testing.assert_close(scene.approx_cam_centres[1], torch.tensor([-1.5, 0, 0]))
    assert scene.sorted_frame_indices.tolist() == [1, 0]
    assert summary["observations_updated"] == 2


def test_joint_ba_controller_commits_to_scene_map_and_shared_landmarks(scene_class):
    from poses.local_ba import CovisibilityLocalBA
    from test_local_ba import make_scene

    fixture_scene, _ = make_scene(7, noisy=True)
    scene = _scene(scene_class)
    scene.keyframes = fixture_scene.keyframes
    for keyframe in scene.keyframes:
        pose = keyframe.get_Rt()
        keyframe.approx_centre = torch.linalg.inv(pose)[:3, 3]
        keyframe.desc_kpts.depth = keyframe.desc_kpts.pts3d[:, 2].clone()
        def set_pose(value, kf=keyframe):
            kf.Rt = value.clone()
            kf.approx_centre = torch.linalg.inv(value)[:3, 3]
        keyframe.set_Rt = set_pose
    scene.approx_cam_centres = torch.stack([kf.approx_centre for kf in scene.keyframes])
    scene.valid_Rt_cache = torch.ones(7, dtype=torch.bool)
    scene.gaussian_owner_keyframe_ids = torch.tensor([6, 5, 4], dtype=torch.int32)
    scene.anchors[0].keyframe_ids = list(range(7))
    before_map = scene.xyz.clone()
    old_fixed = {i: scene.keyframes[i].get_Rt().clone() for i in (0, 1)}
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_iterations=10, local_ba_every=1)).run(scene, 6)
    assert event["accepted"], event
    assert event["fixed_keyframe_ids"] == [0, 1]
    assert event["optimized_keyframe_ids"] == [2, 3, 4, 5, 6]
    assert event["map_correction"]["poses_updated"] > 1
    assert event["map_correction"]["gaussians_warped"] == 3
    assert not torch.allclose(scene.xyz, before_map)
    for i, pose in old_fixed.items():
        torch.testing.assert_close(scene.keyframes[i].get_Rt(), pose)
    for feature_id in range(80):
        fitted = scene.keyframes[0].desc_kpts.pts3d[feature_id]
        for keyframe in scene.keyframes[1:]:
            torch.testing.assert_close(keyframe.desc_kpts.pts3d[feature_id], fitted)


@pytest.mark.parametrize("bad_point", [[float("nan"), 0, 4], [0, 0, -1]])
def test_invalid_result_does_not_partially_apply_pose_or_map(scene_class, bad_point):
    scene = _scene(scene_class)
    before = scene.gaussian_params["xyz"]["val"].clone()
    old_pose = scene.keyframes[1].get_Rt()
    new_pose = old_pose.clone()
    new_pose[0, 3] += 0.5
    with pytest.raises(ValueError):
        scene.apply_local_ba_result({1: new_pose}, [([(0, 0)], torch.tensor(bad_point))])
    torch.testing.assert_close(scene.keyframes[1].get_Rt(), old_pose)
    torch.testing.assert_close(scene.gaussian_params["xyz"]["val"], before)
    assert scene.valid_Rt_cache.all()


def test_offloaded_anchor_receives_same_owner_correction(scene_class):
    scene = _scene(scene_class)
    anchor = SimpleNamespace(
        gaussian_params=_params([[1, 2, 3], [4, 5, 6]]), keyframe_ids=[0],
        gaussian_owner_keyframe_ids=torch.tensor([1, 0]),
    )
    scene.anchors.insert(0, anchor)
    new_pose = scene.keyframes[1].get_Rt()
    new_pose[0, 3] += 0.5
    summary = scene.apply_local_ba_result({1: new_pose}, [])
    torch.testing.assert_close(anchor.gaussian_params["xyz"]["val"], torch.tensor([[0.5, 2, 3], [4, 5, 6]]))
    assert summary["gaussians_warped"] == 3


def test_missing_archived_owners_rejects_commit_before_geometry_changes(scene_class):
    scene = _scene(scene_class)
    scene.anchors.append(SimpleNamespace(gaussian_params=_params([[1, 2, 3]]), keyframe_ids=[1]))
    new_pose = scene.keyframes[1].get_Rt()
    new_pose[0, 3] += 0.5
    before = scene.gaussian_params["xyz"]["val"].clone()
    with pytest.raises(ValueError, match="without Gaussian ownership"):
        scene.apply_local_ba_result({1: new_pose}, [])
    torch.testing.assert_close(scene.gaussian_params["xyz"]["val"], before)
    assert scene.keyframes[1].get_Rt()[0, 3] == 1


def test_batched_shared_landmarks_fill_all_observations_and_preserve_confidence(scene_class):
    scene = _scene(scene_class)
    count = 300
    for keyframe in scene.keyframes:
        keyframe.desc_kpts = SimpleNamespace(
            pts3d=torch.zeros(count, 3), has_pt3d=torch.zeros(count, dtype=torch.bool),
            depth=torch.zeros(count), pts_conf=torch.full((count,), float("nan")),
        )
        keyframe.desc_kpts.pts_conf[0] = 0.9
    points = torch.stack([torch.linspace(-1, 1, count), torch.zeros(count), torch.full((count,), 7.0)], dim=1)
    updates = [([(0, index), (1, index)], point) for index, point in enumerate(points)]
    summary = scene.apply_local_ba_result({}, updates)
    for keyframe in scene.keyframes:
        torch.testing.assert_close(keyframe.desc_kpts.pts3d, points)
        torch.testing.assert_close(keyframe.desc_kpts.depth, torch.full((count,), 7.0))
        assert keyframe.desc_kpts.has_pt3d.all()
        assert keyframe.desc_kpts.pts_conf[0] == pytest.approx(0.9)
        assert (keyframe.desc_kpts.pts_conf[1:] == 1).all()
    assert summary["landmarks_updated"] == count
    assert summary["observations_updated"] == 2 * count


@pytest.mark.parametrize("angle", [0.0, math.pi / 2, math.pi, -math.pi + 1e-5])
def test_quaternion_composition_matches_matrix_composition(scene_class, angle):
    axis = torch.tensor([1.0, 2.0, -3.0])
    axis /= torch.linalg.vector_norm(axis)
    delta = torch.cat([torch.tensor([math.cos(angle / 2)]), axis * math.sin(angle / 2)])
    q = torch.tensor([[0.3, 0.7, -0.2, 0.5], [1.0, 0.0, 0.0, 0.0]])
    rotation = build_rotation(delta[None])[0]
    actual = scene_class._rotate_gaussian_quaternions(q, rotation)
    torch.testing.assert_close(build_rotation(actual), rotation @ build_rotation(q), atol=1e-6, rtol=1e-6)


def test_creation_owners_survive_pruning_and_explicit_merged_owner(scene_class):
    scene = _scene(scene_class)
    scene.gaussian_optimization_counts = torch.tensor([1, 2, 3], dtype=torch.int32)
    scene.gaussian_observed_keyframe_ids = torch.tensor([[0, 1, -1], [1, 0, -1], [1, 0, -1]], dtype=torch.int32)
    scene.gaussian_observed_keyframe_counts = torch.tensor([2, 2, 2], dtype=torch.int16)

    def add_and_prune(extensions, mask):
        for name in scene.gaussian_params:
            scene.gaussian_params[name]["val"] = torch.cat([scene.gaussian_params[name]["val"][mask], extensions[name]])

    scene.optimizer = SimpleNamespace(add_and_prune=add_and_prune)
    extensions = {"xyz": torch.tensor([[7.0, 8, 9]]), "rotation": torch.tensor([[1.0, 0, 0, 0]])}
    scene._apply_gaussian_add_and_prune(extensions, torch.tensor([True, False, True]),
                                      extension_observer_ids=torch.tensor([[0, 1, -1]]),
                                      extension_owner_ids=torch.tensor([1]))
    assert scene.gaussian_owner_keyframe_ids.tolist() == [0, 1, 1]

