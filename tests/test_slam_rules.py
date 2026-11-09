import random
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from slam_backend import (
    SLAMBackend,
    WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS,
    WINDOW_MAPPING_SAMPLE_ITERATIONS,
)
from slam import (
    refresh_keyframe_pose_records,
    seed_everything,
    should_estimate_pose_for_frame,
)
from slam_frontend import KeyframeDecision, KeyframeSelector, SLAMFrontend
from utils import (
    anchor_poses_to_first_frame,
    combine_mapping_losses,
    get_sampling_response,
    is_test_hold_frame,
    normalize_keyframe_sampling_probabilities,
    resize_longest_side,
    sample_keyframe_id_with_probabilities,
    sample_window_keyframe_id,
    valid_8_neighbor_mean,
    window_keyframe_sampling_probabilities,
)


def _make_fake_clidd_root(tmp_path):
    root = tmp_path / "CLIDD-main"
    (root / "model").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "weights" / "S64.pth").write_bytes(b"fake")
    (root / "clidd.py").write_text(
        """
import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseModel(nn.Module):
    def forward(self, x):
        h, w = x.shape[-2:]
        desc = (
            torch.zeros(1, 8, max(h // 2, 1), max(w // 2, 1), device=x.device),
            torch.zeros(1, 24, max(h // 8, 1), max(w // 8, 1), device=x.device),
            torch.zeros(1, 32, max(h // 32, 1), max(w // 32, 1), device=x.device),
        )
        return desc, torch.zeros(1, 1, h, w, device=x.device)

    def sample(self, dense, kpts, *, align_corners=False):
        count = kpts.shape[1]
        values = torch.arange(count * 64, device=kpts.device, dtype=torch.float32).reshape(1, count, 64)
        return F.normalize(values + 1.0, dim=-1)


class CLIDD(nn.Module):
    def __init__(self, cfg, top_k, radius=2, score=-5):
        super().__init__()
        self.cfg = cfg
        self.top_k = top_k
        self.model = _DenseModel()

    def forward(self, x):
        device = x.device
        kpts = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
        desc = F.normalize(torch.eye(3, 64, device=device), dim=-1)
        if self.top_k is not None:
            kpts = kpts[: self.top_k]
            desc = desc[: self.top_k]
        return [{"keypoints": kpts, "descriptors": desc}]
""",
        encoding="utf-8",
    )
    return root


def test_clidd_detector_pads_sparse_features(monkeypatch, tmp_path):
    from poses.feature_detector import Detector

    root = _make_fake_clidd_root(tmp_path)
    args = SimpleNamespace(
        clidd_root=str(root),
        clidd_model="S64",
        clidd_radius=2,
        clidd_score=-5.0,
        device="cpu",
    )

    desc_kpts = Detector(5, 64, 64, args)(torch.zeros(3, 64, 64))

    assert desc_kpts.kpts.shape == (5, 2)
    assert desc_kpts.feats.shape == (5, 64)
    assert desc_kpts.valid.tolist() == [True, True, True, False, False]
    assert torch.all(desc_kpts.kpts[3:] == 0)


def test_clidd_match_ignores_padded_descriptors():
    from poses.matcher import match

    feats1 = torch.zeros(4, 64)
    feats2 = torch.zeros(4, 64)
    feats1[0, 0] = 1.0
    feats1[2, 1] = 1.0
    feats2[1, 0] = 1.0
    feats2[3, 1] = 1.0

    idx, idx_other, mask = match(feats1, feats2, beta=20.0)

    assert idx[mask].tolist() == [0, 2]
    assert idx_other[mask].tolist() == [1, 3]


def test_clidd_dense_extractor_outputs_half_descriptor_map(tmp_path):
    from scene.dense_extractor import DenseExtractor

    root = _make_fake_clidd_root(tmp_path)
    args = SimpleNamespace(
        clidd_root=str(root),
        clidd_model="S64",
        clidd_radius=2,
        clidd_score=-5.0,
        clidd_dense_stride=8,
        device="cpu",
    )

    feat_map = DenseExtractor(64, 64, args)(torch.zeros(3, 64, 64))

    assert feat_map.shape == (8, 8, 64)
    assert feat_map.dtype == torch.float16
    assert feat_map.is_contiguous()


def test_clidd_install_validation_reports_missing_weights(tmp_path):
    from poses.clidd_adapter import validate_clidd_install

    root = tmp_path / "CLIDD-main"
    (root / "model").mkdir(parents=True)
    (root / "weights").mkdir()
    (root / "clidd.py").write_text("class CLIDD: pass\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="S64.pth"):
        validate_clidd_install(root, "S64")


def test_keyframe_selection_uses_motion_and_matches_only():
    selector = KeyframeSelector(
        image_width=1000,
        min_displacement_ratio=0.02,
        min_num_inliers=100,
    )
    first = selector.decide(10, 0.0, 0)
    assert first.selected and first.reason == "first"
    selector.mark_selected(10)

    threshold_count = selector.decide(11, 21.0, 100)
    assert threshold_count.selected
    assert threshold_count.reason == "motion"

    small_motion = selector.decide(11, 19.0, 100)
    assert not small_motion.selected
    assert small_motion.reason == "small_motion"

    few_matches = selector.decide(11, 21.0, 99)
    assert not few_matches.selected
    assert few_matches.reason == "few_matches"

    long_gap = selector.decide(20, 0.0, 100)
    assert not long_gap.selected
    assert long_gap.reason == "small_motion"


def test_pose_estimation_runs_only_for_keyframes_or_test_hold():
    ordinary = KeyframeDecision(False, "small_motion", 0.0, 200, 20.0)
    keyframe = KeyframeDecision(True, "motion", 21.0, 200, 20.0)

    assert not should_estimate_pose_for_frame(ordinary, {"is_test": False})
    assert should_estimate_pose_for_frame(keyframe, {"is_test": False})
    assert should_estimate_pose_for_frame(ordinary, {"is_test": True})


def test_seed_everything_resets_global_random_sources():
    seed_everything(123)
    python_value = random.random()
    numpy_value = np.random.rand()
    torch_value = torch.rand(1)

    seed_everything(123)

    assert random.random() == python_value
    assert np.random.rand() == numpy_value
    assert torch.allclose(torch.rand(1), torch_value)


def test_refresh_keyframe_pose_records_uses_optimized_keyframe_poses():
    stale = torch.eye(4)
    updated = torch.eye(4)
    updated[0, 3] = 3.0

    class FakeKeyframe:
        def __init__(self, Rt):
            self.Rt = Rt

        def get_Rt(self):
            return self.Rt

    records = {
        10: {"estimated_Rt": stale.clone(), "is_keyframe": True},
        20: {"estimated_Rt": stale.clone(), "is_keyframe": True},
    }
    backend = SimpleNamespace(
        keyframe_to_frame_id={0: 10, 1: 20, 2: 30},
        scene_model=SimpleNamespace(
            keyframes=[FakeKeyframe(updated), FakeKeyframe(stale)]
        ),
    )

    refresh_keyframe_pose_records(records, backend)

    assert torch.allclose(records[10]["estimated_Rt"], updated)
    assert torch.allclose(records[20]["estimated_Rt"], stale)
    assert 30 not in records


def test_valid_8_neighbor_mean_counts_only_real_neighbors():
    x = torch.arange(9, dtype=torch.float32).view(3, 3)
    means = valid_8_neighbor_mean(x)
    assert torch.isclose(means[0, 0], torch.tensor((1 + 3 + 4) / 3))
    assert torch.isclose(means[1, 1], torch.tensor((0 + 1 + 2 + 3 + 5 + 6 + 7 + 8) / 8))
    assert torch.isclose(means[0, 1], torch.tensor((0 + 2 + 3 + 4 + 5) / 5))


def test_sampling_response_is_zero_on_flat_image():
    image = torch.full((3, 8, 8), 0.5)
    kernel = torch.ones(1, 1, 1, 1)

    assert torch.count_nonzero(get_sampling_response(image, kernel)) == 0
    assert torch.count_nonzero(get_sampling_response(image, kernel, "scharr")) == 0


def test_scharr_sampling_response_prefers_texture_edges():
    image = torch.zeros(3, 8, 8)
    image[:, :, 4:] = 1.0
    kernel = torch.ones(1, 1, 1, 1)

    response = get_sampling_response(image, kernel)

    assert response[2:6, 3:5].max() > 0
    assert response[2:6, 3:5].mean() > response[2:6, 1:3].mean()


def test_test_hold_selects_positive_multiples_only():
    selected = [idx for idx in range(25) if is_test_hold_frame(idx, 8)]
    assert selected == [8, 16, 24]
    assert not is_test_hold_frame(0, 8)
    assert not is_test_hold_frame(8, -1)


def test_bootstrap_poses_are_anchored_to_first_frame():
    Rts = torch.eye(4).repeat(3, 1, 1)
    Rts[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    Rts[1, :3, 3] = torch.tensor([2.0, 4.0, 6.0])
    Rts[2, :3, 3] = torch.tensor([-1.0, 1.0, 5.0])

    anchored = anchor_poses_to_first_frame(Rts)

    assert torch.allclose(anchored[0], torch.eye(4))
    assert torch.allclose(anchored[1], Rts[1] @ torch.linalg.inv(Rts[0]))
    assert torch.allclose(anchored[2], Rts[2] @ torch.linalg.inv(Rts[0]))


def _make_packet(frame_id, is_test=False):
    return {
        "frame_id": frame_id,
        "image": torch.zeros(3, 4, 5),
        "info": {"is_test": is_test, "name": str(frame_id)},
        "desc_kpts": SimpleNamespace(value=frame_id),
        "timings_ms": {},
    }


@pytest.mark.parametrize("direct_sampling,pose_gate", [(True, True), (False, True), (True, False), (False, False)])
def test_main_runs_miniba_mapping_without_final_pose_refinement(monkeypatch, tmp_path, direct_sampling, pose_gate):
    import slam as slam_module

    def fake_rt(frame_id):
        Rt = torch.eye(4)
        Rt[0, 3] = -0.1 * float(frame_id)
        return Rt

    class FakeDataset:
        def __init__(self):
            self.next_frame = 0

        def __len__(self):
            return 12

        def get_image_size(self):
            return 4, 5

        def getnext(self):
            frame_id = self.next_frame
            self.next_frame += 1
            image = torch.full((3, 4, 5), float(frame_id))
            return image, {"is_test": False, "name": str(frame_id), "Rt": fake_rt(frame_id)}

    class FakeFrontend:
        instances = []

        def __init__(self, width, height, args):
            self.matcher = object()
            self.triangulator = object()
            self.focal_init = torch.tensor([10.0])
            self.accepted = []
            self.bootstrap_descs = []
            self.incremental_calls = []
            FakeFrontend.instances.append(self)

        def extract(self, image):
            return SimpleNamespace(value=int(image[0, 0, 0].item()))

        def decide_keyframe(self, frame_id, desc_kpts):
            return KeyframeDecision(True, "fake", float("inf"), 999, 0.0)

        def accept_keyframe(self, frame_id, desc_kpts):
            self.accepted.append(frame_id)

        def initialize_bootstrap(self, descs):
            self.bootstrap_descs = [desc.value for desc in descs]
            return [fake_rt(frame_id) for frame_id in self.bootstrap_descs], torch.tensor([10.0]), 0.1

        def initialize_incremental(self, scene_model, desc_kpts, frame_id, image, will_be_keyframe, is_test=False):
            self.incremental_calls.append((frame_id, will_be_keyframe, is_test))
            return fake_rt(frame_id)


    class FakeKeyframe:
        def __init__(self, index, frame_id):
            self.index = index
            self.frame_id = frame_id

        def get_Rt(self):
            return fake_rt(self.frame_id)

    class FakeSceneModel:
        def __init__(self):
            self.keyframes = []
            self.n_active_gaussians = 123

        def join_optimization_thread(self):
            pass

        def enable_inference_mode(self):
            pass

        def save(self, *args, **kwargs):
            return {}

    class FakeBackend:
        instances = []

        def __init__(self, width, height, args, matcher, triangulator):
            self.scene_model = FakeSceneModel()
            self.keyframe_to_frame_id = {}
            self.bootstrap_frames = []
            self.incremental_frames = []
            self.finalize_calls = 0
            FakeBackend.instances.append(self)

        def _add_keyframe(self, frame_id):
            keyframe_id = len(self.scene_model.keyframes)
            self.scene_model.keyframes.append(FakeKeyframe(keyframe_id, frame_id))
            self.keyframe_to_frame_id[keyframe_id] = frame_id
            return keyframe_id

        def register_bootstrap(self, packets, Rts, focal):
            self.bootstrap_frames = [packet["frame_id"] for packet in packets]
            keyframe_ids = [self._add_keyframe(frame_id) for frame_id in self.bootstrap_frames]
            return SimpleNamespace(
                phases=[{"phase": "bootstrap_mapping", "iters": 80}],
                timings_ms={"keyframe_registration": 1.0, "gaussian_sampling": 1.0, "mapping_optimization": 1.0},
                keyframe_ids=keyframe_ids,
            )

        def register_incremental_keyframe(self, frame_id, image, info, desc_kpts, Rt, focal):
            self.incremental_frames.append(frame_id)
            keyframe_id = self._add_keyframe(frame_id)
            return SimpleNamespace(
                phases=[{"phase": "mapping", "iters": WINDOW_MAPPING_SAMPLE_ITERATIONS}],
                timings_ms={"keyframe_registration": 1.0, "gaussian_sampling": 1.0, "mapping_optimization": 1.0},
                keyframe_ids=[keyframe_id],
            )


        def finalize_mapping(self):
            self.finalize_calls += 1
            return SLAMBackend.finalize_mapping(self)

    class FakeTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix_str(self, value):
            self.last_postfix = value

    args = SimpleNamespace(
        seed=0,
        device="cpu",
        model_path=str(tmp_path),
        num_keyframes_miniba_bootstrap=8,
        eval_lpips_max_size=512,
        eval_lpips_device="cpu",
        direct_gaussian_sampling=direct_sampling,
        pose_gaussian_gate=pose_gate,
        pose_gaussian_min_update_count=7,
        pose_gaussian_min_observed_keyframes=4,
        densification_interval=20,
        densify_grad_threshold=0.0003,
        percent_dense=0.02,
    )
    captured = {}

    monkeypatch.setattr(slam_module, "get_args", lambda: args)
    monkeypatch.setattr(slam_module, "create_dataset", lambda args: (FakeDataset(), "waymo"))
    monkeypatch.setattr(slam_module, "maybe_seed_from_dataset", lambda args, dataset: None)
    monkeypatch.setattr(slam_module, "SLAMFrontend", FakeFrontend)
    monkeypatch.setattr(slam_module, "SLAMBackend", FakeBackend)
    monkeypatch.setattr(slam_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(
        slam_module,
        "compute_render_metrics",
        lambda *args, **kwargs: ([], {"PSNR": None, "SSIM": None, "LPIPS": None}),
    )
    monkeypatch.setattr(
        slam_module,
        "compute_trajectory_outputs",
        lambda records: ([], 0.0, {"estimated": records}),
    )

    def fake_write_outputs(model_path, metrics, frame_ate_rows, render_rows, poses_json):
        captured["metrics"] = metrics
        captured["poses_json"] = poses_json

    monkeypatch.setattr(slam_module, "write_outputs", fake_write_outputs)

    slam_module.main()

    frontend = FakeFrontend.instances[0]
    backend = FakeBackend.instances[0]
    assert frontend.bootstrap_descs == list(range(8))
    assert frontend.incremental_calls == [(idx, True, False) for idx in range(8, 12)]
    assert backend.bootstrap_frames == list(range(8))
    assert backend.incremental_frames == list(range(8, 12))
    assert backend.finalize_calls == 1
    assert captured["metrics"]["num_keyframes"] == 12
    assert captured["metrics"]["num_gaussians"] == 123
    assert captured["metrics"]["direct_gaussian_sampling"] is direct_sampling
    assert captured["metrics"]["pose_gaussian_gate"] is pose_gate
    assert captured["metrics"]["pose_gaussian_min_update_count"] == 7
    assert captured["metrics"]["pose_gaussian_min_observed_keyframes"] == 4
    assert captured["metrics"]["densification_interval"] == 20
    assert captured["metrics"]["densify_grad_threshold"] == 0.0003
    assert captured["metrics"]["percent_dense"] == 0.02
    assert "pose_refinement" not in captured["metrics"]["avg_keyframe_stage_ms"]
    assert "RMSE ATE final refinement [m]" not in captured["metrics"]
    assert "mode" not in captured["metrics"]


def _make_mapping_backend(num_keyframes):
    class FakeSceneModel:
        def __init__(self, num_keyframes):
            self.keyframes = [SimpleNamespace() for _ in range(num_keyframes)]
            self.calls = []
            self.pose_refinement_calls = []
            self.pose_refinement_iterations = []
            self.call_order = []

        def optimize_slam_window(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            summaries = []
            for phase_name, iters, _, _ in args[1]:
                self.call_order.append(phase_name)
                summaries.append({"phase": phase_name, "iters": iters})
            return summaries

        def optimize_keyframe_poses(self, keyframe_ids, iterations_per_keyframe=1):
            self.call_order.append("pose_refinement")
            self.pose_refinement_calls.append(list(keyframe_ids))
            self.pose_refinement_iterations.append(iterations_per_keyframe)
            return {
                "phase": "pose_refinement",
                "iters": len(keyframe_ids) * iterations_per_keyframe,
                "iterations_per_keyframe": iterations_per_keyframe,
            }

        def optimize_keyframe_poses_once(self, keyframe_ids):
            return self.optimize_keyframe_poses(keyframe_ids, iterations_per_keyframe=1)

        def optimize_window_keyframe_poses_once(self, keyframe_ids):
            return self.optimize_keyframe_poses_once(keyframe_ids)

    backend = object.__new__(SLAMBackend)
    backend.args = SimpleNamespace(
        num_iterations=30,
        slam_first_full_iterations=80,
        slam_first_full_sampling_probs=[0.22, 0.19, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03],
        slam_final_extra_iterations=10,
        slam_latest_keyframe_sample_proba=0.25,
    )
    backend.scene_model = FakeSceneModel(num_keyframes)
    return backend


def _make_register_backend(num_existing=12):
    class FakeKeyframe:
        def __init__(self, index):
            self.index = index
            self.info = {}

    class FakeSceneModel:
        def __init__(self, num_existing):
            self.keyframes = [FakeKeyframe(index) for index in range(num_existing)]
            self.added_gaussians = []
            self.pose_refinement_calls = []
            self.call_order = []
            self.calls = []

        def add_new_gaussians(self, keyframe_id):
            self.call_order.append(("add_gaussians", int(keyframe_id)))
            self.added_gaussians.append(int(keyframe_id))

        def optimize_keyframe_poses(self, keyframe_ids, iterations_per_keyframe=1):
            self.call_order.append(("pose_refinement", tuple(keyframe_ids)))
            self.pose_refinement_calls.append(list(keyframe_ids))
            return {
                "phase": "pose_refinement",
                "iters": len(keyframe_ids) * iterations_per_keyframe,
                "iterations_per_keyframe": iterations_per_keyframe,
                "scheduled_keyframe_ids": [
                    keyframe_id
                    for keyframe_id in keyframe_ids
                    for _ in range(iterations_per_keyframe)
                ],
            }

        def optimize_slam_window(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            summaries = []
            for phase_name, iters, _, _ in args[1]:
                self.call_order.append(("mapping", phase_name))
                summaries.append({"phase": phase_name, "iters": iters})
            return summaries


    backend = object.__new__(SLAMBackend)
    backend.width = 5
    backend.height = 4
    backend.keyframe_to_frame_id = {index: index for index in range(num_existing)}
    backend.args = SimpleNamespace(local_ba_disable=True)
    backend.scene_model = FakeSceneModel(num_existing)
    backend._timed = MethodType(lambda self, func, *args, **kwargs: (func(*args, **kwargs), 1.25), backend)

    def add_keyframe(self, frame_id, image, info, desc_kpts, Rt, focal):
        keyframe = FakeKeyframe(len(self.scene_model.keyframes))
        self.scene_model.call_order.append(("add_keyframe", keyframe.index))
        self.scene_model.keyframes.append(keyframe)
        self.keyframe_to_frame_id[keyframe.index] = int(frame_id)
        return keyframe

    backend.add_keyframe = MethodType(add_keyframe, backend)
    return backend


def _make_register_packets(frame_ids):
    return [_make_packet(frame_id) for frame_id in frame_ids]


@pytest.mark.parametrize("num_existing", [8, 18, 19, 20, 23, 50, 51])
def test_register_incremental_adds_gaussians_before_ordinary_mapping(num_existing):
    backend = _make_register_backend(num_existing)
    packet = _make_packet(frame_id=100)

    summary = backend.register_incremental_keyframe(
        packet["frame_id"], packet["image"], packet["info"], packet["desc_kpts"],
        torch.eye(4), torch.tensor([10.0]),
    )

    assert summary.keyframe_ids == [num_existing]
    assert backend.scene_model.added_gaussians == [num_existing]
    assert backend.scene_model.keyframes[-1].info["source_frame_id"] == 100
    assert backend.scene_model.pose_refinement_calls == []
    assert backend.scene_model.call_order == [
        ("add_keyframe", num_existing),
        ("add_gaussians", num_existing),
        ("mapping", "mapping"),
        ("mapping", "joint_mapping"),
    ]
    assert summary.phases == [
        {"phase": "mapping", "iters": 25},
        {"phase": "joint_mapping", "iters": 25},
    ]
    assert "pose_refinement" not in summary.timings_ms
    assert summary.timings_ms["gaussian_sampling"] == 1.25
    assert summary.timings_ms["mapping_optimization"] == 1.25


@pytest.mark.parametrize("bootstrap_size", [8, 10])
def test_register_bootstrap_adds_gaussians_before_bootstrap_mapping(bootstrap_size):
    backend = _make_register_backend(num_existing=0)
    packets = _make_register_packets(range(bootstrap_size))

    summary = backend.register_bootstrap(
        packets, torch.eye(4).repeat(bootstrap_size, 1, 1),
        torch.tensor([10.0]),
    )

    assert backend.scene_model.pose_refinement_calls == []
    assert backend.scene_model.added_gaussians == list(range(bootstrap_size))
    assert summary.phases == [{"phase": "bootstrap_mapping", "iters": 80}]
    assert summary.keyframe_ids == list(range(bootstrap_size))
    assert "local_ba" not in summary.timings_ms
    assert summary.timings_ms["mapping_optimization"] == 1.25
    assert len(backend.scene_model.calls) == 1
    assert "pose_refinement" not in summary.timings_ms
    assert backend.scene_model.call_order == (
        [("add_keyframe", keyframe_id) for keyframe_id in range(bootstrap_size)]
        + [("add_gaussians", keyframe_id) for keyframe_id in range(bootstrap_size)]
        + [("mapping", "bootstrap_mapping")]
    )


def test_mapping_schedule_samples_25_then_25_steps_after_first_incremental_keyframe():
    backend = _make_mapping_backend(num_keyframes=9)

    summary = backend.run_mapping_schedule()
    args, kwargs = backend.scene_model.calls[0]

    assert summary == [
        {"phase": "mapping", "iters": 25},
        {"phase": "joint_mapping", "iters": 25},
    ]
    assert args[0] == list(range(9))
    assert args[1] == [("mapping", 25, False, True), ("joint_mapping", 25, True, True)]
    assert WINDOW_MAPPING_SAMPLE_ITERATIONS == 25
    assert WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS == 25
    assert kwargs["latest_keyframe_sample_proba"] == 0.25
    assert not kwargs.get("optimize_unsampled_keyframes_once", False)
    assert "deterministic_keyframe_iterations" not in kwargs
    assert "keyframe_sampling_probabilities" not in kwargs
    assert backend.scene_model.call_order == ["mapping", "joint_mapping"]
    assert backend.scene_model.pose_refinement_calls == []


@pytest.mark.parametrize("iterations", [80, 123])
def test_bootstrap_mapping_runs_weighted_steps_and_covers_unsampled_frames(iterations):
    backend = _make_mapping_backend(num_keyframes=8)
    backend.args.slam_first_full_iterations = iterations

    summary = backend.run_bootstrap_mapping_schedule()
    args, kwargs = backend.scene_model.calls[0]

    assert summary == [{"phase": "bootstrap_mapping", "iters": iterations}]
    assert args[0] == list(range(8))
    assert args[1] == [("bootstrap_mapping", iterations, True, True)]
    assert kwargs["keyframe_sampling_probabilities"] == [0.22, 0.19, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03]
    assert kwargs["optimize_unsampled_keyframes_once"] is True
    assert backend.scene_model.call_order == ["bootstrap_mapping"]
    assert backend.scene_model.pose_refinement_calls == []


def test_mapping_schedule_samples_all_50_keyframes_when_legacy_window_first_fills():
    backend = _make_mapping_backend(num_keyframes=50)

    summary = backend.run_mapping_schedule()
    args, kwargs = backend.scene_model.calls[0]

    assert summary == [
        {"phase": "mapping", "iters": 25},
        {"phase": "joint_mapping", "iters": 25},
    ]
    assert args[0] == list(range(50))
    assert args[1] == [("mapping", 25, False, True), ("joint_mapping", 25, True, True)]
    assert kwargs["latest_keyframe_sample_proba"] == 0.25
    assert not kwargs.get("optimize_unsampled_keyframes_once", False)
    assert "deterministic_keyframe_iterations" not in kwargs
    assert "keyframe_sampling_probabilities" not in kwargs
    assert backend.scene_model.call_order == ["mapping", "joint_mapping"]
    assert backend.scene_model.pose_refinement_calls == []


def test_mapping_schedule_samples_all_registered_keyframes():
    backend = _make_mapping_backend(num_keyframes=51)

    summary = backend.run_mapping_schedule()
    args, kwargs = backend.scene_model.calls[0]

    assert summary == [
        {"phase": "mapping", "iters": 25},
        {"phase": "joint_mapping", "iters": 25},
    ]
    assert args[0] == list(range(51))
    assert args[1] == [("mapping", 25, False, True), ("joint_mapping", 25, True, True)]
    assert kwargs["latest_keyframe_sample_proba"] == 0.25
    assert not kwargs.get("optimize_unsampled_keyframes_once", False)
    assert "deterministic_keyframe_iterations" not in kwargs
    assert len(backend.scene_model.calls) == 1
    assert backend.scene_model.call_order == ["mapping", "joint_mapping"]
    assert backend.scene_model.pose_refinement_calls == []


def test_final_mapping_schedule_is_disabled_after_all_incremental_keyframes():
    backend = _make_mapping_backend(num_keyframes=51)

    summary = backend.run_final_mapping_schedule()

    assert summary == []
    assert backend.scene_model.calls == []


@pytest.mark.parametrize("num_keyframes", [0, 1, 8, 51])
def test_finalize_mapping_is_noop_with_or_without_keyframes(num_keyframes):
    backend = _make_mapping_backend(num_keyframes=num_keyframes)

    summary = backend.finalize_mapping()

    assert summary.phases == []
    assert summary.timings_ms == {}
    assert backend.scene_model.call_order == []
    assert backend.scene_model.calls == []
    assert backend.scene_model.pose_refinement_calls == []
    assert backend.scene_model.pose_refinement_iterations == []


def test_final_mapping_skips_when_no_post_full_keyframe_exists():
    backend = _make_mapping_backend(num_keyframes=50)

    assert backend.run_final_mapping_schedule() == []
    assert backend.scene_model.calls == []


def _import_scene_model_with_stubs(monkeypatch):
    import importlib
    import sys
    import types

    fused_ssim_module = types.ModuleType("fused_ssim")
    fused_ssim_module.fused_ssim = lambda *args, **kwargs: torch.tensor(0.5)

    raster_module = types.ModuleType("diff_gaussian_rasterization")
    raster_module.GaussianRasterizationSettings = object
    raster_module.GaussianRasterizer = object
    raster_module.adamUpdate = lambda *args, **kwargs: None
    raster_module.adamUpdateBasic = lambda *args, **kwargs: None

    simple_knn_module = types.ModuleType("simple_knn")
    simple_knn_c_module = types.ModuleType("simple_knn._C")
    simple_knn_c_module.distIndex2 = lambda *args, **kwargs: None

    feature_detector_module = types.ModuleType("poses.feature_detector")
    feature_detector_module.DescribedKeypoints = object
    matcher_module = types.ModuleType("poses.matcher")
    matcher_module.Matcher = object
    guided_mvs_module = types.ModuleType("poses.guided_mvs")
    guided_mvs_module.GuidedMVS = object
    keyframe_module = types.ModuleType("scene.keyframe")
    keyframe_module.Keyframe = object
    anchor_module = types.ModuleType("scene.anchor")
    anchor_module.Anchor = object

    monkeypatch.setitem(sys.modules, "fused_ssim", fused_ssim_module)
    monkeypatch.setitem(sys.modules, "diff_gaussian_rasterization", raster_module)
    monkeypatch.setitem(sys.modules, "simple_knn", simple_knn_module)
    monkeypatch.setitem(sys.modules, "simple_knn._C", simple_knn_c_module)
    monkeypatch.setitem(sys.modules, "poses.feature_detector", feature_detector_module)
    monkeypatch.setitem(sys.modules, "poses.matcher", matcher_module)
    monkeypatch.setitem(sys.modules, "poses.guided_mvs", guided_mvs_module)
    monkeypatch.setitem(sys.modules, "scene.keyframe", keyframe_module)
    monkeypatch.setitem(sys.modules, "scene.anchor", anchor_module)
    monkeypatch.delitem(sys.modules, "scene.optimizers", raising=False)
    monkeypatch.delitem(sys.modules, "scene.scene_model", raising=False)
    return importlib.import_module("scene.scene_model")


def _make_sampling_scene_model(scene_model, seed):
    model = object.__new__(scene_model.SceneModel)
    model.slam_window_sampling_seed = seed
    model.slam_window_sampling_rng = np.random.RandomState(seed)
    model.slam_latest_keyframe_sample_proba = 0.25

    def optimization_step(**kwargs):
        return {"loss": float(kwargs["keyframe_id"])}

    model.optimization_step = optimization_step
    return model


@pytest.mark.parametrize("latest_probability", [0.25, 1.0])
def test_mapping_schedule_runs_real_sampler_in_25_then_25_step_phases(
    monkeypatch, latest_probability
):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    backend = _make_mapping_backend(num_keyframes=9)
    backend.args.slam_latest_keyframe_sample_proba = latest_probability
    model = _make_sampling_scene_model(scene_model, seed=123)
    model.keyframes = backend.scene_model.keyframes
    backend.scene_model = model
    calls = []

    def optimization_step(**kwargs):
        calls.append(kwargs)
        return {"loss": float(len(calls))}

    model.optimization_step = optimization_step
    summaries = backend.run_mapping_schedule()

    assert len(calls) == 50
    assert [call["optimize_pose"] for call in calls] == [False] * 25 + [True] * 25
    assert all(call["optimize_gaussians"] is True for call in calls)
    assert [phase["phase"] for phase in summaries] == ["mapping", "joint_mapping"]
    assert [phase["max_iters"] for phase in summaries] == [25, 25]
    assert [phase["iters"] for phase in summaries] == [25, 25]
    assert [phase["sampled_iters"] for phase in summaries] == [25, 25]
    assert [phase["optimize_pose"] for phase in summaries] == [False, True]
    assert all(phase["optimize_gaussians"] is True for phase in summaries)
    assert [(phase["initial_loss"], phase["final_loss"]) for phase in summaries] == [
        (1.0, 25.0), (26.0, 50.0),
    ]
    for phase, phase_calls in zip(summaries, [calls[:25], calls[25:]]):
        sampled_ids = phase["sampled_keyframe_ids"]
        assert sampled_ids == [call["keyframe_id"] for call in phase_calls]
        assert phase["coverage_iters"] == 0
        assert phase["coverage_keyframe_ids"] == []
        assert phase["scheduled_keyframe_ids"] == []
        assert phase["sampling_seed"] == 123
        assert set(sampled_ids) <= set(range(9))
        assert 8 in sampled_ids
        if latest_probability == 1.0:
            assert set(sampled_ids) == {8}
        else:
            assert any(keyframe_id < 8 for keyframe_id in sampled_ids)


def test_optimize_slam_window_sampling_is_reproducible_with_same_seed(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    first = _make_sampling_scene_model(scene_model, seed=123)
    second = _make_sampling_scene_model(scene_model, seed=123)

    first_summary = scene_model.SceneModel.optimize_slam_window(
        first,
        [10, 11, 12, 13, 14],
        [("mapping", 20, True, False)],
        latest_keyframe_sample_proba=0.25,
    )
    np.random.rand(100)
    second_summary = scene_model.SceneModel.optimize_slam_window(
        second,
        [10, 11, 12, 13, 14],
        [("mapping", 20, True, False)],
        latest_keyframe_sample_proba=0.25,
    )

    assert first_summary[0]["sampled_keyframe_ids"] == second_summary[0]["sampled_keyframe_ids"]
    assert first_summary[0]["sampling_seed"] == 123
    assert second_summary[0]["sampling_seed"] == 123


@pytest.mark.parametrize("explicit_thresholds", [False, True])
def test_pose_optimization_gaussian_mask_requires_updates_and_distinct_views(
    monkeypatch, explicit_thresholds
):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {"xyz": {"val": torch.zeros(4, 3)}}
    model.gaussian_optimization_counts = torch.tensor([4, 5, 5, 8], dtype=torch.int32)
    model.gaussian_observed_keyframe_ids = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, -1],
            [1, 2, 3],
            [1, -1, -1],
        ],
        dtype=torch.int32,
    )
    model.gaussian_observed_keyframe_counts = torch.tensor([3, 2, 3, 1], dtype=torch.int16)
    if explicit_thresholds:
        model.pose_gaussian_min_update_count = 5
        model.pose_gaussian_min_observed_keyframes = 3

    mask = scene_model.SceneModel.pose_optimization_gaussian_mask(model)

    assert mask.tolist() == [False, False, True, False]

    scene_model.SceneModel._increment_gaussian_optimization_counts(
        model,
        torch.tensor([True, False, True, False]),
    )

    assert model.gaussian_optimization_counts.tolist() == [5, 5, 6, 8]
    assert scene_model.SceneModel.pose_optimization_gaussian_mask(model).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_pose_optimization_gaussian_mask_allows_new_gaussians_with_zero_thresholds(
    monkeypatch
):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {"xyz": {"val": torch.zeros(2, 3)}}
    model.pose_gaussian_min_update_count = 0
    model.pose_gaussian_min_observed_keyframes = 0

    assert model.pose_optimization_gaussian_mask().tolist() == [True, True]


def test_gaussian_observed_keyframes_count_distinct_keyframes(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {"xyz": {"val": torch.zeros(2, 3)}}
    model.gaussian_observed_keyframe_ids = torch.full((2, 3), -1, dtype=torch.int32)
    model.gaussian_observed_keyframe_counts = torch.zeros(2, dtype=torch.int16)
    model.pose_gaussian_min_observed_keyframes = 3

    visibility = torch.tensor([True, False])
    scene_model.SceneModel._mark_gaussians_observed_by_keyframe(model, visibility, 7)
    scene_model.SceneModel._mark_gaussians_observed_by_keyframe(model, visibility, 7)
    scene_model.SceneModel._mark_gaussians_observed_by_keyframe(model, visibility, 8)
    scene_model.SceneModel._mark_gaussians_observed_by_keyframe(model, visibility, 9)
    scene_model.SceneModel._mark_gaussians_observed_by_keyframe(model, visibility, 10)

    assert model.gaussian_observed_keyframe_ids.tolist() == [
        [7, 8, 9],
        [-1, -1, -1],
    ]
    assert model.gaussian_observed_keyframe_counts.tolist() == [3, 0]


def test_gaussian_observed_keyframes_can_track_five_distinct_views(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {"xyz": {"val": torch.zeros(1, 3)}}
    model.pose_gaussian_min_observed_keyframes = 5
    model.gaussian_observed_keyframe_ids = torch.full((1, 3), -1, dtype=torch.int32)
    model.gaussian_observed_keyframe_counts = torch.zeros(1, dtype=torch.int16)

    visibility = torch.tensor([True])
    for keyframe_id in range(5):
        scene_model.SceneModel._mark_gaussians_observed_by_keyframe(
            model,
            visibility,
            keyframe_id,
        )

    assert model.gaussian_observed_keyframe_ids.tolist() == [[0, 1, 2, 3, 4]]
    assert model.gaussian_observed_keyframe_counts.tolist() == [5]


def test_gaussian_optimization_counts_follow_add_and_prune(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {
        "xyz": {"val": torch.arange(9, dtype=torch.float32).view(3, 3)}
    }
    model.gaussian_optimization_counts = torch.tensor([3, 12, 7], dtype=torch.int32)
    model.gaussian_observed_keyframe_ids = torch.tensor(
        [[1, -1, -1], [2, 3, -1], [4, 5, 6]],
        dtype=torch.int32,
    )
    model.gaussian_observed_keyframe_counts = torch.tensor([1, 2, 3], dtype=torch.int16)
    model.pose_gaussian_min_observed_keyframes = 3

    class FakeOptimizer:
        def add_and_prune(self, extension_tensors, valid_mask):
            old_xyz = model.gaussian_params["xyz"]["val"]
            model.gaussian_params["xyz"]["val"] = torch.cat(
                [old_xyz[valid_mask], extension_tensors["xyz"]],
                dim=0,
            )

    model.optimizer = FakeOptimizer()
    extension_tensors = {"xyz": torch.zeros(2, 3)}

    scene_model.SceneModel._apply_gaussian_add_and_prune(
        model,
        extension_tensors,
        torch.tensor([False, True, True]),
        extension_observer_ids=torch.tensor(
            [[9, -1, -1], [10, 11, -1]],
            dtype=torch.int32,
        ),
    )

    assert model.gaussian_params["xyz"]["val"].shape[0] == 4
    assert model.gaussian_optimization_counts.tolist() == [12, 7, 0, 0]
    assert model.gaussian_observed_keyframe_ids.tolist() == [
        [2, 3, -1],
        [4, 5, 6],
        [9, -1, -1],
        [10, 11, -1],
    ]
    assert model.gaussian_observed_keyframe_counts.tolist() == [2, 3, 1, 2]


@pytest.mark.parametrize("is_test", [False, True])
@pytest.mark.parametrize("optimize_gaussians", [False, True])
def test_optimization_step_uses_mature_gaussians_for_pose_branch(
    monkeypatch, optimize_gaussians, is_test
):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    model.gaussian_params = {
        "xyz": {"val": torch.ones(3, 3, requires_grad=True)}
    }
    model.gaussian_optimization_counts = torch.tensor([5, 0, 0], dtype=torch.int32)
    model.gaussian_observed_keyframe_ids = torch.tensor(
        [[0, 1, 2], [-1, -1, -1], [-1, -1, -1]],
        dtype=torch.int32,
    )
    model.gaussian_observed_keyframe_counts = torch.tensor([3, 0, 0], dtype=torch.int16)
    model.pose_gaussian_min_update_count = 5
    model.pose_gaussian_min_observed_keyframes = 3
    model.lambda_dssim = 0.0
    model.geometry_loss_weight = 0.0
    model.valid_Rt_cache = torch.ones(1, dtype=torch.bool)
    model.last_trained_id = -1

    class FakeOptimizer:
        def zero_grad(self):
            model.gaussian_params["xyz"]["val"].grad = None

        def step(self, visibility, n_radii):
            self.visibility = visibility
            self.n_radii = n_radii

    model.optimizer = FakeOptimizer()

    pose_param = torch.nn.Parameter(torch.tensor(0.25))
    depth_scale = torch.nn.Parameter(torch.tensor(0.5))
    keyframe = SimpleNamespace(
        index=0,
        device=SimpleNamespace(type="cuda"),
        pyr_lvl=0,
        image_pyr=[torch.zeros(3, 2, 2)],
        mask_pyr=None,
        depth_loss_weight=1.0,
        desc_kpts=None,
        info={"is_test": is_test},
        pose_param=pose_param,
        depth_scale=depth_scale,
        step_called=False,
        schedule_steps=0,
    )

    def zero_grad():
        pose_param.grad = None
        depth_scale.grad = None

    def get_mono_idepth(lvl):
        return depth_scale * torch.ones(1, 2, 2)

    def step(optimize_only_pose=False):
        keyframe.step_called = True

    def advance_mapping_schedule():
        keyframe.schedule_steps += 1

    keyframe.zero_grad = zero_grad
    keyframe.get_mono_idepth = get_mono_idepth
    keyframe.step = step
    keyframe.advance_mapping_schedule = advance_mapping_schedule
    model.keyframes = [keyframe]
    render_calls = []

    def render_from_id(
        keyframe_id,
        pyr_lvl=0,
        bg=None,
        detach_pose=False,
        gaussian_mask=None,
        detach_gaussians=False,
        **kwargs,
    ):
        render_calls.append({
            "gaussian_mask": gaussian_mask,
            "detach_pose": detach_pose,
            "detach_gaussians": detach_gaussians,
        })
        gaussian_value = model.gaussian_params["xyz"]["val"].sum()
        if gaussian_mask is not None:
            gaussian_value = model.gaussian_params["xyz"]["val"][gaussian_mask].sum()
        if detach_gaussians:
            gaussian_value = gaussian_value.detach()
        pose_value = pose_param
        if detach_pose:
            pose_value = pose_value.detach()
        value = gaussian_value + pose_value
        return {
            "render": value.expand(3, 2, 2),
            "invdepth": (value + 1.0).expand(1, 2, 2),
            "visibility_filter": torch.tensor([True, False, True]),
            "radii": torch.ones(3),
            "screenspace_points": torch.zeros_like(model.gaussian_params["xyz"]["val"]),
        }

    model.render_from_id = render_from_id

    result = scene_model.SceneModel.optimization_step(
        model,
        keyframe_id=0,
        optimize_pose=True,
        optimize_gaussians=optimize_gaussians,
        optimize_only_pose=not optimize_gaussians,
    )

    assert result["pose_gaussians_used"] == 1
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert depth_scale.grad is not None and depth_scale.grad.abs().item() > 0
    assert result["depth"] == result["render_depth"] > 0
    assert keyframe.step_called is True
    assert keyframe.schedule_steps == int(optimize_gaussians)
    assert len(render_calls) == (2 if optimize_gaussians else 1)
    if optimize_gaussians:
        assert render_calls[0] == {
            "gaussian_mask": None,
            "detach_pose": True,
            "detach_gaussians": False,
        }
    pose_render = render_calls[-1]
    assert pose_render["gaussian_mask"].tolist() == [True, False, False]
    assert pose_render["detach_gaussians"] is True
    assert pose_render["detach_pose"] is False
    gaussian_step_ran = optimize_gaussians and not is_test
    assert hasattr(model.optimizer, "visibility") is gaussian_step_ran
    assert model.gaussian_optimization_counts.tolist() == (
        [6, 0, 1] if gaussian_step_ran else [5, 0, 0]
    )
    assert model.gaussian_observed_keyframe_ids.tolist() == [
        [0, 1, 2],
        [-1, -1, -1],
        [0, -1, -1] if gaussian_step_ran else [-1, -1, -1],
    ]
    assert model.gaussian_observed_keyframe_counts.tolist() == (
        [3, 0, 1] if gaussian_step_ran else [3, 0, 0]
    )


def test_optimize_slam_window_sampling_changes_with_different_seed(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    first = _make_sampling_scene_model(scene_model, seed=123)
    second = _make_sampling_scene_model(scene_model, seed=456)

    first_summary = scene_model.SceneModel.optimize_slam_window(
        first,
        [10, 11, 12, 13, 14],
        [("mapping", 20, True, False)],
        latest_keyframe_sample_proba=0.25,
    )
    second_summary = scene_model.SceneModel.optimize_slam_window(
        second,
        [10, 11, 12, 13, 14],
        [("mapping", 20, True, False)],
        latest_keyframe_sample_proba=0.25,
    )

    assert first_summary[0]["sampled_keyframe_ids"] != second_summary[0]["sampled_keyframe_ids"]


def test_optimize_slam_window_explicit_probabilities_use_independent_rng(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    first = _make_sampling_scene_model(scene_model, seed=99)
    second = _make_sampling_scene_model(scene_model, seed=99)

    first_summary = scene_model.SceneModel.optimize_slam_window(
        first,
        [10, 11, 12],
        [("bootstrap_mapping", 12, True, False)],
        keyframe_sampling_probabilities=[0.5, 0.3, 0.2],
    )
    np.random.rand(100)
    second_summary = scene_model.SceneModel.optimize_slam_window(
        second,
        [10, 11, 12],
        [("bootstrap_mapping", 12, True, False)],
        keyframe_sampling_probabilities=[0.5, 0.3, 0.2],
    )

    assert first_summary[0]["sampled_keyframe_ids"] == second_summary[0]["sampled_keyframe_ids"]
    assert first_summary[0]["sampling_seed"] == 99


def test_optimize_slam_window_explicit_probabilities_change_with_different_seed(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    first = _make_sampling_scene_model(scene_model, seed=99)
    second = _make_sampling_scene_model(scene_model, seed=100)

    first_summary = scene_model.SceneModel.optimize_slam_window(
        first,
        [10, 11, 12],
        [("bootstrap_mapping", 30, True, False)],
        keyframe_sampling_probabilities=[0.5, 0.3, 0.2],
    )
    second_summary = scene_model.SceneModel.optimize_slam_window(
        second,
        [10, 11, 12],
        [("bootstrap_mapping", 30, True, False)],
        keyframe_sampling_probabilities=[0.5, 0.3, 0.2],
    )

    assert first_summary[0]["sampled_keyframe_ids"] != second_summary[0]["sampled_keyframe_ids"]


def test_optimize_slam_window_catches_up_unsampled_keyframes(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    calls = []

    def optimization_step(**kwargs):
        calls.append(kwargs)
        return {"loss": float(len(calls))}

    model.optimization_step = optimization_step

    summary = scene_model.SceneModel.optimize_slam_window(
        model,
        [10, 11, 12],
        [("mapping", 2, True, False)],
        keyframe_sampling_probabilities=[1.0, 0.0, 0.0],
        optimize_unsampled_keyframes_once=True,
    )

    assert [call["keyframe_id"] for call in calls] == [10, 10, 11, 12]
    assert all(call["optimize_pose"] is True for call in calls)
    assert all(call["optimize_gaussians"] is False for call in calls)
    assert summary[0]["iters"] == 4
    assert summary[0]["sampled_iters"] == 2
    assert summary[0]["coverage_iters"] == 2
    assert summary[0]["sampled_keyframe_ids"] == [10, 10]
    assert summary[0]["coverage_keyframe_ids"] == [11, 12]
    assert summary[0]["initial_loss"] == 1.0
    assert summary[0]["final_loss"] == 4.0


def test_optimize_keyframe_poses_once_runs_pose_only_refinement(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    calls = []

    def optimization_step(**kwargs):
        calls.append(kwargs)
        return {"loss": float(len(calls))}

    model.optimization_step = optimization_step

    summary = scene_model.SceneModel.optimize_keyframe_poses_once(
        model, [10, 11, 12]
    )

    assert [call["keyframe_id"] for call in calls] == [10, 11, 12]
    assert all(call["optimize_pose"] is True for call in calls)
    assert all(call["optimize_gaussians"] is False for call in calls)
    assert all(call["optimize_only_pose"] is True for call in calls)
    assert summary["phase"] == "pose_refinement"
    assert summary["iters"] == 3
    assert summary["scheduled_keyframe_ids"] == [10, 11, 12]
    assert summary["optimize_only_pose"] is True
    assert summary["initial_loss"] == 1.0
    assert summary["final_loss"] == 3.0


def test_optimize_keyframe_poses_runs_multiple_pose_only_iterations(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    calls = []

    def optimization_step(**kwargs):
        calls.append(kwargs)
        return {"loss": float(len(calls))}

    model.optimization_step = optimization_step

    summary = scene_model.SceneModel.optimize_keyframe_poses(
        model, [10, 11], iterations_per_keyframe=5
    )

    assert [call["keyframe_id"] for call in calls] == [10] * 5 + [11] * 5
    assert all(call["optimize_pose"] is True for call in calls)
    assert all(call["optimize_gaussians"] is False for call in calls)
    assert all(call["optimize_only_pose"] is True for call in calls)
    assert summary["phase"] == "pose_refinement"
    assert summary["max_iters"] == 10
    assert summary["iters"] == 10
    assert summary["iterations_per_keyframe"] == 5
    assert summary["scheduled_keyframe_ids"] == [10] * 5 + [11] * 5
    assert summary["initial_loss"] == 1.0
    assert summary["final_loss"] == 10.0


def test_keyframe_pose_only_step_freezes_non_pose_state(monkeypatch):
    import importlib
    import sys
    import types

    raster_module = types.ModuleType("diff_gaussian_rasterization")
    raster_module.adamUpdate = lambda *args, **kwargs: None
    raster_module.adamUpdateBasic = lambda *args, **kwargs: None
    feature_detector_module = types.ModuleType("poses.feature_detector")
    feature_detector_module.DescribedKeypoints = object
    triangulator_module = types.ModuleType("poses.triangulator")
    triangulator_module.Triangulator = object
    dense_extractor_module = types.ModuleType("scene.dense_extractor")
    dense_extractor_module.DenseExtractor = object
    mono_depth_module = types.ModuleType("scene.mono_depth")
    mono_depth_module.MonoDepthEstimator = object
    mono_depth_module.align_depth = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "diff_gaussian_rasterization", raster_module)
    monkeypatch.setitem(sys.modules, "poses.feature_detector", feature_detector_module)
    monkeypatch.setitem(sys.modules, "poses.triangulator", triangulator_module)
    monkeypatch.setitem(sys.modules, "scene.dense_extractor", dense_extractor_module)
    monkeypatch.setitem(sys.modules, "scene.mono_depth", mono_depth_module)
    monkeypatch.delitem(sys.modules, "scene.optimizers", raising=False)
    monkeypatch.delitem(sys.modules, "scene.keyframe", raising=False)

    keyframe_module = importlib.import_module("scene.keyframe")
    keyframe = object.__new__(keyframe_module.Keyframe)

    params = {
        "rW2C": {"val": torch.nn.Parameter(torch.zeros(3, 2))},
        "tW2C": {"val": torch.nn.Parameter(torch.zeros(3))},
        "depth_scale": {"val": torch.nn.Parameter(torch.ones(1))},
        "depth_offset": {"val": torch.nn.Parameter(torch.zeros(1))},
        "exposure": {"val": torch.nn.Parameter(torch.eye(3, 4))},
    }
    for param in params.values():
        param["val"].grad = torch.ones_like(param["val"])

    class FakeOptimizer:
        def __init__(self, params):
            self.params = params

        def step(self):
            for param in self.params.values():
                value = param["val"]
                if value.grad is not None:
                    value.data.add_(value.grad)

    keyframe.optimizer = FakeOptimizer(params)
    keyframe.depth_loss_weight = 2.0
    keyframe.depth_loss_weight_decay = 0.5
    keyframe.num_steps = 4
    keyframe.pyr_lvl = 1

    before_depth_scale = params["depth_scale"]["val"].detach().clone()
    before_depth_offset = params["depth_offset"]["val"].detach().clone()
    before_exposure = params["exposure"]["val"].detach().clone()

    keyframe_module.Keyframe.step(keyframe, optimize_only_pose=True)

    assert torch.allclose(params["rW2C"]["val"], torch.ones(3, 2))
    assert torch.allclose(params["tW2C"]["val"], torch.ones(3))
    assert torch.allclose(params["depth_scale"]["val"], before_depth_scale)
    assert torch.allclose(params["depth_offset"]["val"], before_depth_offset)
    assert torch.allclose(params["exposure"]["val"], before_exposure)
    assert keyframe.depth_loss_weight == 2.0
    assert keyframe.num_steps == 4
    assert keyframe.pyr_lvl == 1


def test_optimize_slam_window_uses_deterministic_keyframe_iterations(monkeypatch):
    scene_model = _import_scene_model_with_stubs(monkeypatch)
    model = object.__new__(scene_model.SceneModel)
    calls = []

    def optimization_step(**kwargs):
        calls.append(kwargs)
        return {"loss": float(len(calls))}

    model.optimization_step = optimization_step

    summary = scene_model.SceneModel.optimize_slam_window(
        model,
        [10, 11, 12],
        [("mapping", 6, True, False)],
        latest_keyframe_sample_proba=0.25,
        keyframe_sampling_probabilities=[1.0, 0.0, 0.0],
        optimize_unsampled_keyframes_once=True,
        deterministic_keyframe_iterations=[1, 2, 3],
    )

    assert [call["keyframe_id"] for call in calls] == [10, 11, 11, 12, 12, 12]
    assert all(call["optimize_pose"] is True for call in calls)
    assert all(call["optimize_gaussians"] is False for call in calls)
    assert summary[0]["iters"] == 6
    assert summary[0]["sampled_iters"] == 0
    assert summary[0]["coverage_iters"] == 0
    assert summary[0]["sampled_keyframe_ids"] == []
    assert summary[0]["coverage_keyframe_ids"] == []
    assert summary[0]["scheduled_keyframe_ids"] == [10, 11, 11, 12, 12, 12]
    assert summary[0]["initial_loss"] == 1.0
    assert summary[0]["final_loss"] == 6.0


def test_window_keyframe_sampling_uses_latest_25_percent_by_default():
    class FakeRng:
        def __init__(self, random_value, choice_index=0):
            self.random_value = random_value
            self.choice_index = choice_index

        def random(self):
            return self.random_value

        def choice(self, choices):
            return choices[self.choice_index]

    probabilities = window_keyframe_sampling_probabilities(list(range(5)))
    assert probabilities[4] == 0.25
    for keyframe_id in range(4):
        assert probabilities[keyframe_id] == 0.75 / 4

    assert sample_window_keyframe_id([0, 1, 2], rng=FakeRng(0.1)) == 2
    assert sample_window_keyframe_id([0, 1, 2], rng=FakeRng(0.9, choice_index=1)) == 1


def test_explicit_keyframe_sampling_probabilities_are_ordered_by_window():
    class FakeRng:
        def __init__(self, random_value):
            self.random_value = random_value

        def random(self):
            return self.random_value

    probabilities = normalize_keyframe_sampling_probabilities([10, 11, 12], [2, 3, 5])
    assert probabilities == {10: 0.2, 11: 0.3, 12: 0.5}
    assert sample_keyframe_id_with_probabilities(
        [10, 11, 12], [0.22, 0.19, 0.59], rng=FakeRng(0.21)
    ) == 10
    assert sample_keyframe_id_with_probabilities(
        [10, 11, 12], [0.22, 0.19, 0.59], rng=FakeRng(0.22)
    ) == 11
    assert sample_keyframe_id_with_probabilities(
        [10, 11, 12], [0.22, 0.19, 0.59], rng=FakeRng(0.42)
    ) == 12


@pytest.mark.parametrize("depth_weight", [0.0, 0.5, 2.0])
def test_mapping_loss_weights_photometric_depth_and_geometry_values_and_gradients(depth_weight):
    l1_loss = torch.tensor(2.0, requires_grad=True)
    ssim_loss = torch.tensor(4.0, requires_grad=True)
    depth_loss = torch.tensor(8.0, requires_grad=True)
    geometry_loss = torch.tensor(16.0, requires_grad=True)
    total_loss, photometric_loss = combine_mapping_losses(
        l1_loss,
        ssim_loss,
        depth_loss,
        geometry_loss,
        lambda_dssim=0.25,
        render_depth_weight=depth_weight,
        geometry_weight=0.125,
    )

    assert torch.isclose(photometric_loss, torch.tensor(2.5))
    assert total_loss.item() == pytest.approx(4.5 + depth_weight * 8.0)
    total_loss.backward()
    assert l1_loss.grad.item() == pytest.approx(0.75)
    assert ssim_loss.grad.item() == pytest.approx(0.25)
    assert depth_loss.grad.item() == pytest.approx(depth_weight)
    assert geometry_loss.grad.item() == pytest.approx(0.125)


def test_resize_longest_side_only_downsamples_large_images():
    image = torch.zeros(3, 1000, 1500)
    resized = resize_longest_side(image, 512)

    assert resized.shape == (3, 341, 512)
    assert resize_longest_side(image, 0).shape == image.shape
    assert resize_longest_side(torch.zeros(3, 100, 120), 512).shape == (3, 100, 120)


def test_evaluate_test_frame_does_not_delete_missing_invdepth(monkeypatch):
    import importlib
    import sys
    import types

    fused_ssim_module = types.ModuleType("fused_ssim")
    fused_ssim_module.fused_ssim = lambda *args, **kwargs: torch.tensor(0.5)

    raster_module = types.ModuleType("diff_gaussian_rasterization")
    raster_module.GaussianRasterizationSettings = object
    raster_module.GaussianRasterizer = object
    raster_module.adamUpdate = lambda *args, **kwargs: None
    raster_module.adamUpdateBasic = lambda *args, **kwargs: None

    simple_knn_module = types.ModuleType("simple_knn")
    simple_knn_c_module = types.ModuleType("simple_knn._C")
    simple_knn_c_module.distIndex2 = lambda *args, **kwargs: None

    feature_detector_module = types.ModuleType("poses.feature_detector")
    feature_detector_module.DescribedKeypoints = object
    matcher_module = types.ModuleType("poses.matcher")
    matcher_module.Matcher = object
    guided_mvs_module = types.ModuleType("poses.guided_mvs")
    guided_mvs_module.GuidedMVS = object
    keyframe_module = types.ModuleType("scene.keyframe")
    keyframe_module.Keyframe = object
    anchor_module = types.ModuleType("scene.anchor")
    anchor_module.Anchor = object

    monkeypatch.setitem(sys.modules, "fused_ssim", fused_ssim_module)
    monkeypatch.setitem(sys.modules, "diff_gaussian_rasterization", raster_module)
    monkeypatch.setitem(sys.modules, "simple_knn", simple_knn_module)
    monkeypatch.setitem(sys.modules, "simple_knn._C", simple_knn_c_module)
    monkeypatch.setitem(sys.modules, "poses.feature_detector", feature_detector_module)
    monkeypatch.setitem(sys.modules, "poses.matcher", matcher_module)
    monkeypatch.setitem(sys.modules, "poses.guided_mvs", guided_mvs_module)
    monkeypatch.setitem(sys.modules, "scene.keyframe", keyframe_module)
    monkeypatch.setitem(sys.modules, "scene.anchor", anchor_module)
    monkeypatch.delitem(sys.modules, "scene.optimizers", raising=False)
    monkeypatch.delitem(sys.modules, "scene.scene_model", raising=False)

    scene_model = importlib.import_module("scene.scene_model")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    model = object.__new__(scene_model.SceneModel)
    keyframe = SimpleNamespace(
        index=0,
        info={"is_test": True},
        image_pyr=[torch.full((3, 4, 5), 0.5)],
        mask_pyr=None,
    )
    model.keyframes = [keyframe]
    model.harmonize_test_exposure = lambda: None
    model.render_from_id = lambda keyframe_id, pyr_lvl=0: {
        "render": torch.zeros(3, 4, 5)
    }

    metrics = scene_model.SceneModel.evaluate(
        model,
        eval_poses=False,
        with_LPIPS=False,
        all=True,
    )

    assert set(metrics) == {"PSNR", "SSIM"}
    assert metrics["SSIM"] == 0.5
