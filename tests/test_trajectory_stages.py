from types import SimpleNamespace

import numpy as np
import pytest
import torch

import slam as slam_module
from slam_backend import SLAMBackend
from slam_frontend import KeyframeDecision


def _pose(x):
    Rt = torch.eye(4)
    Rt[0, 3] = float(x)
    return Rt


def test_pose_record_and_stage_snapshots_own_pose_storage():
    estimated = _pose(2).requires_grad_()
    gt = _pose(3)
    records = {}
    slam_module.add_pose_record(records, 20, {"Rt": gt}, estimated, is_keyframe=True)
    slam_module.add_pose_record(records, 10, {}, _pose(1))
    records[20]["metadata"] = {"values": [1]}

    original_snapshot = slam_module.snapshot_pose_records(records)
    frontend_pose = _pose(4).requires_grad_()
    frontend_snapshot = slam_module.snapshot_pose_records(records, {20: frontend_pose})
    with torch.no_grad():
        estimated[0, 3] = 20
        gt[0, 3] = 30
        frontend_pose[0, 3] = 40

    assert records[20]["estimated_Rt"][0, 3].item() == 2
    assert records[20]["gt_Rt"][0, 3].item() == 3
    assert not records[20]["estimated_Rt"].requires_grad
    assert [record["frame_id"] for record in original_snapshot] == [10, 20]
    assert frontend_snapshot[0]["estimated_Rt"] is None
    assert frontend_snapshot[1]["estimated_Rt"][0, 3].item() == 4
    assert not frontend_snapshot[1]["estimated_Rt"].requires_grad

    records[20]["estimated_Rt"][0, 3] = 200
    records[20]["gt_Rt"][0, 3] = 300
    records[20]["metadata"]["values"].append(2)
    assert original_snapshot[1]["estimated_Rt"][0, 3].item() == 2
    assert original_snapshot[1]["gt_Rt"][0, 3].item() == 3
    assert original_snapshot[1]["metadata"]["values"] == [1]
    assert frontend_snapshot[1]["metadata"]["values"] == [1]


def test_pose_record_clones_numpy_inputs():
    estimated = np.eye(4, dtype=np.float32)
    gt = np.eye(4, dtype=np.float32)
    records = {}
    slam_module.add_pose_record(records, 0, {"Rt": gt}, estimated)

    estimated[0, 3] = 10
    gt[0, 3] = 20

    assert records[0]["estimated_Rt"][0, 3] == 0
    assert records[0]["gt_Rt"][0, 3] == 0


@pytest.mark.parametrize("with_gt", [True, False])
def test_main_preserves_two_trajectory_stages_and_saves_online_poses(monkeypatch, tmp_path, with_gt):
    selected_ids = {0, 1, 2, 4, 5, 7}
    captured = {"stage_calls": 0, "render_calls": 0}

    class FakeDataset:
        def __init__(self):
            self.next_frame = 0

        def __len__(self):
            return 8

        def get_image_size(self):
            return 4, 5

        def getnext(self):
            frame_id = self.next_frame
            self.next_frame += 1
            info = {"is_test": frame_id == 6, "name": str(frame_id), "timestamp": frame_id * 0.1}
            if with_gt:
                info["Rt"] = _pose(frame_id)
            return torch.full((3, 4, 5), float(frame_id)), info

    class FakeFrontend:
        def __init__(self, width, height, args):
            self.matcher = object()
            self.triangulator = object()
            self.focal_init = torch.tensor([10.0])

        def extract(self, image):
            return SimpleNamespace(value=int(image[0, 0, 0].item()))

        def decide_keyframe(self, frame_id, desc_kpts):
            return KeyframeDecision(frame_id in selected_ids, "fake", 1.0, 999, 0.0)

        def accept_keyframe(self, frame_id, desc_kpts):
            pass

        def initialize_bootstrap(self, descs):
            return [_pose(desc.value) for desc in descs], torch.tensor([10.0]), 0.1

        def initialize_incremental(self, scene_model, desc_kpts, frame_id, image, will_be_keyframe, is_test=False):
            return None if frame_id == 7 else _pose(frame_id)

    class FakeKeyframe:
        def __init__(self, Rt):
            self.Rt = Rt

        def get_Rt(self):
            return self.Rt

    class FakeSceneModel:
        def __init__(self):
            self.keyframes = []
            self.n_active_gaussians = 123

        def join_optimization_thread(self):
            pass

        def enable_inference_mode(self):
            captured["inference_poses"] = [kf.get_Rt().clone() for kf in self.keyframes]

        def save(self, *args, **kwargs):
            captured["saved_poses"] = [kf.get_Rt().clone() for kf in self.keyframes]
            return {}

    class FakeBackend:
        def __init__(self, width, height, args, matcher, triangulator):
            self.scene_model = FakeSceneModel()
            self.keyframe_to_frame_id = {}
            captured["backend"] = self

        def _register(self, frame_id, Rt):
            keyframe_id = len(self.scene_model.keyframes)
            # CPU tensors share storage unless trajectory capture clones them.
            Rt[0, 3] += 100
            self.scene_model.keyframes.append(FakeKeyframe(Rt))
            self.keyframe_to_frame_id[keyframe_id] = frame_id
            return keyframe_id

        def register_bootstrap(self, packets, Rts, focal):
            keyframe_ids = [self._register(packet["frame_id"], Rt) for packet, Rt in zip(packets, Rts)]
            for keyframe in self.scene_model.keyframes:
                # Bootstrap mapping replaces pose storage while frontend Rts remain separate.
                keyframe.Rt = keyframe.Rt.clone()
                keyframe.Rt[0, 3] += 20
            return SimpleNamespace(timings_ms={}, keyframe_ids=keyframe_ids)

        def register_incremental_keyframe(self, frame_id, image, info, desc_kpts, Rt, focal):
            # Later online mapping changes the poses of older keyframes too.
            for keyframe in self.scene_model.keyframes:
                keyframe.Rt[0, 3] += 10
            keyframe_id = self._register(frame_id, Rt)
            return SimpleNamespace(timings_ms={}, keyframe_ids=[keyframe_id])

        def finalize_mapping(self):
            captured["finalize_calls"] = captured.get("finalize_calls", 0) + 1
            return SLAMBackend.finalize_mapping(self)

    class FakeTqdm:
        def __init__(self, iterable, desc=None):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix_str(self, value):
            pass

    args = SimpleNamespace(
        seed=0,
        device="cpu",
        model_path=str(tmp_path),
        num_keyframes_miniba_bootstrap=2,
        eval_lpips_max_size=512,
        eval_lpips_device="cpu",
    )

    def fake_render_metrics(scene_model, dataset, records, **kwargs):
        captured["render_calls"] += 1
        captured["render_records"] = records
        return [], {"PSNR": 30.0, "SSIM": 0.9, "LPIPS": 0.1}

    def fake_write_stage_outputs(model_path, stages):
        captured["stage_calls"] += 1
        captured["stages"] = stages
        return {stage: {"RMSE ATE [m]": float(index) if with_gt else None} for index, stage in enumerate(stages)}

    def fake_write_outputs(model_path, metrics, frame_ate_rows, render_rows, poses_json):
        captured["metrics"] = metrics
        captured["legacy_poses"] = poses_json["records"]

    real_refresh = slam_module.refresh_keyframe_pose_records

    def capture_refresh(records, backend):
        assert not captured.get("finalize_calls"), "No pose refresh is needed after the no-op finalizer"
        real_refresh(records, backend)
        if "bootstrap_records" not in captured:
            captured["bootstrap_records"] = slam_module.snapshot_pose_records(records)

    monkeypatch.setattr(slam_module, "get_args", lambda: args)
    monkeypatch.setattr(slam_module, "create_dataset", lambda args: (FakeDataset(), "waymo"))
    monkeypatch.setattr(slam_module, "maybe_seed_from_dataset", lambda args, dataset: None)
    monkeypatch.setattr(slam_module, "SLAMFrontend", FakeFrontend)
    monkeypatch.setattr(slam_module, "SLAMBackend", FakeBackend)
    monkeypatch.setattr(slam_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(slam_module, "compute_render_metrics", fake_render_metrics)
    monkeypatch.setattr(slam_module, "compute_trajectory_outputs", lambda records: ([], 1.0 if with_gt else None, {"records": records}))
    monkeypatch.setattr(slam_module, "write_trajectory_stage_outputs", fake_write_stage_outputs)
    monkeypatch.setattr(slam_module, "write_outputs", fake_write_outputs)
    monkeypatch.setattr(slam_module, "refresh_keyframe_pose_records", capture_refresh)

    slam_module.main()

    assert captured["stage_calls"] == 1
    assert captured["render_calls"] == 1
    assert captured["finalize_calls"] == 1
    stages = captured["stages"]
    assert list(stages) == ["frontend_miniba", "online_mapping"]
    assert [record["estimated_Rt"][0, 3].item() for record in captured["bootstrap_records"]] == [120, 121]
    expected_online = {0: 150, 1: 151, 2: 122, 4: 114, 5: 105, 6: 6}
    for stage, records in stages.items():
        assert [record["frame_id"] for record in records] == list(range(8))
        assert {record["frame_id"] for record in records if record["is_keyframe"]} == selected_ids
        assert {record["frame_id"] for record in records if record["ate_selected"]} == selected_ids
        assert {record["frame_id"] for record in records if record["render_selected"]} == {6}
        assert records[3]["estimated_Rt"] is None
        assert records[7]["estimated_Rt"] is None
        assert records[7]["status"] == "keyframe_pose_failed"
        for frame_id, online_x in expected_online.items():
            expected_x = frame_id if stage == "frontend_miniba" else online_x
            assert records[frame_id]["estimated_Rt"][0, 3].item() == expected_x
        assert all((record["gt_Rt"] is not None) == with_gt for record in records)

    # General trajectory files, rendering and the saved scene all use online poses.
    for online_record, legacy_record, render_record in zip(
        stages["online_mapping"], captured["legacy_poses"], captured["render_records"]
    ):
        if online_record["estimated_Rt"] is None:
            assert legacy_record["estimated_Rt"] is None
            assert render_record["estimated_Rt"] is None
        else:
            assert torch.equal(online_record["estimated_Rt"], legacy_record["estimated_Rt"])
            assert torch.equal(online_record["estimated_Rt"], render_record["estimated_Rt"])
    for keyframe_id, frame_id in captured["backend"].keyframe_to_frame_id.items():
        expected_pose = stages["online_mapping"][frame_id]["estimated_Rt"]
        assert torch.equal(captured["inference_poses"][keyframe_id], expected_pose)
        assert torch.equal(captured["saved_poses"][keyframe_id], expected_pose)
    assert captured["metrics"]["RMSE ATE [m]"] == (1.0 if with_gt else None)
    assert captured["metrics"]["RMSE ATE frontend mini-BA [m]"] == (0.0 if with_gt else None)
    assert captured["metrics"]["RMSE ATE online mapping [m]"] == (1.0 if with_gt else None)
    assert "RMSE ATE final refinement [m]" not in captured["metrics"]
    assert "pose_refinement" not in captured["metrics"]["avg_keyframe_stage_ms"]

    # Stage storage must remain independent even after further backend activity.
    captured["backend"].scene_model.keyframes[0].Rt[0, 3] += 10000
    assert stages["frontend_miniba"][0]["estimated_Rt"][0, 3].item() == 0
    assert stages["online_mapping"][0]["estimated_Rt"][0, 3].item() == 150
    stages["frontend_miniba"][0]["estimated_Rt"][0, 3] = -100
    assert stages["online_mapping"][0]["estimated_Rt"][0, 3].item() == 150
