import csv
import json

import numpy as np
import pytest
import torch

from slam_outputs import (
    TRAJECTORY_STAGES,
    compute_trajectory_outputs,
    record_is_ate_selected,
    record_is_render_selected,
    write_trajectory_stage_outputs,
)


OBSOLETE_STAGE_FILES = (
    "slam_poses_final_refinement.json",
    "slam_frame_ate_final_refinement.csv",
)


def _w2c_at(x):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [x, 0, 0]
    return np.linalg.inv(c2w).astype(np.float32)


def test_compute_trajectory_outputs_sim3_alignment():
    records = [
        {"frame_id": 0, "name": "0", "estimated_Rt": _w2c_at(0), "gt_Rt": _w2c_at(0), "status": "ok", "is_keyframe": True},
        {"frame_id": 1, "name": "1", "estimated_Rt": _w2c_at(1), "gt_Rt": _w2c_at(2), "status": "ok", "is_keyframe": True},
        {"frame_id": 2, "name": "2", "estimated_Rt": _w2c_at(2), "gt_Rt": _w2c_at(4), "status": "ok", "is_keyframe": True},
    ]
    rows, rmse, poses = compute_trajectory_outputs(records)
    assert rmse is not None
    assert rmse < 1e-5
    assert all(row["ate_m"] is not None for row in rows)
    assert poses["sim3"]["scale"] > 1.9


def test_trajectory_metrics_ignore_non_keyframe_registered_frames():
    records = [
        {"frame_id": 0, "name": "0", "estimated_Rt": _w2c_at(0), "gt_Rt": _w2c_at(0), "status": "keyframe", "is_keyframe": True},
        {"frame_id": 1, "name": "1", "estimated_Rt": _w2c_at(100), "gt_Rt": _w2c_at(0), "status": "tracked"},
        {"frame_id": 8, "name": "8", "estimated_Rt": _w2c_at(1), "gt_Rt": _w2c_at(1), "status": "tracked", "is_test": True},
    ]
    rows, rmse, poses = compute_trajectory_outputs(records)
    by_frame = {row["frame_id"]: row for row in rows}

    assert rmse is not None
    assert rmse < 1e-5
    assert by_frame[1]["ate_m"] is None
    assert by_frame[1]["status"] == "not_evaluated"
    assert by_frame[8]["ate_m"] is None
    assert by_frame[8]["ate_selected"] is False
    assert poses["estimated"][1]["ate_selected"] is False


def test_metric_selection_helpers_are_separate():
    keyframe = {"is_keyframe": True, "is_test": False}
    test_frame = {"is_keyframe": False, "is_test": True}
    ordinary = {"is_keyframe": False, "is_test": False}

    assert record_is_ate_selected(keyframe)
    assert not record_is_render_selected(keyframe)
    assert not record_is_ate_selected(test_frame)
    assert record_is_render_selected(test_frame)
    assert not record_is_ate_selected(ordinary)
    assert not record_is_render_selected(ordinary)


def _pose_record(frame_id, x, gt_x=None, yaw_deg=0, **metadata):
    angle = np.radians(yaw_deg)
    c2w = np.eye(4)
    c2w[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    c2w[:3, 3] = [x, 0, 0]
    return {
        "frame_id": frame_id,
        "timestamp": float(frame_id),
        "name": f"{frame_id}.png",
        "estimated_Rt": np.linalg.inv(c2w),
        "gt_Rt": _w2c_at(gt_x) if gt_x is not None else None,
        "is_keyframe": True,
        "status": "keyframe",
        **metadata,
    }


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_compute_trajectory_outputs_retains_raw_poses_without_gt():
    records = [
        _pose_record(0, 2),
        {"frame_id": 1, "estimated_Rt": None, "gt_Rt": _w2c_at(1)},
    ]
    rows, rmse, poses = compute_trajectory_outputs(records)

    assert rmse is None
    assert [row["status"] for row in rows] == ["keyframe", "no_pose"]
    assert poses["sim3"] is None
    assert len(poses["estimated"]) == len(poses["gt"]) == len(poses["aligned"]) == 2
    np.testing.assert_array_equal(poses["estimated"][0]["Rt_w2c"], records[0]["estimated_Rt"])
    assert poses["estimated"][0]["name"] == "0.png"
    assert poses["estimated"][1]["Rt_w2c"] is None
    assert poses["gt"][0]["Rt_w2c"] is None
    assert poses["gt"][1]["Rt_w2c"] is not None
    assert all(pose["c2w"] is None for pose in poses["aligned"])


def test_stage_outputs_independent_alignment_and_raw_deltas(tmp_path):
    stages = {
        "frontend_miniba": [_pose_record(i, i, 2 * i) for i in range(3)],
        "online_mapping": [_pose_record(i, 2 * i + 3, 2 * i, yaw_deg=10) for i in range(3)],
    }
    stages["frontend_miniba"][0]["estimated_Rt"] = torch.tensor(stages["frontend_miniba"][0]["estimated_Rt"])
    metrics = write_trajectory_stage_outputs(tmp_path, stages)

    assert TRAJECTORY_STAGES == ("frontend_miniba", "online_mapping")
    assert set(metrics) == set(TRAJECTORY_STAGES)
    for stage in TRAJECTORY_STAGES:
        assert metrics[stage]["RMSE ATE [m]"] == pytest.approx(0, abs=1e-6)
        assert metrics[stage]["n_registered_keyframes"] == metrics[stage]["n_evaluated_keyframes"] == 3
        exported = json.loads((tmp_path / f"slam_poses_{stage}.json").read_text())
        assert exported["stage"] == stage
        assert exported["estimated"][1]["timestamp"] == 1.0
        np.testing.assert_allclose(exported["estimated"][1]["Rt_w2c"], stages[stage][1]["estimated_Rt"])
        assert len(exported["aligned"]) == 3
        assert len(_read_csv(tmp_path / f"slam_frame_ate_{stage}.csv")) == 3
    assert metrics["frontend_miniba"]["alignment"]["scale"] == pytest.approx(2)
    assert metrics["online_mapping"]["alignment"]["scale"] == pytest.approx(1)
    assert metrics["online_mapping"]["alignment"]["translation"][0] == pytest.approx(-3)
    summary = json.loads((tmp_path / "slam_trajectory_metrics.json").read_text())
    assert summary["stages"] == metrics
    assert "independently" in summary["definitions"]["alignment"]
    assert "saving and evaluation" in summary["definitions"]["online_mapping"]
    assert "final_refinement" not in summary["definitions"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "slam_poses_frontend_miniba.json", "slam_poses_online_mapping.json",
        "slam_frame_ate_frontend_miniba.csv", "slam_frame_ate_online_mapping.csv",
        "slam_trajectory_metrics.json", "slam_trajectory_comparison.csv",
    }

    comparison = _read_csv(tmp_path / "slam_trajectory_comparison.csv")
    assert [row["frame_id"] for row in comparison] == ["0", "1", "2"]
    for i, row in enumerate(comparison):
        assert float(row["ate_m_online_mapping"]) == pytest.approx(0, abs=1e-6)
        assert float(row["camera_center_delta_raw_frontend_to_online"]) == pytest.approx(3 + i)
        assert float(row["orientation_delta_deg_frontend_to_online"]) == pytest.approx(10)
        assert float(row["rotation_error_deg_online_mapping"]) == pytest.approx(10)
        assert not any("final_refinement" in name or "online_to_final" in name for name in row)

    # Exporting does not rewrite selection flags or retain input tensor buffers.
    assert "ate_selected" not in stages["frontend_miniba"][0]
    stages["frontend_miniba"][0]["estimated_Rt"][0, 3] = 100
    saved_frontend = json.loads((tmp_path / "slam_poses_frontend_miniba.json").read_text())
    assert saved_frontend["estimated"][0]["Rt_w2c"][0][3] == 0


def test_stage_outputs_partial_gt_and_common_registered_keyframes(tmp_path):
    records = [
        _pose_record(7, 0, 0),
        _pose_record(2, 1),
        _pose_record(8, 2, 4),
        _pose_record(9, 3, 6, estimated_Rt=None),
        _pose_record(10, 100, 0, is_keyframe=False, ate_selected=True),
    ]
    stages = {stage: [dict(record) for record in records] for stage in TRAJECTORY_STAGES}
    stages["online_mapping"] = [record for record in stages["online_mapping"] if record["frame_id"] != 8]
    metrics = write_trajectory_stage_outputs(tmp_path, stages)

    assert metrics["frontend_miniba"]["n_registered_keyframes"] == 3
    assert metrics["frontend_miniba"]["n_evaluated_keyframes"] == 2
    assert metrics["online_mapping"]["n_registered_keyframes"] == 2
    assert metrics["online_mapping"]["n_evaluated_keyframes"] == 1
    rows = {row["frame_id"]: row for row in _read_csv(tmp_path / "slam_frame_ate_frontend_miniba.csv")}
    assert rows["2"]["ate_m"] == rows["9"]["ate_m"] == rows["10"]["ate_m"] == ""
    assert rows["10"]["status"] == "not_evaluated"
    assert rows["10"]["ate_selected"] == "False"
    comparison = _read_csv(tmp_path / "slam_trajectory_comparison.csv")
    assert [row["frame_id"] for row in comparison] == ["7", "2"]
    assert comparison[1]["ate_m_frontend_miniba"] == ""
    assert comparison[1]["camera_center_delta_raw_frontend_to_online"] == "0.0"
    assert records[-1]["ate_selected"] is True


def test_stage_outputs_without_gt_still_export_trajectories_and_deltas(tmp_path):
    stages = {
        "frontend_miniba": [_pose_record(0, 1)],
        "online_mapping": [_pose_record(0, 2)],
    }
    metrics = write_trajectory_stage_outputs(tmp_path, stages)

    for stage in TRAJECTORY_STAGES:
        assert metrics[stage]["RMSE ATE [m]"] is None
        assert metrics[stage]["alignment"] is None
        assert metrics[stage]["n_registered_keyframes"] == 1
        assert metrics[stage]["n_evaluated_keyframes"] == 0
        saved = json.loads((tmp_path / f"slam_poses_{stage}.json").read_text())
        assert len(saved["estimated"]) == 1
        assert saved["estimated"][0]["Rt_w2c"] is not None
        assert saved["gt"][0]["Rt_w2c"] is None
        assert saved["aligned"][0]["c2w"] is None
    comparison = _read_csv(tmp_path / "slam_trajectory_comparison.csv")
    assert comparison[0]["ate_m_frontend_miniba"] == ""
    assert float(comparison[0]["camera_center_delta_raw_frontend_to_online"]) == pytest.approx(1)


def test_stage_outputs_empty_snapshots_write_headers_and_null_metrics(tmp_path):
    metrics = write_trajectory_stage_outputs(tmp_path, {stage: [] for stage in TRAJECTORY_STAGES})
    assert all(metric["RMSE ATE [m]"] is None for metric in metrics.values())
    assert all(metric["n_registered_keyframes"] == 0 for metric in metrics.values())
    assert _read_csv(tmp_path / "slam_trajectory_comparison.csv") == []
    with (tmp_path / "slam_trajectory_comparison.csv").open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames
    assert fields == [
        "frame_id", "timestamp", "name",
        "ate_m_frontend_miniba", "rotation_error_deg_frontend_miniba",
        "ate_m_online_mapping", "rotation_error_deg_online_mapping",
        "camera_center_delta_raw_frontend_to_online", "orientation_delta_deg_frontend_to_online",
    ]


@pytest.mark.parametrize("stage_names", [
    (), ("frontend_miniba",), ("online_mapping",), ("final_refinement",),
    ("frontend_miniba", "online_mapping", "final_refinement"),
    ("frontend_miniba", "online_mapping", "unknown"),
])
def test_stage_outputs_require_exactly_two_snapshots_without_removing_old_files(tmp_path, stage_names):
    for name in OBSOLETE_STAGE_FILES:
        (tmp_path / name).write_text("old result", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected trajectory stages"):
        write_trajectory_stage_outputs(tmp_path, {name: [] for name in stage_names})
    assert {path.name for path in tmp_path.iterdir()} == set(OBSOLETE_STAGE_FILES)
    assert all((tmp_path / name).read_text(encoding="utf-8") == "old result" for name in OBSOLETE_STAGE_FILES)


def test_stage_outputs_only_remove_obsolete_files_in_the_current_output_directory(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    nested_history = current / "history"
    nested_history.mkdir()
    other_run = tmp_path / "other_run"
    other_run.mkdir()
    for directory in (current, nested_history, other_run):
        for name in OBSOLETE_STAGE_FILES:
            (directory / name).write_text("old result", encoding="utf-8")
    for name in ("notes.txt", "slam_poses_final_refinement.json.bak"):
        (current / name).write_text("keep", encoding="utf-8")
    (current / "slam_trajectory_metrics.json").write_text('{"final_refinement": {}}', encoding="utf-8")
    (current / "slam_trajectory_comparison.csv").write_text("online_to_final\n1\n", encoding="utf-8")

    snapshots = {stage: [] for stage in TRAJECTORY_STAGES}
    metrics = write_trajectory_stage_outputs(current, snapshots)

    assert all(not (current / name).exists() for name in OBSOLETE_STAGE_FILES)
    for directory in (nested_history, other_run):
        assert all((directory / name).read_text(encoding="utf-8") == "old result" for name in OBSOLETE_STAGE_FILES)
    for name in ("notes.txt", "slam_poses_final_refinement.json.bak"):
        assert (current / name).read_text(encoding="utf-8") == "keep"
    assert json.loads((current / "slam_trajectory_metrics.json").read_text())["stages"] == metrics
    assert "online_to_final" not in (current / "slam_trajectory_comparison.csv").read_text()
    # A repeated export also succeeds after the obsolete files have gone.
    assert write_trajectory_stage_outputs(current, snapshots) == metrics


def test_stage_outputs_preserve_directories_with_obsolete_filenames(tmp_path):
    for name in OBSOLETE_STAGE_FILES:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "keep.txt").write_text("keep", encoding="utf-8")

    write_trajectory_stage_outputs(tmp_path, {stage: [] for stage in TRAJECTORY_STAGES})

    for name in OBSOLETE_STAGE_FILES:
        assert (tmp_path / name).is_dir()
        assert (tmp_path / name / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_stage_outputs_unlink_obsolete_symlinks_without_touching_targets(tmp_path, target_kind):
    current = tmp_path / "current"
    current.mkdir()
    target = tmp_path / "target"
    if target_kind == "directory":
        target.mkdir()
        (target / "keep.txt").write_text("keep", encoding="utf-8")
    elif target_kind == "file":
        target.write_text("keep", encoding="utf-8")
    try:
        for name in OBSOLETE_STAGE_FILES:
            (current / name).symlink_to(target, target_is_directory=target_kind == "directory")
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Symlinks unavailable: {error}")

    write_trajectory_stage_outputs(current, {stage: [] for stage in TRAJECTORY_STAGES})

    assert all(not (current / name).is_symlink() for name in OBSOLETE_STAGE_FILES)
    assert all(not (current / name).exists() for name in OBSOLETE_STAGE_FILES)
    if target_kind == "directory":
        assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    elif target_kind == "file":
        assert target.read_text(encoding="utf-8") == "keep"
    else:
        assert not target.exists()


def test_stage_outputs_keep_obsolete_files_when_new_summary_write_fails(tmp_path, monkeypatch):
    import slam_outputs

    for name in OBSOLETE_STAGE_FILES:
        (tmp_path / name).write_text("old result", encoding="utf-8")
    real_dump = slam_outputs.json.dump

    def fail_summary(payload, stream, **kwargs):
        if "definitions" in payload and "stages" in payload:
            raise OSError("summary write failed")
        return real_dump(payload, stream, **kwargs)

    monkeypatch.setattr(slam_outputs.json, "dump", fail_summary)
    with pytest.raises(OSError, match="summary write failed"):
        write_trajectory_stage_outputs(tmp_path, {stage: [] for stage in TRAJECTORY_STAGES})
    assert all((tmp_path / name).read_text(encoding="utf-8") == "old result" for name in OBSOLETE_STAGE_FILES)


def test_write_outputs_emits_core_results_only(tmp_path):
    from slam_outputs import write_outputs

    poses = {"estimated": [], "gt": [], "aligned": [], "sim3": None}
    write_outputs(tmp_path, {"RMSE ATE [m]": None}, [], [], poses)

    assert {path.name for path in tmp_path.iterdir()} == {
        "slam_metrics.json",
        "slam_frame_ate.csv",
        "slam_render_metrics.csv",
        "slam_poses.json",
    }
    assert json.loads((tmp_path / "slam_poses.json").read_text()) == poses
