import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataloaders.kitti_dataset import discover_kitti_sequences
from scripts import run_all_slam, run_kitti_slam


@pytest.mark.parametrize("local_ba_options,expected_local_ba", [
    ([], (10, 5, 2)),
    (["--local_ba_window_size", "9", "--local_ba_recent_keyframes", "6",
      "--local_ba_fixed_keyframes", "2"], (9, 6, 2)),
])
def test_server_kitti_batch_runs_in_order_and_checkpoints_failures(
    monkeypatch, tmp_path, local_ba_options, expected_local_ba
):
    dataset = tmp_path / "KITTI"
    for name in ("07", "00", "02"):
        sequence = dataset / name
        (sequence / "rgb").mkdir(parents=True)
        (sequence / "calib.txt").write_text(
            "P2: 100 0 50 0 0 100 30 0 0 0 1 0\n", encoding="utf-8"
        )
    output = tmp_path / "results"
    summary_path = output / "slam_batch_summary.json"

    def discover(root):
        assert root == "/mnt/data_23/zzy/dataset/KITTI"
        return discover_kitti_sequences(dataset)

    monkeypatch.setattr(run_all_slam, "discover_kitti_sequences", discover)
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda *a, **kw: None)
    monkeypatch.setattr(run_all_slam, "check_slam_arguments", lambda *a: None)
    launched = []

    def run(command, cwd, env):
        # Each launch must see the previous sequence's saved result.
        if launched:
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            assert [row["sequence"] for row in saved] == launched
        sequence = Path(command[command.index("-s") + 1]).name
        assert command[command.index("--dataset_type") + 1] == "kitti"
        assert command[command.index("--kitti_camera") + 1] == "2"
        assert tuple(int(command[command.index(option) + 1]) for option in (
            "--local_ba_window_size", "--local_ba_recent_keyframes", "--local_ba_fixed_keyframes",
        )) == expected_local_ba
        assert Path(command[command.index("-m") + 1]) == output / "kitti" / sequence
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        launched.append(sequence)
        return SimpleNamespace(returncode=1 if sequence == "00" else 0)

    monkeypatch.setattr(run_all_slam.subprocess, "run", run)
    run_kitti_slam.main(["--output_root", str(output), "--viewer_mode", "none", *local_ba_options])

    assert launched == ["00", "02", "07"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [row["returncode"] for row in summary] == [1, 0, 0]
    assert (output / "slam_batch_summary.csv").is_file()
    assert (output / "slam_batch_summary.xlsx").is_file()


@pytest.mark.parametrize("options", [
    ["--local_ba_window_size", "15"],
    ["--local_ba_recent_keyframes", "3"],
    ["--local_ba_recent_keyframes", "4"],
    ["--local_ba_fixed_keyframes", "1"],
    ["--local_ba_window_size", "9", "--local_ba_recent_keyframes", "8"],
    ["--local_ba_disable", "--local_ba_recent_keyframes", "3"],
])
def test_kitti_entry_rejects_invalid_local_ba_settings_before_dispatch(monkeypatch, tmp_path, options):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Invalid local BA settings must be rejected before discovery or dispatch")

    monkeypatch.setattr(run_all_slam, "discover_jobs", unexpected_dispatch)
    monkeypatch.setattr(run_all_slam, "run_sequence", unexpected_dispatch)
    output_root = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        run_kitti_slam.main(["--output_root", str(output_root), *options])
    assert error.value.code == 2
    assert not output_root.exists()
