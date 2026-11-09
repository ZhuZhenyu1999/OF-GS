import json
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

from args import get_args
from dataloaders.dl3dv_dataset import DL3DVDataset, discover_dl3dv_sequences
from scripts import run_all_slam
from slam import create_dataset, detect_dataset_type, maybe_seed_from_dataset


def make_sequence(root):
    (root / "rgb").mkdir(parents=True)
    intrinsics = dict(fx=0.8, fy=0.9, cx=0.5, cy=0.4)
    (root / "intrinsics.json").write_text(json.dumps(intrinsics))
    cameras = []
    for i in range(3):
        name = f"frame_{i+1:05d}.png"
        cv2.imwrite(str(root / "rgb" / name), np.zeros((10, 20, 3), dtype=np.uint8))
        cameras.append(dict(image_name=name, cam_quat=[0, 0, 2**-0.5, 2**-0.5],
                            cam_trans=[i, 0, 0], **intrinsics))
    # Camera metadata order must not determine frame/pose association.
    (root / "cameras.json").write_text(json.dumps(cameras[::-1]))
    return root


def arguments(root, **overrides):
    values = dict(source_path=str(root), device="cpu", downsampling=1,
                  start_at=0, max_frames=-1, test_hold=2, pose_format="auto")
    return SimpleNamespace(**{**values, **overrides})


def test_calibration_poses_resize_and_frame_selection(tmp_path):
    root = make_sequence(tmp_path / "1")
    dataset = DL3DVDataset(arguments(root, start_at=1, max_frames=1, downsampling=2))
    image, info = dataset.getnext()
    assert image.shape == (3, 5, 10)
    assert info["source_index"] == 1
    assert info["name"] == "frame_00002.png"
    np.testing.assert_allclose(info["K_np"], [[8, 0, 4.75], [0, 4.5, 1.75], [0, 0, 1]])
    np.testing.assert_allclose(info["gt_c2w_np"][:3, 3], [1, 0, 0])
    np.testing.assert_allclose(info["gt_c2w_np"][:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(info["Rt"] @ info["gt_c2w"], np.eye(4), atol=1e-6)
    with pytest.raises(StopIteration):
        dataset.getnext()


def test_pose_direction_and_quaternion_override(tmp_path):
    root = make_sequence(tmp_path / "1")
    cameras = json.loads((root / "cameras.json").read_text())
    for camera in cameras:
        camera["cam_quat"] = [1, 0, 0, 0]
    (root / "cameras.json").write_text(json.dumps(cameras))
    dataset = DL3DVDataset(arguments(root, pose_format="w2c", dl3dv_quaternion_order="wxyz"))
    _, info = dataset[1]
    np.testing.assert_allclose(info["Rt_np"][:3, 3], [1, 0, 0])
    np.testing.assert_allclose(info["gt_c2w_np"][:3, 3], [-1, 0, 0])
    np.testing.assert_allclose(info["Rt_np"][:3, :3], np.eye(3))


def test_optional_and_unmatched_gt(tmp_path):
    root = make_sequence(tmp_path / "1")
    cameras = json.loads((root / "cameras.json").read_text())
    (root / "cameras.json").write_text(json.dumps(cameras[:1]))
    dataset = DL3DVDataset(arguments(root))
    assert "Rt_np" not in dataset.infos[0]
    assert "Rt_np" in dataset.infos[2]
    (root / "cameras.json").unlink()
    assert all("Rt_np" not in info for info in DL3DVDataset(arguments(root)).infos)


def test_invalid_calibration_and_duplicate_cameras(tmp_path):
    root = make_sequence(tmp_path / "1")
    cameras = json.loads((root / "cameras.json").read_text())
    (root / "cameras.json").write_text(json.dumps(cameras + cameras[:1]))
    with pytest.raises(ValueError, match="Duplicate"):
        DL3DVDataset(arguments(root))
    (root / "intrinsics.json").write_text(json.dumps(dict(fx=-1, fy=1, cx=0.5, cy=0.5)))
    with pytest.raises(ValueError, match="positive focal"):
        DL3DVDataset(arguments(root))


def test_auto_detection_and_intrinsics_initialization(tmp_path, monkeypatch):
    import sys

    root = make_sequence(tmp_path / "dl3dv" / "1")
    monkeypatch.setattr(sys, "argv", ["slam.py", "-s", str(root.parent), "--device", "cpu"])
    args = get_args()
    assert detect_dataset_type(args) == "dl3dv"
    dataset, kind = create_dataset(args)
    maybe_seed_from_dataset(args, dataset)
    assert kind == "dl3dv"
    np.testing.assert_allclose(args.init_intrinsics, [[16, 0, 10], [0, 9, 4], [0, 0, 1]])


def test_three_sequence_batch_and_child_arguments(tmp_path):
    root = tmp_path / "dl3dv"
    for name in ("3", "1", "2"):
        make_sequence(root / name)
    assert [path.name for path in discover_dl3dv_sequences(root)] == ["1", "2", "3"]
    assert discover_dl3dv_sequences(root / "1") == [root / "1"]
    output = tmp_path / "results"
    run_all_slam.main(["--dataset_type", "dl3dv", "--dl3dv_root", str(root),
                       "--output_root", str(output), "--cuda_visible_devices", "0", "--dry_run",
                       "--pose_format", "w2c", "--dl3dv_quaternion_order", "wxyz"])
    summary = json.loads((output / "slam_batch_summary.json").read_text())
    assert [row["sequence"] for row in summary] == ["1", "2", "3"]
    for row in summary:
        assert row["dataset_type"] == "dl3dv"
        assert Path(row["output_dir"]) == output / "dl3dv" / row["sequence"]
        assert "--dataset_type dl3dv" in row["command"]
        assert "--camera_name" not in row["command"]
        assert "--waymo_image_model" not in row["command"]
    parser_args = run_all_slam.build_parser().parse_args(["--dataset_type", "dl3dv"])
    command = run_all_slam.build_sequence_command(root / "1", output, "FRONT",
        run_all_slam.build_default_slam_args(parser_args), [], dataset_type="dl3dv")
    run_all_slam.check_slam_arguments(command)
    assert run_all_slam.profile_defaults("server", "dl3dv") == "/mnt/data_23/zzy/dataset/dl3dv"

