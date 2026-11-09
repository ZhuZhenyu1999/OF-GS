from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from dataloaders.tum_dataset import TUMDataset, associate_timestamps, quat_xyzw_to_rotmat
from dataloaders.waymo_dataset import WaymoDataset, parse_waymo_calib


def _write_rgb(path: Path, shape=(4, 6)):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    cv2.imwrite(str(path), image)


def _args(source_path, dataset_type):
    return Namespace(
        source_path=str(source_path),
        dataset_type=dataset_type,
        camera_name="FRONT",
        waymo_image_model="auto",
        start_at=0,
        max_frames=-1,
        downsampling=1.0,
        device="cpu",
        test_hold=-1,
        pose_format="auto",
        tum_max_dt=0.02,
        tum_fx=-1.0,
        tum_fy=-1.0,
        tum_cx=-1.0,
        tum_cy=-1.0,
    )


def test_tum_timestamp_association_and_loader(tmp_path):
    import cv2

    root = tmp_path / "rgbd_dataset_freiburg1_xyz"
    _write_rgb(root / "rgb" / "1305031102.175304.png")
    root.joinpath("depth").mkdir(parents=True)
    cv2.imwrite(str(root / "depth" / "1305031102.160407.png"), np.ones((4, 6), dtype=np.uint16))
    (root / "rgb.txt").write_text("1305031102.175304 rgb/1305031102.175304.png\n", encoding="utf-8")
    (root / "depth.txt").write_text("1305031102.160407 depth/1305031102.160407.png\n", encoding="utf-8")
    (root / "groundtruth.txt").write_text("1305031102.170000 1 0 0 0 0 0 1\n", encoding="utf-8")

    assert associate_timestamps([(0.0, "a")], [(0.01, "b")], 0.02) == [(0, 0, 0.01)]
    assert np.allclose(quat_xyzw_to_rotmat([0, 0, 0, 1]), np.eye(3))

    dataset = TUMDataset(_args(root, "tum"))
    image, info = dataset.load_frame(0)
    assert image.shape == (3, 4, 6)
    assert info["depth_path"].endswith("1305031102.160407.png")
    assert torch.allclose(info["Rt"][:3, 3], torch.tensor([-1.0, 0.0, 0.0]))


def test_waymo_calib_gt_loader(tmp_path):
    root = tmp_path / "waymo_seq"
    camera = root / "FRONT"
    _write_rgb(camera / "rgb" / "000000.png")
    (camera / "calib").mkdir(parents=True)
    (camera / "gt").mkdir(parents=True)
    (camera / "calib" / "000000.txt").write_text(
        "fx: 100\nfy: 101\ncx: 3\ncy: 2\nk1: 0\n", encoding="utf-8"
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1, 2, 3]
    np.savetxt(camera / "gt" / "000000.txt", c2w)

    calib = parse_waymo_calib(camera / "calib" / "000000.txt")
    assert calib["fx"] == 100
    assert calib["fy"] == 101

    dataset = WaymoDataset(_args(root, "waymo"))
    image, info = dataset.load_frame(0)
    assert image.shape == (3, 4, 6)
    assert torch.allclose(info["focal"], torch.tensor([100.0, 101.0]))
    assert torch.allclose(info["K"], torch.tensor([[100.0, 0, 3], [0, 101.0, 2], [0, 0, 1]]))
    assert info["intrinsics_calibrated"]
    assert torch.allclose(info["Rt"][:3, 3], torch.tensor([-1.0, -2.0, -3.0]))


def _write_waymo_calib(camera, stem, fx=100.0, fy=101.0, cx=3.1, cy=2.2, distortion=0.0):
    (camera / "calib").mkdir(parents=True, exist_ok=True)
    (camera / "calib" / f"{stem}.txt").write_text(
        f"fx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\nk1: {distortion}\nk2: 0\np1: 0\np2: 0\nk3: 0\n",
        encoding="utf-8",
    )


def test_waymo_resize_intrinsics_follow_actual_dimensions_and_pixel_centres(tmp_path):
    camera = tmp_path / "FRONT"
    _write_rgb(camera / "rgb" / "000000.png", shape=(5, 7))
    _write_waymo_calib(camera, "000000")
    args = _args(tmp_path, "waymo")
    args.downsampling = 2.0
    dataset = WaymoDataset(args)
    image, info = dataset.load_frame(0)

    assert image.shape == (3, 2, 4)
    sx, sy = 4 / 7, 2 / 5
    assert np.allclose(info["K_np"], [[100 * sx, 0, (3.1 + 0.5) * sx - 0.5],
                                      [0, 101 * sy, (2.2 + 0.5) * sy - 0.5], [0, 0, 1]])
    assert torch.allclose(info["focal"], info["K"].diagonal()[:2])


def test_waymo_stem_association_survives_start_at_and_missing_metadata(tmp_path):
    import cv2

    camera = tmp_path / "FRONT"
    for stem in ("000000", "000001", "000002"):
        _write_rgb(camera / "rgb" / f"{stem}.png")
    for stem, focal in (("000000", 100), ("000002", 200)):
        _write_waymo_calib(camera, stem, fx=focal)
        (camera / "gt").mkdir(exist_ok=True)
        pose = np.eye(4)
        pose[0, 3] = int(stem)
        np.savetxt(camera / "gt" / f"{stem}.txt", pose)
        (camera / "depth").mkdir(exist_ok=True)
        cv2.imwrite(str(camera / "depth" / f"{stem}.png"), np.ones((4, 6), np.uint16))
    args = _args(tmp_path, "waymo")
    args.start_at = 1
    dataset = WaymoDataset(args)

    _, missing = dataset.load_frame(0)
    assert missing["frame_stem"] == "000001"
    assert missing["source_index"] == 1
    assert missing["calib_path"] is None
    assert missing["gt_path"] is None
    assert missing["depth_path"] is None
    assert not missing["intrinsics_calibrated"]
    assert "Rt" not in missing
    _, matched = dataset.load_frame(1)
    assert matched["calib_path"].endswith("000002.txt")
    assert matched["depth_path"].endswith("000002.png")
    assert matched["K"][0, 0] == 200
    assert matched["gt_c2w"][0, 3] == 2


def test_waymo_unknown_png_model_requires_explicit_distortion_contract(tmp_path):
    camera = tmp_path / "FRONT"
    _write_rgb(camera / "rgb" / "000000.png")
    _write_waymo_calib(camera, "000000", distortion=0.1)
    with pytest.raises(ValueError, match="PNG export model is unknown"):
        WaymoDataset(_args(tmp_path, "waymo"))

    args = _args(tmp_path, "waymo")
    args.waymo_image_model = "pinhole"
    image, info = WaymoDataset(args).load_frame(0)
    assert image.shape == (3, 4, 6)
    assert not info["undistorted_by_loader"]
    assert info["source_calib"]["k1"] == 0.1
    assert info["calib"]["k1"] == 0.0
    assert torch.allclose(image[:, 0, 0], torch.tensor([30, 20, 10]) / 255)


def test_waymo_raw_opencv_images_are_rectified_with_matching_k_and_validity_mask(tmp_path):
    import cv2

    camera = tmp_path / "FRONT"
    path = camera / "rgb" / "000000.png"
    path.parent.mkdir(parents=True)
    raw_bgr = np.random.default_rng(0).integers(0, 256, size=(23, 31, 3), dtype=np.uint8)
    cv2.imwrite(str(path), raw_bgr)
    _write_waymo_calib(camera, "000000", fx=20, fy=23, cx=13.3, cy=9.4, distortion=0.4)
    args = _args(tmp_path, "waymo")
    args.waymo_image_model = "opencv"
    image, info = WaymoDataset(args).load_frame(0)

    K = np.array([[20, 0, 13.3], [0, 23, 9.4], [0, 0, 1]], dtype=np.float64)
    raw_bgra = np.concatenate((raw_bgr, np.full((23, 31, 1), 255, np.uint8)), axis=-1)
    expected = cv2.undistort(raw_bgra, K, np.array([0.4, 0, 0, 0, 0]), None, K)
    expected = cv2.cvtColor(expected, cv2.COLOR_BGRA2RGBA)
    assert info["undistorted_by_loader"]
    assert info["image_model"] == "pinhole"
    assert np.allclose(info["K_np"], K)
    assert torch.equal(image, torch.from_numpy(expected[..., :3].copy()).permute(2, 0, 1).float() / 255)
    assert torch.equal(info["mask"], torch.from_numpy(expected[..., 3].copy())[None].float() / 255)
    assert bool((info["mask"] == 0).any())


def test_waymo_calibration_numpy_and_unlabelled_distortion_vectors(tmp_path):
    K = np.array([[100, 0, 3.1], [0, 101, 2.2], [0, 0, 1]], dtype=np.float64)
    np.save(tmp_path / "matrix.npy", K)
    assert parse_waymo_calib(tmp_path / "matrix.npy") == {"fx": 100, "fy": 101, "cx": 3.1, "cy": 2.2}
    (tmp_path / "vector.txt").write_text("100 101 3.1 2.2 0.1 0.2 0.3 0.4 0.5", encoding="utf-8")
    parsed = parse_waymo_calib(tmp_path / "vector.txt")
    assert parsed["fy"] == 101
    assert parsed["k1"] == 0.1
    assert parsed["k3"] == 0.5
