from argparse import Namespace

import numpy as np
import pytest
import torch

from dataloaders.kitti_dataset import KITTIDataset, discover_kitti_sequences, parse_kitti_calib


def _args(root, **overrides):
    values = dict(source_path=str(root), kitti_camera="2", start_at=0, max_frames=-1,
                  downsampling=1.0, device="cpu", test_hold=-1, pose_format="auto")
    values.update(overrides)
    return Namespace(**values)


def _calib(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "calib.txt").write_text(
        "P0: 90 0 3.0 0 0 91 2.0 0 0 0 1 0\n"
        "P2: 100 0 3.1 42 0 101 2.2 0.2 0 0 1 0.006\n",
        encoding="utf-8",
    )


def _image(root, stem="000000", shape=(5, 7), folder="rgb", channels=3):
    import cv2

    image = np.empty((*shape, channels), dtype=np.uint8) if channels > 1 else np.full(shape, 77, np.uint8)
    if channels > 1:
        image[..., :3] = [10, 20, 30]
        if channels == 4:
            image[..., 3] = 123
    path = root / folder / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _pose(root, stem="000000", rows=4):
    # A nonidentity rotation catches inversion and unintended axis conversions.
    c2w = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float64)
    (root / "gt").mkdir(exist_ok=True)
    np.savetxt(root / "gt" / f"{stem}.txt", c2w[:rows])
    return c2w


@pytest.mark.parametrize("rows", [3, 4])
def test_rectified_calibration_and_pose_conversion(tmp_path, rows):
    root = tmp_path / "07"
    _calib(root)
    _image(root)
    c2w = _pose(root, rows=rows)

    dataset = KITTIDataset(_args(root))
    image, info = dataset.getnext()

    assert image.shape == (3, 5, 7)
    assert torch.allclose(image[:, 0, 0], torch.tensor([30, 20, 10]) / 255)
    assert np.allclose(info["K_np"], [[100, 0, 3.1], [0, 101, 2.2], [0, 0, 1]])
    assert torch.allclose(info["focal"], torch.tensor([100., 101.]))
    assert np.allclose(info["gt_c2w"], c2w)
    assert np.allclose(info["Rt"], np.linalg.inv(c2w))
    assert info["intrinsics_calibrated"]
    assert info["dataset_type"] == "kitti"
    with pytest.raises(StopIteration):
        dataset.getnext()


def test_w2c_override_preserves_input_pose(tmp_path):
    _calib(tmp_path)
    _image(tmp_path)
    w2c = _pose(tmp_path)
    _, info = KITTIDataset(_args(tmp_path, pose_format="w2c"))[0]
    assert np.allclose(info["Rt"], w2c)
    assert np.allclose(info["gt_c2w"], np.linalg.inv(w2c))
    assert info["gt_pose_format"] == "w2c"


def test_resize_uses_actual_dimensions_and_pixel_centres(tmp_path):
    _calib(tmp_path)
    _image(tmp_path)
    dataset = KITTIDataset(_args(tmp_path, downsampling=2))
    image, info = dataset[0]
    assert dataset.get_image_size() == (2, 4)
    assert image.shape == (3, 2, 4)
    sx, sy = 4 / 7, 2 / 5
    assert np.allclose(info["K_np"], [[100 * sx, 0, (3.1 + .5) * sx - .5],
                                      [0, 101 * sy, (2.2 + .5) * sy - .5], [0, 0, 1]])


def test_frame_filtering_and_gt_association_use_stems(tmp_path):
    _calib(tmp_path)
    for stem in ("000000", "000001", "000002", "000003"):
        _image(tmp_path, stem)
    _pose(tmp_path, "000000")
    c2w = _pose(tmp_path, "000002")
    dataset = KITTIDataset(_args(tmp_path, start_at=1, max_frames=2))
    _, missing = dataset[0]
    _, matched = dataset[1]
    assert len(dataset) == 2
    assert missing["source_index"] == 1
    assert missing["frame_stem"] == "000001"
    assert missing["gt_path"] is None
    assert "Rt" not in missing
    assert matched["frame_id"] == 1
    assert matched["gt_path"].endswith("000002.txt")
    assert np.allclose(matched["gt_c2w_np"], c2w)
    matched["K_np"][0, 0] = -1
    assert dataset.infos[1]["K_np"][0, 0] == 100


@pytest.mark.parametrize("folder,camera,channels", [("image_0", "0", 1), ("image_02/data", "2", 4)])
def test_selected_camera_layout_grayscale_and_alpha(tmp_path, folder, camera, channels):
    _calib(tmp_path)
    _image(tmp_path, folder=folder, channels=channels)
    dataset = KITTIDataset(_args(tmp_path, kitti_camera=camera))
    image, info = dataset[0]
    assert image.shape == (3, 5, 7)
    assert info["K"][0, 0] == (90 if camera == "0" else 100)
    if channels == 1:
        assert torch.allclose(image, torch.full_like(image, 77 / 255))
    else:
        assert torch.allclose(info["mask"], torch.full((1, 5, 7), 123 / 255))


def test_selected_camera_directory_takes_precedence_over_rgb_export(tmp_path):
    _calib(tmp_path)
    _image(tmp_path, folder="rgb")
    camera_path = _image(tmp_path, folder="image_0", channels=1)
    dataset = KITTIDataset(_args(tmp_path, kitti_camera="0"))
    image, info = dataset[0]
    assert dataset.rgb_dir == tmp_path / "image_0"
    assert info["image_path"] == str(camera_path)
    assert info["K"][0, 0] == 90
    assert torch.allclose(image, torch.full_like(image, 77 / 255))


def test_sequence_discovery_supports_parent_and_odometry_root(tmp_path):
    direct = tmp_path / "07"
    nested = tmp_path / "sequences" / "08"
    for root in (direct, nested):
        _calib(root)
        _image(root)
    unrelated = tmp_path / "waymo" / "FRONT"
    (unrelated / "rgb").mkdir(parents=True)
    assert discover_kitti_sequences(direct) == [direct]
    assert discover_kitti_sequences(tmp_path) == [direct, nested]
    assert discover_kitti_sequences(tmp_path / "sequences") == [nested]
    assert discover_kitti_sequences(unrelated) == []
    assert discover_kitti_sequences(tmp_path / "missing") == []
    assert KITTIDataset(_args(tmp_path / "sequences")).sequence_root == nested


def test_missing_or_invalid_selected_calibration_fails(tmp_path):
    _calib(tmp_path)
    with pytest.raises(ValueError, match="Missing KITTI camera projection P3"):
        parse_kitti_calib(tmp_path / "calib.txt", "3")
    (tmp_path / "calib.txt").write_text("P2: 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="12 numeric values"):
        parse_kitti_calib(tmp_path / "calib.txt")
    (tmp_path / "calib.txt").write_text("P2: nan 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid rectified pinhole"):
        parse_kitti_calib(tmp_path / "calib.txt")


def test_invalid_pose_is_rejected_with_path(tmp_path):
    _calib(tmp_path)
    _image(tmp_path)
    _pose(tmp_path)
    np.savetxt(tmp_path / "gt" / "000000.txt", np.zeros((4, 4)))
    with pytest.raises(ValueError, match="Invalid rigid KITTI pose.*000000.txt"):
        KITTIDataset(_args(tmp_path))


def test_changed_image_dimensions_and_empty_selection_fail(tmp_path):
    _calib(tmp_path)
    _image(tmp_path)
    _image(tmp_path, "000001", shape=(6, 7))
    dataset = KITTIDataset(_args(tmp_path))
    with pytest.raises(ValueError, match="has size.*expected"):
        dataset[1]
    with pytest.raises(FileNotFoundError, match="after start_at/max_frames"):
        KITTIDataset(_args(tmp_path, start_at=2))
