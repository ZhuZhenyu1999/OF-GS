"""Verify CLI dispatch and calibrated initialization without loading CUDA models."""

import sys

import cv2
import numpy as np
import pytest

from args import get_args
from slam import create_dataset, detect_dataset_type, maybe_seed_from_dataset


def _sequence(root):
    (root / "rgb").mkdir(parents=True)
    cv2.imwrite(str(root / "rgb" / "000000.png"), np.zeros((8, 12, 3), dtype=np.uint8))
    (root / "calib.txt").write_text("P2: 10 0 5 0 0 11 3 0 0 0 1 0\n", encoding="utf-8")
    return root


def _args(monkeypatch, source, *options):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(source), "-m", str(source / "output"),
        "--device", "cpu", *options,
    ])
    return get_args()


@pytest.mark.parametrize("use_parent", [False, True])
@pytest.mark.parametrize("explicit_type", [False, True])
def test_kitti_dispatch_seeds_complete_intrinsics(monkeypatch, tmp_path, use_parent, explicit_type):
    sequence = _sequence(tmp_path / "KITTI" / "07")
    source = sequence.parent if use_parent else sequence
    options = ("--dataset_type", "kitti") if explicit_type else ()
    args = _args(monkeypatch, source, *options)
    # This layout is also accepted by the broad Waymo rgb-folder discovery.
    assert detect_dataset_type(args) == "kitti"
    dataset, dataset_type = create_dataset(args)
    image, info = dataset.getnext()
    maybe_seed_from_dataset(args, dataset)

    assert dataset_type == "kitti"
    assert image.shape == (3, 8, 12)
    assert info["intrinsics_calibrated"]
    np.testing.assert_allclose(args.init_intrinsics, [[10, 0, 5], [0, 11, 3], [0, 0, 1]])
    assert args.init_focal == pytest.approx(10)
    assert "Rt" not in info  # GT is optional for running SLAM.


def test_multiple_kitti_sequences_select_first(monkeypatch, tmp_path, capsys):
    _sequence(tmp_path / "09")
    first = _sequence(tmp_path / "07")
    args = _args(monkeypatch, tmp_path)
    dataset, dataset_type = create_dataset(args)
    assert dataset_type == "kitti"
    assert args.source_path == str(first)
    assert dataset.image_paths[0].parent.parent == first
    assert "Multiple KITTI sequences found" in capsys.readouterr().out


def test_waymo_detection_remains_available(monkeypatch, tmp_path):
    (tmp_path / "FRONT" / "rgb").mkdir(parents=True)
    assert detect_dataset_type(_args(monkeypatch, tmp_path)) == "waymo"


def test_tum_detection_remains_available(monkeypatch, tmp_path):
    (tmp_path / "rgb.txt").write_text("", encoding="utf-8")
    assert detect_dataset_type(_args(monkeypatch, tmp_path)) == "tum"


def test_plain_images_remain_nvs(monkeypatch, tmp_path):
    (tmp_path / "images").mkdir()
    assert detect_dataset_type(_args(monkeypatch, tmp_path)) == "nvs"
