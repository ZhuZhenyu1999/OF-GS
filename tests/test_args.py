import sys
from pathlib import Path

import pytest

from args import GAUSSIAN_ABLATION_DEFAULTS, gaussian_ablation_config, get_args


@pytest.mark.parametrize("camera", ["0", "1", "2", "3"])
def test_kitti_arguments(monkeypatch, tmp_path, camera):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path),
        "--dataset_type", "kitti", "--kitti_camera", camera,
    ])
    parsed = get_args()
    assert parsed.dataset_type == "kitti"
    assert parsed.kitti_camera == camera


def test_init_sampling_method_defaults_to_scharr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["slam.py", "--source_path", str(tmp_path), "--model_path", str(tmp_path)],
    )

    parsed = get_args()

    assert parsed.init_sampling_method == "scharr"
    assert not hasattr(parsed, "slam_window_size")
    assert parsed.slam_first_full_iterations == 80
    assert parsed.slam_first_full_sampling_probs == [0.22, 0.19, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03]
    assert parsed.seed == 0
    assert parsed.kitti_camera == "2"
    assert parsed.slam_window_sampling_seed is None
    assert parsed.num_kpts == 8000
    assert Path(parsed.clidd_root).name == "CLIDD-main"
    assert parsed.clidd_model == "S64"
    assert parsed.clidd_radius == 2
    assert parsed.clidd_score == -5.0
    assert parsed.clidd_match_beta == 20.0
    assert parsed.clidd_dense_stride == 8
    assert parsed.pose_gaussian_min_update_count == 5
    assert parsed.pose_gaussian_min_observed_keyframes == 3
    assert gaussian_ablation_config(parsed) == GAUSSIAN_ABLATION_DEFAULTS


def test_init_sampling_method_accepts_scharr(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--init_sampling_method",
            "scharr",
        ],
    )

    parsed = get_args()

    assert parsed.init_sampling_method == "scharr"


def test_seed_arguments_are_parsed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--seed",
            "123",
            "--slam_window_sampling_seed",
            "456",
        ],
    )

    parsed = get_args()

    assert parsed.seed == 123
    assert parsed.slam_window_sampling_seed == 456


def test_pose_gaussian_maturity_arguments_parse(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--pose_gaussian_min_update_count",
            "9",
            "--pose_gaussian_min_observed_keyframes",
            "4",
        ],
    )

    parsed = get_args()

    assert parsed.pose_gaussian_min_update_count == 9
    assert parsed.pose_gaussian_min_observed_keyframes == 4


def test_clidd_arguments_parse(monkeypatch, tmp_path):
    clidd_root = tmp_path / "CLIDD-main"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--clidd_root",
            str(clidd_root),
            "--clidd_model",
            "N64",
            "--clidd_radius",
            "3",
            "--clidd_score",
            "-4.5",
            "--clidd_match_beta",
            "25",
            "--clidd_dense_stride",
            "4",
        ],
    )

    parsed = get_args()

    assert parsed.clidd_root == str(clidd_root)
    assert parsed.clidd_model == "N64"
    assert parsed.clidd_radius == 3
    assert parsed.clidd_score == -4.5
    assert parsed.clidd_match_beta == 25
    assert parsed.clidd_dense_stride == 4


def test_bootstrap_mapping_arguments_are_parsed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--slam_first_full_iterations", "123",
            "--slam_first_full_sampling_probs",
            "0.3", "0.2", "0.15", "0.1", "0.08", "0.07", "0.06", "0.04",
        ],
    )

    parsed = get_args()
    assert parsed.slam_first_full_iterations == 123
    assert parsed.slam_first_full_sampling_probs == [0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.06, 0.04]


def test_init_sampling_method_rejects_unknown_value(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "slam.py",
            "--source_path",
            str(tmp_path),
            "--model_path",
            str(tmp_path),
            "--init_sampling_method",
            "canny",
        ],
    )

    with pytest.raises(SystemExit):
        get_args()


def test_local_ba_defaults_and_overrides(monkeypatch, tmp_path):
    argv = ["slam.py", "-s", str(tmp_path), "-m", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    parsed = get_args()
    expected = {
        "local_ba_window_size": 10,
        "local_ba_recent_keyframes": 5,
        "local_ba_fixed_keyframes": 2,
        "local_ba_max_points": 1500,
        "local_ba_min_shared_points": 30,
        "local_ba_min_tracks": 50,
        "local_ba_iterations": 15,
        "local_ba_every": 5,
        "local_ba_huber_delta": 3.0,
        "local_ba_max_reprojection_error": 8.0,
        "local_ba_min_parallax_deg": 1.0,
        "local_ba_max_pose_change_ratio": 0.2,
        "local_ba_max_rotation_deg": 5.0,
    }
    assert not parsed.local_ba_disable
    assert {name: getattr(parsed, name) for name in expected} == expected

    monkeypatch.setattr(
        sys, "argv", argv + ["--local_ba_disable", "--local_ba_window_size", "9",
                            "--local_ba_recent_keyframes", "6", "--local_ba_fixed_keyframes", "2",
                            "--local_ba_every", "2", "--local_ba_huber_delta", "2.5"]
    )
    parsed = get_args()
    assert parsed.local_ba_disable
    assert parsed.local_ba_window_size == 9
    assert parsed.local_ba_recent_keyframes == 6
    assert parsed.local_ba_fixed_keyframes == 2
    assert parsed.local_ba_every == 2
    assert parsed.local_ba_huber_delta == 2.5


@pytest.mark.parametrize("options", [
    ["--local_ba_fixed_keyframes", "1"],
    ["--local_ba_recent_keyframes", "3"],
    ["--local_ba_recent_keyframes", "4"],
    ["--local_ba_recent_keyframes", "11"],
    ["--local_ba_window_size", "15"],
    ["--local_ba_window_size", "11"],
    ["--local_ba_window_size", "6"],
    ["--local_ba_window_size", "9", "--local_ba_recent_keyframes", "8"],
    ["--local_ba_fixed_keyframes", "6"],
    ["--local_ba_disable", "--local_ba_recent_keyframes", "3"],
    ["--local_ba_every", "0"],
    ["--local_ba_min_parallax_deg", "0"],
    ["--local_ba_max_rotation_deg", "-1"],
    ["--local_ba_huber_delta", "nan"],
])
def test_local_ba_rejects_invalid_settings(monkeypatch, tmp_path, options):
    monkeypatch.setattr(
        sys, "argv", ["slam.py", "-s", str(tmp_path), "-m", str(tmp_path)] + options
    )
    with pytest.raises(SystemExit):
        get_args()


@pytest.mark.parametrize("window,recent,fixed", [(7, 5, 2), (10, 8, 2), (10, 5, 5)])
def test_local_ba_accepts_window_filled_by_recent_frames_and_fixed_anchors(
    monkeypatch, tmp_path, window, recent, fixed
):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path),
        "--local_ba_window_size", str(window),
        "--local_ba_recent_keyframes", str(recent),
        "--local_ba_fixed_keyframes", str(fixed),
    ])

    parsed = get_args()

    assert (parsed.local_ba_window_size, parsed.local_ba_recent_keyframes,
            parsed.local_ba_fixed_keyframes) == (window, recent, fixed)


@pytest.mark.parametrize("direct", ["on", "off"])
@pytest.mark.parametrize("gate", ["on", "off"])
def test_gaussian_ablation_switches_store_booleans(monkeypatch, tmp_path, direct, gate):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path),
        "--direct_gaussian_sampling", direct, "--pose_gaussian_gate", gate,
    ])
    parsed = get_args()
    assert parsed.direct_gaussian_sampling is (direct == "on")
    assert parsed.pose_gaussian_gate is (gate == "on")


def test_gaussian_ablation_boundary_values_and_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path),
        "--direct_gaussian_sampling", "off", "--direct_gaussian_sampling", "on",
        "--pose_gaussian_gate", "off", "--pose_gaussian_gate", "on",
        "--pose_gaussian_min_update_count", "0",
        "--pose_gaussian_min_observed_keyframes", "0",
        "--densification_interval", "1", "--densify_grad_threshold", "0",
        "--percent_dense", "0.025",
    ])
    assert gaussian_ablation_config(get_args()) == {
        "direct_gaussian_sampling": True,
        "pose_gaussian_gate": True,
        "pose_gaussian_min_update_count": 0,
        "pose_gaussian_min_observed_keyframes": 0,
        "densification_interval": 1,
        "densify_grad_threshold": 0.0,
        "percent_dense": 0.025,
    }


@pytest.mark.parametrize("option,value", [
    ("direct_gaussian_sampling", "true"),
    ("pose_gaussian_gate", "yes"),
    ("pose_gaussian_min_update_count", "-1"),
    ("pose_gaussian_min_observed_keyframes", "-1"),
    ("pose_gaussian_min_update_count", "1.5"),
    ("densification_interval", "0"),
    ("densification_interval", "-2"),
    ("densification_interval", "2.5"),
    ("densify_grad_threshold", "-0.001"),
    ("densify_grad_threshold", "nan"),
    ("densify_grad_threshold", "inf"),
    ("percent_dense", "0"),
    ("percent_dense", "-1"),
    ("percent_dense", "nan"),
    ("percent_dense", "inf"),
])
def test_gaussian_ablation_rejects_invalid_settings(monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path), "--" + option, value,
    ])
    with pytest.raises(SystemExit):
        get_args()


def test_gaussian_ablation_config_defaults_for_legacy_namespace():
    from argparse import Namespace

    assert gaussian_ablation_config(Namespace()) == GAUSSIAN_ABLATION_DEFAULTS
    assert gaussian_ablation_config(Namespace(pose_gaussian_gate=False))["pose_gaussian_gate"] is False
