import csv
import json
from pathlib import Path

import pytest
from openpyxl import load_workbook

from scripts import run_all_slam


@pytest.mark.parametrize(
    "profile, expected_root",
    [
        ("server", "/mnt/data_23/zzy/dataset/waymo"),
        ("local", "C:/Users/Administrator/Desktop/dataset/waymo"),
    ],
)
def test_run_all_slam_profile_dry_run(monkeypatch, tmp_path, profile, expected_root):
    discovered = []

    def discover_waymo(root, camera_name):
        assert root == expected_root
        assert camera_name == "FRONT"
        discovered.append("waymo")
        return [tmp_path / "100613"]

    monkeypatch.setattr(run_all_slam, "discover_waymo_sequences", discover_waymo)
    output_root = tmp_path / "results"

    run_all_slam.main(
        [
            "--profile", profile,
            "--cuda_visible_devices", "0",
            "--init_sampling_method", "scharr",
            "--output_root", str(output_root),
            "--dry_run",
        ]
    )

    assert discovered == ["waymo"]
    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert {record["dataset_type"] for record in summary} == {"waymo"}
    assert all(record["returncode"] == 0 for record in summary)
    assert all("CUDA_VISIBLE_DEVICES=0" in record["command"] for record in summary)
    assert all("--init_sampling_method scharr" in record["command"] for record in summary)
    assert all("--dataset_type waymo" in record["command"] for record in summary)


@pytest.mark.parametrize("profile", ["server", "local", "custom"])
def test_run_all_slam_dry_run_writes_excel(tmp_path, profile):
    waymo_root = tmp_path / "waymo"
    (waymo_root / "100613" / "FRONT" / "rgb").mkdir(parents=True)

    output_root = tmp_path / "results"
    run_all_slam.main(
        [
            "--profile",
            profile,
            "--waymo_root",
            str(waymo_root),
            "--output_root",
            str(output_root),
            "--dry_run",
        ]
    )

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert all(record["returncode"] == 0 for record in summary)

    waymo_record = next(record for record in summary if record["dataset_type"] == "waymo")
    assert "--camera_name FRONT" in waymo_record["command"]
    assert "CUDA_VISIBLE_DEVICES=2" in waymo_record["command"]
    assert "--test_hold 8" in waymo_record["command"]
    assert "--num_kpts 8000" in waymo_record["command"]
    assert "--slam_window_size" not in waymo_record["command"]
    assert "--seed 0" in waymo_record["command"]
    assert "--mono_depth_backend da2" in waymo_record["command"]
    assert "--waymo_image_model auto" in waymo_record["command"]

    workbook = load_workbook(output_root / "slam_batch_summary.xlsx", read_only=True)
    assert set(workbook.sheetnames) == {"per_sequence", "aggregate", "failures"}
    assert [row[0] for row in workbook["aggregate"].iter_rows(min_row=2, values_only=True)] == [
        "waymo", "ALL"
    ]

    headers = [cell.value for cell in workbook["per_sequence"][1]]
    for column in (
        "dataset_type",
        "sequence",
        "output_dir",
        "returncode",
        "PSNR",
        "SSIM",
        "LPIPS",
        "TIME",
        "ATE",
        "num_gaussians",
    ):
        assert column in headers


def test_run_all_slam_requires_waymo_sequences(tmp_path):
    with pytest.raises(FileNotFoundError, match="No Waymo sequences discovered"):
        run_all_slam.main(
            [
                "--profile", "custom",
                "--output_root", str(tmp_path / "results"),
                "--dry_run",
            ]
        )


def test_write_excel_summary_records_failures(tmp_path):
    excel_path = tmp_path / "summary.xlsx"
    run_all_slam.write_excel_summary(
        excel_path,
        [
            {
                "dataset_type": "waymo",
                "sequence": "100613",
                "sequence_path": "/mnt/data_23/zzy/dataset/waymo/100613",
                "output_dir": "results/waymo/100613",
                "returncode": 1,
                "command": "python slam.py ...",
            }
        ],
    )

    workbook = load_workbook(excel_path, read_only=True)
    failures = workbook["failures"]
    assert failures.max_row == 2
    headers = [cell.value for cell in failures[1]]
    values = dict(zip(headers, [cell.value for cell in failures[2]]))
    assert values["dataset_type"] == "waymo"
    assert values["sequence"] == "100613"
    assert values["returncode"] == 1


def test_run_all_slam_summary_only_reads_existing_metrics(tmp_path):
    waymo_root = tmp_path / "waymo"
    (waymo_root / "100613" / "FRONT" / "rgb").mkdir(parents=True)

    output_root = tmp_path / "results"
    sequence_out = output_root / "waymo" / "100613"
    sequence_out.mkdir(parents=True)
    (sequence_out / "slam_metrics.json").write_text(
        json.dumps(
            {
                "PSNR": 26.1,
                "SSIM": 0.86,
                "LPIPS": 0.19,
                "time": 84.2,
                "RMSE ATE [m]": 0.85,
                "num_frames": 198,
                "num_keyframes": 70,
                "num_gaussians": 12345,
            }
        ),
        encoding="utf-8",
    )

    run_all_slam.main(
        [
            "--profile",
            "custom",
            "--waymo_root",
            str(waymo_root),
            "--output_root",
            str(output_root),
            "--summary_only",
        ]
    )

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["returncode"] == 0
    assert summary[0]["TIME"] == 84.2
    assert summary[0]["ATE"] == 0.85
    assert summary[0]["num_gaussians"] == 12345

    workbook = load_workbook(output_root / "slam_batch_summary.xlsx", read_only=True)
    headers = [cell.value for cell in workbook["per_sequence"][1]]
    values = dict(zip(headers, [cell.value for cell in workbook["per_sequence"][2]]))
    assert values["num_gaussians"] == 12345

    aggregate_headers = [cell.value for cell in workbook["aggregate"][1]]
    aggregate_values = dict(zip(aggregate_headers, [cell.value for cell in workbook["aggregate"][2]]))
    assert aggregate_values["num_gaussians"] == 12345


def test_run_all_slam_skip_existing_reuses_metrics(tmp_path):
    waymo_root = tmp_path / "waymo"
    (waymo_root / "100613" / "FRONT" / "rgb").mkdir(parents=True)

    output_root = tmp_path / "results"
    sequence_out = output_root / "waymo" / "100613"
    sequence_out.mkdir(parents=True)
    (sequence_out / "slam_metrics.json").write_text(
        json.dumps({"PSNR": 30.0, "time": 10.0, "RMSE ATE [m]": 0.1}),
        encoding="utf-8",
    )

    run_all_slam.main(
        [
            "--profile",
            "custom",
            "--waymo_root",
            str(waymo_root),
            "--output_root",
            str(output_root),
            "--skip_existing",
        ]
    )

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["returncode"] == 0
    assert summary[0]["PSNR"] == 30.0
    assert summary[0]["TIME"] == 10.0
    assert summary[0]["ATE"] == 0.1


def test_batch_only_args_are_not_forwarded_to_slam(tmp_path):
    waymo_root = tmp_path / "waymo"
    sequence = waymo_root / "100613"
    (sequence / "FRONT" / "rgb").mkdir(parents=True)

    record = run_all_slam.run_sequence(
        sequence_path=sequence,
        output_dir=tmp_path / "out",
        camera_name="FRONT",
        slam_args=run_all_slam.build_default_slam_args(run_all_slam.build_parser().parse_args([])),
        extra_args=run_all_slam.sanitize_extra_args(["--skip_existing"]),
        cuda_visible_devices="2",
        dry_run=True,
    )

    assert "--skip_existing" not in record["command"]


def test_run_all_slam_forwards_window_sampling_seed(tmp_path):
    waymo_root = tmp_path / "waymo"
    (waymo_root / "100613" / "FRONT" / "rgb").mkdir(parents=True)

    output_root = tmp_path / "results"
    run_all_slam.main(
        [
            "--profile",
            "custom",
            "--waymo_root",
            str(waymo_root),
            "--output_root",
            str(output_root),
            "--dry_run",
            "--seed",
            "123",
            "--slam_window_sampling_seed",
            "456",
        ]
    )

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    command = summary[0]["command"]

    assert "--seed 123" in command
    assert "--slam_window_sampling_seed 456" in command


@pytest.mark.parametrize("image_model", ["auto", "pinhole", "opencv"])
def test_run_all_slam_forwards_image_model(image_model):
    args = run_all_slam.build_parser().parse_args(["--waymo_image_model", image_model])
    slam_args = run_all_slam.build_default_slam_args(args)
    assert slam_args[slam_args.index("--waymo_image_model") + 1] == image_model


def test_run_all_slam_checks_sources_once_before_first_dispatch(monkeypatch, tmp_path):
    jobs = [tmp_path / "100613", tmp_path / "163453"]
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: jobs)
    calls = []

    def check(root, *, check_local_ba, check_trajectory_stages):
        assert root == run_all_slam.ROOT
        assert check_local_ba is True
        assert check_trajectory_stages is True
        calls.append("check")

    def run(sequence_path, *args):
        calls.append(sequence_path.name)
        return {"dataset_type": "waymo", "sequence": sequence_path.name, "returncode": 0}

    monkeypatch.setattr(run_all_slam, "check_calibration_sources", check)
    monkeypatch.setattr(run_all_slam, "run_sequence", run)
    run_all_slam.main(["--output_root", str(tmp_path / "results")])
    assert calls == ["check", "100613", "163453"]


def test_run_all_slam_stale_calibration_stops_before_any_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_all_slam, "discover_jobs", lambda *args: [tmp_path / "100613", tmp_path / "163453"]
    )

    def stale(root, **kwargs):
        raise RuntimeError("MiniBA graph focal sample is [1], expected (n_cams, 2)")

    monkeypatch.setattr(run_all_slam, "check_calibration_sources", stale)
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *args: pytest.fail("SLAM was launched"))
    with pytest.raises(RuntimeError, match="MiniBA graph focal sample"):
        run_all_slam.main(["--output_root", str(tmp_path / "results")])


@pytest.mark.parametrize("mode", ["--dry_run", "--summary_only", "--skip_existing"])
def test_run_all_slam_no_source_check_without_actual_training(monkeypatch, tmp_path, mode):
    sequence = tmp_path / "100613"
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [sequence])
    monkeypatch.setattr(
        run_all_slam, "check_calibration_sources", lambda *args: pytest.fail("Unexpected source check")
    )
    monkeypatch.setattr(
        run_all_slam, "check_slam_arguments", lambda *args: pytest.fail("Unexpected argument check")
    )
    output_root = tmp_path / "results"
    existing = output_root / "waymo" / sequence.name
    existing.mkdir(parents=True)
    (existing / "slam_metrics.json").write_text("{}", encoding="utf-8")
    run_all_slam.main(["--output_root", str(output_root), mode])


def test_run_all_slam_checks_after_skipped_result_before_first_training(monkeypatch, tmp_path):
    jobs = [tmp_path / "100613", tmp_path / "163453"]
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: jobs)
    output_root = tmp_path / "results"
    existing = output_root / "waymo" / jobs[0].name
    existing.mkdir(parents=True)
    (existing / "slam_metrics.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda root, **kwargs: calls.append("check"))

    def run(sequence_path, *args):
        calls.append(sequence_path.name)
        return {"dataset_type": "waymo", "sequence": sequence_path.name, "returncode": 0}

    monkeypatch.setattr(run_all_slam, "run_sequence", run)
    run_all_slam.main(["--output_root", str(output_root), "--skip_existing"])
    assert calls == ["check", "163453"]


@pytest.mark.parametrize("extra", [[], ["--local_ba_disable", "--local_ba_window_size", "9",
                                       "--local_ba_recent_keyframes", "6", "--local_ba_fixed_keyframes", "2",
                                       "--local_ba_every", "2", "--local_ba_huber_delta", "2.5"]])
def test_batch_local_ba_options_roundtrip_to_single_sequence(monkeypatch, tmp_path, extra):
    import sys
    from args import get_args

    batch = run_all_slam.build_parser().parse_args(extra)
    forwarded = run_all_slam.build_default_slam_args(batch)
    monkeypatch.setattr(
        sys, "argv", ["slam.py", "-s", str(tmp_path), "-m", str(tmp_path)] + forwarded
    )
    single = get_args()
    for name in run_all_slam.LOCAL_BA_VALUE_OPTIONS + ("local_ba_disable",):
        assert getattr(single, name) == getattr(batch, name)
    assert (single.local_ba_window_size, single.local_ba_recent_keyframes,
            single.local_ba_fixed_keyframes) == ((9, 6, 2) if extra else (10, 5, 2))


@pytest.mark.parametrize("dataset_type", ["waymo", "kitti"])
@pytest.mark.parametrize("options", [
    ["--local_ba_fixed_keyframes", "1"],
    ["--local_ba_recent_keyframes", "3"],
    ["--local_ba_recent_keyframes", "4"],
    ["--local_ba_window_size", "15"],
    ["--local_ba_window_size", "11"],
    ["--local_ba_window_size", "6"],
    ["--local_ba_window_size", "9", "--local_ba_recent_keyframes", "8"],
    ["--local_ba_fixed_keyframes", "6"],
    ["--local_ba_disable", "--local_ba_recent_keyframes", "3"],
])
def test_batch_rejects_invalid_local_ba_settings_before_dispatch(
    monkeypatch, tmp_path, dataset_type, options
):
    def unexpected_dispatch(*args, **kwargs):
        pytest.fail("Invalid local BA settings must be rejected before discovery or dispatch")

    monkeypatch.setattr(run_all_slam, "discover_jobs", unexpected_dispatch)
    monkeypatch.setattr(run_all_slam, "run_sequence", unexpected_dispatch)
    output_root = tmp_path / "results"
    with pytest.raises(SystemExit) as error:
        run_all_slam.main([
            "--dataset_type", dataset_type, "--output_root", str(output_root), *options,
        ])
    assert error.value.code == 2
    assert not output_root.exists()


@pytest.mark.parametrize(
    "profile, expected_root",
    [
        ("server", "/mnt/data_23/zzy/dataset/KITTI"),
        ("local", "C:/Users/Administrator/Desktop/dataset/KITTI"),
    ],
)
def test_kitti_profile_dry_run_uses_dataset_defaults(monkeypatch, tmp_path, profile, expected_root):
    def discover_kitti(root):
        assert root == expected_root
        return [tmp_path / "07"]

    monkeypatch.setattr(run_all_slam, "discover_kitti_sequences", discover_kitti)
    monkeypatch.chdir(tmp_path)
    run_all_slam.main(["--dataset_type", "kitti", "--profile", profile, "--dry_run"])

    output_root = tmp_path / run_all_slam.DEFAULT_KITTI_OUTPUT_ROOT
    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    record = summary[0]
    assert record["dataset_type"] == "kitti"
    assert record["sequence"] == "07"
    assert record["output_dir"] == str(
        Path(run_all_slam.DEFAULT_KITTI_OUTPUT_ROOT) / "kitti" / "07"
    )
    assert "--dataset_type kitti" in record["command"]
    assert "--kitti_camera 2" in record["command"]
    assert "--camera_name" not in record["command"]
    assert "--waymo_image_model" not in record["command"]

    workbook = load_workbook(output_root / "slam_batch_summary.xlsx", read_only=True)
    assert [row[0] for row in workbook["aggregate"].iter_rows(min_row=2, values_only=True)] == [
        "kitti", "ALL"
    ]
    workbook.close()


@pytest.mark.parametrize("single_sequence", [False, True])
def test_kitti_discovers_custom_layout_and_forwards_camera(tmp_path, single_sequence):
    kitti_root = tmp_path / "KITTI"
    sequence = kitti_root / "07"
    (sequence / "rgb").mkdir(parents=True)
    (sequence / "gt").mkdir()
    (sequence / "calib.txt").write_text("P2: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
    output_root = tmp_path / "results"

    run_all_slam.main([
        "--dataset_type", "kitti", "--profile", "custom",
        "--kitti_root", str(sequence if single_sequence else kitti_root),
        "--kitti_camera", "0", "--output_root", str(output_root), "--dry_run",
    ])

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 1
    assert summary[0]["sequence_path"] == str(sequence)
    assert summary[0]["output_dir"] == str(output_root / "kitti" / "07")
    assert "--kitti_camera 0" in summary[0]["command"]


@pytest.mark.parametrize("mode", ["--summary_only", "--skip_existing"])
def test_kitti_existing_metrics_keep_dataset_label(monkeypatch, tmp_path, mode):
    sequence = tmp_path / "KITTI" / "07"
    monkeypatch.setattr(run_all_slam, "discover_kitti_sequences", lambda root: [sequence])
    monkeypatch.setattr(
        run_all_slam, "check_calibration_sources", lambda *args: pytest.fail("Unexpected source check")
    )
    output_root = tmp_path / "results"
    sequence_out = output_root / "kitti" / "07"
    sequence_out.mkdir(parents=True)
    (sequence_out / "slam_metrics.json").write_text(
        json.dumps({"PSNR": 25.0, "time": 12.0, "RMSE ATE [m]": 0.2}), encoding="utf-8"
    )

    run_all_slam.main([
        "--dataset_type", "kitti", "--kitti_root", str(sequence),
        "--output_root", str(output_root), mode,
    ])

    summary = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["dataset_type"] == "kitti"
    assert summary[0]["returncode"] == 0
    assert summary[0]["TIME"] == 12.0
    assert summary[0]["ATE"] == 0.2
    assert "--dataset_type kitti" in summary[0]["command"]
    assert "--kitti_camera 2" in summary[0]["command"]


def test_run_all_slam_requires_kitti_sequences(tmp_path):
    with pytest.raises(FileNotFoundError, match="No KITTI sequences discovered"):
        run_all_slam.main([
            "--dataset_type", "kitti", "--profile", "custom",
            "--output_root", str(tmp_path / "results"), "--dry_run",
        ])


@pytest.mark.parametrize("direct", ["on", "off"])
@pytest.mark.parametrize("gate", ["on", "off"])
def test_batch_gaussian_options_roundtrip_and_dry_run_exports(monkeypatch, tmp_path, direct, gate):
    import sys
    from args import gaussian_ablation_config, get_args

    options = [
        "--direct_gaussian_sampling", direct, "--pose_gaussian_gate", gate,
        "--pose_gaussian_min_update_count", "7", "--pose_gaussian_min_observed_keyframes", "4",
        "--densification_interval", "25", "--densify_grad_threshold", "0.0005", "--percent_dense", "0.02",
    ]
    batch = run_all_slam.build_parser().parse_args(options)
    forwarded = run_all_slam.build_default_slam_args(batch)
    monkeypatch.setattr(sys, "argv", ["slam.py", "-s", str(tmp_path), "-m", str(tmp_path)] + forwarded)
    expected = gaussian_ablation_config(get_args())
    assert expected == gaussian_ablation_config(batch)

    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [tmp_path / "100613"])
    output_root = tmp_path / "results"
    run_all_slam.main(options + ["--output_root", str(output_root), "--dry_run"])
    record = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))[0]
    assert {name: record[name] for name in expected} == expected
    assert f"--direct_gaussian_sampling {direct}" in record["command"]
    assert f"--pose_gaussian_gate {gate}" in record["command"]
    with (output_root / "slam_batch_summary.csv").open(newline="", encoding="utf-8") as f:
        csv_record = next(csv.DictReader(f))
    assert {name: csv_record[name] for name in expected} == {name: str(value) for name, value in expected.items()}
    workbook = load_workbook(output_root / "slam_batch_summary.xlsx", read_only=True)
    headers, row = list(workbook["per_sequence"].iter_rows(values_only=True))
    excel_record = dict(zip(headers, row))
    assert {name: excel_record[name] for name in expected} == expected
    workbook.close()


@pytest.mark.parametrize("mode", ["--summary_only", "--skip_existing"])
@pytest.mark.parametrize("has_saved_config", [False, True])
def test_batch_historical_ablation_config_comes_from_saved_metrics(monkeypatch, tmp_path, mode, has_saved_config):
    from args import GAUSSIAN_ABLATION_DEFAULTS

    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [tmp_path / "100613"])
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda *args, **kwargs: pytest.fail("Unexpected training"))
    output_root = tmp_path / "results"
    output_dir = output_root / "waymo" / "100613"
    output_dir.mkdir(parents=True)
    metrics = {"PSNR": 25.0}
    if has_saved_config:
        metrics.update(GAUSSIAN_ABLATION_DEFAULTS)
    (output_dir / "slam_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    run_all_slam.main([
        "--output_root", str(output_root), mode,
        "--direct_gaussian_sampling", "off", "--pose_gaussian_gate", "off",
        "--pose_gaussian_min_update_count", "9", "--densification_interval", "100",
    ])
    record = json.loads((output_root / "slam_batch_summary.json").read_text(encoding="utf-8"))[0]
    assert {name: record[name] for name in GAUSSIAN_ABLATION_DEFAULTS} == {
        name: value if has_saved_config else None for name, value in GAUSSIAN_ABLATION_DEFAULTS.items()
    }
    workbook = load_workbook(output_root / "slam_batch_summary.xlsx", read_only=True)
    headers, row = list(workbook["per_sequence"].iter_rows(values_only=True))
    excel_record = dict(zip(headers, row))
    assert excel_record["direct_gaussian_sampling"] is (True if has_saved_config else None)
    assert excel_record["pose_gaussian_gate"] is (True if has_saved_config else None)
    workbook.close()


@pytest.mark.parametrize("options", [
    ["--pose_gaussian_min_update_count", "-1"],
    ["--pose_gaussian_min_observed_keyframes", "-1"],
    ["--densification_interval", "0"],
    ["--densify_grad_threshold", "nan"],
    ["--percent_dense", "0"],
    ["--direct_gaussian_sampling", "false"],
    ["--pose_gaussian_gate", "true"],
])
def test_batch_rejects_invalid_gaussian_settings_before_dispatch(monkeypatch, tmp_path, options):
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: pytest.fail("Invalid options dispatched"))
    with pytest.raises(SystemExit):
        run_all_slam.main(options + ["--output_root", str(tmp_path / "results")])
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("direct", ["on", "off"])
@pytest.mark.parametrize("gate", ["on", "off"])
def test_actual_single_sequence_parser_accepts_ablation_command(tmp_path, capsys, direct, gate):
    batch = run_all_slam.build_parser().parse_args([
        "--direct_gaussian_sampling", direct, "--pose_gaussian_gate", gate,
    ])
    command = run_all_slam.build_sequence_command(
        tmp_path / "106762", tmp_path / "output", "FRONT",
        run_all_slam.build_default_slam_args(batch), ["--viewer_mode", "none"],
    )
    run_all_slam.check_slam_arguments(command)
    assert "SLAM argument interface OK" in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_batch_stale_disk_parser_stops_before_any_sequence(monkeypatch, tmp_path):
    # The running batch module can be current while the child reads an older
    # args.py from disk. Reproduce the server's missing on/off options.
    (tmp_path / "args.py").write_text(
        "import argparse, sys\n"
        "assert 'torch' not in sys.modules\n"
        "def get_args():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('-s')\n"
        "    parser.add_argument('-m')\n"
        "    return parser.parse_args()\n", encoding="utf-8",
    )
    monkeypatch.setattr(run_all_slam, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [
        tmp_path / "100613", tmp_path / "106762",
    ])
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda *a, **kw: None)
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *a, **kw: pytest.fail("SLAM was launched"))
    with pytest.raises(RuntimeError, match="no SLAM sequence was started") as error:
        run_all_slam.main(["--output_root", str(tmp_path / "results"),
                           "--direct_gaussian_sampling", "on", "--pose_gaussian_gate", "on"])
    assert "unrecognized arguments" in str(error.value)
    assert "--direct_gaussian_sampling on --pose_gaussian_gate on" in str(error.value)
    assert str(tmp_path / "args.py") in str(error.value)
    assert not (tmp_path / "results/slam_batch_summary.json").exists()


def test_batch_argument_typo_stops_before_any_sequence(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [tmp_path / "106762"])
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda *a, **kw: None)
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *a, **kw: pytest.fail("SLAM was launched"))
    with pytest.raises(RuntimeError, match="unrecognized arguments: --viewer_mdoe none"):
        run_all_slam.main(["--output_root", str(tmp_path / "results"), "--viewer_mdoe", "none"])

