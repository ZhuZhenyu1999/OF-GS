import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from copy import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders.waymo_dataset import discover_waymo_sequences
from dataloaders.kitti_dataset import discover_kitti_sequences
from dataloaders.dl3dv_dataset import discover_dl3dv_sequences
from args import (
    GAUSSIAN_ABLATION_OPTIONS,
    LOCAL_BA_VALUE_OPTIONS,
    add_gaussian_ablation_arguments,
    add_local_ba_arguments,
    gaussian_ablation_config,
    validate_gaussian_ablation_arguments,
    validate_local_ba_arguments,
)
from scripts.calibration_preflight import check_calibration_sources


PROFILE_WAYMO_ROOTS = {
    "server": "/mnt/data_23/zzy/dataset/waymo",
    "local": "C:/Users/Administrator/Desktop/dataset/waymo",
    "custom": "",
}
PROFILE_KITTI_ROOTS = {
    "server": "/mnt/data_23/zzy/dataset/KITTI",
    "local": "C:/Users/Administrator/Desktop/dataset/KITTI",
    "custom": "",
}
PROFILE_DL3DV_ROOTS = {
    "server": "/mnt/data_23/zzy/dataset/dl3dv",
    "local": "C:/Users/Administrator/Desktop/dataset/dl3dv",
    "custom": "",
}
DEFAULT_DL3DV_OUTPUT_ROOT = "results/dl3dv_testhold8_v2"
DEFAULT_OUTPUT_ROOT = "results/waymo_testhold8_v2"
DEFAULT_KITTI_OUTPUT_ROOT = "results/kitti_testhold8_v2"

PER_SEQUENCE_COLUMNS = [
    "dataset_type",
    "sequence",
    "sequence_path",
    "output_dir",
    "returncode",
    "PSNR",
    "SSIM",
    "LPIPS",
    "TIME",
    "ATE",
    "FPS",
    "num_frames",
    "num_keyframes",
    "num_gaussians",
    *GAUSSIAN_ABLATION_OPTIONS,
    "command",
]
AGGREGATE_METRICS = [
    "PSNR",
    "SSIM",
    "LPIPS",
    "TIME",
    "ATE",
    "FPS",
    "num_frames",
    "num_keyframes",
    "num_gaussians",
]
AGGREGATE_COLUMNS = ["dataset_type", "num_sequences", "num_success", "num_failed"] + AGGREGATE_METRICS
FAILURE_COLUMNS = ["dataset_type", "sequence", "sequence_path", "output_dir", "returncode", "command"]
BATCH_ONLY_EXTRA_FLAGS = {"--summary_only", "--skip_existing"}


def profile_defaults(profile, dataset_type="waymo"):
    roots = {"waymo": PROFILE_WAYMO_ROOTS, "kitti": PROFILE_KITTI_ROOTS,
             "dl3dv": PROFILE_DL3DV_ROOTS}[dataset_type]
    return roots.get(profile, "")


def build_parser():
    parser = argparse.ArgumentParser(description="Batch runner for TEST2-SLAM Waymo, KITTI and DL3DV sequences.")
    parser.add_argument("--profile", choices=["server", "local", "custom"], default="server")
    parser.add_argument("--dataset_type", choices=["waymo", "kitti", "dl3dv"], default="waymo")
    parser.add_argument("--dl3dv_root", type=str, default=None,
                        help="DL3DV root containing 1/, 2/, 3/, or a single sequence directory.")
    parser.add_argument("--waymo_root", type=str, default=None)
    parser.add_argument("--kitti_root", type=str, default=None,
                        help="KITTI root or a single sequence directory such as /mnt/data_23/zzy/dataset/KITTI/07.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Defaults to results/<dataset>_testhold8_v2.")
    parser.add_argument("--excel_path", type=str, default="")
    parser.add_argument("--camera_name", type=str, default="FRONT")
    parser.add_argument("--kitti_camera", choices=["0", "1", "2", "3"], default="2",
                        help="KITTI projection matrix to use from calib.txt (default: P2).")
    parser.add_argument(
        "--waymo_image_model", choices=["auto", "pinhole", "opencv"], default="auto",
        help="Image/calibration convention; auto refuses nonzero distortion without an explicit choice.",
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="2")
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only read existing per-sequence slam_metrics.json files and write batch summaries.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse an existing sequence output when slam_metrics.json is present instead of rerunning it.",
    )

    parser.add_argument("--test_hold", type=int, default=8)
    parser.add_argument("--num_kpts", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slam_window_sampling_seed", type=int, default=None)
    parser.add_argument("--mono_depth_backend", choices=["da2", "da3mono"], default="da2")
    add_local_ba_arguments(parser)
    add_gaussian_ablation_arguments(parser)
    return parser


def build_default_slam_args(args):
    slam_args = [
        "--test_hold",
        str(args.test_hold),
        "--num_kpts",
        str(args.num_kpts),
        "--seed",
        str(args.seed),
        "--mono_depth_backend",
        str(args.mono_depth_backend),
    ]
    if getattr(args, "dataset_type", "waymo") == "waymo":
        slam_args.extend(["--waymo_image_model", args.waymo_image_model])
    if args.slam_window_sampling_seed is not None:
        slam_args.extend(
            ["--slam_window_sampling_seed", str(args.slam_window_sampling_seed)]
        )
    for name in LOCAL_BA_VALUE_OPTIONS:
        slam_args.extend(["--" + name, str(getattr(args, name))])
    if args.local_ba_disable:
        slam_args.append("--local_ba_disable")
    for name, value in gaussian_ablation_config(args).items():
        if isinstance(value, bool):
            value = "on" if value else "off"
        slam_args.extend(["--" + name, str(value)])
    return slam_args


def sanitize_extra_args(extra_args):
    sanitized = []
    removed = []
    for arg in extra_args:
        if arg in BATCH_ONLY_EXTRA_FLAGS:
            removed.append(arg)
            continue
        sanitized.append(arg)
    if removed:
        print(
            f"Warning: removed batch-only option(s) from slam.py arguments: {' '.join(removed)}",
            file=sys.stderr,
        )
    return sanitized


def command_to_string(cmd, cuda_visible_devices=""):
    command = " ".join(shlex.quote(str(part)) for part in cmd)
    if cuda_visible_devices:
        return f"CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_visible_devices))} {command}"
    return command


def run_sequence(
    sequence_path,
    output_dir,
    camera_name,
    slam_args,
    extra_args,
    cuda_visible_devices="",
    dry_run=False,
    dataset_type="waymo",
    kitti_camera="2",
):
    cmd = build_sequence_command(
        sequence_path, output_dir, camera_name, slam_args, extra_args, dataset_type, kitti_camera
    )

    command = command_to_string(cmd, cuda_visible_devices)
    print(command)

    record = {
        "dataset_type": dataset_type,
        "sequence": sequence_path.name,
        "sequence_path": str(sequence_path),
        "output_dir": str(output_dir),
        "returncode": 0,
        "command": command,
        **dict.fromkeys(GAUSSIAN_ABLATION_OPTIONS),
    }
    if dry_run:
        config_parser = argparse.ArgumentParser(add_help=False)
        add_gaussian_ablation_arguments(config_parser)
        requested_args, _ = config_parser.parse_known_args(list(slam_args) + list(extra_args))
        validate_gaussian_ablation_arguments(requested_args, config_parser)
        record.update(gaussian_ablation_config(requested_args))
        return record

    env = os.environ.copy()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    record["returncode"] = result.returncode

    metrics_path = output_dir / "slam_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        record.update(metrics)
        record["dataset_type"] = dataset_type
        record["TIME"] = metrics.get("time")
        record["ATE"] = metrics.get("RMSE ATE [m]")
    return record


def build_sequence_command(
    sequence_path, output_dir, camera_name, slam_args, extra_args,
    dataset_type="waymo", kitti_camera="2",
):
    cmd = [
        sys.executable,
        str(ROOT / "slam.py"),
        "-s",
        str(sequence_path),
        "--dataset_type",
        dataset_type,
    ]
    if dataset_type == "kitti":
        cmd.extend(["--kitti_camera", kitti_camera])
    elif dataset_type == "waymo":
        cmd.extend(["--camera_name", camera_name])
    cmd += list(slam_args) + list(extra_args) + ["-m", str(output_dir)]
    return cmd


def check_slam_arguments(command):
    """Parse the actual child arguments from disk without loading SLAM/CUDA."""
    probe = (
        "import runpy, sys; sys.argv[0] = 'slam.py'; "
        "runpy.run_path('args.py')['get_args']()"
    )
    try:
        result = subprocess.run(
            [command[0], "-S", "-c", probe, *command[2:]],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"SLAM argument preflight could not check {ROOT / 'args.py'}; "
            f"no SLAM sequence was started: {exc}"
        ) from exc
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise RuntimeError(
            f"SLAM argument preflight failed for {ROOT / 'args.py'}; "
            "no SLAM sequence was started.\n"
            f"{details}\n"
            "Check the arguments and synchronize the Gaussian ablation update "
            "(args.py, slam.py, scene/scene_model.py and scripts/run_all_slam.py) "
            "into this checkout, then restart Python."
        )
    print(f"SLAM argument interface OK: {ROOT / 'args.py'}")


def existing_sequence_record(
    sequence_path,
    output_dir,
    camera_name,
    slam_args,
    extra_args,
    cuda_visible_devices="",
    dataset_type="waymo",
    kitti_camera="2",
):
    cmd = build_sequence_command(
        sequence_path, output_dir, camera_name, slam_args, extra_args, dataset_type, kitti_camera
    )

    record = {
        "dataset_type": dataset_type,
        "sequence": sequence_path.name,
        "sequence_path": str(sequence_path),
        "output_dir": str(output_dir),
        "returncode": 1,
        "command": command_to_string(cmd, cuda_visible_devices),
        # Historical results must describe their saved run, not this invocation.
        **dict.fromkeys(GAUSSIAN_ABLATION_OPTIONS),
    }

    metrics_path = output_dir / "slam_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        record.update(metrics)
        record["dataset_type"] = dataset_type
        record["TIME"] = metrics.get("time")
        record["ATE"] = metrics.get("RMSE ATE [m]")
        record["returncode"] = 0
    return record


def discover_jobs(waymo_root, camera_name, dataset_type="waymo"):
    if not waymo_root:
        return []
    if dataset_type == "dl3dv":
        return discover_dl3dv_sequences(waymo_root)
    if dataset_type == "kitti":
        return discover_kitti_sequences(waymo_root)
    return discover_waymo_sequences(waymo_root, camera_name)


def make_csv_fieldnames(summary):
    preferred = PER_SEQUENCE_COLUMNS + [
        "time",
        "RMSE ATE [m]",
        "num_registered_frames",
        "num_ate_selected_frames",
        "num_ate_registered_frames",
        "num_render_selected_frames",
        "num_render_registered_frames",
        "num_gaussians",
    ]
    seen = set()
    fieldnames = []
    for key in preferred + sorted({key for record in summary for key in record.keys()}):
        if key in seen:
            continue
        if any(key in record for record in summary):
            fieldnames.append(key)
            seen.add(key)
    return fieldnames


def write_json_csv_summary(output_root, summary):
    json_path = output_root / "slam_batch_summary.json"
    csv_path = output_root / "slam_batch_summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = make_csv_fieldnames(summary)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    return json_path, csv_path


def metric_value(record, metric):
    if metric == "TIME":
        return record.get("TIME", record.get("time"))
    if metric == "ATE":
        return record.get("ATE", record.get("RMSE ATE [m]"))
    return record.get(metric)


def excel_record(record):
    row = {column: record.get(column) for column in PER_SEQUENCE_COLUMNS}
    for metric in ("PSNR", "SSIM", "LPIPS", "TIME", "ATE", "FPS"):
        row[metric] = metric_value(record, metric)
    return row


def numeric_values(records, metric):
    values = []
    for record in records:
        value = metric_value(record, metric)
        if isinstance(value, bool) or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def aggregate_row(dataset_type, records):
    successful = [record for record in records if record.get("returncode") == 0]
    row = {
        "dataset_type": dataset_type,
        "num_sequences": len(records),
        "num_success": len(successful),
        "num_failed": len(records) - len(successful),
    }
    for metric in AGGREGATE_METRICS:
        values = numeric_values(successful, metric)
        row[metric] = sum(values) / len(values) if values else None
    return row


def write_rows(ws, columns, rows):
    ws.append(columns)
    for row in rows:
        ws.append([row.get(column) for column in columns])
    ws.freeze_panes = "A2"
    for cell in ws[1]:
        font = copy(cell.font)
        font.bold = True
        cell.font = font
    for index, column in enumerate(columns, start=1):
        max_len = len(column)
        for row in rows:
            value = row.get(column)
            if value is not None:
                max_len = max(max_len, len(str(value)))
        ws.column_dimensions[ws.cell(row=1, column=index).column_letter].width = min(max_len + 2, 80)


def write_excel_summary(excel_path, summary):
    try:
        from openpyxl import Workbook
    except ModuleNotFoundError as exc:
        if exc.name == "openpyxl":
            print(
                "Warning: openpyxl is not installed; skipped Excel summary. "
                "Install it with `pip install openpyxl` and rerun with --summary_only.",
                file=sys.stderr,
            )
            return None
        raise

    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    per_sequence = wb.active
    per_sequence.title = "per_sequence"
    write_rows(per_sequence, PER_SEQUENCE_COLUMNS, [excel_record(record) for record in summary])

    aggregate = wb.create_sheet("aggregate")
    dataset_types = sorted({record.get("dataset_type", "waymo") for record in summary}) or ["waymo"]
    aggregate_rows = [
        aggregate_row(dataset_type, [record for record in summary if record.get("dataset_type", "waymo") == dataset_type])
        for dataset_type in dataset_types
    ] + [aggregate_row("ALL", summary)]
    write_rows(aggregate, AGGREGATE_COLUMNS, aggregate_rows)

    failures = wb.create_sheet("failures")
    failure_rows = [record for record in summary if record.get("returncode") != 0]
    write_rows(failures, FAILURE_COLUMNS, failure_rows)

    wb.save(excel_path)
    return excel_path


def main(argv=None):
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    validate_local_ba_arguments(args, parser)
    validate_gaussian_ablation_arguments(args, parser)
    extra_args = sanitize_extra_args(extra_args)

    explicit_root = getattr(args, args.dataset_type + "_root")
    dataset_root = explicit_root if explicit_root is not None else profile_defaults(args.profile, args.dataset_type)

    default_output_root = {"waymo": DEFAULT_OUTPUT_ROOT, "kitti": DEFAULT_KITTI_OUTPUT_ROOT,
                           "dl3dv": DEFAULT_DL3DV_OUTPUT_ROOT}[args.dataset_type]
    output_root = Path(args.output_root if args.output_root is not None else default_output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    excel_path = Path(args.excel_path) if args.excel_path else output_root / "slam_batch_summary.xlsx"

    jobs = discover_jobs(dataset_root, args.camera_name, args.dataset_type)
    if args.max_sequences > 0:
        jobs = jobs[: args.max_sequences]
    if not jobs:
        dataset_label = {"waymo": "Waymo", "kitti": "KITTI", "dl3dv": "DL3DV"}[args.dataset_type]
        raise FileNotFoundError(f"No {dataset_label} sequences discovered under {dataset_root!r}.")

    slam_args = build_default_slam_args(args)
    summary = []
    calibration_checked = False
    dataset_options = {"dataset_type": args.dataset_type} if args.dataset_type != "waymo" else {}
    if args.dataset_type == "kitti":
        dataset_options["kitti_camera"] = args.kitti_camera
    print(f"Dataset root: {dataset_root}; {len(jobs)} sequences (serial execution)")
    for job_index, seq in enumerate(jobs, start=1):
        print(f"[{job_index}/{len(jobs)}] {args.dataset_type}/{seq.name}", flush=True)
        out_dir = output_root / args.dataset_type / seq.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.summary_only:
            summary.append(
                existing_sequence_record(
                    seq,
                    out_dir,
                    args.camera_name,
                    slam_args,
                    extra_args,
                    args.cuda_visible_devices,
                    **dataset_options,
                )
            )
        elif args.skip_existing and (out_dir / "slam_metrics.json").exists():
            print(f"Skipping existing result: {out_dir}")
            summary.append(
                existing_sequence_record(
                    seq,
                    out_dir,
                    args.camera_name,
                    slam_args,
                    extra_args,
                    args.cuda_visible_devices,
                    **dataset_options,
                )
            )
        else:
            if not args.dry_run and not calibration_checked:
                print(f"SLAM Python: {sys.executable}")
                check_calibration_sources(
                    ROOT, check_local_ba=not args.local_ba_disable, check_trajectory_stages=True
                )
                check_slam_arguments(build_sequence_command(
                    seq, out_dir, args.camera_name, slam_args, extra_args,
                    **dataset_options,
                ))
                calibration_checked = True
            summary.append(
                run_sequence(
                    seq,
                    out_dir,
                    args.camera_name,
                    slam_args,
                    extra_args,
                    args.cuda_visible_devices,
                    args.dry_run,
                    **dataset_options,
                )
            )

        # Preserve completed results even if a later sequence is interrupted.
        write_json_csv_summary(output_root, summary)

    json_path, csv_path = write_json_csv_summary(output_root, summary)
    xlsx_path = write_excel_summary(excel_path, summary)
    if xlsx_path is None:
        print(f"Wrote {json_path} and {csv_path}")
    else:
        print(f"Wrote {json_path}, {csv_path}, and {xlsx_path}")


if __name__ == "__main__":
    main()
