from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import warnings

import numpy as np
import torch

from utils import psnr, resize_longest_side


def tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def w2c_to_c2w(Rt):
    Rt = tensor_to_numpy(Rt).astype(np.float64)
    return np.linalg.inv(Rt)


def rotation_error_deg(R_a, R_b):
    cos = (np.trace(R_a.T @ R_b) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def umeyama_sim3(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] < 3:
        R = np.eye(3)
        scale = 1.0
        t = target.mean(axis=0) - source.mean(axis=0)
        return scale, R, t

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src0 = source - mu_src
    tgt0 = target - mu_tgt
    cov = (tgt0.T @ src0) / source.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = np.mean(np.sum(src0 * src0, axis=1))
    scale = float(np.trace(np.diag(D) @ S) / max(var_src, 1e-12))
    t = mu_tgt - scale * (R @ mu_src)
    return scale, R, t


def align_c2w_poses_sim3(estimated_c2w, gt_c2w):
    est_centres = np.stack([pose[:3, 3] for pose in estimated_c2w], axis=0)
    gt_centres = np.stack([pose[:3, 3] for pose in gt_c2w], axis=0)
    scale, R, t = umeyama_sim3(est_centres, gt_centres)
    aligned = []
    for pose in estimated_c2w:
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R @ pose[:3, :3]
        out[:3, 3] = scale * (R @ pose[:3, 3]) + t
        aligned.append(out)
    return aligned, {"scale": scale, "rotation": R.tolist(), "translation": t.tolist()}


def record_is_ate_selected(record):
    return bool(record.get("ate_selected", record.get("is_keyframe", False)))


def record_is_render_selected(record):
    return bool(record.get("render_selected", record.get("is_test", False)))


def compute_trajectory_outputs(pose_records):
    valid = [
        record
        for record in pose_records
        if record_is_ate_selected(record)
        and record.get("estimated_Rt") is not None
        and record.get("gt_Rt") is not None
    ]
    if not valid:
        frame_rows = [
            {
                "frame_id": record["frame_id"],
                "timestamp": record.get("timestamp"),
                "name": record.get("name"),
                "ate_m": None,
                "rotation_error_deg": None,
                "status": record.get("status", "no_pose"),
                "is_keyframe": bool(record.get("is_keyframe", False)),
                "is_test": bool(record.get("is_test", False)),
                "ate_selected": record_is_ate_selected(record),
                "render_selected": record_is_render_selected(record),
            }
            for record in pose_records
        ]
        aligned_by_frame = {}
        sim3 = None
        rmse = None
    else:
        estimated_c2w = [w2c_to_c2w(record["estimated_Rt"]) for record in valid]
        gt_c2w = [w2c_to_c2w(record["gt_Rt"]) for record in valid]
        aligned_c2w, sim3 = align_c2w_poses_sim3(estimated_c2w, gt_c2w)
        aligned_by_frame = {record["frame_id"]: pose for record, pose in zip(valid, aligned_c2w)}
        gt_by_frame = {record["frame_id"]: pose for record, pose in zip(valid, gt_c2w)}

        squared = []
        frame_rows = []
        for record in pose_records:
            frame_id = record["frame_id"]
            selected = record_is_ate_selected(record)
            if selected and frame_id in aligned_by_frame:
                aligned = aligned_by_frame[frame_id]
                gt = gt_by_frame[frame_id]
                ate = float(np.linalg.norm(aligned[:3, 3] - gt[:3, 3]))
                rot = float(rotation_error_deg(aligned[:3, :3], gt[:3, :3]))
                squared.append(ate * ate)
                status = "ok"
            else:
                ate = None
                rot = None
                status = record.get("status", "no_gt_or_pose") if selected else "not_evaluated"
            frame_rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp": record.get("timestamp"),
                    "name": record.get("name"),
                    "ate_m": ate,
                    "rotation_error_deg": rot,
                    "status": status,
                    "is_keyframe": bool(record.get("is_keyframe", False)),
                    "is_test": bool(record.get("is_test", False)),
                    "ate_selected": selected,
                    "render_selected": record_is_render_selected(record),
                }
            )

        rmse = math.sqrt(sum(squared) / len(squared)) if squared else None
    poses_json = {
        "estimated": [
            {
                "frame_id": record["frame_id"],
                "timestamp": record.get("timestamp"),
                "name": record.get("name"),
                "Rt_w2c": tensor_to_numpy(record["estimated_Rt"]).tolist() if record.get("estimated_Rt") is not None else None,
                "status": record.get("status"),
                "is_keyframe": bool(record.get("is_keyframe", False)),
                "is_test": bool(record.get("is_test", False)),
                "ate_selected": record_is_ate_selected(record),
                "render_selected": record_is_render_selected(record),
            }
            for record in pose_records
        ],
        "gt": [
            {
                "frame_id": record["frame_id"],
                "timestamp": record.get("timestamp"),
                "name": record.get("name"),
                "Rt_w2c": tensor_to_numpy(record["gt_Rt"]).tolist() if record.get("gt_Rt") is not None else None,
                "ate_selected": record_is_ate_selected(record),
                "render_selected": record_is_render_selected(record),
            }
            for record in pose_records
        ],
        "aligned": [
            {
                "frame_id": record["frame_id"],
                "timestamp": record.get("timestamp"),
                "name": record.get("name"),
                "c2w": aligned_by_frame[record["frame_id"]].tolist() if record["frame_id"] in aligned_by_frame else None,
                "ate_selected": record_is_ate_selected(record),
                "render_selected": record_is_render_selected(record),
            }
            for record in pose_records
        ],
        "sim3": sim3,
    }
    return frame_rows, rmse, poses_json


TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping")
_FRAME_ATE_FIELDS = [
    "frame_id", "timestamp", "name", "ate_m", "rotation_error_deg", "status",
    "is_keyframe", "is_test", "ate_selected", "render_selected",
]


def write_trajectory_stage_outputs(out_dir, stage_records):
    """Export frontend and final online pose snapshots without mutating their records."""
    if set(stage_records) != set(TRAJECTORY_STAGES):
        raise ValueError(f"Expected trajectory stages: {', '.join(TRAJECTORY_STAGES)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "frontend_miniba": "Keyframe poses immediately after frontend mini-BA, before backend mapping.",
        "online_mapping": "Final keyframe poses after all online mapping, used directly for saving and evaluation.",
        "pose_convention": "Raw estimated and GT Rt_w2c matrices map world to camera; aligned c2w matrices map camera to world.",
        "evaluation": "Keyframes only, with an estimated pose and ground-truth pose; translation RMSE ATE in meters.",
        "alignment": "Each stage is independently Sim(3)-aligned to its paired GT camera centers. With fewer than 3 pairs, only mean translation is aligned.",
        "comparison": "Rows are registered keyframes present in both stages. ATE and rotation errors use each stage's alignment; camera-center and orientation deltas use raw unaligned poses.",
        "raw_delta_units": "Camera-center deltas use the source trajectory coordinate units, with no guaranteed metric scale; orientation deltas are degrees.",
    }
    metrics = {}
    records_by_stage = {}
    rows_by_stage = {}
    for stage in TRAJECTORY_STAGES:
        # Preserve raw records and their buffers; stage ATE always uses keyframes.
        records = [dict(record, ate_selected=bool(record.get("is_keyframe", False))) for record in stage_records[stage]]
        rows, rmse, poses = compute_trajectory_outputs(records)
        metrics[stage] = {
            "RMSE ATE [m]": rmse,
            "n_registered_keyframes": sum(record["ate_selected"] and record.get("estimated_Rt") is not None for record in records),
            "n_evaluated_keyframes": sum(row["ate_m"] is not None for row in rows),
            "alignment": poses["sim3"],
        }
        poses["stage"] = stage
        poses["definition"] = definitions[stage]
        with open(out_dir / f"slam_poses_{stage}.json", "w", encoding="utf-8") as f:
            json.dump(poses, f, indent=2)
        with open(out_dir / f"slam_frame_ate_{stage}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FRAME_ATE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        records_by_stage[stage] = {
            record["frame_id"]: record
            for record in records
            if record["ate_selected"] and record.get("estimated_Rt") is not None
        }
        rows_by_stage[stage] = {row["frame_id"]: row for row in rows}

    comparison_fields = ["frame_id", "timestamp", "name"]
    for stage in TRAJECTORY_STAGES:
        comparison_fields.extend([f"ate_m_{stage}", f"rotation_error_deg_{stage}"])
    transitions = (
        ("frontend_miniba", "online_mapping", "frontend_to_online"),
    )
    for _, _, label in transitions:
        comparison_fields.extend([f"camera_center_delta_raw_{label}", f"orientation_delta_deg_{label}"])
    common_frames = set.intersection(*(set(records_by_stage[stage]) for stage in TRAJECTORY_STAGES))
    with open(out_dir / "slam_trajectory_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison_fields)
        writer.writeheader()
        # Follow frontend acquisition order instead of sorting arbitrary frame IDs.
        for frame_id, record in records_by_stage["frontend_miniba"].items():
            if frame_id not in common_frames:
                continue
            row = {"frame_id": frame_id, "timestamp": record.get("timestamp"), "name": record.get("name")}
            raw_c2w = {}
            for stage in TRAJECTORY_STAGES:
                frame_row = rows_by_stage[stage][frame_id]
                row[f"ate_m_{stage}"] = frame_row["ate_m"]
                row[f"rotation_error_deg_{stage}"] = frame_row["rotation_error_deg"]
                raw_c2w[stage] = w2c_to_c2w(records_by_stage[stage][frame_id]["estimated_Rt"])
            for before, after, label in transitions:
                row[f"camera_center_delta_raw_{label}"] = float(np.linalg.norm(raw_c2w[after][:3, 3] - raw_c2w[before][:3, 3]))
                row[f"orientation_delta_deg_{label}"] = rotation_error_deg(raw_c2w[before][:3, :3], raw_c2w[after][:3, :3])
            writer.writerow(row)
    with open(out_dir / "slam_trajectory_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"definitions": definitions, "stages": metrics}, f, indent=2)
    # Retire only the old stage files after the replacement outputs succeed.
    # Unlink links themselves; preserve directories and their contents.
    for name in ("slam_poses_final_refinement.json", "slam_frame_ate_final_refinement.csv"):
        obsolete_path = out_dir / name
        if obsolete_path.is_symlink() or obsolete_path.is_file():
            obsolete_path.unlink(missing_ok=True)
    return metrics


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _load_lpips_model(device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    try:
        import lpips
        import sys

        original_stdout = sys.stdout
        with open(os.devnull, "w") as devnull, warnings.catch_warnings():
            sys.stdout = devnull
            warnings.simplefilter("ignore")
            model = lpips.LPIPS(net="vgg").to(device).eval()
        sys.stdout = original_stdout
        return model, device
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device == "cuda":
            torch.cuda.empty_cache()
            return _load_lpips_model("cpu")
        return None, None
    except Exception:
        return None, None
    finally:
        if "original_stdout" in locals():
            sys.stdout = original_stdout


def _compute_lpips_value(lpips_model, lpips_device, render, image, max_size):
    render_for_lpips = resize_longest_side(render.clamp(0, 1), max_size)
    image_for_lpips = resize_longest_side(image.clamp(0, 1), max_size)
    return lpips_model(
        render_for_lpips.to(lpips_device)[None],
        image_for_lpips.to(lpips_device)[None],
    ).item()


@torch.no_grad()
def compute_render_metrics(
    scene_model,
    dataset,
    pose_records,
    with_lpips=True,
    lpips_max_size=512,
    lpips_device="cpu",
):
    try:
        from fused_ssim import fused_ssim
    except Exception:
        fused_ssim = None

    lpips_model = None
    if with_lpips:
        lpips_model, lpips_device = _load_lpips_model(lpips_device)

    rows = []
    if scene_model.n_active_gaussians == 0:
        return rows, {"PSNR": None, "SSIM": None, "LPIPS": None}

    for record in pose_records:
        if not record_is_render_selected(record) or record.get("estimated_Rt") is None:
            continue
        frame_id = record["frame_id"]
        try:
            image, info = dataset.load_frame(frame_id) if hasattr(dataset, "load_frame") else dataset[frame_id]
        except Exception:
            continue
        image = image.cuda()
        Rt = torch.tensor(tensor_to_numpy(record["estimated_Rt"]), dtype=torch.float32, device="cuda")
        from utils import intrinsics_from_info
        intrinsics = intrinsics_from_info(
            info, scene_model.f, scene_model.width, scene_model.height, device="cuda"
        )
        render_pkg = scene_model.render(
            scene_model.width,
            scene_model.height,
            Rt.transpose(0, 1),
            1.0,
            bg=torch.zeros(3, device="cuda"),
            intrinsics=intrinsics,
        )
        render = render_pkg["render"].clamp(0, 1)
        mask = info.get("mask")
        if mask is not None:
            mask = (mask.cuda() > 1 - 1e-6).expand_as(image)
            render_eval = render[mask]
            image_eval = image[mask]
            ssim_render = render * mask
            ssim_image = image * mask
        else:
            render_eval = render
            image_eval = image
            ssim_render = render
            ssim_image = image

        row = {
            "frame_id": frame_id,
            "timestamp": record.get("timestamp"),
            "name": record.get("name"),
            "is_keyframe": bool(record.get("is_keyframe", False)),
            "is_test": bool(record.get("is_test", False)),
            "render_selected": record_is_render_selected(record),
            "psnr": _safe_float(psnr(render_eval, image_eval)),
            "ssim": None,
            "lpips": None,
        }
        if fused_ssim is not None:
            row["ssim"] = _safe_float(fused_ssim(ssim_render[None], ssim_image[None], train=False).item())
        if lpips_model is not None:
            try:
                row["lpips"] = _safe_float(
                    _compute_lpips_value(
                        lpips_model,
                        lpips_device,
                        render,
                        image,
                        lpips_max_size,
                    )
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and lpips_device == "cuda":
                    del lpips_model
                    torch.cuda.empty_cache()
                    lpips_model, lpips_device = _load_lpips_model("cpu")
                    if lpips_model is not None:
                        row["lpips"] = _safe_float(
                            _compute_lpips_value(
                                lpips_model,
                                lpips_device,
                                render,
                                image,
                                lpips_max_size,
                            )
                        )
                else:
                    raise
        rows.append(row)
        del render_pkg, render, image, Rt
        if mask is not None:
            del mask
        torch.cuda.empty_cache()

    aggregate = {}
    for metric in ("psnr", "ssim", "lpips"):
        values = [row[metric] for row in rows if row[metric] is not None]
        aggregate[metric.upper()] = float(np.mean(values)) if values else None
    del lpips_model
    torch.cuda.empty_cache()
    return rows, aggregate


def write_outputs(out_dir, metrics, frame_ate_rows, render_rows, poses_json):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "slam_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "slam_frame_ate.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "timestamp",
                "name",
                "ate_m",
                "rotation_error_deg",
                "status",
                "is_keyframe",
                "is_test",
                "ate_selected",
                "render_selected",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_ate_rows)

    with open(out_dir / "slam_render_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_id", "timestamp", "name", "is_keyframe", "is_test", "render_selected", "psnr", "ssim", "lpips"],
        )
        writer.writeheader()
        writer.writerows(render_rows)

    with open(out_dir / "slam_poses.json", "w", encoding="utf-8") as f:
        json.dump(poses_json, f, indent=2)

