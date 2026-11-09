import copy
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from args import gaussian_ablation_config, get_args
from dataloaders.image_dataset import ImageDataset
from dataloaders.dl3dv_dataset import DL3DVDataset, discover_dl3dv_sequences
from dataloaders.kitti_dataset import KITTIDataset, discover_kitti_sequences
from dataloaders.tum_dataset import TUMDataset, discover_tum_sequences
from dataloaders.waymo_dataset import WaymoDataset, discover_waymo_sequences
from slam_backend import SLAMBackend
from slam_frontend import SLAMFrontend
from scripts.calibration_preflight import check_trajectory_sources
from slam_outputs import (
    compute_render_metrics,
    compute_trajectory_outputs,
    write_outputs,
    write_trajectory_stage_outputs,
)


KEYFRAME_TIMING_STAGES = [
    "image_loading",
    "feature_extraction",
    "keyframe_selection",
    "pose_estimation",
    "keyframe_registration",
    "local_ba",
    "gaussian_sampling",
    "mapping_optimization",
]


def detect_dataset_type(args):
    if args.dataset_type != "auto":
        return args.dataset_type
    if discover_tum_sequences(args.source_path):
        return "tum"
    # KITTI exports can also contain rgb/, so inspect calib.txt before the
    # broader Waymo rgb-folder detector.
    if discover_kitti_sequences(args.source_path):
        return "kitti"
    if discover_dl3dv_sequences(args.source_path):
        return "dl3dv"
    if discover_waymo_sequences(args.source_path, args.camera_name):
        return "waymo"
    return "nvs"


def create_dataset(args):
    dataset_type = detect_dataset_type(args)
    if dataset_type == "dl3dv":
        sequences = discover_dl3dv_sequences(args.source_path)
        if len(sequences) > 1:
            print(f"Multiple DL3DV sequences found; using first sequence: {sequences[0]}")
            args.source_path = str(sequences[0])
        return DL3DVDataset(args), "dl3dv"
    if dataset_type == "kitti":
        sequences = discover_kitti_sequences(args.source_path)
        if len(sequences) > 1 and os.path.abspath(str(sequences[0])) != os.path.abspath(args.source_path):
            print(f"Multiple KITTI sequences found; using first sequence: {sequences[0]}")
            args.source_path = str(sequences[0])
        return KITTIDataset(args), "kitti"
    if dataset_type == "tum":
        sequences = discover_tum_sequences(args.source_path)
        if len(sequences) > 1 and os.path.abspath(str(sequences[0])) != os.path.abspath(args.source_path):
            print(f"Multiple TUM sequences found; using first sequence: {sequences[0]}")
            args.source_path = str(sequences[0])
        return TUMDataset(args), "tum"
    if dataset_type == "waymo":
        sequences = discover_waymo_sequences(args.source_path, args.camera_name)
        if len(sequences) > 1 and os.path.abspath(str(sequences[0])) != os.path.abspath(args.source_path):
            print(f"Multiple Waymo sequences found; using first sequence: {sequences[0]}")
            args.source_path = str(sequences[0])
        return WaymoDataset(args), "waymo"
    return ImageDataset(args), "nvs"


def maybe_seed_from_dataset(args, dataset):
    args.init_intrinsics = None
    focal = None
    if hasattr(dataset, "infos") and len(dataset.infos) > 0:
        first_info = next(iter(dataset.infos.values())) if isinstance(dataset.infos, dict) else dataset.infos[0]
        if isinstance(first_info, dict) and first_info.get("intrinsics_calibrated", True):
            K = first_info.get("K_np")
            if K is None:
                K = first_info.get("K")
            if K is not None:
                args.init_intrinsics = np.asarray(K.detach().cpu() if torch.is_tensor(K) else K, dtype=np.float32)
        if isinstance(first_info, dict) and "focal_np" in first_info:
            focal = float(np.asarray(first_info["focal_np"]).reshape(-1)[0])
    if args.init_focal <= 0 and focal is not None and focal > 0:
        args.init_focal = focal


def clone_pose(Rt):
    return Rt.detach().cpu().clone() if torch.is_tensor(Rt) else copy.deepcopy(Rt)


def snapshot_pose_records(records, pose_estimates=None):
    """Freeze a trajectory; optionally substitute immutable frontend estimates."""
    snapshot = copy.deepcopy([records[frame_id] for frame_id in sorted(records)])
    if pose_estimates is not None:
        for record in snapshot:
            record["estimated_Rt"] = clone_pose(pose_estimates.get(record["frame_id"]))
    return snapshot


def add_pose_record(records, frame_id, info, estimated_Rt=None, status="pending", is_keyframe=False):
    gt_Rt = info.get("Rt")
    is_test = bool(info.get("is_test", False))
    is_keyframe = bool(is_keyframe)
    records[frame_id] = {
        "frame_id": frame_id,
        "timestamp": info.get("timestamp"),
        "name": info.get("name", str(frame_id)),
        "estimated_Rt": clone_pose(estimated_Rt),
        "gt_Rt": clone_pose(gt_Rt),
        "status": status,
        "is_keyframe": is_keyframe,
        "is_test": is_test,
        "ate_selected": is_keyframe,
        "render_selected": is_test,
    }


@torch.no_grad()
def refresh_keyframe_pose_records(records, backend):
    for keyframe_id, frame_id in backend.keyframe_to_frame_id.items():
        if frame_id not in records or keyframe_id >= len(backend.scene_model.keyframes):
            continue
        records[frame_id]["estimated_Rt"] = (
            backend.scene_model.keyframes[keyframe_id].get_Rt().detach().cpu().clone()
        )


def should_estimate_pose_for_frame(decision, info):
    return bool(decision.selected or info.get("is_test", False))


class KeyframeTimingStats:
    def __init__(self):
        self.totals_ms = {stage: 0.0 for stage in KEYFRAME_TIMING_STAGES}

    def add(self, stage: str, elapsed_ms: float):
        self.totals_ms[stage] = self.totals_ms.get(stage, 0.0) + float(elapsed_ms)

    def add_many(self, timings_ms):
        if not timings_ms:
            return
        for stage, elapsed_ms in timings_ms.items():
            self.add(stage, elapsed_ms)

    def averages(self, num_keyframes: int):
        if num_keyframes <= 0:
            return {}
        return {
            stage: self.totals_ms.get(stage, 0.0) / num_keyframes
            for stage in KEYFRAME_TIMING_STAGES
        }


def sync_cuda_if_needed(args):
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(args, func, *call_args, **call_kwargs):
    sync_cuda_if_needed(args)
    start = time.perf_counter()
    result = func(*call_args, **call_kwargs)
    sync_cuda_if_needed(args)
    return result, (time.perf_counter() - start) * 1000.0


def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_slam(args, viewer=None):
    check_trajectory_sources(os.path.dirname(os.path.abspath(__file__)))
    publisher = None
    if viewer is not None:
        from visualization.snapshot import SnapshotPublisher
        publisher = SnapshotPublisher(viewer, max_points=args.viewer_max_points,
                                      max_size=args.viewer_image_size,
                                      interval=args.viewer_interval)
    ablation_config = gaussian_ablation_config(args)
    generation_mode = "direct sampling" if ablation_config["direct_gaussian_sampling"] else "gradient clone/split densification"
    gate_mode = "on" if ablation_config["pose_gaussian_gate"] else "off"
    print(f"Gaussian generation: {generation_mode}; pose Gaussian gate: {gate_mode}")
    print("Gaussian ablation configuration:", json.dumps(ablation_config, sort_keys=True))
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TEST2-SLAM tracking and Gaussian mapping.")
    os.makedirs(args.model_path, exist_ok=True)

    dataset, dataset_type = create_dataset(args)
    if getattr(args, "export_da2_frames", None):
        from depth_export import export_da2_images, select_requested_images
        selected_images = select_requested_images(dataset.image_paths, args.export_da2_frames)
        if viewer is not None:
            viewer.set_status("initializing", "Exporting requested Depth Anything V2 predictions...")
        export_dir = os.path.join(args.model_path, "da2_depth")
        export_da2_images(selected_images, export_dir, device=args.device)
        print("Wrote Depth Anything V2 relative depth outputs to:", export_dir, flush=True)
    if viewer is not None:
        viewer.publish({"total_frames": len(dataset), "status": "initializing",
                        "message": "Loading feature, depth and mapping modules..."})
    maybe_seed_from_dataset(args, dataset)
    height, width = dataset.get_image_size()

    print(f"Dataset: {dataset_type}, frames: {len(dataset)}, size: {width}x{height}")
    print("Initializing TEST2-SLAM modules...")
    frontend = SLAMFrontend(width, height, args)
    backend = SLAMBackend(width, height, args, frontend.matcher, frontend.triangulator)
    focal = frontend.focal_init

    bootstrap_packets = []
    pose_records = {}
    frontend_pose_estimates = {}
    bootstrapped = False
    processed_frames = 0
    selected_keyframes = 0
    keyframe_timing = KeyframeTimingStats()
    start_time = time.time()

    def progress_postfix():
        return (
            f"kf={selected_keyframes}, "
            f"gs={backend.scene_model.n_active_gaussians}"
        )

    pbar = tqdm(range(len(dataset)), desc="TEST2-SLAM")

    def publish_frame(force=False, status=None):
        if publisher is None:
            return
        publisher.update(
            backend.scene_model, frame_id, len(dataset), image, info,
            Rt=pose_records[frame_id].get("estimated_Rt"), force=force,
            status=status or ("running" if bootstrapped else "bootstrapping"),
        )

    for frame_id in pbar:
        if publisher is not None:
            publisher.wait_if_paused()
        (image, info), load_ms = timed_call(args, dataset.getnext)
        info = copy.copy(info)
        info["frame_id"] = frame_id
        if "K" in info or "K_np" in info:
            frontend.set_frame_info(info)
            focal = frontend.focal
        add_pose_record(pose_records, frame_id, info, status="unregistered")
        desc_kpts, feature_ms = timed_call(args, frontend.extract, image)
        decision, selection_ms = timed_call(
            args,
            frontend.decide_keyframe,
            frame_id,
            desc_kpts,
        )
        processed_frames += 1

        if not bootstrapped:
            if decision.selected:
                frontend.accept_keyframe(frame_id, desc_kpts)
                bootstrap_packets.append(
                    {
                        "frame_id": frame_id,
                        "image": image,
                        "info": info,
                        "desc_kpts": desc_kpts,
                        "decision": decision,
                        "timings_ms": {
                            "image_loading": load_ms,
                            "feature_extraction": feature_ms,
                            "keyframe_selection": selection_ms,
                        },
                    }
                )
                selected_keyframes += 1
                pose_records[frame_id]["status"] = f"bootstrap_{decision.reason}"
                pose_records[frame_id]["is_keyframe"] = True
                pose_records[frame_id]["ate_selected"] = True

            if len(bootstrap_packets) == args.num_keyframes_miniba_bootstrap:
                bootstrap_infos = [packet["info"] for packet in bootstrap_packets]
                bootstrap_kwargs = {"infos": bootstrap_infos} if any(
                    "K" in info or "K_np" in info for info in bootstrap_infos
                ) else {}
                bootstrap_result, pose_ms = timed_call(
                    args,
                    frontend.initialize_bootstrap,
                    [packet["desc_kpts"] for packet in bootstrap_packets],
                    **bootstrap_kwargs,
                )
                Rts, f, residual = bootstrap_result
                focal = f
                for packet, Rt in zip(bootstrap_packets, Rts):
                    frontend_pose_estimates[packet["frame_id"]] = clone_pose(Rt)
                mapping_summary = backend.register_bootstrap(bootstrap_packets, Rts, f)
                for packet in bootstrap_packets:
                    keyframe_timing.add_many(packet.get("timings_ms"))
                keyframe_timing.add("pose_estimation", pose_ms)
                keyframe_timing.add_many(mapping_summary.timings_ms)
                for packet, Rt in zip(bootstrap_packets, Rts):
                    add_pose_record(
                        pose_records,
                        packet["frame_id"],
                        packet["info"],
                        Rt,
                        status="bootstrap_keyframe",
                        is_keyframe=True,
                    )
                refresh_keyframe_pose_records(pose_records, backend)
                bootstrapped = True
                print(f"Bootstrap complete with residual {float(residual):.4f}")
            pbar.set_postfix_str(
                f"bootstrap {len(bootstrap_packets)}/{args.num_keyframes_miniba_bootstrap}, kf={selected_keyframes}"
            )
            publish_frame(force=bootstrapped)
            continue

        is_test_frame = bool(info.get("is_test", False))
        will_be_keyframe = decision.selected
        if not should_estimate_pose_for_frame(decision, info):
            pose_records[frame_id]["status"] = "skipped_pose"
            pbar.set_postfix_str(progress_postfix())
            publish_frame()
            continue
        Rt, pose_ms = timed_call(
            args,
            frontend.initialize_incremental,
            backend.scene_model,
            desc_kpts,
            frame_id,
            image,
            will_be_keyframe,
            is_test=is_test_frame,
        )
        if Rt is None:
            status = "keyframe_pose_failed" if will_be_keyframe else "pose_failed"
            pose_records[frame_id]["status"] = status
            if will_be_keyframe:
                pose_records[frame_id]["is_keyframe"] = True
                pose_records[frame_id]["ate_selected"] = True
        else:
            frontend_pose_estimates[frame_id] = clone_pose(Rt)
            add_pose_record(
                pose_records,
                frame_id,
                info,
                Rt,
                status="keyframe" if will_be_keyframe else "tracked",
                is_keyframe=will_be_keyframe,
            )
            if will_be_keyframe:
                mapping_summary = backend.register_incremental_keyframe(
                    frame_id, image, info, desc_kpts, Rt, focal
                )
                keyframe_timing.add_many(
                    {
                        "image_loading": load_ms,
                        "feature_extraction": feature_ms,
                        "keyframe_selection": selection_ms,
                        "pose_estimation": pose_ms,
                    }
                )
                keyframe_timing.add_many(mapping_summary.timings_ms)
                frontend.accept_keyframe(frame_id, desc_kpts)
                selected_keyframes += 1

        pbar.set_postfix_str(
            progress_postfix()
        )
        publish_frame()

    if viewer is not None:
        viewer.set_status("finalizing", "评测与保存")
    backend.scene_model.join_optimization_thread()

    # Capture all online updates, including updates to older keyframes.
    refresh_keyframe_pose_records(pose_records, backend)
    frontend_records = snapshot_pose_records(pose_records, frontend_pose_estimates)
    online_records = snapshot_pose_records(pose_records)
    final_mapping_summary = backend.finalize_mapping()
    keyframe_timing.add_many(final_mapping_summary.timings_ms)
    if processed_frames:
        # Freeze the last view before inference mode / saving can move anchors.
        publish_frame(force=True, status="finalizing")
    viewer_time = publisher.elapsed_seconds if publisher is not None else 0.0
    reconstruction_time = max(0.0, time.time() - start_time - viewer_time)
    trajectory_metrics = write_trajectory_stage_outputs(
        args.model_path,
        {
            "frontend_miniba": frontend_records,
            "online_mapping": online_records,
        },
    )

    if len(backend.scene_model.keyframes) > 0:
        backend.scene_model.enable_inference_mode()
        backend.scene_model.save(args.model_path, reconstruction_time, processed_frames)

    ordered_records = online_records
    frame_ate_rows, rmse_ate, poses_json = compute_trajectory_outputs(ordered_records)
    render_rows, render_aggregate = compute_render_metrics(
        backend.scene_model,
        dataset,
        ordered_records,
        lpips_max_size=args.eval_lpips_max_size,
        lpips_device=args.eval_lpips_device,
    )
    num_keyframes = len(backend.scene_model.keyframes)
    local_ba_events = getattr(getattr(backend, "local_ba", None), "events", [])
    avg_keyframe_stage_ms = keyframe_timing.averages(num_keyframes)
    metrics = {
        **ablation_config,
        "viewer_mode": getattr(args, "viewer_mode", "none"),
        "viewer_overhead_seconds": viewer_time,
        "PSNR": render_aggregate.get("PSNR"),
        "SSIM": render_aggregate.get("SSIM"),
        "LPIPS": render_aggregate.get("LPIPS"),
        "time": reconstruction_time,
        "FPS": processed_frames / reconstruction_time if reconstruction_time > 0 else None,
        "RMSE ATE [m]": rmse_ate,
        "RMSE ATE frontend mini-BA [m]": trajectory_metrics["frontend_miniba"]["RMSE ATE [m]"],
        "RMSE ATE online mapping [m]": trajectory_metrics["online_mapping"]["RMSE ATE [m]"],
        "num_frames": processed_frames,
        "num_registered_frames": sum(1 for record in ordered_records if record.get("estimated_Rt") is not None),
        "num_ate_selected_frames": sum(1 for record in ordered_records if record.get("ate_selected", False)),
        "num_ate_registered_frames": sum(
            1 for record in ordered_records
            if record.get("ate_selected", False) and record.get("estimated_Rt") is not None
        ),
        "num_render_selected_frames": sum(1 for record in ordered_records if record.get("render_selected", False)),
        "num_render_registered_frames": sum(
            1 for record in ordered_records
            if record.get("render_selected", False) and record.get("estimated_Rt") is not None
        ),
        "num_keyframes": num_keyframes,
        "num_local_ba_attempts": len(local_ba_events),
        "num_local_ba_accepted": sum(event["accepted"] for event in local_ba_events),
        "num_gaussians": backend.scene_model.n_active_gaussians,
        "avg_keyframe_total_ms": sum(avg_keyframe_stage_ms.values()) if avg_keyframe_stage_ms else None,
        "avg_keyframe_stage_ms": avg_keyframe_stage_ms,
    }
    write_outputs(args.model_path, metrics, frame_ate_rows, render_rows, poses_json)
    with open(os.path.join(args.model_path, "slam_local_ba.json"), "w", encoding="utf-8") as f:
        json.dump(local_ba_events, f, indent=2, allow_nan=False)

    print("Wrote TEST2-SLAM outputs to:", args.model_path)
    printable_metrics = {
        key: value for key, value in metrics.items() if key != "avg_keyframe_stage_ms"
    }
    print(
        ", ".join(
            f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in printable_metrics.items()
        )
    )
    if avg_keyframe_stage_ms:
        print(
            "Average keyframe stage time [ms]: "
            + ", ".join(
                f"{stage}: {avg_keyframe_stage_ms[stage]:.3f}"
                for stage in KEYFRAME_TIMING_STAGES
            )
        )


def main():
    args = get_args()
    mode = getattr(args, "viewer_mode", "none")
    if mode == "none":
        return run_slam(args)

    from visualization.server import LiveViewer
    viewer = LiveViewer(args.ip, args.port, open_browser=(mode == "local")).start()
    print(f"Live SLAM viewer: {viewer.url}", flush=True)
    if args.ip in ("0.0.0.0", "::"):
        print(f"Remote access: http://<server-address>:{viewer.port}", flush=True)
    else:
        print(f"Remote server: forward local port {viewer.port} to 127.0.0.1:{viewer.port} over SSH.", flush=True)
    finished = False
    try:
        run_slam(args, viewer)
        finished = True
        viewer.set_status("complete", f"Results saved to {args.model_path}")
        if args.viewer_keep_alive:
            print("SLAM finished. Viewer remains available; press Ctrl+C to close.", flush=True)
            viewer.run_until_closed()
    except KeyboardInterrupt:
        viewer.set_status("stopped", "Stopped from the terminal.")
        if not finished:
            raise
    except Exception as exc:
        viewer.set_status("error", str(exc))
        raise
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
