from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

import torch

from poses.local_ba import CovisibilityLocalBA


WINDOW_MAPPING_SAMPLE_ITERATIONS = 25
WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS = 25


def validate_local_ba_scene_interface(scene_class):
    """Catch a mixed checkout before allocating depth models or starting SLAM."""
    required = (
        "apply_local_ba_result", "_fit_gaussian_owner_keyframe_ids",
        "_rotate_gaussian_quaternions", "_apply_gaussian_add_and_prune",
    )
    missing = [name for name in required if not callable(getattr(scene_class, name, None))]
    add_and_prune = getattr(scene_class, "_apply_gaussian_add_and_prune", None)
    if callable(add_and_prune) and "extension_owner_ids" not in inspect.signature(add_and_prune).parameters:
        missing.append("_apply_gaussian_add_and_prune(..., extension_owner_ids=...)")
    if missing:
        try:
            source = inspect.getsourcefile(scene_class) or inspect.getfile(scene_class)
        except (TypeError, OSError):
            source = scene_class.__module__ + "." + scene_class.__name__
        raise RuntimeError(
            "Local BA deployment is incomplete: the loaded SceneModel is missing "
            + ", ".join(missing) + ". Loaded class source: " + str(source)
            + ". Synchronize scene/scene_model.py together with poses/local_ba.py, "
            "poses/local_ba_solver.py and slam_backend.py, then restart Python."
        )


@dataclass
class MappingSummary:
    phases: list[dict]
    timings_ms: dict[str, float] | None = None
    keyframe_ids: list[int] | None = None


class SLAMBackend:
    """Mapping over registered keyframes and Gaussian sampling."""
    def __init__(self, width: int, height: int, args, matcher, triangulator):
        from scene.dense_extractor import DenseExtractor
        from scene.keyframe import Keyframe
        from scene.mono_depth import MonoDepthEstimator
        from scene.scene_model import SceneModel

        if not getattr(args, "local_ba_disable", False):
            validate_local_ba_scene_interface(SceneModel)
        self.width = width
        self.height = height
        self.args = args
        self.Keyframe = Keyframe
        self.triangulator = triangulator
        dense_extractor_params = inspect.signature(DenseExtractor).parameters
        if "args" in dense_extractor_params or len(dense_extractor_params) >= 3:
            self.dense_extractor = DenseExtractor(width, height, args)
        else:
            self.dense_extractor = DenseExtractor(width, height)
        self.depth_estimator = MonoDepthEstimator(width, height, args)
        self.scene_model = SceneModel(width, height, args, matcher)
        self.keyframe_to_frame_id: dict[int, int] = {}
        self.local_ba = CovisibilityLocalBA(args)

    def _all_keyframe_ids(self) -> list[int]:
        return list(range(len(self.scene_model.keyframes)))

    def _sync_cuda(self):
        if getattr(self.args, "device", "cuda").startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _timed(self, func, *args, **kwargs):
        self._sync_cuda()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        self._sync_cuda()
        return result, (time.perf_counter() - start) * 1000.0

    def _keyframe_info(self, info: dict):
        copied = dict(info)
        if "mask" in copied and torch.is_tensor(copied["mask"]):
            copied["mask"] = copied["mask"].clone()
        return copied

    def _make_focal_tensor(self, focal, device=None):
        device = device or getattr(self.args, "device", "cuda")
        focal = torch.as_tensor(focal, dtype=torch.float32, device=device).reshape(-1)
        if focal.numel() == 1:
            focal = focal.repeat(2)
        if focal.numel() != 2:
            raise ValueError("Expected focal length as a scalar or [fx, fy]")
        return focal

    def add_keyframe(self, frame_id: int, image, info, desc_kpts, Rt, focal):
        keyframe_id = len(self.scene_model.keyframes)
        keyframe = self.Keyframe(
            image,
            self._keyframe_info(info),
            desc_kpts,
            Rt,
            keyframe_id,
            self._make_focal_tensor(focal, device=Rt.device),
            self.dense_extractor,
            self.depth_estimator,
            self.triangulator,
            self.args,
        )
        self.scene_model.add_keyframe(keyframe, keyframe.f)
        self.keyframe_to_frame_id[keyframe_id] = frame_id
        return keyframe

    def register_bootstrap(self, frame_packets: list[dict], Rts, focal):
        timings_ms = {
            "keyframe_registration": 0.0,
            "gaussian_sampling": 0.0,
            "mapping_optimization": 0.0,
        }
        keyframe_ids = []
        for packet, Rt in zip(frame_packets, Rts):
            keyframe, elapsed_ms = self._timed(
                self.add_keyframe,
                packet["frame_id"],
                packet["image"],
                packet["info"],
                packet["desc_kpts"],
                Rt,
                focal,
            )
            timings_ms["keyframe_registration"] += elapsed_ms
            keyframe.info["source_frame_id"] = packet["frame_id"]
            keyframe_ids.append(keyframe.index)

        for keyframe_id in keyframe_ids:
            _, elapsed_ms = self._timed(self.scene_model.add_new_gaussians, keyframe_id)
            timings_ms["gaussian_sampling"] += elapsed_ms
        phases, elapsed_ms = self._timed(self.run_bootstrap_mapping_schedule)
        timings_ms["mapping_optimization"] += elapsed_ms
        return MappingSummary(phases=phases, timings_ms=timings_ms, keyframe_ids=keyframe_ids)

    def register_incremental_keyframe(self, frame_id: int, image, info, desc_kpts, Rt, focal):
        timings_ms = {
            "keyframe_registration": 0.0,
            "gaussian_sampling": 0.0,
            "mapping_optimization": 0.0,
        }
        keyframe, elapsed_ms = self._timed(self.add_keyframe, frame_id, image, info, desc_kpts, Rt, focal)
        timings_ms["keyframe_registration"] += elapsed_ms
        keyframe.info["source_frame_id"] = frame_id
        ba_phase, elapsed_ms = self._timed(self.run_local_ba, keyframe.index)
        timings_ms["local_ba"] = elapsed_ms
        _, elapsed_ms = self._timed(self.scene_model.add_new_gaussians, keyframe.index)
        timings_ms["gaussian_sampling"] += elapsed_ms
        phases, elapsed_ms = self._timed(self.run_mapping_schedule)
        phases = ([ba_phase] if ba_phase is not None else []) + list(phases)
        timings_ms["mapping_optimization"] += elapsed_ms
        return MappingSummary(phases=phases, timings_ms=timings_ms, keyframe_ids=[keyframe.index])

    def run_local_ba(self, newest_id):
        if getattr(self.args, "local_ba_disable", False):
            return None
        if not hasattr(self, "local_ba"):
            self.local_ba = CovisibilityLocalBA(self.args)
        return self.local_ba.run(self.scene_model, newest_id)


    def run_bootstrap_mapping_schedule(self):
        keyframe_ids = self._all_keyframe_ids()
        if len(keyframe_ids) == 0:
            return []
        return self.scene_model.optimize_slam_window(
            keyframe_ids,
            [("bootstrap_mapping", getattr(self.args, "slam_first_full_iterations", 80), True, True)],
            keyframe_sampling_probabilities=getattr(
                self.args,
                "slam_first_full_sampling_probs",
                [0.22, 0.19, 0.17, 0.14, 0.11, 0.08, 0.06, 0.03],
            ),
            optimize_unsampled_keyframes_once=True,
        )


    def run_mapping_schedule(self):
        keyframe_ids = self._all_keyframe_ids()
        if len(keyframe_ids) == 0:
            return []

        return self.scene_model.optimize_slam_window(
            keyframe_ids,
            [
                ("mapping", WINDOW_MAPPING_SAMPLE_ITERATIONS, False, True),
                ("joint_mapping", WINDOW_JOINT_MAPPING_SAMPLE_ITERATIONS, True, True),
            ],
            latest_keyframe_sample_proba=getattr(
                self.args, "slam_latest_keyframe_sample_proba", 0.25
            ),
        )

    def run_final_mapping_schedule(self):
        return []

    def finalize_mapping(self):
        """Compatibility hook; completed online mapping is the final result."""
        return MappingSummary(phases=[], timings_ms={})
