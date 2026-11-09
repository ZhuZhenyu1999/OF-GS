from dataclasses import dataclass
import inspect
from typing import Optional

import torch
from utils import focal_pair, intrinsics_from_info


@dataclass
class KeyframeDecision:
    selected: bool
    reason: str
    median_displacement_px: float
    match_count: int
    threshold_px: float


class KeyframeSelector:
    """Implements the requested keyframe policy independent of CUDA modules."""
    def __init__(
        self,
        image_width: int,
        min_displacement_ratio: float = 0.02,
        min_num_inliers: int = 100,
    ):
        self.image_width = image_width
        self.min_displacement_ratio = min_displacement_ratio
        self.min_num_inliers = min_num_inliers
        self.last_keyframe_frame_id: Optional[int] = None

    @property
    def threshold_px(self):
        return self.min_displacement_ratio * self.image_width

    def mark_selected(self, frame_id: int):
        self.last_keyframe_frame_id = frame_id

    def decide(
        self,
        frame_id: int,
        median_displacement_px: float,
        match_count: int,
    ):
        if self.last_keyframe_frame_id is None:
            return KeyframeDecision(True, "first", median_displacement_px, match_count, self.threshold_px)

        motion_ok = median_displacement_px > self.threshold_px
        matches_ok = match_count >= self.min_num_inliers
        if motion_ok and matches_ok:
            return KeyframeDecision(True, "motion", median_displacement_px, match_count, self.threshold_px)
        if not matches_ok:
            return KeyframeDecision(False, "few_matches", median_displacement_px, match_count, self.threshold_px)
        return KeyframeDecision(False, "small_motion", median_displacement_px, match_count, self.threshold_px)


@dataclass
class TrackingResult:
    frame_id: int
    desc_kpts: object
    decision: KeyframeDecision
    pose_w2c: Optional[torch.Tensor] = None
    pose_success: bool = False


class SLAMFrontend:
    """
    CLIDD tracking and mini-BA pose initialization wrapper.

    Heavy CUDA-backed modules are imported only when this class is instantiated
    so unit tests can exercise keyframe logic on CPU-only machines.
    """
    def __init__(self, width: int, height: int, args):
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("slam.py requires CUDA for CLIDD, RANSAC, mini-BA, and Gaussian mapping.")

        from poses.feature_detector import Detector
        from poses.matcher import Matcher
        from poses.pose_initializer import PoseInitializer
        from poses.triangulator import Triangulator

        self.width = width
        self.height = height
        self.args = args
        self.max_error = max(args.match_max_error * width, 1.5)
        self.matcher = Matcher(args.fundmat_samples, self.max_error)
        self.matcher.match_beta = float(getattr(args, "clidd_match_beta", 20.0))
        self.triangulator = Triangulator(
            args.num_kpts, args.num_prev_keyframes_miniba_incr, self.max_error
        )
        self.pose_initializer = PoseInitializer(
            width, height, self.triangulator, self.matcher, 2 * self.max_error, args
        )
        detector_params = inspect.signature(Detector).parameters
        if "args" in detector_params or len(detector_params) >= 4:
            self.detector = Detector(args.num_kpts, width, height, args)
        else:
            self.detector = Detector(args.num_kpts, width, height)
        self.selector = KeyframeSelector(
            width,
            min_displacement_ratio=args.min_displacement,
            min_num_inliers=args.min_num_inliers,
        )
        self.last_keyframe_desc = None

    @property
    def focal_init(self):
        return self.pose_initializer.f_init

    @property
    def focal(self):
        return self.pose_initializer.f

    def set_focal(self, focal):
        # Known calibration takes priority over a network/focal estimate.
        if self.pose_initializer.calibrated:
            return
        f = focal_pair(focal, device=self.pose_initializer.centre.device, dtype=torch.float32).reshape(2)
        K = self.pose_initializer.K.clone()
        K[0, 0], K[1, 1] = f[0], f[1]
        self.pose_initializer.set_intrinsics(K)

    def set_frame_info(self, info):
        if "K" not in info and "K_np" not in info:
            return
        K = intrinsics_from_info(
            info, self.pose_initializer.f, self.width, self.height,
            device=self.pose_initializer.centre.device,
        )
        self.pose_initializer.set_intrinsics(K)
        self.pose_initializer.calibrated = bool(info.get("intrinsics_calibrated", True))

    def extract(self, image):
        return self.detector(image)

    @torch.no_grad()
    def decide_keyframe(self, frame_id: int, desc_kpts):
        if self.last_keyframe_desc is None:
            decision = self.selector.decide(frame_id, float("inf"), 0)
            return decision
        matches = self.matcher(desc_kpts, self.last_keyframe_desc)
        if len(matches.kpts) == 0:
            median_displacement = 0.0
        else:
            dist = torch.linalg.vector_norm(matches.kpts - matches.kpts_other, dim=-1)
            median_displacement = float(dist.median().item())
        return self.selector.decide(
            frame_id,
            median_displacement,
            len(matches.kpts),
        )

    def accept_keyframe(self, frame_id: int, desc_kpts):
        self.selector.mark_selected(frame_id)
        self.last_keyframe_desc = desc_kpts

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list, infos=None):
        return self.pose_initializer.initialize_bootstrap(desc_kpts_list, infos=infos)

    @torch.no_grad()
    def initialize_incremental(self, scene_model, desc_kpts, frame_id, image, will_be_keyframe, is_test=False):
        prev_keyframes = scene_model.keyframes[-self.args.num_prev_keyframes_miniba_incr :]
        for keyframe in prev_keyframes:
            keyframe.update_3dpts(scene_model.keyframes)
        planned_index = len(scene_model.keyframes) if will_be_keyframe else -1
        return self.pose_initializer.initialize_incremental(
            prev_keyframes,
            desc_kpts,
            planned_index,
            is_test,
            image,
            update_matches=will_be_keyframe,
        )
