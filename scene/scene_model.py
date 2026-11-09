#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import Namespace
import gc
import os
import json
import math
import threading
import time
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from fused_ssim import fused_ssim
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from simple_knn._C import distIndex2
from poses.feature_detector import DescribedKeypoints
from poses.matcher import Matcher
from poses.guided_mvs import GuidedMVS
from scene.optimizers import SparseGaussianAdam
from scene.keyframe import Keyframe
from scene.anchor import Anchor
from utils import (
    RGB2SH,
    build_rotation,
    combine_mapping_losses,
    depth2points,
    focal2fov,
    getProjectionMatrix,
    getProjectionMatrixFromIntrinsics,
    intrinsics_from_info,
    resize_intrinsics,
    get_sampling_response,
    inverse_sigmoid,
    align_poses,
    make_torch_sampler,
    pts2px,
    psnr,
    resize_longest_side,
    rotation_distance,
    sample_window_keyframe_id,
    valid_8_neighbor_mean,
)
from dataloaders.read_write_model import write_model


def sample_keyframe_id_with_explicit_probabilities(window_ids, probabilities, rng=None):
    ids = [int(window_id) for window_id in window_ids]
    if len(ids) == 0:
        raise ValueError("Cannot sample a keyframe from an empty window")

    if isinstance(probabilities, dict):
        weights = [float(probabilities.get(window_id, 0.0)) for window_id in ids]
    else:
        weights = [float(probability) for probability in probabilities]
        if len(weights) != len(ids):
            raise ValueError("Sampling probabilities must match the keyframe window size")

    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Sampling probabilities must be finite non-negative values")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("At least one sampling probability must be positive")

    rng = np.random if rng is None else rng
    roll = float(rng.random())
    cumulative = 0.0
    for window_id, weight in zip(ids, weights / total):
        cumulative += float(weight)
        if roll < cumulative:
            return window_id
    return ids[-1]


class SceneModel:
    """
    Scene Model class that contains the scene's Gaussians, anchors, keyframes, and methods for rendering and optimization.
    """
    def __init__(
        self,
        width: int,
        height: int,
        args: Namespace,
        matcher: Matcher = None,
        inference_mode: bool = False,
    ):
        """
        Args:
            width: Width of the image.
            height: Height of the image.
            args: Arguments for the scene model. Should always have anchor_overlap, and training parameters if inference_mode is False.
            matcher: Matcher for the scene model. Defaults to None (if inference_mode is True).
            inference_mode: Whether we load the scene for visualization. Defaults to False.
        """
        self.width = width
        self.height = height
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device="cuda")
        self.anchor_overlap = args.anchor_overlap
        self.optimization_thread = None
        self.eval_lpips_max_size = getattr(args, "eval_lpips_max_size", 512)
        self.eval_lpips_device = getattr(args, "eval_lpips_device", "cpu")
        self.lpips = None
        self.lpips_device = None
        self.lpips_unavailable = False

        if not inference_mode:
            self.num_prev_keyframes_check = args.num_prev_keyframes_check
            self.active_sh_degree = args.sh_degree
            self.max_sh_degree = args.sh_degree
            self.lambda_dssim = args.lambda_dssim
            self.geometry_loss_weight = getattr(args, "geometry_loss_weight", 1.0)
            self.init_sampling_method = getattr(args, "init_sampling_method", "scharr")
            self.init_proba_scaler = args.init_proba_scaler
            self.direct_gaussian_sampling = getattr(args, "direct_gaussian_sampling", True)
            self.pose_gaussian_gate = getattr(args, "pose_gaussian_gate", True)
            self.densification_interval = int(getattr(args, "densification_interval", 50))
            self.densify_grad_threshold = float(getattr(args, "densify_grad_threshold", 2e-4))
            self.percent_dense = float(getattr(args, "percent_dense", 0.01))
            self.gaussian_mapping_update_count = 0
            self.slam_neighbor_penalty_weight = getattr(args, "slam_neighbor_penalty_weight", 0.2)
            self.max_active_keyframes = args.max_active_keyframes
            self.n_kept_frames = 20
            self.use_last_frame_proba = args.use_last_frame_proba
            self.slam_latest_keyframe_sample_proba = getattr(
                args, "slam_latest_keyframe_sample_proba", 0.25
            )
            global_seed = int(getattr(args, "seed", 0))
            window_sampling_seed = getattr(args, "slam_window_sampling_seed", None)
            self.slam_window_sampling_seed = (
                global_seed if window_sampling_seed is None else int(window_sampling_seed)
            )
            self.slam_window_sampling_rng = np.random.RandomState(
                self.slam_window_sampling_seed
            )
            self.active_frames_cpu = []
            self.active_frames_gpu = []
            self.gaussian_optimization_counts = torch.empty(0, device="cuda", dtype=torch.int32)
            self.pose_gaussian_min_update_count = int(
                getattr(args, "pose_gaussian_min_update_count", 5)
            )
            self.pose_gaussian_min_observed_keyframes = int(
                getattr(args, "pose_gaussian_min_observed_keyframes", 3)
            )
            self.pose_gaussian_min_optimization_count = self.pose_gaussian_min_update_count
            observer_slots = max(1, self.pose_gaussian_min_observed_keyframes)
            self.gaussian_observed_keyframe_ids = torch.empty(
                0,
                observer_slots,
                device="cuda",
                dtype=torch.int32,
            )
            self.gaussian_observed_keyframe_counts = torch.empty(0, device="cuda", dtype=torch.int16)
            # Keep creation ownership separate from the capped set of observers.
            # Local BA moves each Gaussian with its owner camera's correction.
            self.gaussian_owner_keyframe_ids = torch.empty(0, device="cuda", dtype=torch.int32)
            self.guided_mvs = GuidedMVS(args)
            self.lr_dict = {
                "xyz": {
                    "lr_init": args.position_lr_init,
                    "lr_decay": args.position_lr_decay,
                }
            }

            ## Initialize Gaussian parameters
            self.gaussian_params = {
                "xyz": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.position_lr_init,
                },
                "f_dc": {
                    "val": torch.empty(0, 1, 3, device="cuda"),
                    "lr": args.feature_lr,
                },
                "f_rest": {
                    "val": torch.empty(
                        0,
                        (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
                        3,
                        device="cuda",
                    ),
                    "lr": args.feature_lr / 20.0,
                },
                "scaling": {
                    "val": torch.empty(0, 3, device="cuda"),
                    "lr": args.scaling_lr,
                },
                "rotation": {
                    "val": torch.empty(0, 4, device="cuda"),
                    "lr": args.rotation_lr,
                },
                "opacity": {
                    "val": torch.empty(0, 1, device="cuda"),
                    "lr": args.opacity_lr,
                },
            }
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors = [self.active_anchor]
            ## Initialize optimizer
            self.reset_optimizer()

        self.keyframes = []
        self.anchor_weights = [1.0]
        self.f = torch.full((2,), 0.7 * width, device=self.centre.device)
        self.init_intrinsics()

        self.approx_cam_centres = None
        self.gt_Rts = torch.empty(0, 4, 4, device="cuda")
        self.gt_Rts_mask = torch.empty(0, device="cuda", dtype=bool)
        self.gt_f = self.f
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, device="cuda", dtype=torch.bool)
        self.sorted_frame_indices = None
        self.last_trained_id = 0
        self.valid_keyframes = torch.empty(0, dtype=torch.bool)
        self.lock = threading.Lock()
        self.inference_mode = inference_mode

        ## Initialize helpers for Gaussian initialization
        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, width), torch.arange(0, height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )

    def reset_optimizer(self):
        for key in self.gaussian_params:
            if not self.gaussian_params[key]["val"].requires_grad:
                self.gaussian_params[key]["val"].requires_grad = True
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )
        # A different anchor may have the same number of Gaussians. Its gradient
        # history must not be mistaken for the previous anchor's history.
        self._reset_gaussian_densification_stats()

    def _reset_gaussian_densification_stats(self):
        self.gaussian_gradient_accum = self.xyz.new_zeros(self.xyz.shape[0])
        self.gaussian_gradient_counts = torch.zeros(
            self.xyz.shape[0], device=self.xyz.device, dtype=torch.int32
        )

    def _fit_gaussian_densification_stats(self):
        n = self.xyz.shape[0]
        accum = getattr(self, "gaussian_gradient_accum", None)
        counts = getattr(self, "gaussian_gradient_counts", None)
        if accum is None or counts is None or accum.shape != (n,) or counts.shape != (n,):
            self._reset_gaussian_densification_stats()
        else:
            self.gaussian_gradient_accum = accum.to(device=self.xyz.device, dtype=self.xyz.dtype)
            self.gaussian_gradient_counts = counts.to(device=self.xyz.device, dtype=torch.int32)
        return self.gaussian_gradient_accum, self.gaussian_gradient_counts

    @torch.no_grad()
    def _accumulate_gaussian_densification_stats(self, render_pkg):
        accum, counts = self._fit_gaussian_densification_stats()
        points = render_pkg.get("screenspace_points")
        gradient = None if points is None else points.grad
        visibility = render_pkg.get("visibility_filter")
        if gradient is None or visibility is None:
            return
        if (
            gradient.ndim != 2
            or gradient.shape[0] != accum.shape[0]
            or gradient.shape[1] < 2
            or visibility.shape != accum.shape
        ):
            return
        gradient_norm = torch.linalg.vector_norm(gradient[:, :2], dim=-1)
        valid = visibility.to(device=accum.device, dtype=torch.bool) & torch.isfinite(gradient_norm)
        accum[valid] += gradient_norm[valid]
        counts[valid] += 1

    def _gradient_densification_extent(self):
        centres = getattr(self, "approx_cam_centres", None)
        if centres is None or centres.numel() == 0:
            return self.xyz.new_tensor(1e-6)
        centres = centres.to(device=self.xyz.device, dtype=self.xyz.dtype)
        centres = centres[torch.isfinite(centres).all(dim=-1)]
        if centres.shape[0] == 0:
            return self.xyz.new_tensor(1e-6)
        radius = torch.linalg.vector_norm(centres - centres.mean(dim=0), dim=-1).max()
        return (1.1 * radius).clamp_min(1e-6)

    @torch.no_grad()
    def densify_gaussians_from_gradients(self):
        """Clone small high-gradient Gaussians and split large ones into two."""
        accum, counts = self._fit_gaussian_densification_stats()
        average_gradient = accum / counts.clamp_min(1)
        scales = self.scaling
        max_scale = scales.max(dim=-1).values
        selected = (
            (counts > 0)
            & torch.isfinite(average_gradient)
            & (average_gradient >= getattr(self, "densify_grad_threshold", 2e-4))
            & torch.isfinite(scales).all(dim=-1)
        )
        size_threshold = getattr(self, "percent_dense", 0.01) * self._gradient_densification_extent()
        clone_mask = selected & (max_scale <= size_threshold)
        split_mask = selected & (max_scale > size_threshold)
        clone_ids = torch.where(clone_mask)[0]
        split_ids = torch.where(split_mask)[0]
        child_ids = torch.cat([clone_ids, split_ids.repeat_interleave(2)])
        result = {
            "cloned": int(clone_ids.numel()),
            "split": int(split_ids.numel()),
            "added": int(child_ids.numel()),
        }
        if child_ids.numel() > 0:
            extension_tensors = {
                name: param["val"].detach()[child_ids].clone()
                for name, param in self.gaussian_params.items()
            }
            split_start = clone_ids.numel()
            if split_ids.numel() > 0:
                split_children = child_ids[split_start:]
                offsets = torch.randn_like(scales[split_children]) * scales[split_children]
                rotations = build_rotation(self.gaussian_params["rotation"]["val"][split_children])
                extension_tensors["xyz"][split_start:] += torch.bmm(
                    rotations, offsets.unsqueeze(-1)
                ).squeeze(-1)
                extension_tensors["scaling"][split_start:] -= math.log(1.6)
            owners = self._fit_gaussian_owner_keyframe_ids()[child_ids]
            with self.lock:
                # Children start their own optimization/observation history but
                # stay attached to the source camera for local BA corrections.
                self._apply_gaussian_add_and_prune(
                    extension_tensors,
                    ~split_mask,
                    extension_owner_ids=owners,
                )
        self._reset_gaussian_densification_stats()
        return result

    def _maybe_densify_gaussians(self, render_pkg):
        if getattr(self, "direct_gaussian_sampling", True) or getattr(self, "inference_mode", False):
            return None
        visibility = render_pkg.get("visibility_filter")
        if visibility is None or not bool(visibility.any()):
            return None
        self._accumulate_gaussian_densification_stats(render_pkg)
        self.gaussian_mapping_update_count = getattr(self, "gaussian_mapping_update_count", 0) + 1
        interval = int(getattr(self, "densification_interval", 50))
        if self.gaussian_mapping_update_count % interval == 0:
            return self.densify_gaussians_from_gradients()
        return None

    def _fit_gaussian_optimization_counts(self):
        n = self.xyz.shape[0]
        counts = getattr(self, "gaussian_optimization_counts", None)
        if counts is None:
            counts = torch.empty(0, device=self.xyz.device, dtype=torch.int32)
        counts = counts.to(device=self.xyz.device, dtype=torch.int32)
        if counts.shape[0] != n:
            fitted = torch.zeros(n, device=self.xyz.device, dtype=torch.int32)
            keep = min(n, counts.shape[0])
            if keep > 0:
                fitted[:keep] = counts[:keep]
            counts = fitted
            self.gaussian_optimization_counts = counts
        return counts

    def _gaussian_observer_slot_count(self):
        threshold = int(getattr(self, "pose_gaussian_min_observed_keyframes", 3))
        return max(1, threshold)

    def _fit_gaussian_owner_keyframe_ids(self):
        n = self.xyz.shape[0]
        owners = getattr(self, "gaussian_owner_keyframe_ids", None)
        if owners is None:
            # Older in-memory scenes store the creation frame in observer slot 0.
            owners = self._fit_gaussian_observed_keyframe_ids()[:, 0].clone()
        owners = owners.to(device=self.xyz.device, dtype=torch.int32)
        if owners.shape != (n,):
            fitted = torch.full((n,), -1, device=self.xyz.device, dtype=torch.int32)
            keep = min(n, owners.numel())
            fitted[:keep] = owners.reshape(-1)[:keep]
            owners = fitted
        self.gaussian_owner_keyframe_ids = owners
        return owners

    def _fit_gaussian_observed_keyframe_ids(self):
        n = self.xyz.shape[0]
        slot_count = self._gaussian_observer_slot_count()
        observed = getattr(self, "gaussian_observed_keyframe_ids", None)
        if observed is None:
            observed = torch.empty(0, slot_count, device=self.xyz.device, dtype=torch.int32)
        observed = observed.to(device=self.xyz.device, dtype=torch.int32)
        if observed.ndim == 1:
            observed = observed[:, None]
        if observed.shape[0] != n or observed.shape[1] != slot_count:
            fitted = torch.full(
                (n, slot_count),
                -1,
                device=self.xyz.device,
                dtype=torch.int32,
            )
            keep_rows = min(n, observed.shape[0])
            keep_cols = min(slot_count, observed.shape[1]) if observed.ndim == 2 else 0
            if keep_rows > 0 and keep_cols > 0:
                fitted[:keep_rows, :keep_cols] = observed[:keep_rows, :keep_cols]
            observed = fitted
            self.gaussian_observed_keyframe_ids = observed
        return observed

    def _fit_gaussian_observed_keyframe_counts(self):
        n = self.xyz.shape[0]
        counts = getattr(self, "gaussian_observed_keyframe_counts", None)
        if counts is None:
            observed = self._fit_gaussian_observed_keyframe_ids()
            counts = (observed >= 0).sum(dim=1).to(dtype=torch.int16)
        counts = counts.to(device=self.xyz.device, dtype=torch.int16)
        if counts.shape[0] != n:
            fitted = torch.zeros(n, device=self.xyz.device, dtype=torch.int16)
            keep = min(n, counts.shape[0])
            if keep > 0:
                fitted[:keep] = counts[:keep]
            counts = fitted
            self.gaussian_observed_keyframe_counts = counts
        return counts

    def _empty_gaussian_observer_ids(self, count, device=None):
        device = self.xyz.device if device is None else device
        return torch.full(
            (int(count), self._gaussian_observer_slot_count()),
            -1,
            device=device,
            dtype=torch.int32,
        )

    def _observer_ids_for_keyframe(self, count, keyframe_id, device=None):
        observed = self._empty_gaussian_observer_ids(count, device=device)
        if int(count) > 0 and keyframe_id is not None:
            observed[:, 0] = int(keyframe_id)
        return observed

    def _normalize_gaussian_observer_ids(self, observer_ids, count, device=None):
        device = self.xyz.device if device is None else device
        slot_count = self._gaussian_observer_slot_count()
        if observer_ids is None:
            return self._empty_gaussian_observer_ids(count, device=device)
        observer_ids = observer_ids.to(device=device, dtype=torch.int32)
        if observer_ids.ndim == 1:
            observer_ids = observer_ids[:, None]
        normalized = torch.full(
            (int(count), slot_count),
            -1,
            device=device,
            dtype=torch.int32,
        )
        keep_rows = min(int(count), observer_ids.shape[0])
        keep_cols = min(slot_count, observer_ids.shape[1])
        if keep_rows > 0 and keep_cols > 0:
            normalized[:keep_rows, :keep_cols] = observer_ids[:keep_rows, :keep_cols]
        return normalized

    @torch.no_grad()
    def _apply_gaussian_add_and_prune(
        self,
        extension_tensors,
        valid_mask,
        extension_counts=None,
        extension_observer_ids=None,
        extension_owner_ids=None,
    ):
        old_counts = self._fit_gaussian_optimization_counts()
        old_observed = self._fit_gaussian_observed_keyframe_ids()
        old_observed_counts = self._fit_gaussian_observed_keyframe_counts()
        old_owners = self._fit_gaussian_owner_keyframe_ids()
        old_gradient_accum, old_gradient_counts = self._fit_gaussian_densification_stats()
        valid_mask = valid_mask.to(device=old_counts.device, dtype=torch.bool)
        extension_count = next(iter(extension_tensors.values())).shape[0]
        if extension_counts is None:
            extension_counts = torch.zeros(extension_count, device=old_counts.device, dtype=torch.int32)
        else:
            extension_counts = extension_counts.to(device=old_counts.device, dtype=torch.int32).contiguous()
        extension_observer_ids = self._normalize_gaussian_observer_ids(
            extension_observer_ids,
            extension_count,
            device=old_counts.device,
        )
        extension_observed_counts = (extension_observer_ids >= 0).sum(dim=1).to(dtype=torch.int16)
        if extension_owner_ids is None:
            extension_owner_ids = extension_observer_ids[:, 0].clone()
        else:
            extension_owner_ids = extension_owner_ids.to(device=old_counts.device, dtype=torch.int32)
            if extension_owner_ids.shape != (extension_count,):
                raise ValueError("Gaussian owners must match the number of added Gaussians")
        self.optimizer.add_and_prune(extension_tensors, valid_mask)
        self.gaussian_optimization_counts = torch.cat(
            [old_counts[valid_mask], extension_counts],
            dim=0,
        ).contiguous()
        self.gaussian_observed_keyframe_ids = torch.cat(
            [old_observed[valid_mask], extension_observer_ids],
            dim=0,
        ).contiguous()
        self.gaussian_observed_keyframe_counts = torch.cat(
            [old_observed_counts[valid_mask], extension_observed_counts],
            dim=0,
        ).contiguous()
        self.gaussian_owner_keyframe_ids = torch.cat(
            [old_owners[valid_mask], extension_owner_ids], dim=0
        ).contiguous()
        self.gaussian_gradient_accum = torch.cat(
            [old_gradient_accum[valid_mask], old_gradient_accum.new_zeros(extension_count)]
        ).contiguous()
        self.gaussian_gradient_counts = torch.cat(
            [old_gradient_counts[valid_mask], old_gradient_counts.new_zeros(extension_count)]
        ).contiguous()

    @torch.no_grad()
    def _increment_gaussian_optimization_counts(self, visibility):
        counts = self._fit_gaussian_optimization_counts()
        if counts.numel() == 0:
            return
        visibility = visibility.to(device=counts.device, dtype=torch.bool)
        if visibility.shape[0] != counts.shape[0]:
            fitted = torch.zeros_like(counts, dtype=torch.bool)
            keep = min(fitted.shape[0], visibility.shape[0])
            if keep > 0:
                fitted[:keep] = visibility[:keep]
            visibility = fitted
        counts[visibility] += 1

    @torch.no_grad()
    def _mark_gaussians_observed_by_keyframe(self, visibility, keyframe_id):
        observed = self._fit_gaussian_observed_keyframe_ids()
        observed_counts = self._fit_gaussian_observed_keyframe_counts()
        if observed.numel() == 0 or keyframe_id is None:
            return
        visibility = visibility.to(device=observed.device, dtype=torch.bool)
        if visibility.shape[0] != observed.shape[0]:
            fitted = torch.zeros(observed.shape[0], device=observed.device, dtype=torch.bool)
            keep = min(fitted.shape[0], visibility.shape[0])
            if keep > 0:
                fitted[:keep] = visibility[:keep]
            visibility = fitted
        visible_rows = torch.where(visibility)[0]
        if visible_rows.numel() == 0:
            return
        keyframe_id = int(keyframe_id)
        rows = observed[visible_rows]
        already_seen = (rows == keyframe_id).any(dim=1)
        has_empty_slot = (rows < 0).any(dim=1)
        rows_to_update = visible_rows[~already_seen & has_empty_slot]
        if rows_to_update.numel() == 0:
            return
        empty_cols = observed[rows_to_update] < 0
        first_empty = empty_cols.float().argmax(dim=1).long()
        observed[rows_to_update, first_empty] = keyframe_id
        observed_counts[rows_to_update] = torch.clamp(
            observed_counts[rows_to_update] + 1,
            max=self._gaussian_observer_slot_count(),
        )

    def _merge_gaussian_observer_ids(self, observer_rows):
        slot_count = self._gaussian_observer_slot_count()
        observer_rows = observer_rows.to(device=self.xyz.device, dtype=torch.int32)
        if observer_rows.ndim == 2:
            observer_rows = observer_rows[:, None, :]
        merged = torch.full(
            (observer_rows.shape[0], slot_count),
            -1,
            device=observer_rows.device,
            dtype=torch.int32,
        )
        for row_idx in range(observer_rows.shape[0]):
            write_col = 0
            for value in observer_rows[row_idx].reshape(-1).tolist():
                value = int(value)
                if value < 0:
                    continue
                if value in merged[row_idx, :write_col].tolist():
                    continue
                merged[row_idx, write_col] = value
                write_col += 1
                if write_col >= slot_count:
                    break
        return merged

    def pose_optimization_gaussian_mask(self):
        counts = self._fit_gaussian_optimization_counts()
        observed_counts = self._fit_gaussian_observed_keyframe_counts()
        if not getattr(self, "pose_gaussian_gate", True):
            return torch.ones_like(counts, dtype=torch.bool)
        min_update_count = int(
            getattr(
                self,
                "pose_gaussian_min_update_count",
                getattr(self, "pose_gaussian_min_optimization_count", 5),
            )
        )
        min_observed = int(getattr(self, "pose_gaussian_min_observed_keyframes", 3))
        return (counts >= min_update_count) & (observed_counts >= min_observed)

    def release_training_state(self):
        self.join_optimization_thread()
        if hasattr(self, "optimizer"):
            self.optimizer = None

        for param in getattr(self, "gaussian_params", {}).values():
            value = param.get("val")
            if torch.is_tensor(value):
                value.grad = None
                value.requires_grad_(False)
            param.pop("exp_avg", None)
            param.pop("exp_avg_sq", None)

        for keyframe in getattr(self, "keyframes", []):
            if hasattr(keyframe, "optimizer"):
                keyframe.optimizer = None
            for attr_name in ("rW2C", "tW2C", "exposure", "depth_scale", "depth_offset"):
                value = getattr(keyframe, attr_name, None)
                if torch.is_tensor(value):
                    value.grad = None

        gc.collect()
        torch.cuda.empty_cache()

    def _clear_lpips_model(self):
        self.lpips = None
        self.lpips_device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_lpips_model(self, device=None):
        if self.lpips_unavailable:
            return None

        device = device or self.eval_lpips_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        if self.lpips is not None and self.lpips_device == device:
            return self.lpips

        self._clear_lpips_model()
        try:
            import lpips
            import sys

            original_stdout = sys.stdout
            with open(os.devnull, "w") as devnull, warnings.catch_warnings():
                sys.stdout = devnull
                warnings.simplefilter("ignore")
                model = lpips.LPIPS(net="vgg").to(device).eval()
            sys.stdout = original_stdout
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device == "cuda":
                self._clear_lpips_model()
                return self._get_lpips_model(device="cpu")
            self.lpips_unavailable = True
            return None
        except Exception:
            self.lpips_unavailable = True
            return None
        finally:
            if "original_stdout" in locals():
                sys.stdout = original_stdout

        self.lpips = model
        self.lpips_device = device
        return self.lpips

    def _lpips_value(self, image, gt_image):
        model = self._get_lpips_model()
        if model is None:
            return None

        image_for_lpips = resize_longest_side(
            image.clamp(0, 1),
            self.eval_lpips_max_size,
        )
        gt_for_lpips = resize_longest_side(
            gt_image.clamp(0, 1),
            self.eval_lpips_max_size,
        )
        try:
            return model(
                image_for_lpips.to(self.lpips_device)[None],
                gt_for_lpips.to(self.lpips_device)[None],
            ).item()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.lpips_device == "cuda":
                self._clear_lpips_model()
                model = self._get_lpips_model(device="cpu")
                if model is None:
                    return None
                return model(image_for_lpips.cpu()[None], gt_for_lpips.cpu()[None]).item()
            raise

    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    @classmethod
    def from_scene(cls, scene_dir: str, args):
        with open(os.path.join(scene_dir, "metadata.json")) as f:
            metadata = json.load(f)

        width = metadata["config"]["width"]
        height = metadata["config"]["height"]
        scene_model = cls(width, height, args, inference_mode=True)
        scene_model.active_sh_degree = metadata["config"]["sh_degree"]
        scene_model.max_sh_degree = metadata["config"]["sh_degree"]

        # Load anchors
        scene_model.anchors = []
        for i in range(len(metadata["anchors"])):
            scene_model.anchors.append(
                Anchor.from_ply(
                    os.path.join(scene_dir, "point_clouds", f"anchor_{i}.ply"),
                    torch.tensor(metadata["anchors"][i]["position"]),
                    metadata["config"]["sh_degree"],
                )
            )

        scene_model.active_anchor = scene_model.anchors[0]

        # Load keyframes
        for i in range(len(metadata["keyframes"])):
            keyframe = Keyframe.from_json(
                metadata["keyframes"][i], i, height=height, width=width,
                fallback_f=metadata["config"].get("f"),
            )
            scene_model.add_keyframe(keyframe)

        return scene_model

    @property
    def first_active_frame(self):
        return self.active_anchor.keyframe_ids[0]

    @property
    def last_active_frame(self):
        return self.active_anchor.keyframe_ids[-1]

    @property
    def n_active_keyframes(self):
        return self.last_active_frame - self.first_active_frame + 1

    def compute_geometric_loss(
        self,
        keyframe: Keyframe,
        rendered_invdepth,
        lvl: int = 0,
        detach_pose: bool = False,
    ):
        """Weighted reprojection and rendered-depth consistency with sparse landmarks."""
        if keyframe.desc_kpts is None or rendered_invdepth is None:
            return self.xyz.new_zeros(())

        desc_kpts = keyframe.desc_kpts
        if desc_kpts.kpts.device.type != "cuda":
            desc_kpts.to("cuda")

        valid_mask = desc_kpts.has_pt3d & (desc_kpts.pts_conf > 0)
        if not valid_mask.any():
            return self.xyz.new_zeros(())

        world_points = desc_kpts.pts3d[valid_mask].detach()
        target_uv = desc_kpts.kpts[valid_mask].detach()
        weights = desc_kpts.pts_conf[valid_mask].detach().float().clamp_min(0)

        R = keyframe.get_R()
        t = keyframe.get_t()
        if detach_pose:
            R = R.detach()
            t = t.detach()
        cam_points = world_points @ R.T + t
        z = cam_points[:, 2]
        finite_mask = torch.isfinite(cam_points).all(dim=-1) & torch.isfinite(target_uv).all(dim=-1)
        depth_mask = finite_mask & (z > 1e-6)
        if not depth_mask.any():
            return self.xyz.new_zeros(())

        cam_points = cam_points[depth_mask]
        target_uv = target_uv[depth_mask]
        weights = weights[depth_mask]
        z = z[depth_mask]

        projected_uv = pts2px(cam_points, keyframe.f, keyframe.centre)
        pixel_scale = float(max(keyframe.width, keyframe.height))
        reprojection_loss = F.smooth_l1_loss(
            projected_uv / pixel_scale,
            target_uv / pixel_scale,
            reduction="none",
        ).sum(dim=-1)

        render_height, render_width = rendered_invdepth.shape[-2:]
        sample_uv = (target_uv + 0.5) / (2**lvl) - 0.5
        in_bounds = (
            (sample_uv[:, 0] >= 0)
            & (sample_uv[:, 0] <= render_width - 1)
            & (sample_uv[:, 1] >= 0)
            & (sample_uv[:, 1] <= render_height - 1)
        )
        if not in_bounds.any():
            sparse_depth_loss = torch.zeros_like(reprojection_loss)
        else:
            sampler = make_torch_sampler(
                sample_uv[in_bounds].view(1, 1, -1, 2),
                render_width,
                render_height,
            )
            sampled_invdepth = F.grid_sample(
                rendered_invdepth[None],
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            target_invdepth = 1 / z[in_bounds].clamp_min(1e-6)
            sparse_depth_loss = torch.zeros_like(reprojection_loss)
            sparse_depth_loss[in_bounds] = F.smooth_l1_loss(
                sampled_invdepth,
                target_invdepth,
                reduction="none",
            )

        point_loss = reprojection_loss + sparse_depth_loss
        return (point_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def optimization_step(
        self,
        finetuning=False,
        keyframe_id=None,
        optimize_pose=True,
        optimize_gaussians=True,
        optimize_only_pose=False,
    ):
        if len(self.xyz) == 0:
            return None
        # Select which keyframe to train on
        # We train on the latest keyframe with self.use_last_frame_proba probability or a random keyframe otherwise
        if keyframe_id is None:
            if (
                np.random.rand() > self.use_last_frame_proba
                or self.last_trained_id == -1
                or finetuning
            ):
                keyframe_id = np.random.choice(self.active_frames_gpu)
            else:
                keyframe_id = -1
        keyframe = self.keyframes[keyframe_id]
        if keyframe.device.type != "cuda":
            keyframe.to("cuda")
        keyframe_index = int(keyframe.index)
        lvl = keyframe.pyr_lvl

        # Zero gradients
        keyframe.zero_grad()
        self.optimizer.zero_grad()

        gt_image = keyframe.image_pyr[lvl]

        def compute_loss(render_pkg, detach_geometric_pose=False):
            image = render_pkg["render"]
            invdepth = render_pkg["invdepth"]
            target_image = gt_image
            target_idepth = keyframe.get_mono_idepth(lvl)

            if keyframe.mask_pyr is not None:
                image = image * keyframe.mask_pyr[lvl]
                target_image = target_image * keyframe.mask_pyr[lvl]
                invdepth = invdepth * keyframe.mask_pyr[lvl]
                target_idepth = target_idepth * keyframe.mask_pyr[lvl]

            l1_loss = (image - target_image).abs().mean()
            ssim_loss = 1 - fused_ssim(image[None], target_image[None])
            render_depth_loss = (invdepth - target_idepth).abs().mean()
            geometry_loss = self.compute_geometric_loss(
                keyframe,
                invdepth,
                lvl,
                detach_pose=detach_geometric_pose,
            )
            loss, photometric_loss = combine_mapping_losses(
                l1_loss,
                ssim_loss,
                render_depth_loss,
                geometry_loss,
                self.lambda_dssim,
                keyframe.depth_loss_weight,
                getattr(self, "geometry_loss_weight", 1.0),
            )
            return {
                "loss": loss,
                "photometric": photometric_loss,
                "l1": l1_loss,
                "ssim": ssim_loss,
                "render_depth": render_depth_loss,
                "geometry": geometry_loss,
            }

        bg = torch.rand(3, device="cuda")
        render_pkg = None
        full_metrics = None
        if optimize_gaussians or not optimize_pose:
            # Camera poses are constant in the Gaussian pass. Joint optimization
            # updates them separately through the pose-only render below.
            render_pkg = self.render_from_id(
                keyframe_id,
                pyr_lvl=lvl,
                bg=bg,
                detach_pose=True,
            )
            full_metrics = compute_loss(
                render_pkg,
                detach_geometric_pose=True,
            )
            full_metrics["loss"].backward()

        pose_render_pkg = None
        pose_metrics = None
        pose_gaussians_used = 0
        if optimize_pose:
            pose_gaussian_mask = self.pose_optimization_gaussian_mask()
            pose_gaussians_used = int(pose_gaussian_mask.sum().item())
            if pose_gaussians_used > 0:
                pose_render_pkg = self.render_from_id(
                    keyframe_id,
                    pyr_lvl=lvl,
                    bg=bg,
                    gaussian_mask=pose_gaussian_mask,
                    detach_gaussians=True,
                )
                pose_metrics = compute_loss(pose_render_pkg)
                pose_metrics["loss"].backward()

        reported_metrics = full_metrics if full_metrics is not None else pose_metrics
        if reported_metrics is None:
            return None

        # Optimizers
        with torch.no_grad():
            # Pose optimization
            pose_step_ran = optimize_pose and pose_metrics is not None
            if pose_step_ran:
                keyframe.step(optimize_only_pose=optimize_only_pose)

            # Preserve the joint-optimization schedule and also advance it when
            # mapping holds all keyframe parameters fixed. Pose-only refinement
            # and fixed-pose test views must not change the mapping schedule.
            if not optimize_only_pose and (
                pose_step_ran
                or (optimize_gaussians and not keyframe.info["is_test"])
            ):
                keyframe.advance_mapping_schedule()

            # Skip the scene optimization if the current keyframe is a test keyframe
            if optimize_gaussians and render_pkg is not None and not keyframe.info["is_test"]:
                # Scene Gaussian optimization
                self.optimizer.step(
                    render_pkg["visibility_filter"], render_pkg["radii"].shape[0]
                )
                self._increment_gaussian_optimization_counts(
                    render_pkg["visibility_filter"]
                )
                self._mark_gaussians_observed_by_keyframe(
                    render_pkg["visibility_filter"],
                    keyframe.index,
                )
                if any(param["val"].grad is not None for param in self.gaussian_params.values()):
                    self._maybe_densify_gaussians(render_pkg)

            latest_pkg = render_pkg if render_pkg is not None else pose_render_pkg
            keyframe.latest_invdepth = latest_pkg["invdepth"].detach()

        if pose_step_ran:
            self.valid_Rt_cache[keyframe_index] = False
        self.last_trained_id = keyframe_id
        return {
            "loss": float(reported_metrics["loss"].detach().item()),
            "photometric": float(reported_metrics["photometric"].detach().item()),
            "l1": float(reported_metrics["l1"].detach().item()),
            "ssim": float(reported_metrics["ssim"].detach().item()),
            "render_depth": float(reported_metrics["render_depth"].detach().item()),
            "depth": float(reported_metrics["render_depth"].detach().item()),
            "geometry": float(reported_metrics["geometry"].detach().item()),
            "keyframe_id": keyframe_index,
            "optimize_pose": bool(pose_step_ran),
            "optimize_gaussians": bool(optimize_gaussians),
            "optimize_only_pose": bool(optimize_only_pose),
            "pose_gaussians_used": pose_gaussians_used,
        }

    def optimize_slam_window(
        self,
        keyframe_ids,
        phases,
        latest_keyframe_sample_proba=None,
        keyframe_sampling_probabilities=None,
        optimize_unsampled_keyframes_once=False,
        deterministic_keyframe_iterations=None,
    ):
        summaries = []
        keyframe_ids = [int(k) for k in keyframe_ids]
        if len(keyframe_ids) == 0:
            return summaries

        deterministic_keyframe_iterations = (
            None
            if deterministic_keyframe_iterations is None
            else [int(count) for count in deterministic_keyframe_iterations]
        )
        if deterministic_keyframe_iterations is not None:
            if len(deterministic_keyframe_iterations) != len(keyframe_ids):
                raise ValueError("Deterministic keyframe iterations must match the keyframe window size")
            if any(count < 0 for count in deterministic_keyframe_iterations):
                raise ValueError("Deterministic keyframe iterations must be non-negative")

        if latest_keyframe_sample_proba is None:
            latest_keyframe_sample_proba = getattr(
                self, "slam_latest_keyframe_sample_proba", 0.25
            )
        sampling_rng = getattr(self, "slam_window_sampling_rng", None)
        if sampling_rng is None:
            sampling_seed = int(getattr(self, "slam_window_sampling_seed", 0))
            self.slam_window_sampling_seed = sampling_seed
            self.slam_window_sampling_rng = np.random.RandomState(sampling_seed)
            sampling_rng = self.slam_window_sampling_rng
        sampling_seed = int(getattr(self, "slam_window_sampling_seed", 0))
        for name, max_iters, optimize_pose, optimize_gaussians in phases:
            losses = []
            ran_iters = 0
            sampled_iters = 0
            coverage_iters = 0
            sampled_keyframe_ids = []
            sampled_keyframe_id_set = set()
            coverage_keyframe_ids = []
            scheduled_keyframe_ids = []
            if deterministic_keyframe_iterations is not None:
                for keyframe_id, keyframe_iters in zip(keyframe_ids, deterministic_keyframe_iterations):
                    for _ in range(keyframe_iters):
                        scheduled_keyframe_ids.append(int(keyframe_id))
                        result = self.optimization_step(
                            keyframe_id=keyframe_id,
                            optimize_pose=optimize_pose,
                            optimize_gaussians=optimize_gaussians,
                        )
                        if result is None:
                            continue
                        loss = result["loss"]
                        losses.append(loss)
                        ran_iters += 1
            else:
                for _ in range(int(max_iters)):
                    if keyframe_sampling_probabilities is None:
                        keyframe_id = sample_window_keyframe_id(
                            keyframe_ids,
                            latest_probability=latest_keyframe_sample_proba,
                            rng=sampling_rng,
                        )
                    else:
                        keyframe_id = sample_keyframe_id_with_explicit_probabilities(
                            keyframe_ids,
                            keyframe_sampling_probabilities,
                            rng=sampling_rng,
                        )
                    sampled_keyframe_ids.append(int(keyframe_id))
                    sampled_keyframe_id_set.add(int(keyframe_id))
                    result = self.optimization_step(
                        keyframe_id=keyframe_id,
                        optimize_pose=optimize_pose,
                        optimize_gaussians=optimize_gaussians,
                    )
                    if result is None:
                        continue
                    loss = result["loss"]
                    losses.append(loss)
                    ran_iters += 1
                    sampled_iters += 1
                if optimize_unsampled_keyframes_once:
                    for keyframe_id in keyframe_ids:
                        if keyframe_id in sampled_keyframe_id_set:
                            continue
                        coverage_keyframe_ids.append(int(keyframe_id))
                        result = self.optimization_step(
                            keyframe_id=keyframe_id,
                            optimize_pose=optimize_pose,
                            optimize_gaussians=optimize_gaussians,
                        )
                        if result is None:
                            continue
                        loss = result["loss"]
                        losses.append(loss)
                        ran_iters += 1
                        coverage_iters += 1
            summaries.append(
                {
                    "phase": name,
                    "max_iters": int(max_iters),
                    "iters": ran_iters,
                    "sampled_iters": sampled_iters,
                    "coverage_iters": coverage_iters,
                    "sampled_keyframe_ids": sampled_keyframe_ids,
                    "coverage_keyframe_ids": coverage_keyframe_ids,
                    "scheduled_keyframe_ids": scheduled_keyframe_ids,
                    "sampling_seed": sampling_seed,
                    "optimize_pose": bool(optimize_pose),
                    "optimize_gaussians": bool(optimize_gaussians),
                    "initial_loss": losses[0] if losses else None,
                    "final_loss": losses[-1] if losses else None,
                }
            )
        return summaries

    def optimize_keyframe_poses(self, keyframe_ids, iterations_per_keyframe=1):
        keyframe_ids = [int(k) for k in keyframe_ids]
        iterations_per_keyframe = max(0, int(iterations_per_keyframe))
        losses = []
        scheduled_keyframe_ids = []
        ran_iters = 0
        for keyframe_id in keyframe_ids:
            for _ in range(iterations_per_keyframe):
                scheduled_keyframe_ids.append(int(keyframe_id))
                result = self.optimization_step(
                    keyframe_id=keyframe_id,
                    optimize_pose=True,
                    optimize_gaussians=False,
                    optimize_only_pose=True,
                )
                if result is None:
                    continue
                losses.append(result["loss"])
                ran_iters += 1

        return {
            "phase": "pose_refinement",
            "max_iters": len(keyframe_ids) * iterations_per_keyframe,
            "iters": ran_iters,
            "iterations_per_keyframe": iterations_per_keyframe,
            "sampled_iters": 0,
            "coverage_iters": 0,
            "sampled_keyframe_ids": [],
            "coverage_keyframe_ids": [],
            "scheduled_keyframe_ids": scheduled_keyframe_ids,
            "optimize_pose": True,
            "optimize_gaussians": False,
            "optimize_only_pose": True,
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
        }

    def optimize_keyframe_poses_once(self, keyframe_ids):
        return self.optimize_keyframe_poses(keyframe_ids, iterations_per_keyframe=1)

    def optimize_window_keyframe_poses_once(self, keyframe_ids):
        return self.optimize_keyframe_poses_once(keyframe_ids)

    def optimization_loop(self, n_iters: int, run_until_interupt: bool = False):
        """
        Runs at least n_iters optimization steps.
        If run_until_interupt, also runs until join_optimization_thread is called (Useful to run the optimization until the next keyframe is added in streaming mode).
        """
        self.interupt_optimization = False
        i = 0
        while i < n_iters or (run_until_interupt and not self.interupt_optimization): 
            self.optimization_step()
            i += 1
        
    def join_optimization_thread(self):
        """
        Interupts the optimization loop and waits for the thread to finish.
        """
        if self.optimization_thread is not None:
            self.interupt_optimization = True
            self.optimization_thread.join()
            self.optimization_thread = None
    
    def optimize_async(self, n_iters: int):
        """
        Starts an optimization thread that runs at least n_iters optimization steps.
        """
        self.join_optimization_thread()
        self.optimization_thread = threading.Thread(
            target=self.optimization_loop, args=(n_iters, True)
        )
        self.optimization_thread.start()

    @torch.no_grad()
    def harmonize_test_exposure(self):
        """Harmonizes the exposure matrices of test keyframes by averaging the exposure of the previous and next keyframes."""
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.info["is_test"]:
                idxm = index - 1 if index != 0 else 1
                idxp = (
                    index + 1
                    if index != len(self.keyframes) - 1
                    else len(self.keyframes) - 2
                )
                keyframe.exposure = (
                    self.keyframes[idxm].exposure + self.keyframes[idxp].exposure
                ) / 2

    @torch.no_grad()
    def evaluate(self, eval_poses=False, with_LPIPS=False, all=False):
        # Make sure test keyframes have similar exposure matrices compared to their neighbors
        self.harmonize_test_exposure()

        # Compute image quality metrics
        metrics = {"PSNR": 0, "SSIM": 0}
        if with_LPIPS:
            metrics["LPIPS"] = 0
        n_test_frames = 0
        n_lpips_frames = 0
        start_index = 0 if all else self.active_anchor.keyframe_ids[0]
        for index, keyframe in enumerate(self.keyframes[start_index:]):
            if keyframe.info["is_test"]:
                gt_image = keyframe.image_pyr[0].cuda()
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = render_pkg["render"]
                mask = (
                    keyframe.mask_pyr[0].cuda()
                    if keyframe.mask_pyr is not None
                    else torch.ones_like(image[:1] > 0)
                )
                mask = mask.expand_as(image)
                image = image * mask
                gt_image = gt_image * mask
                metrics["PSNR"] += psnr(image[mask], gt_image[mask])
                metrics["SSIM"] += fused_ssim(
                    image[None], gt_image[None], train=False
                ).item()
                if with_LPIPS:
                    lpips_value = self._lpips_value(image, gt_image)
                    if lpips_value is not None:
                        metrics["LPIPS"] += lpips_value
                        n_lpips_frames += 1
                n_test_frames += 1
                del render_pkg, image, gt_image, mask

        if n_test_frames > 0:
            for metric in metrics:
                if metric == "LPIPS":
                    metrics[metric] = metrics[metric] / n_lpips_frames if n_lpips_frames > 0 else None
                else:
                    metrics[metric] /= n_test_frames
        else:
            metrics = {}

        if with_LPIPS:
            self._clear_lpips_model()

        # Compute pose errors
        if eval_poses:
            Rts = self.get_Rts()
            gt_Rts = self.get_gt_Rts(align=False)
            if len(Rts) == len(gt_Rts):
                Rts_aligned = torch.linalg.inv(align_poses(Rts, gt_Rts))
                gt_Rts = torch.linalg.inv(gt_Rts)
                R_error = rotation_distance(Rts_aligned[:, :3, :3], gt_Rts[:, :3, :3])
                t_error = (Rts_aligned[:, :3, 3] - gt_Rts[:, :3, 3]).norm(dim=-1)

                metrics["R°"] = R_error.mean().item() * 180 / math.pi
                metrics["t"] = t_error.mean().item()

        return metrics

    @torch.no_grad()
    def save_test_frames(self, out_dir):
        self.harmonize_test_exposure()
        os.makedirs(out_dir, exist_ok=True)
        for keyframe in self.keyframes:
            if keyframe.info["is_test"]:
                render_pkg = self.render_from_id(keyframe.index, pyr_lvl=0)
                image = torch.clamp(render_pkg["render"], 0, 1) * 255
                image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                is_jpeg = os.path.splitext(keyframe.info["name"])[-1].lower() in [
                    ".jpg",
                    ".jpeg",
                ]
                write_flag = [int(cv2.IMWRITE_JPEG_QUALITY), 100] if is_jpeg else []
                cv2.imwrite(
                    os.path.join(out_dir, keyframe.info["name"]), image, write_flag
                )

    def render_from_id(
        self,
        keyframe_id,
        pyr_lvl=0,
        scaling_modifier=1,
        bg=None,
        gaussian_mask=None,
        detach_pose=False,
        detach_gaussians=False,
    ):
        """
        Render the scene from a given keyframe id at a specified resolution level (pyr_lvl).
        Applies the exposure matrix of the keyframe to the rendered image.
        """
        keyframe = self.keyframes[keyframe_id]
        view_matrix = keyframe.get_Rt()
        if detach_pose:
            view_matrix = view_matrix.detach()
        view_matrix = view_matrix.transpose(0, 1)
        scale = 2**pyr_lvl
        width, height = keyframe.width // scale, keyframe.height // scale
        render_pkg = self.render(
            width,
            height,
            view_matrix,
            scaling_modifier,
            bg,
            gaussian_mask=gaussian_mask,
            detach_gaussians=detach_gaussians,
            intrinsics=resize_intrinsics(keyframe.get_intrinsics(), 1 / scale, 1 / scale),
        )
        render_pkg["render"] = (
            keyframe.exposure[:3, :3] @ render_pkg["render"].view(3, -1)
        ) + keyframe.exposure[:3, 3, None]
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def render(
        self,
        width: int,
        height: int,
        view_matrix: torch.Tensor,
        scaling_modifier: float,
        bg: torch.Tensor = None,
        top_view: bool = False,
        fov_x: float = None,
        fov_y: float = None,
        gaussian_mask=None,
        detach_gaussians=False,
        intrinsics=None,
    ):
        cam_centre = view_matrix.detach().inverse()[3, :3]
        device = view_matrix.device
        if bg is None:
            bg = torch.zeros(3, device=device, dtype=view_matrix.dtype)

        if intrinsics is not None and (fov_x is not None or fov_y is not None):
            raise ValueError("Provide either intrinsics or a field of view")
        # Explicit intrinsics are expressed at the requested rendering resolution.
        if fov_x is None and fov_y is None:
            if intrinsics is None:
                K = resize_intrinsics(self.K, width / self.width, height / self.height)
            else:
                K = torch.as_tensor(intrinsics, device=view_matrix.device, dtype=view_matrix.dtype)
            tanfovx = float(width / (2 * K[0, 0]))
            tanfovy = float(height / (2 * K[1, 1]))
            projection_matrix = getProjectionMatrixFromIntrinsics(
                znear=0.01, zfar=100.0, K=K, width=width, height=height
            ).transpose(0, 1).contiguous()
        # Use the provided FOV values
        elif fov_x is not None and fov_y is not None:
            tanfovx = math.tan(fov_x * 0.5)
            tanfovy = math.tan(fov_y * 0.5)
            projection_matrix = (
                getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov_x, fovY=fov_y)
                .transpose(0, 1)
                .to(device=device, dtype=view_matrix.dtype)
            )
        else:
            raise ValueError("Both fov_x and fov_y should be provided or neither.")

        raster_settings = GaussianRasterizationSettings(
            height,
            width,
            tanfovx,
            tanfovy,
            bg,
            1 if top_view else scaling_modifier,
            projection_matrix,
            self.active_sh_degree,
            cam_centre,
            False,
            False,
        )
        rasterizer = GaussianRasterizer(raster_settings)
        with self.lock:
            # Load and blend anchors if in inference mode 
            if self.inference_mode and not top_view:
                self.gaussian_params, self.anchor_weights = Anchor.blend(cam_centre, self.anchors, self.anchor_overlap)
            if gaussian_mask is None:
                xyz = self.xyz
                f_dc = self.f_dc
                f_rest = self.f_rest
                scaling = self.scaling
                opacity = self.opacity
                rotation = self.rotation
            else:
                gaussian_mask = gaussian_mask.to(device=self.xyz.device, dtype=torch.bool)
                xyz = self.xyz[gaussian_mask]
                f_dc = self.f_dc[gaussian_mask]
                f_rest = self.f_rest[gaussian_mask]
                scaling = self.scaling[gaussian_mask]
                opacity = self.opacity[gaussian_mask]
                rotation = self.rotation[gaussian_mask]
            if detach_gaussians:
                xyz = xyz.detach()
                f_dc = f_dc.detach()
                f_rest = f_rest.detach()
                scaling = scaling.detach()
                opacity = opacity.detach()
                rotation = rotation.detach()
            screenspace_points = torch.zeros_like(
                xyz,
                requires_grad=(
                    not self.inference_mode
                    and torch.is_grad_enabled()
                    and not detach_gaussians
                ),
            )
            if xyz.shape[0] > 0:
                # Set constant scaling and opacity to visualize the Gaussians' positions in the top view
                if top_view:
                    scaling = torch.ones_like(scaling) * scaling_modifier
                    opacity = torch.ones_like(opacity)
                else:
                    scaling = scaling
                    opacity = opacity
                raster_out = rasterizer(
                    xyz,
                    screenspace_points,
                    opacity,
                    f_dc,
                    f_rest,
                    scaling,
                    rotation,
                    view_matrix,
                )
                color, invdepth, mainGaussID, radii = raster_out
            else:
                # If no Gaussians are present, return empty tensors
                color = torch.zeros(3, height, width, device=device)
                invdepth = torch.zeros(1, height, width, device=device)
                mainGaussID = -torch.ones(
                    2, height, width, device=device, dtype=torch.int32
                )
                radii = torch.zeros(0, device=device, dtype=torch.int32)
        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }

    def get_closest_by_cam(self, cam_centre, k=3):
        closest_anchors = []
        closest_anchors_ids = []
        offset = 0
        approx_cam_centres = self.approx_cam_centres.clone()
        for l in range(min(k, len(self.anchors))):
            if approx_cam_centres.shape[0] == 0:
                break
            dists = torch.linalg.norm(approx_cam_centres - cam_centre[None], dim=-1)
            min_dist, min_id = torch.min(dists, dim=0)

            if min_dist < 1e9:
                for anchor_id, anchor in enumerate(self.anchors):
                    if min_id in anchor.keyframe_ids:
                        closest_anchors.append(anchor)
                        closest_anchors_ids.append(anchor_id)
                        approx_cam_centres[
                            anchor.keyframe_ids[0] : anchor.keyframe_ids[-1] + 1
                        ] = 1e9
                        break

        return closest_anchors, closest_anchors_ids

    @torch.no_grad()
    def get_prev_keyframes(self, n: int, update_3dpts: bool, desc_kpts: DescribedKeypoints = None):
        """
        Get the n previous keyframes that are the closest to the last
        If desc_kpts is not None, we find the previous keyframes that have the most matches with desc_kpts. The search window is given by self.num_prev_keyframes_check
        """
        # Make sure the optimization thread is not running
        self.join_optimization_thread()

        # Look for the previous keyframes with the most matches with desc_kpts (if provided)
        if desc_kpts is not None and len(self.keyframes) > n:
            n_ckecks = min(self.num_prev_keyframes_check, len(self.keyframes))
            keyframes_indices_to_check = self.sorted_frame_indices[:n_ckecks]
            n_matches = torch.zeros(len(keyframes_indices_to_check), device="cuda")
            for i, index in enumerate(keyframes_indices_to_check):
                n_matches[i] = self.matcher.evaluate_match(
                    self.keyframes[index].desc_kpts, desc_kpts
                )
            _, top_indices = torch.topk(n_matches, n)
            prev_keyframes_indices = keyframes_indices_to_check[top_indices.cpu()]
        # If desc_kpts is not provided, we take the n closest keyframes
        else:
            prev_keyframes_indices = self.sorted_frame_indices[:n]
        prev_keyframes = [self.keyframes[i] for i in prev_keyframes_indices]

        # Re-run triangulation if necessary
        if update_3dpts:
            for keyframe in prev_keyframes:
                keyframe.update_3dpts(self.keyframes)
        return prev_keyframes

    def get_Rts(self):
        invalid_ids = torch.where(~self.valid_Rt_cache)[0]
        if len(invalid_ids) > 0:
            for keyframe_id in invalid_ids:
                self.cached_Rts[keyframe_id] = self.keyframes[keyframe_id].get_Rt()
            self.valid_Rt_cache[invalid_ids] = True
        return self.cached_Rts

    @staticmethod
    def _rotate_gaussian_quaternions(quaternions, rotation):
        """Left-compose a world rotation with Gaussian (w, x, y, z) rotations."""
        r = rotation.to(quaternions)
        q_abs = torch.sqrt(torch.clamp(torch.stack([
            1 + r[0, 0] + r[1, 1] + r[2, 2],
            1 + r[0, 0] - r[1, 1] - r[2, 2],
            1 - r[0, 0] + r[1, 1] - r[2, 2],
            1 - r[0, 0] - r[1, 1] + r[2, 2],
        ]), min=0))
        candidates = torch.stack([
            torch.stack([q_abs[0] ** 2, r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]),
            torch.stack([r[2, 1] - r[1, 2], q_abs[1] ** 2, r[1, 0] + r[0, 1], r[0, 2] + r[2, 0]]),
            torch.stack([r[0, 2] - r[2, 0], r[1, 0] + r[0, 1], q_abs[2] ** 2, r[2, 1] + r[1, 2]]),
            torch.stack([r[1, 0] - r[0, 1], r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], q_abs[3] ** 2]),
        ])
        best = q_abs.argmax()
        delta = candidates[best] / (2 * q_abs[best].clamp_min(1e-8))
        q = F.normalize(quaternions, dim=-1)
        w, x, y, z = q.unbind(dim=-1)
        a, b, c, d = delta.unbind()
        return F.normalize(torch.stack([
            a * w - b * x - c * y - d * z,
            a * x + b * w + c * z - d * y,
            a * y - b * z + c * w + d * x,
            a * z + b * y - c * x + d * w,
        ], dim=-1), dim=-1)

    @torch.no_grad()
    def apply_local_ba_result(self, pose_updates, landmark_updates):
        """Commit accepted local BA poses and shared landmarks without tearing the map.

        ``pose_updates`` maps keyframe IDs to world-to-camera matrices.
        Each landmark is ``(observations, xyz)`` where observations are
        ``(keyframe_id, keypoint_index)`` pairs. The caller performs residual and
        motion acceptance checks; this method validates the entire result before
        changing scene state. No ground-truth information is used.
        """
        self.join_optimization_thread()
        prepared_poses, corrections = {}, {}
        for keyframe_id, value in pose_updates.items():
            keyframe_id = int(keyframe_id)
            if not 0 <= keyframe_id < len(self.keyframes):
                raise ValueError("Local BA pose references an unknown keyframe")
            old = self.keyframes[keyframe_id].get_Rt().detach()
            new = torch.as_tensor(value, device=old.device, dtype=old.dtype).detach().clone()
            if new.shape != (4, 4) or not torch.isfinite(new).all():
                raise ValueError("Local BA poses must be finite 4 x 4 matrices")
            rotation = new[:3, :3]
            if (not torch.allclose(new[3], new.new_tensor([0, 0, 0, 1]), atol=1e-5, rtol=0)
                    or not torch.allclose(rotation.T @ rotation, torch.eye(3, device=new.device, dtype=new.dtype), atol=1e-3, rtol=0)
                    or not torch.isclose(torch.linalg.det(rotation), new.new_tensor(1), atol=1e-3, rtol=0)):
                raise ValueError("Local BA poses must contain proper rigid rotations")
            prepared_poses[keyframe_id] = new
            if not torch.allclose(old, new, atol=1e-7, rtol=0):
                # T_new D X = T_old X: preserve geometry in the owner camera.
                corrections[keyframe_id] = torch.linalg.inv(new) @ old

        point_batches, observation_groups, seen_observations = {}, {}, set()
        landmark_count, point_count = 0, 0
        for observations, point in landmark_updates:
            point = torch.as_tensor(point).detach()
            if point.shape != (3,):
                raise ValueError("Local BA landmarks must be finite 3D points")
            point_batches.setdefault(point.device, []).append((point_count, point))
            has_observation = False
            for keyframe_id, keypoint_id in observations:
                keyframe_id, keypoint_id = int(keyframe_id), int(keypoint_id)
                if not 0 <= keyframe_id < len(self.keyframes):
                    raise ValueError("Local BA landmark references an unknown keyframe")
                keyframe = self.keyframes[keyframe_id]
                desc = keyframe.desc_kpts
                if desc is None or not 0 <= keypoint_id < len(desc.pts3d):
                    raise ValueError("Local BA landmark references an unknown keypoint")
                observation = (keyframe_id, keypoint_id)
                if observation in seen_observations:
                    raise ValueError("Local BA landmarks must have disjoint observations")
                seen_observations.add(observation)
                observation_groups.setdefault(keyframe_id, []).append((keypoint_id, point_count))
                has_observation = True
            landmark_count += int(has_observation)
            point_count += 1

        # Snapshot once per source device, rather than synchronizing CUDA for
        # every landmark and every observation in a typical 1,500-track window.
        points_cpu = torch.empty((point_count, 3), dtype=torch.float64)
        for batch in point_batches.values():
            ids = torch.tensor([index for index, _ in batch], dtype=torch.long)
            points_cpu[ids] = torch.stack([point for _, point in batch]).to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(points_cpu).all():
            raise ValueError("Local BA landmarks must be finite 3D points")
        prepared_observations = []
        for keyframe_id, observations in observation_groups.items():
            pose = prepared_poses.get(keyframe_id)
            if pose is None:
                pose = self.keyframes[keyframe_id].get_Rt().detach()
            pose_cpu = pose.to(device="cpu", dtype=torch.float64)
            ids = torch.tensor(observations, dtype=torch.long)
            frame_points = points_cpu[ids[:, 1]]
            depths = frame_points @ pose_cpu[2, :3] + pose_cpu[2, 3]
            if not torch.isfinite(depths).all() or (depths <= 1e-6).any():
                raise ValueError("Local BA landmark is behind an observing camera")
            prepared_observations.append((keyframe_id, ids[:, 0], frame_points, depths))

        map_sets = [(self.gaussian_params, self._fit_gaussian_owner_keyframe_ids())]
        # Offloaded anchors keep their own ownership snapshot and may contain
        # duplicated geometry from the overlap with the active map.
        for anchor in getattr(self, "anchors", []):
            if anchor.gaussian_params is self.gaussian_params:
                continue
            owners = getattr(anchor, "gaussian_owner_keyframe_ids", None)
            if owners is None:
                # Archived maps can retain overlap Gaussians whose owner is no
                # longer in anchor.keyframe_ids, so membership is insufficient.
                if corrections and anchor.gaussian_params["xyz"]["val"].shape[0] > 0:
                    raise ValueError("Cannot correct an archived map without Gaussian ownership")
                continue
            if owners.shape != (anchor.gaussian_params["xyz"]["val"].shape[0],):
                raise ValueError("Archived Gaussian ownership does not match map size")
            map_sets.append((anchor.gaussian_params, owners))

        moved_gaussians = 0
        with self.lock:
            for params, owners in map_sets:
                xyz, rotations = params["xyz"]["val"], params["rotation"]["val"]
                owners = owners.to(device=xyz.device)
                for keyframe_id, correction in corrections.items():
                    selected = owners == keyframe_id
                    count = int(selected.sum().item())
                    if count == 0:
                        continue
                    delta = correction.to(xyz)
                    xyz[selected] = xyz[selected] @ delta[:3, :3].T + delta[:3, 3]
                    rotations[selected] = self._rotate_gaussian_quaternions(rotations[selected], delta[:3, :3])
                    for name in ("xyz", "rotation"):
                        for moment in ("exp_avg", "exp_avg_sq"):
                            if moment in params[name]:
                                params[name][moment][selected] = 0
                        params[name]["val"].grad = None
                    if params is self.gaussian_params:
                        gradient_accum, gradient_counts = self._fit_gaussian_densification_stats()
                        gradient_accum[selected] = 0
                        gradient_counts[selected] = 0
                    moved_gaussians += count

            for keyframe_id, new in prepared_poses.items():
                keyframe = self.keyframes[keyframe_id]
                correction = corrections.get(keyframe_id)
                desc = keyframe.desc_kpts
                if correction is not None and desc is not None:
                    valid = desc.has_pt3d & torch.isfinite(desc.pts3d).all(dim=-1)
                    delta = correction.to(desc.pts3d)
                    desc.pts3d[valid] = desc.pts3d[valid] @ delta[:3, :3].T + delta[:3, 3]
                keyframe.set_Rt(new)
                if correction is not None:
                    optimizer = getattr(keyframe, "optimizer", None)
                    if optimizer is not None:
                        for name in ("rW2C", "tW2C"):
                            state = optimizer.params.get(name, {})
                            for moment in ("exp_avg", "exp_avg_sq"):
                                if moment in state:
                                    state[moment].zero_()
                            if "val" in state:
                                state["val"].grad = None
                keyframe.latest_invdepth = None
                self.valid_Rt_cache[keyframe_id] = False
                if self.approx_cam_centres is not None:
                    self.approx_cam_centres[keyframe_id] = keyframe.approx_centre.to(self.approx_cam_centres)

            for keyframe_id, ids, points, depths in prepared_observations:
                desc = self.keyframes[keyframe_id].desc_kpts
                ids = ids.to(device=desc.pts3d.device)
                desc.pts3d[ids] = points.to(desc.pts3d)
                desc.depth[ids] = depths.to(desc.depth)
                desc.has_pt3d[ids] = True
                # Retain triangulation confidence, including strong inliers.
                confidence = desc.pts_conf[ids]
                desc.pts_conf[ids] = torch.where(torch.isfinite(confidence) & (confidence > 0), confidence, torch.ones_like(confidence))

            if corrections:
                # A fixed camera can also see corrected map geometry.
                for keyframe in self.keyframes:
                    keyframe.latest_invdepth = None
                if self.approx_cam_centres is not None and len(self.keyframes):
                    distances = torch.linalg.vector_norm(self.approx_cam_centres - self.approx_cam_centres[-1], dim=-1)
                    self.sorted_frame_indices = torch.argsort(distances).cpu()
                for anchor in getattr(self, "anchors", []):
                    ids = getattr(anchor, "keyframe_ids", [])
                    if ids and self.approx_cam_centres is not None:
                        anchor.position = self.approx_cam_centres[ids].mean(dim=0)
        return {
            "poses_updated": len(corrections),
            "landmarks_updated": landmark_count,
            "observations_updated": len(seen_observations),
            "gaussians_warped": moved_gaussians,
        }

    def get_gt_Rts(self, align):
        n_poses = min(self.gt_Rts_mask.shape[0], self.cached_Rts.shape[0])
        if align and n_poses > 0:
            Rts = self.get_Rts()[:n_poses][self.gt_Rts_mask[:n_poses]]
            return align_poses(self.gt_Rts[: len(Rts)], Rts)
        else:
            return self.gt_Rts

    def make_dummy_ext_tensor(self):
        return {
            name: self.gaussian_params[name]["val"][:0].detach()
            for name in self.gaussian_params
        }

    def reset(self, keyframe_id: int = -1):
        """Remove the Gaussians that are visible in the given keyframe."""
        valid_mask = self.opacity[:, 0] > 0.05
        render_pkg = self.render_from_id(keyframe_id)
        valid_mask[render_pkg["visibility_filter"]] = False
        self._apply_gaussian_add_and_prune(self.make_dummy_ext_tensor(), valid_mask)

    @torch.no_grad()
    def add_new_gaussians(self, keyframe_id: int = -1):
        """Use the given keyframe to add new Gaussians to the scene model."""
        keyframe = self.keyframes[keyframe_id]
        ## align the keyframe's depth
        if keyframe.desc_kpts.has_pt3d.sum() == 0:
            keyframe.update_3dpts(self.keyframes)
        keyframe.align_depth()

        # Skip if the keyframe is a test keyframe
        if keyframe.info["is_test"]:
            return

        ## Get the pixel-wise probability to add a Gaussian
        img = keyframe.image_pyr[0]
        img = F.avg_pool2d(img, 2)
        img = F.interpolate(
            img[None], (self.height, self.width), mode="bilinear", align_corners=True
        )[0]
        init_proba = get_sampling_response(
            img, self.disc_kernel, self.init_sampling_method
        ) # eq. 1

        if keyframe.mask_pyr is not None:
            dilated_mask = (
                F.conv2d(
                    keyframe.mask_pyr[0][None].float(), self.disc_kernel, padding="same"
                )[0, 0]
                >= 0.99
            )
            init_proba *= dilated_mask

        if getattr(self, "direct_gaussian_sampling", True):
            ## Compute the penalty based on the rendering from the new keyframe's point of view
            penalty = torch.zeros_like(init_proba)
            rendered_depth = None
            if self.xyz.shape[0] > 0:
                render_pkg = self.render_from_id(keyframe_id)
                render = render_pkg["render"]
                rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
                penalty = get_sampling_response(
                    render, self.disc_kernel, self.init_sampling_method
                )

            ## Define which pixels should become Gaussians
            init_proba *= self.init_proba_scaler
            penalty *= self.init_proba_scaler
            neighbor_penalty = valid_8_neighbor_mean(penalty)
            sample_proba = (
                init_proba
                - penalty
                - self.slam_neighbor_penalty_weight * neighbor_penalty
            ).clamp_min(0)
            sample_mask = torch.rand_like(init_proba) < sample_proba

            sampled_uv = self.uv[sample_mask]
            ## Initialize positions
            prev_KFs = self.get_prev_keyframes(
                self.guided_mvs.n_cams + 1, update_3dpts=False
            )
            for i, prev_keyframe in enumerate(prev_KFs):
                if keyframe.index == prev_keyframe.index:
                    prev_KFs.pop(i)
                    break
            if sampled_uv.shape[0] > 0:
                depth, accurate_mask = self.guided_mvs(sampled_uv, keyframe, prev_KFs)
                valid_mask = (keyframe.sample_conf(sampled_uv) > 0.5) & (depth > 1e-6)
                sample_mask[sample_mask.clone()] = valid_mask
                depth = depth[valid_mask]
                sampled_uv = sampled_uv[valid_mask]
                accurate_mask = accurate_mask[valid_mask]
            else:
                depth = torch.empty(0, device=img.device)
                accurate_mask = torch.empty(0, device=img.device, dtype=torch.bool)

            # Remove Gaussians that are coarser than the newpoints
            if len(self.xyz) > 0:
                main_gaussians_map = render_pkg["mainGaussID"]
                accurate_sample_mask = sample_mask.clone()
                accurate_sample_mask[accurate_sample_mask.clone()] = accurate_mask
                selected_main_gaussians = main_gaussians_map[:, accurate_sample_mask]
                ids, counts = torch.unique(
                    selected_main_gaussians[selected_main_gaussians >= 0],
                    return_counts=True,
                )
                valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
                valid_gs_mask[ids] = counts < 10
                with self.lock:
                    self._apply_gaussian_add_and_prune(
                        self.make_dummy_ext_tensor(), valid_gs_mask
                    )
                render_pkg = self.render_from_id(keyframe_id)
                rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

            # Check for occlusions
            if rendered_depth is not None:
                valid_mask = depth < rendered_depth[sample_mask]
                sample_mask[sample_mask.clone()] = valid_mask
                depth = depth[valid_mask]
                sampled_uv = sampled_uv[valid_mask]
                accurate_mask = accurate_mask[valid_mask]

        else:
            # Sparse triangulated landmarks seed new regions in gradient mode.
            # Image response is retained only for their existing scale formula.
            init_proba *= self.init_proba_scaler
            sample_mask = torch.zeros_like(init_proba, dtype=torch.bool)
            sampled_uv = img.new_empty((0, 2))
            depth = img.new_empty(0)
            accurate_mask = torch.empty(0, device=img.device, dtype=torch.bool)

        # Get the samples' 3D positions
        new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), keyframe.f, keyframe.centre)
        new_pts = (new_pts - keyframe.get_t()) @ keyframe.get_R()
        # Add points from matching
        match_pts = keyframe.desc_kpts.pts3d[keyframe.desc_kpts.has_pt3d]
        new_pts = torch.cat([new_pts, match_pts], dim=0)

        ## Initialize Colour
        f_dc = img[:, sample_mask]
        match_sampler = keyframe.desc_kpts.kpts[keyframe.desc_kpts.has_pt3d]
        match_sampler = make_torch_sampler(match_sampler, self.width, self.height)
        if match_sampler.shape[0] > 0:
            match_colors = F.grid_sample(
                img[None],
                match_sampler[None, None],
                mode="bilinear",
                align_corners=True,
            ).view(3, -1)
        else:
            match_colors = torch.empty(3, 0, device=img.device)
        f_dc = torch.cat([f_dc, match_colors], dim=1)
        f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

        ## Initialize Scales
        sampled_init_proba = init_proba[sample_mask]
        if match_sampler.shape[0] > 0:
            match_init_proba = F.grid_sample(
                init_proba[None, None],
                match_sampler[None, None],
                mode="bilinear",
                align_corners=True,
            ).view(-1)
        else:
            match_init_proba = torch.empty(0, device=img.device)
        sampled_init_proba = torch.cat([sampled_init_proba, match_init_proba], dim=0)
        # Expected distance to the nearest neighbour (eq. 4)
        scales = 1 / (torch.sqrt(sampled_init_proba))
        scales.clamp_(1, self.width / 10)
        # Scale by the distance to the camera centre
        # An isotropic Gaussian uses the area-equivalent focal length.
        scales.mul_(1 / torch.sqrt(keyframe.f.prod()))
        scales *= torch.linalg.vector_norm(
            new_pts - keyframe.approx_centre[None], dim=-1
        )
        scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

        ## Initialize opacities
        opacities = torch.ones(f_dc.shape[0], 1, device=img.device)
        # Lower inital opacity depending for innacurate points
        opacities[: sampled_uv.shape[0]] *= (
            0.07 * accurate_mask[..., None] + 0.02 * ~accurate_mask[..., None]
        )
        # High opacity for triangulated Gaussians
        opacities[sampled_uv.shape[0] :] *= 0.2
        opacities = inverse_sigmoid(opacities)

        ## Initialize SH, rotations as identity
        f_rest = torch.zeros(
            f_dc.shape[0],
            (self.max_sh_degree + 1) * (self.max_sh_degree + 1) - 1,
            3,
            device=img.device,
        )
        rots = torch.zeros(f_dc.shape[0], 4, device=img.device)
        rots[:, 0] = 1

        ## Get which Gaussians should be pruned
        if self.xyz.shape[0] > 0:
            # Only keep Gaussians with non neglectible opacity
            valid_gs_mask = self.opacity[:, 0] > 0.05

            # Discard huge Gaussians
            dist = torch.linalg.vector_norm(
                self.xyz - keyframe.approx_centre[None], dim=-1
            )
            screen_size = keyframe.f.max() * self.scaling.max(dim=-1)[0] / dist
            valid_gs_mask *= screen_size < 0.5 * self.width
        else:
            valid_gs_mask = torch.ones(0, device=img.device, dtype=torch.bool)

        ## Append the new Gaussians
        extension_tensors = {
            "xyz": new_pts,
            "f_dc": f_dc,
            "f_rest": f_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        with self.lock:
            extension_observer_ids = self._observer_ids_for_keyframe(
                new_pts.shape[0],
                keyframe.index,
                device=new_pts.device,
            )
            self._apply_gaussian_add_and_prune(
                extension_tensors,
                valid_gs_mask,
                extension_observer_ids=extension_observer_ids,
            )

    def init_intrinsics(self, K=None):
        self.K = intrinsics_from_info(
            {"K": K} if K is not None else {}, self.f, self.width, self.height,
            device=self.centre.device, dtype=self.centre.dtype,
        )
        self.f = self.K[[0, 1], [0, 1]].clone()
        self.centre = self.K[:2, 2].clone()
        self.FoVx = focal2fov(float(self.f[0]), self.width)
        self.FoVy = focal2fov(float(self.f[1]), self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = (
            getProjectionMatrixFromIntrinsics(
                znear=0.01, zfar=100.0, K=self.K, width=self.width, height=self.height
            )
            .transpose(0, 1)
            .contiguous()
        )

    def move_rand_keyframe_to_cpu(self):
        """Move a random keyframe to CPU memory"""
        n_kept_frames = min(getattr(self, "n_kept_frames", 20), len(self.active_frames_gpu) - 1)
        candidates = self.active_frames_gpu[:-n_kept_frames] if n_kept_frames > 0 else self.active_frames_gpu
        if len(candidates) == 0:
            return
        frame_id = np.random.choice(candidates)
        self.keyframes[frame_id].to("cpu")
        self.active_frames_cpu.append(frame_id)
        self.active_frames_gpu.remove(frame_id) 

    def move_rand_keyframe_to_gpu(self):
        """Move a random keyframe to GPU memory"""
        if len(self.active_frames_cpu) > 0:
            frame_id = np.random.choice(self.active_frames_cpu)
            self.keyframes[frame_id].to("cuda")
            self.active_frames_gpu.insert(0, frame_id)
            self.active_frames_cpu.remove(frame_id) 

    def add_keyframe(self, keyframe: Keyframe, f=None):
        """Add a keyframe to the scene, add and prune Gaussians"""

        # Make sure training is not running 
        self.join_optimization_thread()

        ## Add the keyframe and update the indices (sorted by distance to last keyframe)
        self.keyframes.append(keyframe)
        if self.approx_cam_centres is None:
            self.approx_cam_centres = keyframe.approx_centre[None]
        else:
            self.approx_cam_centres = torch.cat(
                [self.approx_cam_centres, keyframe.approx_centre[None]], dim=0
            )
        dist_to_last = torch.linalg.vector_norm(
            self.approx_cam_centres - keyframe.approx_centre[None], dim=-1
        )
        self.sorted_frame_indices = torch.argsort(dist_to_last).cpu()

        ## Update intrinsics
        self.init_intrinsics(keyframe.get_intrinsics(self.width, self.height))

        ## Update cached Rts for the viewer
        self.cached_Rts = torch.cat(
            [self.cached_Rts, keyframe.get_Rt().unsqueeze(0)], dim=0
        )
        self.valid_Rt_cache = torch.cat(
            [self.valid_Rt_cache, self.valid_Rt_cache.new_ones(1)], dim=0
        )
        gt_pose = keyframe.info.get("Rt", None)
        if gt_pose is not None:
            self.gt_Rts = torch.cat([self.gt_Rts, gt_pose.unsqueeze(0)], dim=0)
        self.gt_Rts_mask = torch.cat(
            [
                self.gt_Rts_mask,
                torch.Tensor([gt_pose is not None]).to(self.gt_Rts_mask),
            ],
            dim=0,
        )
        self.gt_f = keyframe.info.get("focal", self.f)

        if not self.inference_mode:
            ## Add keyframe to the active anchor
            self.active_anchor.add_keyframe(keyframe)
            self.active_frames_gpu.append(keyframe.index)

            ## Clear memory if there are many keyframes
            if len(self.active_frames_gpu) > self.max_active_keyframes:
                self.move_rand_keyframe_to_cpu()
                # Reshuffle the active keyframes and clear cache
                if len(self.active_frames_cpu) % 5 == 0:
                    self.move_rand_keyframe_to_cpu()
                    self.move_rand_keyframe_to_gpu()

                    gc.collect()
                    torch.cuda.empty_cache()

    def enable_inference_mode(self):
        """Enable inference mode and sets the anchor position to the mean of the active keyframes."""
        self.inference_mode = True
        self.update_anchor()
        self.release_training_state()

    def update_anchor(self, n_left_frames: int = 0):
        """Update the anchor position and remove the last n_left_frames keyframes from the active anchor."""
        anchor_position = self.approx_cam_centres[
            self.first_active_frame : self.last_active_frame - n_left_frames
        ].mean(dim=0)
        self.active_anchor.position = anchor_position
        if n_left_frames > 0:
            self.active_anchor.keyframes = self.active_anchor.keyframes[:-n_left_frames]
            self.active_anchor.keyframe_ids = self.active_anchor.keyframe_ids[
                :-n_left_frames
            ]

    def place_anchor_if_needed(self):
        """Check if many Gaussians appear small on the screen. If so, place a new anchor. and merge the Gaussians."""
        small_prop_thresh = 0.4
        k = 3
        if (
            self.xyz.shape[0] > 0
            and self.first_active_frame < len(self.keyframes) - 2 * self.n_kept_frames
        ):
            with torch.no_grad():
                dist = torch.linalg.vector_norm(
                    self.xyz - self.approx_cam_centres[-1][None], dim=-1
                )
                screen_size = self.f.max() * self.scaling.mean(dim=-1) / dist
                small_mask = screen_size < 1
                small_prop = small_mask.float().mean()

            if small_prop > small_prop_thresh:
                with torch.no_grad():
                    small_mask = screen_size < 1.5
                    # Update anchor positions using the camera poses used to optimize it
                    self.update_anchor(self.n_kept_frames)

                    ## Merge fine Gaussians for the current active set
                    # Select a subset and get their nearest neighbours for merging
                    small_gaussians = {
                        name: self.gaussian_params[name]["val"][small_mask]
                        for name in self.gaussian_params
                    }
                    xyz = small_gaussians["xyz"].contiguous()
                    _, nn_idx = distIndex2(xyz, k)
                    nn_idx = nn_idx.view(-1, k)
                    perm = torch.randperm(xyz.shape[0], device=xyz.device)
                    idx = perm[: (xyz.shape[0] // (k + 1))]
                    selected_nn_idx = torch.cat([idx[..., None], nn_idx[idx]], dim=-1)
                    source_counts = self._fit_gaussian_optimization_counts()
                    small_indices = torch.where(small_mask)[0]
                    selected_source_idx = small_indices[selected_nn_idx]
                    source_observers = self._fit_gaussian_observed_keyframe_ids()
                    source_owners = self._fit_gaussian_owner_keyframe_ids()

                    # Compute merging weights based on contribution to the rendering
                    weights = self.gaussian_params["opacity"]["val"][
                        selected_source_idx, 0
                    ].sigmoid() * (screen_size[selected_source_idx] ** 2)
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                    dominant_source = selected_source_idx.gather(1, weights.argmax(dim=1, keepdim=True))[:, 0]
                    merged_owners = source_owners[dominant_source]
                    weights.unsqueeze_(-1)

                    # Merge the Gaussians by averaging their parameters
                    merged_gaussians = {
                        "xyz": (self.gaussian_params["xyz"]['val'][selected_source_idx, :] * weights).sum(dim=1),
                        "f_dc": (self.gaussian_params["f_dc"]['val'][selected_source_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "f_rest": (self.gaussian_params["f_rest"]['val'][selected_source_idx, :] * weights.unsqueeze(-1)).sum(dim=1),
                        "opacity": inverse_sigmoid(self.gaussian_params["opacity"]['val'][selected_source_idx, :].sigmoid() * weights).sum(dim=1),
                        "scaling": torch.log((torch.exp(self.gaussian_params["scaling"]['val'][selected_source_idx, :]) * weights * (k+1)).sum(dim=1)),
                        "rotation": (self.gaussian_params["rotation"]['val'][selected_source_idx, :] * weights).sum(dim=1),
                    }
                    merged_counts = source_counts[selected_source_idx].max(dim=1).values
                    merged_observers = self._merge_gaussian_observer_ids(
                        source_observers[selected_source_idx]
                    )

                    # Offload the previous Gaussians to the CPU
                    self.active_anchor.gaussian_owner_keyframe_ids = source_owners.detach().cpu().clone()
                    self.active_anchor.duplicate_param_dict()
                    self.active_anchor.to("cpu", with_keyframes=True)

                    ## Add the merged Gaussians to the set of Gaussians and reset the optimizer
                    with self.lock:
                        self._apply_gaussian_add_and_prune(
                            merged_gaussians,
                            ~small_mask,
                            extension_counts=merged_counts,
                            extension_observer_ids=merged_observers,
                            extension_owner_ids=merged_owners,
                        )

                    # Create a new active anchor with the merged Gaussians
                    self.active_anchor = Anchor(
                        self.gaussian_params,
                        self.approx_cam_centres[-1],
                        self.keyframes[-self.n_kept_frames :],
                    )
                    self.active_anchor.gaussian_owner_keyframe_ids = self.gaussian_owner_keyframe_ids
                    self.anchors.append(self.active_anchor)
                    self.active_frames_gpu = [kf.index for kf in self.active_anchor.keyframes]
                    self.active_frames_cpu = []

                    # Visualization
                    self.anchor_weights = np.zeros(len(self.anchors))
                    self.anchor_weights[-1] = 1.0

                gc.collect()
                torch.cuda.empty_cache()

    def save(self, path: str, reconstruction_time: float = 0, n_frames: int = 0):
        # Get metrics
        metrics = {
            "num anchors": len(self.anchors),
            "num keyframes": len(self.keyframes),
        }
        if reconstruction_time > 0:
            metrics["time"] = reconstruction_time
            if n_frames > 0:
                metrics["FPS"] = n_frames / reconstruction_time
        metrics.update(self.evaluate(True, True, True))

        if path == "":
            print("No path provided, skipping save")
            return metrics

        # Save anchors
        pcd_path = os.path.join(path, "point_clouds")
        os.makedirs(pcd_path, exist_ok=True)
        for index, anchor in enumerate(self.anchors):
            anchor.save_ply(os.path.join(pcd_path, f"anchor_{index}.ply"))

        # Save metadata
        metadata = {
            "config": {
                "width": self.width,
                "height": self.height,
                "sh_degree": self.max_sh_degree,
                "f": self.f.detach().cpu().tolist(),
                "K": self.K.detach().cpu().tolist(),
            },
            "anchors": [
                {
                    "position": anchor.position.cpu().numpy().tolist(),
                }
                for anchor in self.anchors
            ],
            "keyframes": [keyframe.to_json() for keyframe in self.keyframes],
        }
        metadata = {**metrics, **metadata}

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        # Save renders of test views
        self.save_test_frames(os.path.join(path, "test_images"))

        # Saving cameras with COLMAP format
        images = {}
        cameras = {}
        colmap_save_path = os.path.join(path, "colmap")
        os.makedirs(colmap_save_path, exist_ok=True)
        for index, keyframe in enumerate(self.keyframes):
            camera, image = keyframe.to_colmap(index)
            cameras[index] = camera
            images[index] = image
        write_model(cameras, images, {}, colmap_save_path, ext=".bin")

        return metrics

    def get_closest_keyframe(
        self, position: torch.Tensor, count: int = 1
    ) -> list[Keyframe]:
        dists = torch.linalg.vector_norm(
            self.approx_cam_centres - position[None], dim=-1
        )
        closest_ids = dists.argsort()[:count]
        return [self.keyframes[closest_id] for closest_id in closest_ids]

    def finetune_epoch(self):
        """
        Go through all anchors and optimize them one by one.
        This is used for finetuning after the initial training.
        """
        self.anchor_weights = np.zeros(len(self.anchors))
        for anchor_id, anchor in enumerate(self.anchors):
            self.active_anchor = anchor
            # Load the anchor and make its parameters optimizable
            anchor.to("cuda", with_keyframes=True)
            self.gaussian_params = anchor.gaussian_params
            self.anchor_weights[anchor_id] = 1
            self.reset_optimizer()

            # # Ensure other anchors are on cpu to save memory
            # if anchor_id >= 1:
            #     self.anchors[anchor_id-1].to("cpu", with_keyframes=True)

            # Optimize the anchor by going through its keyframes
            for _ in range(len(anchor.keyframes)):
                self.optimization_step(finetuning=True)

            # Update the anchor and store it on cpu
            anchor.gaussian_params = self.gaussian_params
            self.anchor_weights[anchor_id] = 0
