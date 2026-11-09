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

from __future__ import annotations
from argparse import Namespace
import torch
import torch.nn.functional as F

from poses.feature_detector import DescribedKeypoints
from poses.triangulator import Triangulator
from scene.dense_extractor import DenseExtractor
from scene.mono_depth import MonoDepthEstimator, align_depth
from scene.optimizers import BaseAdam
from utils import sample, sixD2mtx, make_torch_sampler, depth2points, intrinsics_from_info, resize_intrinsics
from dataloaders.read_write_model import Camera, BaseImage, rotmat2qvec


class Keyframe:
    """
    A keyframe in the scene, containing the image, camera parameters, and other information used for optimization.
    """
    def __init__(
        self,
        image: torch.Tensor,
        info: dict,
        desc_kpts: DescribedKeypoints,
        Rt: torch.Tensor,
        index: int,
        f: torch.Tensor,
        feat_extractor: DenseExtractor,
        depth_estimator: MonoDepthEstimator,
        triangulator: Triangulator,
        args: Namespace,
        inference_mode: bool = False,
    ):
        self.image_pyr = [image]
        self.width = image.shape[2] if image is not None else info["width"]
        self.height = image.shape[1] if image is not None else info["height"]
        self.K = intrinsics_from_info(
            info, f, self.width, self.height, device=Rt.device, dtype=Rt.dtype
        )
        self.f = self.K[[0, 1], [0, 1]].clone()
        self.centre = self.K[:2, 2].clone()
        self.mask_pyr = None
        self.pyr_lvl = 0
        if not inference_mode: # Only extract depth and feature maps in training mode
            self.triangulator = triangulator
            self.feat_map = feat_extractor(image)
            self.mono_idepth, self.mono_depth_conf = depth_estimator(image)
            self.depth_loss_weight = args.depth_loss_weight_init
            self.depth_loss_weight_decay = args.depth_loss_weight_decay
            # Build the multiscale pyramids
            for _ in range(args.pyr_levels - 1):
                self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
            self.mask_pyr = info.pop("mask", None)
            self.pyr_lvl = args.pyr_levels - 1
            if self.mask_pyr is not None:
                self.mask_pyr = [self.mask_pyr.to(image.device)]
                for _ in range(args.pyr_levels - 1):
                    self.mask_pyr.append(F.avg_pool2d(self.mask_pyr[-1], 2))
                for i in range(len(self.mask_pyr)):
                    self.mask_pyr[i] = self.mask_pyr[i] > (1 - 1e-6)

        self.index = index

        self.idepth_pyr = None

        self.latest_invdepth = None
        self.desc_kpts = desc_kpts
        self.info = info
        self.is_test = info["is_test"]

        # Optimizable parameters
        self.rW2C = torch.nn.Parameter(Rt[:3, :2].clone())
        self.tW2C = torch.nn.Parameter(Rt[:3, 3].clone())
        exposure = torch.eye(3, 4, device=Rt.device, dtype=Rt.dtype)
        self.exposure = torch.nn.Parameter(exposure)
        self.depth_scale = torch.nn.Parameter(torch.ones(1, device=Rt.device, dtype=Rt.dtype))
        self.depth_offset = torch.nn.Parameter(torch.zeros(1, device=Rt.device, dtype=Rt.dtype))

        # Optimizer
        if not inference_mode: # Only create optimizer in training mode
            params = {
                "rW2C": {"val": self.rW2C, "lr": args.lr_poses},
                "tW2C": {"val": self.tW2C, "lr": args.lr_poses},
                "depth_scale": {
                    "val": self.depth_scale,
                    "lr": args.lr_depth_scale_offset,
                },
                "depth_offset": {
                    "val": self.depth_offset,
                    "lr": args.lr_depth_scale_offset,
                },
            }
            if not info["is_test"]:
                params["exposure"] = {"val": self.exposure, "lr": args.lr_exposure}
            self.optimizer = BaseAdam(params, betas=(0.8, 0.99))
            self.num_steps = 0

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def _make_idepth_pyramid(self, mono_idepth):
        idepth_pyr = [
            F.interpolate(
                mono_idepth,
                (self.height, self.width),
                mode="bilinear",
                align_corners=True,
            )[0]
        ]
        for _ in range(len(self.image_pyr) - 1):
            idepth_pyr.append(F.avg_pool2d(idepth_pyr[-1], 2))
        return idepth_pyr

    def to(self, device: str, only_train=False):
        if self.device.type == device:
            return
        for i in range(len(self.image_pyr)):
            self.image_pyr[i] = self.image_pyr[i].to(device)
            if self.idepth_pyr is not None:
                self.idepth_pyr[i] = self.idepth_pyr[i].to(device)
            if self.mask_pyr is not None:
                self.mask_pyr[i] = self.mask_pyr[i].to(device)
        if not only_train:
            self.feat_map = self.feat_map.to(device)
            self.mono_idepth = self.mono_idepth.to(device)
            self.mono_depth_conf = self.mono_depth_conf.to(device)
            if self.latest_invdepth is not None:
                self.latest_invdepth = self.latest_invdepth.to(device)

    @property
    def device(self):
        return self.image_pyr[0].device if self.image_pyr[0] is not None else self.f.device

    def get_intrinsics(self, width=None, height=None):
        """Return K for this frame, respecting pixel centers when resizing."""
        K = self.K.clone()
        K[0, 0], K[1, 1] = self.f[0], self.f[1]
        K[:2, 2] = self.centre
        if width is not None or height is not None:
            if width is None or height is None:
                raise ValueError("Both width and height are required when resizing intrinsics")
            K = resize_intrinsics(K, width / self.width, height / self.height)
        return K

    def get_R(self):
        return sixD2mtx(self.rW2C)

    def get_t(self):
        return self.tW2C

    def get_Rt(self):
        Rt = torch.eye(4, device=self.tW2C.device, dtype=self.tW2C.dtype)
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def set_Rt(self, Rt: torch.Tensor):
        self.rW2C.data.copy_(Rt[:3, :2])
        self.tW2C.data.copy_(Rt[:3, 3])

        self.approx_centre = -Rt[:3, :3].T @ Rt[:3, 3]

    def get_centre(self, approx=False):
        if approx:
            return self.approx_centre
        else:
            return -self.get_R().T @ self.get_t()

    @torch.no_grad()
    def update_3dpts(self, all_keyframes: list[Keyframe]):
        """
        Assign a 3D point to each keypoint in the keyframe based on triangulation and the latest rendered depth. 
        """
        unload_desc_kpts = self.desc_kpts.kpts.device.type == "cpu"
        if unload_desc_kpts:
            self.desc_kpts.to("cuda")

        ## Update 3D points using the latest rendered depth
        if self.latest_invdepth is not None:
            uv = self.desc_kpts.kpts
            sampler = make_torch_sampler(uv.view(1, 1, -1, 2), self.width, self.height)
            model_idepth = F.grid_sample(
                self.latest_invdepth[None].cuda(),
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            mono_idepth = F.grid_sample(
                self.get_mono_idepth()[None],
                sampler,
                mode="bilinear",
                align_corners=True,
            )[0, 0, 0]
            mono_conf = F.grid_sample(
                self.mono_depth_conf, sampler, mode="bilinear", align_corners=True
            )[0, 0, 0]
            mono_model_diff = (model_idepth - mono_idepth) ** 2
            var = 0.2
            conf = 0.1 * torch.exp(-mono_model_diff / var) * mono_conf
            depth = 1 / model_idepth.clamp(1e-6, 1e6)
            new_pts = depth2points(uv, depth[..., None], self.f, self.centre)
            new_pts = (new_pts - self.get_t()) @ self.get_R()
            mask = conf > 0
            self.desc_kpts.update_3D_pts(new_pts[mask], depth[mask], conf[mask], mask)

        ## Triangulation
        # Select keyframes to triangulate with based on their locations
        uv, uvs_others, chosen_kfs_ids = self.triangulator.prepare_matches(
            self.desc_kpts
        )
        if len(chosen_kfs_ids) == 0:
            if unload_desc_kpts:
                self.desc_kpts.to("cpu")
            return
        selected_keyframes = [all_keyframes[index] for index in chosen_kfs_ids]
        Rts_others = torch.stack([kf.get_Rt() for kf in selected_keyframes])
        other_focals = torch.stack([kf.f for kf in selected_keyframes])
        other_centres = torch.stack([kf.centre for kf in selected_keyframes])
        if len(Rts_others) < self.triangulator.n_cams:
            padding = self.triangulator.n_cams - len(Rts_others)
            Rts_others = torch.cat(
                [
                    Rts_others,
                    torch.eye(4, device=Rts_others.device)[None].repeat(padding, 1, 1),
                ],
                dim=0,
            )
            other_focals = torch.cat([other_focals, self.f[None].expand(padding, -1)])
            other_centres = torch.cat([other_centres, self.centre[None].expand(padding, -1)])

        # Run the triangulator and update the 3D points
        new_pts, depth, best_dis, valid_matches = self.triangulator(
            uv, uvs_others, self.get_Rt(), Rts_others, self.f, self.centre,
            other_focals=other_focals, other_centres=other_centres,
        )
        self.desc_kpts.update_3D_pts(
            new_pts[valid_matches], depth[valid_matches], 1, valid_matches
        )

        if unload_desc_kpts:
            self.desc_kpts.to("cpu")

    def get_mono_idepth(self, lvl=0):
        if self.idepth_pyr[lvl].device.type != "cuda":
            self.idepth_pyr[lvl] = self.idepth_pyr[lvl].cuda()
        return self.idepth_pyr[lvl] * self.depth_scale + self.depth_offset

    @torch.no_grad()
    def align_depth(self):
        """
        Align the mono depth to the triangulated depth of the keypoints.
        update_3dpts must have been called before this function.
        """
        if (self.desc_kpts.pts_conf > 0).any():
            self.mono_idepth = align_depth(
                self.mono_idepth, self.desc_kpts, self.width, self.height
            )
        self.idepth_pyr = self._make_idepth_pyramid(self.mono_idepth)


    @torch.no_grad()
    def sample_conf(self, uv):
        return sample(
            self.mono_depth_conf, uv.view(1, 1, -1, 2), self.width, self.height
        )[0, 0, 0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    @torch.no_grad()
    def step(self, optimize_pose=True, optimize_only_pose=False):
        saved_states = {}
        if not optimize_pose:
            for name in ("rW2C", "tW2C"):
                if name in self.optimizer.params:
                    saved_states[name] = self.optimizer.params[name]["val"].detach().clone()
                    self.optimizer.params[name]["val"].grad = None
        if optimize_only_pose:
            for name, param in self.optimizer.params.items():
                if name not in ("rW2C", "tW2C"):
                    saved_states[name] = param["val"].detach().clone()
                    param["val"].grad = None

        self.optimizer.step()
        for name, state in saved_states.items():
            self.optimizer.params[name]["val"].data.copy_(state)

    @torch.no_grad()
    def advance_mapping_schedule(self):
        """Advance loss weights and resolution without stepping any parameters."""
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        # decrement pyr_lvl
        if self.num_steps % 5 == 0:
            if self.pyr_lvl > 0:
                self.image_pyr.pop()
                if self.mask_pyr is not None:
                    self.mask_pyr.pop()
                if self.idepth_pyr is not None:
                    self.idepth_pyr.pop()
                self.pyr_lvl -= 1

    def to_json(self):
        info = {
            "is_test": self.info["is_test"],
        }
        if "name" in self.info:
            info["name"] = self.info["name"]
        if "Rt" in self.info:
            info["gt_Rt"] = self.info["Rt"].cpu().numpy().tolist()

        return {
            "info": info,
            "Rt": self.get_Rt().detach().cpu().numpy().tolist(),
            "f": self.f.detach().cpu().tolist(),
            "K": self.get_intrinsics().detach().cpu().tolist(),
        }

    @classmethod
    def from_json(cls, config, index, height, width, fallback_f=None, device="cuda"):
        info = {**config["info"], "width": width, "height": height}
        if "gt_Rt" in info:
            info["Rt"] = torch.tensor(info["gt_Rt"], device=device)
        if "K" in config:
            info["K"] = config["K"]
        keyframe = cls(
            image=None,
            info=info,
            desc_kpts=None,
            Rt=torch.tensor(config["Rt"], device=device),
            index=index,
            f=config.get("f", fallback_f),
            feat_extractor=None,
            depth_estimator=None,
            triangulator=None,
            args=None,
            inference_mode=True,
        )
        return keyframe

    def to_colmap(self, id):
        """
        Convert the keyframe to a colmap camera and image.
        """
        camera = Camera(
            id=id,
            model="PINHOLE",
            width=self.width,
            height=self.height,
            params=[*self.f.detach().cpu().tolist(), *self.centre.detach().cpu().tolist()],
        )

        image = BaseImage(
            id=id,
            name=self.info.get("name", str(id)),
            camera_id=id,
            qvec=-rotmat2qvec(self.get_R().cpu().detach().numpy()),
            tvec=self.get_t().flatten().cpu().detach().numpy(),
            xys=[],
            point3D_ids=[],
        )

        return camera, image
