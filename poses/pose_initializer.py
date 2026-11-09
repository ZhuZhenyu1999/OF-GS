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

import torch
import math

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from utils import anchor_poses_to_first_frame, fov2focal, depth2points, sixD2mtx, intrinsics_from_info
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType

class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        if args.init_focal > 0:
            fallback_f = args.init_focal
        elif args.init_fov > 0:
            fallback_f = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            fallback_f = 0.7 * width

        initial_K = getattr(args, "init_intrinsics", None)
        self.calibrated = initial_K is not None
        self.set_intrinsics(intrinsics_from_info(
            {"K": initial_K} if self.calibrated else {}, fallback_f, width, height, device="cuda"
        ))
        self.f_init = self.f.clone()

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap, not args.fix_focal and not self.calibrated, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

    def set_intrinsics(self, K):
        self.K = K.detach().clone()
        self.f = self.K.diagonal()[:2].clone()
        self.centre = self.K[:2, 2].clone()
        self.intrinsics = torch.cat([self.f, self.centre])

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(
        self,
        desc_kpts_list: list[DescribedKeypoints],
        rebooting=False,
        infos=None,
    ):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list. 
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        fallback_f = self.f if rebooting else self.f_init
        if infos is not None:
            if len(infos) != n_cams:
                raise ValueError("Bootstrap intrinsics must match the number of cameras")
            Ks = torch.stack([
                intrinsics_from_info(info, fallback_f, self.width, self.height, device="cuda")
                for info in infos
            ])
            f_init = Ks.diagonal(dim1=-2, dim2=-1)[:, :2].contiguous()
            centres = Ks[:, :2, 2].contiguous()
            calibrated = any(info.get("intrinsics_calibrated", "K" in info or "K_np" in info) for info in infos)
        else:
            f_init = fallback_f[None].expand(n_cams, 2).clone()
            centres = self.centre[None].expand(n_cams, 2).clone()
            calibrated = self.calibrated
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init[k], centres[k])
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting or calibrated:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, f_init, xyz_init, centres, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, centres, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        if infos is not None:
            for k, info in enumerate(infos):
                if not info.get("intrinsics_calibrated", "K" in info or "K_np" in info):
                    estimated_K = torch.eye(3, device=f.device, dtype=f.dtype)
                    estimated_K[0, 0], estimated_K[1, 1] = f[k, 0], f[k, 1]
                    estimated_K[:2, 2] = centres[k]
                    info["estimated_K"] = estimated_K.detach().clone()
        K = self.K.clone()
        K[0, 0], K[1, 1] = f[-1, 0], f[-1, 1]
        K[:2, 2] = centres[-1]
        self.set_intrinsics(K)

        ## Scale to 0.1 average translation
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts
        Rts = anchor_poses_to_first_frame(Rts)

        return Rts, self.f, final_residual

    @torch.no_grad()
    def initialize_incremental(
        self,
        keyframes: list[Keyframe],
        curr_desc_kpts: DescribedKeypoints,
        index: int,
        is_test: bool,
        curr_img,
        update_matches: bool = True,
    ):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        
        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(
                curr_desc_kpts,
                keyframe.desc_kpts,
                remove_outliers=True,
                update_kpts_flag="all" if update_matches else "",
                kID=index,
                kID_other=keyframe.index,
            )

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)
        if len(xyz) < 4:
            if update_matches:
                for keyframe in keyframes:
                    keyframe.desc_kpts.matches.pop(index, None)
            print("Too few 3D-2D correspondences for pose initialization")
            return None

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            print("Too few inliers for pose initialization")
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None
