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

from poses.clidd_adapter import clidd_descriptor_dim, default_clidd_root, load_clidd_model

class DescribedKeypoints():
    """
    A class to store 2D keypoints, their descriptors, their matches to other images, and their estimated 3D positions.
    """
    @torch.no_grad()
    def __init__(self, kpts, feats):
        self.kpts = kpts
        self.feats = feats
        self.valid = feats.abs().sum(dim=-1) > 0
        self.nvalid = self.valid.sum()
        self.has_pt3d = torch.zeros(kpts.shape[0], dtype=torch.bool, device=kpts.device)
        self.pts_conf = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device)
        self.pts3d = torch.zeros(kpts.shape[0], 3, dtype=torch.float, device=kpts.device)
        self.depth = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device)
        self.matches = {}

    @torch.no_grad()
    def update_matches(self, other_id, matches, swap=False):
        if swap:
            from poses.matcher import Matches
            matches = Matches(matches.kpts_other, matches.kpts, matches.idx_other, matches.idx)
        self.matches[other_id] = matches
 
    @torch.no_grad()
    def update_3D_pts(self, pts3D, depth, conf, idx):
        self.has_pt3d[idx] = True
        self.depth[idx] = depth
        self.pts3d[idx] = pts3D
        self.pts_conf[idx] = conf

    @torch.no_grad()
    def to(self, device):
        self.kpts = self.kpts.to(device)
        self.feats = self.feats.to(device)
        self.valid = self.valid.to(device)
        self.has_pt3d = self.has_pt3d.to(device)
        self.pts_conf = self.pts_conf.to(device)
        self.pts3d = self.pts3d.to(device)
        self.depth = self.depth.to(device)
        for key in self.matches:
            self.matches[key].kpts = self.matches[key].kpts.to(device)
            self.matches[key].kpts_other = self.matches[key].kpts_other.to(device)
            self.matches[key].idx = self.matches[key].idx.to(device)
            self.matches[key].idx_other = self.matches[key].idx_other.to(device)

class Detector():
    @torch.no_grad()
    def __init__(self, top_k, width, height, args=None):
        self.top_k = int(top_k)
        self.width = int(width)
        self.height = int(height)
        self.model_name = getattr(args, "clidd_model", "S64")
        self.desc_dim = clidd_descriptor_dim(self.model_name)
        self.device = torch.device(getattr(args, "device", "cuda"))
        self.extractor = load_clidd_model(
            root=getattr(args, "clidd_root", default_clidd_root()),
            model_name=self.model_name,
            top_k=self.top_k,
            radius=int(getattr(args, "clidd_radius", 2)),
            score=float(getattr(args, "clidd_score", -5.0)),
            device=self.device,
        )

    def _pad_or_truncate(self, kpts, feats):
        kpts = kpts.to(device=self.device, dtype=torch.float32)
        feats = feats.to(device=self.device, dtype=torch.float32)
        if feats.numel() == 0:
            feats = torch.empty((0, self.desc_dim), device=self.device, dtype=torch.float32)
        count = min(kpts.shape[0], self.top_k)
        padded_kpts = torch.zeros((self.top_k, 2), device=self.device, dtype=torch.float32)
        padded_feats = torch.zeros((self.top_k, self.desc_dim), device=self.device, dtype=torch.float32)
        if count > 0:
            padded_kpts[:count] = kpts[:count]
            padded_feats[:count] = feats[:count]
        return padded_kpts, padded_feats

    @torch.no_grad()
    def __call__(self, image):
        output = self.extractor(image[None].to(self.device).float())[0]
        kpts, feats = self._pad_or_truncate(output["keypoints"], output["descriptors"])
        return DescribedKeypoints(kpts, feats)
    
