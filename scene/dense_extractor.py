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
import torch.nn.functional as F

from poses.clidd_adapter import clidd_descriptor_dim, default_clidd_root, load_clidd_model


class DenseExtractor():
    """Extract dense CLIDD descriptor maps from an image."""
    @torch.no_grad()
    def __init__(self, width, height, args=None):
        self.width = int(width)
        self.height = int(height)
        self.model_name = getattr(args, "clidd_model", "S64")
        self.desc_dim = clidd_descriptor_dim(self.model_name)
        self.stride = int(getattr(args, "clidd_dense_stride", 8))
        if self.stride <= 0:
            raise ValueError("--clidd_dense_stride must be positive")
        self.device = torch.device(getattr(args, "device", "cuda"))
        self.extractor = load_clidd_model(
            root=getattr(args, "clidd_root", default_clidd_root()),
            model_name=self.model_name,
            top_k=None,
            radius=int(getattr(args, "clidd_radius", 2)),
            score=float(getattr(args, "clidd_score", -5.0)),
            device=self.device,
        )

    def _preprocess(self, image):
        x = image[None].to(self.device).float()
        _, _, height, width = x.shape
        resized_height = (height // 32) * 32
        resized_width = (width // 32) * 32
        if resized_height <= 0 or resized_width <= 0:
            raise ValueError(f"CLIDD input image is too small: {height}x{width}")
        if resized_height != height or resized_width != width:
            x = F.interpolate(x, (resized_height, resized_width), mode="bilinear", align_corners=True)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x, resized_height, resized_width

    def _dense_grid(self, height, width):
        grid_height = max(height // self.stride, 1)
        grid_width = max(width // self.stride, 1)
        y = (torch.arange(grid_height, device=self.device, dtype=torch.float32) + 0.5) * self.stride - 0.5
        x = (torch.arange(grid_width, device=self.device, dtype=torch.float32) + 0.5) * self.stride - 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kpts = torch.stack([xx, yy], dim=-1).reshape(1, -1, 1, 2)
        size = torch.tensor([width, height], dtype=torch.float32, device=self.device)
        return kpts.add(0.5).div(size).mul(2.0).sub(1.0), grid_height, grid_width

    @torch.no_grad()
    def __call__(self, image):
        x, height, width = self._preprocess(image)
        raw_desc, _ = self.extractor.model(x)
        grid, grid_height, grid_width = self._dense_grid(height, width)
        desc = self.extractor.model.sample([item for item in raw_desc], grid)[0]
        return desc.reshape(grid_height, grid_width, self.desc_dim).to(torch.float16).contiguous()
