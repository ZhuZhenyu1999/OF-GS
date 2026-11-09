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


class Matches:
    """
    A class to store matched keypoints and their indices between two sets of keypoints.
    """
    def __init__(self, kpts, kpts_other, idx, idx_other):
        self.kpts = kpts
        self.kpts_other = kpts_other
        self.idx = idx
        self.idx_other = idx_other


def match(feats1, feats2, beta=20.0):
    valid1 = feats1.abs().sum(dim=-1) > 0
    valid2 = feats2.abs().sum(dim=-1) > 0
    ids1 = torch.where(valid1)[0]
    ids2 = torch.where(valid2)[0]
    if ids1.numel() == 0 or ids2.numel() == 0:
        empty = torch.empty(0, device=feats1.device, dtype=torch.long)
        return empty, empty, torch.empty(0, device=feats1.device, dtype=torch.bool)

    desc1 = feats1[ids1].float()
    desc2 = feats2[ids2].float()
    scores = desc1 @ desc2.t()
    scores = (scores - 1.0).mul(float(beta)).exp()
    scores = scores.square().div(scores.sum(dim=-1, keepdim=True).clamp_min(1e-12))
    scores = scores.div(scores.sum(dim=-2, keepdim=True).clamp_min(1e-12))

    local_idx1 = torch.arange(scores.shape[0], device=scores.device)
    local_match12 = scores.max(dim=1).indices
    local_match21 = scores.max(dim=0).indices
    local_mask = (local_match21[local_match12] == local_idx1)
    local_mask &= scores[local_idx1, local_match12] > 0.01

    return ids1[local_idx1], ids2[local_match12], local_mask


class Matcher:
    @torch.no_grad()
    def __init__(self, fundmat_samples: int, max_error: float, match_beta: float = 20.0):
        """
        Initialize the Matcher.
        Args:
            fundmat_samples (int): Number of RANSAC etimations when estimating inliers with fundamental matrix estimation.
            max_error (float): Maximum error for RANSAC inlier threshold.
        """
        self.max_error = max_error
        self.match_beta = float(match_beta)
        from poses.ransac import EstimatorType, RANSACEstimator
        self.fundmat_estimator = RANSACEstimator(
            fundmat_samples, max_error, EstimatorType.FUNDAMENTAL_8PTS
        )

    def evaluate_match(
        self, desc_kpts: 'DescribedKeypoints', desc_kpts_other: 'DescribedKeypoints'
    ):
        """
        Get the number of matches between two sets of described keypoints.
        """
        _, _, mask = match(desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda(), self.match_beta)
        return mask.sum()

    @torch.no_grad()
    def __call__(
        self,
        desc_kpts: 'DescribedKeypoints',
        desc_kpts_other: 'DescribedKeypoints',
        remove_outliers: bool = False,
        update_kpts_flag: str = "",
        kID: int = -1,
        kID_other: int = -1,
    ):
        """
        Matches keypoints between two sets of described keypoints, with optional outlier removal based on the fundamental RANSAC estimation.
        Args:
            desc_kpts (DescribedKeypoints): Keypoints and descriptors of the first image.
            desc_kpts_other (DescribedKeypoints): Keypoints and descriptors of the second image.
            remove_outliers (bool): Whether to remove outliers using the fundamental matrix.
            update_kpts_flag (str): If "all", updates all matches; if "inliers", updates only inliers.
            kID (int): ID of the first set of keypoints, used for updating matches.
            kID_other (int): ID of the second set of keypoints, used for updating matches.
        Returns:
            Matches: A Matches object containing the matched keypoints and their indices.
        """
        idx, idx_other, mask = match(
            desc_kpts.feats.cuda(), desc_kpts_other.feats.cuda(), self.match_beta
        )
        idx = idx[mask]
        idx_other = idx_other[mask]
        kpts = desc_kpts.kpts[idx]
        kpts_other = desc_kpts_other.kpts[idx_other]
        idx_all = idx
        idx_other_all = idx_other
        kpts_all = kpts
        kpts_other_all = kpts_other

        if remove_outliers:
            F, mask = self.fundmat_estimator(kpts, kpts_other)
            idx = idx[mask]
            idx_other = idx_other[mask]
            kpts = kpts[mask]
            kpts_other = kpts_other[mask]

        if update_kpts_flag == "all":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts_all, kpts_other_all, idx_all, idx_other_all)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other_all, kpts_all, idx_other_all, idx_all)
            )
        elif update_kpts_flag == "inliers":
            assert kID >= 0 and kID_other >= 0
            desc_kpts.update_matches(
                kID_other, Matches(kpts, kpts_other, idx, idx_other)
            )
            desc_kpts_other.update_matches(
                kID, Matches(kpts_other, kpts, idx_other, idx)
            )

        return Matches(kpts, kpts_other, idx, idx_other)
