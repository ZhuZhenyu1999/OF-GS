"""Covisibility windows and shared feature tracks for online local BA.

Only measured keypoints, stored matches and estimated poses enter this module.
Dataset poses/depth are deliberately not part of its interface.
"""
from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import math

import torch

from args import validate_local_ba_window
from poses.local_ba_solver import solve_local_ba


def _training_ids(keyframes):
    return [i for i, kf in enumerate(keyframes)
            if getattr(kf, "desc_kpts", None) is not None
            and not kf.info.get("is_test", False)]


def covisibility_graph(keyframes):
    """Stored pair matches nominate neighbours; track geometry is checked later."""
    eligible = set(_training_ids(keyframes))
    graph = {i: {} for i in eligible}
    for i in sorted(eligible):
        for j, matches in keyframes[i].desc_kpts.matches.items():
            if j not in eligible or i == j:
                continue
            count = int(matches.idx.numel())
            graph[i][j] = max(count, graph[i].get(j, 0))
            graph[j][i] = max(count, graph[j].get(i, 0))
    return graph


def select_local_window(keyframes, newest_id, *, window_size=10, recent_keyframes=5,
                        fixed_keyframes=2, min_shared_points=30):
    """Protect recent poses and include older fixed anchors in the total cap."""
    validate_local_ba_window(window_size, recent_keyframes, fixed_keyframes)
    graph = covisibility_graph(keyframes)
    if newest_id not in graph:
        return [], [], "test_or_missing_keypoints"
    eligible = sorted(i for i in graph if i <= newest_id)
    if len(eligible) < recent_keyframes:
        return [], [], "insufficient_recent_keyframes"
    recent = eligible[-recent_keyframes:]
    selected = list(recent)
    while len(selected) < window_size:
        candidates = []
        for i in eligible:
            if i in selected:
                continue
            connections = [graph[i].get(j, 0) for j in selected]
            if max(connections, default=0) >= min_shared_points:
                candidates.append((sum(connections), i))
        if not candidates:
            break
        selected.append(max(candidates)[1])
    # Only older frames may anchor the window; all recent poses stay variable.
    # A separated second anchor fixes monocular scale.
    anchor_pool = [i for i in sorted(selected) if i not in recent]
    if len(anchor_pool) < fixed_keyframes:
        return [], [], "insufficient_fixed_keyframes"
    first = anchor_pool[0]
    centres = {i: torch.linalg.inv(keyframes[i].get_Rt().detach().cpu().double())[:3, 3]
               for i in selected}
    fixed = [first]
    while len(fixed) < fixed_keyframes:
        remaining = [i for i in anchor_pool if i not in fixed]
        if not remaining:
            return [], [], "insufficient_fixed_keyframes"
        chosen = max(remaining, key=lambda i: (min(float(torch.linalg.vector_norm(
            centres[i] - centres[j])) for j in fixed), -i))
        fixed.append(chosen)
    baseline = max(float(torch.linalg.vector_norm(centres[i] - centres[j]))
                   for i, j in combinations(fixed, 2))
    if baseline <= 1e-6:
        return [], [], "degenerate_anchor_baseline"
    variable = sorted(i for i in selected if i not in fixed)
    return variable, sorted(fixed), None


class _Tracks:
    """Conflict-aware union: a track contains at most one feature per camera."""
    def __init__(self):
        self.parent = {}
        self.members = {}

    def find(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node[0]: node[1]}
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            old = self.parent[node]
            self.parent[node] = root
            node = old
        return root

    def join(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.members[ra].keys() & self.members[rb].keys():
            return
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))


def _spatial_order(tracks, keyframes, keypoint_arrays):
    bins = defaultdict(deque)
    for observations in sorted(tracks, key=lambda obs: (-len(obs), tuple(obs))):
        kf_id, feature_id = max(observations)
        kf = keyframes[kf_id]
        uv = keypoint_arrays[kf_id][feature_id]
        width, height = getattr(kf, "width", 1500), getattr(kf, "height", 1000)
        cell = (min(7, max(0, int(float(uv[0]) * 8 / width))),
                min(5, max(0, int(float(uv[1]) * 6 / height))))
        bins[cell].append(observations)
    ordered = []
    while bins:
        for cell in sorted(list(bins)):
            ordered.append(bins[cell].popleft())
            if not bins[cell]:
                del bins[cell]
    return ordered


def _triangulate(uv, mask, Rts, Ks):
    homogeneous = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)
    rays = torch.einsum("cij,ncj->nci", torch.linalg.inv(Ks), homogeneous)
    P = Rts[:, :3, :]
    rows_u = rays[..., 0, None] * P[None, :, 2] - P[None, :, 0]
    rows_v = rays[..., 1, None] * P[None, :, 2] - P[None, :, 1]
    A = torch.stack([rows_u, rows_v], dim=2) * mask[..., None, None]
    _, _, vh = torch.linalg.svd(A.flatten(1, 2), full_matrices=False)
    xyz_h = vh[:, -1]
    denom = xyz_h[:, 3:]
    denom = torch.where(denom.abs() > 1e-12, denom, torch.full_like(denom, 1e-12))
    return xyz_h[:, :3] / denom


def _project(points, Rts, Ks):
    cam = torch.einsum("cij,nj->nci", Rts[:, :3, :3], points) + Rts[None, :, :3, 3]
    px = torch.einsum("cij,ncj->nci", Ks, cam)
    return px[..., :2] / cam[..., 2:].clamp_min(1e-9), cam[..., 2]


@torch.no_grad()
def build_shared_tracks(keyframes, variable_ids, fixed_ids, *, max_points=1500,
                        max_reprojection_error=8.0, min_parallax_deg=1.0):
    ids = list(variable_ids) + list(fixed_ids)
    selected = set(ids)
    # Snapshot once per frame, rather than synchronizing CUDA for each feature.
    data = {}
    for kf_id in ids:
        desc = keyframes[kf_id].desc_kpts
        masks = getattr(keyframes[kf_id], "mask_pyr", None)
        data[kf_id] = {
            "uv": desc.kpts.detach().cpu().double(),
            "has_point": desc.has_pt3d.detach().cpu(),
            "confidence": desc.pts_conf.detach().cpu(),
            "points": desc.pts3d.detach().cpu().double(),
            "mask": masks[0].detach().cpu().squeeze() if masks else None,
        }
    tracks = _Tracks()
    seen = set()
    # Stronger pair links take precedence when inconsistent matches collide.
    edges = []
    for i in ids:
        for j, matches in keyframes[i].desc_kpts.matches.items():
            pair = tuple(sorted((i, j)))
            if j in selected and i != j and pair not in seen:
                seen.add(pair)
                edges.append((int(matches.idx.numel()), i, j, matches))
    for _, i, j, matches in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
        ni, nj = len(keyframes[i].desc_kpts.kpts), len(keyframes[j].desc_kpts.kpts)
        for a, b in zip(matches.idx.detach().cpu().tolist(), matches.idx_other.detach().cpu().tolist()):
            if 0 <= a < ni and 0 <= b < nj:
                tracks.join((i, a), (j, b))
    shared = [sorted(members.items()) for members in tracks.members.values()
              if len(members) >= 3 and any(i in variable_ids for i in members)]
    shared = _spatial_order(shared, keyframes, {i: d["uv"] for i, d in data.items()})[:max_points * 3]
    Rts = torch.stack([keyframes[i].get_Rt().detach().cpu().double() for i in ids])
    Ks = torch.stack([keyframes[i].K.detach().cpu().double() for i in ids])
    uv = torch.full((len(shared), len(ids), 2), -1.0, dtype=torch.float64)
    seeds = torch.full((len(shared), len(ids), 3), float("nan"), dtype=torch.float64)
    columns = {kf_id: c for c, kf_id in enumerate(ids)}
    for n, observations in enumerate(shared):
        for kf_id, feature_id in observations:
            c = columns[kf_id]
            point_uv = data[kf_id]["uv"][feature_id]
            if not torch.isfinite(point_uv).all() or (point_uv < 0).any():
                continue
            kf = keyframes[kf_id]
            if float(point_uv[0]) >= getattr(kf, "width", float("inf")) or float(point_uv[1]) >= getattr(kf, "height", float("inf")):
                continue
            mask0 = data[kf_id]["mask"]
            if mask0 is not None:
                x, y = point_uv.round().long().tolist()
                if x >= mask0.shape[-1] or y >= mask0.shape[-2] or not bool(mask0[y, x]):
                    continue
            uv[n, c] = point_uv
            if bool(data[kf_id]["has_point"][feature_id]) and float(data[kf_id]["confidence"][feature_id]) > 0:
                seeds[n, c] = data[kf_id]["points"][feature_id]
    valid = (uv >= 0).all(-1)
    if not shared:
        return {"ids": ids, "Rts": Rts, "intrinsics": Ks, "points": seeds[:, 0],
                "observations": uv, "tracks": [], "reason": "no_shared_tracks"}
    xyz = _triangulate(uv, valid, Rts, Ks)
    # Existing triangulated landmarks rescue a track containing a bad pair match.
    candidates = torch.cat([xyz[:, None], seeds], dim=1)
    cam = torch.einsum("cij,nsj->nsci", Rts[:, :3, :3], candidates) + Rts[None, None, :, :3, 3]
    px = torch.einsum("cij,nscj->nsci", Ks, cam)
    errors = torch.linalg.vector_norm(px[..., :2] / cam[..., 2:].clamp_min(1e-9) - uv[:, None], dim=-1)
    good = valid[:, None] & (cam[..., 2] > 1e-6) & torch.isfinite(errors) & (errors <= max_reprojection_error)
    score = good.sum(-1).double() * 1000 - torch.where(good, errors, 0).sum(-1)
    best = score.argmax(-1)
    valid = good[torch.arange(len(shared)), best]
    xyz = _triangulate(uv, valid, Rts, Ks)
    projected, depth = _project(xyz, Rts, Ks)
    valid &= (depth > 1e-6) & torch.isfinite(projected).all(-1)
    valid &= torch.linalg.vector_norm(projected - uv, dim=-1) <= max_reprojection_error
    rays = torch.einsum("cij,ncj->nci", torch.linalg.inv(Ks),
                        torch.cat([uv, torch.ones_like(uv[..., :1])], -1))
    rays = torch.einsum("cji,ncj->nci", Rts[:, :3, :3], rays)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    cosines = torch.einsum("nci,ndi->ncd", rays, rays).clamp(-1, 1)
    pairs_valid = valid[:, :, None] & valid[:, None, :]
    cosines = torch.where(pairs_valid, cosines, torch.ones_like(cosines))
    parallax_ok = cosines.amin(dim=(1, 2)) <= math.cos(math.radians(min_parallax_deg))
    keep = (valid.sum(-1) >= 3) & valid[:, :len(variable_ids)].any(-1) & parallax_ok & torch.isfinite(xyz).all(-1)
    indices = torch.where(keep)[0][:max_points]
    uv[~valid] = -1
    surviving = [[(kf_id, feature_id) for kf_id, feature_id in shared[n]
                  if bool(valid[n, columns[kf_id]])] for n in indices.tolist()]
    return {"ids": ids, "Rts": Rts, "intrinsics": Ks, "points": xyz[indices],
            "observations": uv[indices], "tracks": surviving, "reason": None}


def _connected_observations(observations, min_shared_points=6):
    visible = (observations >= 0).all(-1)
    if not (visible.sum(0) >= 6).all():
        return False
    adjacency = (visible.long().T @ visible.long()) >= min_shared_points
    visited, pending = {0}, [0]
    while pending:
        c = pending.pop()
        for other in torch.where(adjacency[c])[0].tolist():
            if other not in visited:
                visited.add(other)
                pending.append(other)
    return len(visited) == observations.shape[1]


class CovisibilityLocalBA:
    def __init__(self, args):
        self.args = args
        self.events = []

    @torch.no_grad()
    def run(self, scene_model, newest_id):
        args = self.args
        if getattr(args, "local_ba_disable", False):
            return None
        every = int(getattr(args, "local_ba_every", 5))
        if newest_id % every:
            return None
        event = {"phase": "local_bundle_adjustment", "newest_keyframe_id": int(newest_id),
                 "accepted": False, "num_tracks": 0, "iterations": 0}
        variable, fixed, reason = select_local_window(
            scene_model.keyframes, newest_id,
            window_size=getattr(args, "local_ba_window_size", 10),
            recent_keyframes=getattr(args, "local_ba_recent_keyframes", 5),
            fixed_keyframes=getattr(args, "local_ba_fixed_keyframes", 2),
            min_shared_points=int(getattr(args, "local_ba_min_shared_points", 30)))
        event.update(optimized_keyframe_ids=variable, fixed_keyframe_ids=fixed)
        if reason is None:
            problem = build_shared_tracks(
                scene_model.keyframes, variable, fixed,
                max_points=int(getattr(args, "local_ba_max_points", 1500)),
                max_reprojection_error=float(getattr(args, "local_ba_max_reprojection_error", 8.0)),
                min_parallax_deg=float(getattr(args, "local_ba_min_parallax_deg", 1.0)))
            event["num_tracks"] = len(problem["points"])
            baseline_ok = True
            if len(problem["points"]):
                centres = torch.linalg.inv(problem["Rts"])[:, :3, 3]
                anchor_baseline = float(torch.pdist(centres[len(variable):]).max())
                _, depths = _project(problem["points"], problem["Rts"], problem["intrinsics"])
                visible = (problem["observations"] >= 0).all(-1)
                median_depth = float(depths[visible & (depths > 0)].median())
                event.update(anchor_baseline_raw=anchor_baseline, median_depth_raw=median_depth)
                baseline_ok = anchor_baseline / max(median_depth, 1e-12) >= 1e-3
            if event["num_tracks"] < int(getattr(args, "local_ba_min_tracks", 50)):
                reason = "insufficient_valid_tracks"
            elif not baseline_ok:
                reason = "weak_anchor_baseline"
            elif not _connected_observations(problem["observations"], max(6, int(getattr(args, "local_ba_min_shared_points", 30)))):
                reason = "disconnected_or_underconstrained_cameras"
            else:
                device = scene_model.keyframes[newest_id].get_Rt().device
                result = solve_local_ba(
                    problem["Rts"].to(device), problem["points"].to(device),
                    problem["observations"].to(device), problem["intrinsics"].to(device),
                    num_optimized_cameras=len(variable),
                    iterations=int(getattr(args, "local_ba_iterations", 15)),
                    huber_delta=float(getattr(args, "local_ba_huber_delta", 3.0)),
                    max_reprojection_error=float(getattr(args, "local_ba_max_reprojection_error", 8.0)))
                event.update({k: result[k] for k in ("accepted_iterations", "iterations")})
                for name in ("initial_cost", "final_cost"):
                    value = float(result[name])
                    event[name] = value if math.isfinite(value) else None
                reason = result.get("reason")
                if result["success"]:
                    old, new = problem["Rts"].to(device), result["Rts"]
                    old_centres, new_centres = torch.linalg.inv(old)[:, :3, 3], torch.linalg.inv(new)[:, :3, 3]
                    delta = torch.linalg.vector_norm(new_centres[:len(variable)] - old_centres[:len(variable)], dim=-1)
                    relative_R = new[:len(variable), :3, :3] @ old[:len(variable), :3, :3].transpose(-1, -2)
                    angle = torch.acos(((relative_R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)) * (180 / math.pi)
                    _, depths = _project(problem["points"].to(device), old, problem["intrinsics"].to(device))
                    observed = (problem["observations"].to(device) >= 0).all(-1)
                    median_depth = depths[observed & (depths > 0)].median()
                    event.update(max_camera_center_delta_raw=float(delta.max()), max_rotation_delta_deg=float(angle.max()))
                    if float(angle.max()) > float(getattr(args, "local_ba_max_rotation_deg", 5.0)) or float(delta.max()) > float(median_depth) * float(getattr(args, "local_ba_max_pose_change_ratio", 0.2)):
                        reason = "excessive_pose_change"
                    else:
                        final_inliers = result["inlier_mask"].detach().cpu().clone()
                        final_inliers[final_inliers.sum(-1) < 3] = False
                        final_observations = problem["observations"].clone()
                        final_observations[~final_inliers] = -1
                        if not _connected_observations(final_observations, max(6, int(getattr(args, "local_ba_min_shared_points", 30)))):
                            event["reason"] = "insufficient_final_camera_constraints"
                            self.events.append(event)
                            return event
                        fitted_points = result["points"].detach().cpu()
                        updates = []
                        for n, track in enumerate(problem["tracks"]):
                            inlier_observations = [(kf_id, feature_id) for kf_id, feature_id in track
                                if bool(final_inliers[n, problem["ids"].index(kf_id)])]
                            if len(inlier_observations) >= 3:
                                updates.append((inlier_observations, fitted_points[n]))
                        if len(updates) < int(getattr(args, "local_ba_min_tracks", 50)):
                            reason = "insufficient_final_inliers"
                        else:
                            applied = scene_model.apply_local_ba_result(
                                {kf_id: new[c] for c, kf_id in enumerate(variable)}, updates)
                            event.update(accepted=True, num_updated_tracks=len(updates), map_correction=applied)
                            reason = None
        event["reason"] = reason
        self.events.append(event)
        return event
