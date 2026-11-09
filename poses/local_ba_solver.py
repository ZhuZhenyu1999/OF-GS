"""Joint local bundle adjustment with fixed calibration and boundary cameras.

This solver is intentionally independent of the CUDA-graph MiniBA used by the
frontend. Poses are world-to-camera transforms; a left SE(3) increment has
coordinates [rotation_xyz, translation_xyz]. All accepted LM steps decrease the
same Huber objective on the same observations, including the boundary views.
"""

import math

import torch


def _skew(v):
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(-1)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *v.shape[:-1], 3, 3
    )


def _se3_exp(delta):
    """Return the rotation and translation of exp([omega, velocity])."""
    omega, velocity = delta[..., :3], delta[..., 3:]
    theta2 = (omega * omega).sum(-1)
    theta = theta2.sqrt()
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2 * math.pi)).square()
    c = torch.where(
        theta2 > 1e-8,
        (1 - a) / theta2.clamp_min(1e-20),
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
    )
    omega_hat = _skew(omega)
    omega_hat2 = omega_hat @ omega_hat
    eye = torch.eye(3, dtype=delta.dtype, device=delta.device)
    rotation = eye + a[..., None, None] * omega_hat + b[..., None, None] * omega_hat2
    v_matrix = eye + b[..., None, None] * omega_hat + c[..., None, None] * omega_hat2
    translation = (v_matrix @ velocity[..., None]).squeeze(-1)
    return rotation, translation


def _apply_pose_increment(Rts, delta, num_optimized_cameras):
    updated = Rts.clone()
    rotation, translation = _se3_exp(delta)
    updated[:num_optimized_cameras, :3, :3] = (
        rotation @ Rts[:num_optimized_cameras, :3, :3]
    )
    updated[:num_optimized_cameras, :3, 3] = (
        rotation @ Rts[:num_optimized_cameras, :3, 3, None]
    ).squeeze(-1) + translation
    return updated


def _project_with_jacobians(Rts, points, intrinsics):
    """Project Nx3 points and return analytic left-pose / world-point Jacobians."""
    rotations = Rts[:, :3, :3]
    camera_points = torch.einsum("cij,nj->nci", rotations, points) + Rts[None, :, :3, 3]
    homogeneous = torch.einsum("cij,ncj->nci", intrinsics, camera_points)
    denominator = homogeneous[..., 2]
    # Avoid arithmetic exceptions for inactive points. Active depths are checked
    # separately and a trial with a newly invalid projection is always rejected.
    safe_denominator = torch.where(denominator.abs() > 1e-12, denominator, 1e-12)
    pixels = homogeneous[..., :2] / safe_denominator[..., None]
    projection_jacobian = (
        intrinsics[None, :, :2, :]
        - pixels[..., :, None] * intrinsics[None, :, 2:3, :]
    ) / safe_denominator[..., None, None]
    translation_jacobian = torch.eye(3, device=points.device, dtype=points.dtype).expand(
        *camera_points.shape[:-1], 3, 3
    )
    camera_jacobian = torch.cat((-_skew(camera_points), translation_jacobian), -1)
    pose_jacobian = projection_jacobian @ camera_jacobian
    point_jacobian = projection_jacobian @ rotations[None]
    return pixels, camera_points[..., 2], pose_jacobian, point_jacobian


def _huber_cost(residual_norm, delta):
    if delta <= 0:
        return 0.5 * residual_norm.square()
    return torch.where(
        residual_norm <= delta,
        0.5 * residual_norm.square(),
        delta * (residual_norm - 0.5 * delta),
    )


def _cost_on_fixed_mask(pixels, depths, observations, mask, huber_delta):
    """Invalidating an existing observation must never look like a cost decrease."""
    valid = (depths > 1e-6) & torch.isfinite(depths) & torch.isfinite(pixels).all(-1)
    if not bool(valid[mask].all()):
        return pixels.new_tensor(float("inf"))
    residual = pixels[mask] - observations[mask]
    return _huber_cost(torch.linalg.vector_norm(residual, dim=-1), huber_delta).mean()


def _normal_equations(pose_jacobian, point_jacobian, residual, mask, nopt, delta):
    norms = torch.linalg.vector_norm(residual, dim=-1)
    weights = torch.ones_like(norms)
    if delta > 0:
        weights = (delta / norms.clamp_min(1e-12)).clamp_max(1)
    sqrt_weights = torch.where(mask, weights.sqrt(), 0)
    weighted_residual = torch.where(mask[..., None], residual, 0) * sqrt_weights[..., None]
    jp = torch.where(mask[..., None, None], point_jacobian, 0) * sqrt_weights[..., None, None]
    jc = torch.where(mask[:, :nopt, None, None], pose_jacobian[:, :nopt], 0)
    jc = jc * sqrt_weights[:, :nopt, None, None]
    camera_blocks = torch.einsum("ncui,ncuj->cij", jc, jc)
    point_blocks = torch.einsum("ncui,ncuj->nij", jp, jp)
    cross_blocks = torch.einsum("ncui,ncuj->ncij", jc, jp[:, :nopt])
    camera_gradient = torch.einsum("ncui,ncu->ci", jc, weighted_residual[:, :nopt])
    point_gradient = torch.einsum("ncui,ncu->ni", jp, weighted_residual)
    return camera_blocks, point_blocks, cross_blocks, camera_gradient, point_gradient


def _schur_step(blocks, damping):
    camera_blocks, point_blocks, cross_blocks, camera_gradient, point_gradient = blocks
    camera_blocks = camera_blocks.clone()
    point_blocks = point_blocks.clone()
    camera_diagonal = camera_blocks.diagonal(dim1=-2, dim2=-1)
    point_diagonal = point_blocks.diagonal(dim1=-2, dim2=-1)
    camera_diagonal.add_(damping * camera_diagonal.clamp_min(1e-6))
    point_diagonal.add_(damping * point_diagonal.clamp_min(1e-6))
    camera_hessian = torch.block_diag(*camera_blocks.unbind(0))
    cross = cross_blocks.reshape(cross_blocks.shape[0], -1, 3)
    # Solve small 3x3 landmark systems as a batch, without explicitly inverting
    # them. The only dense global solve is the 6*nopt camera Schur complement.
    rhs = torch.cat((cross.transpose(-1, -2), point_gradient[..., None]), -1)
    point_solution, point_info = torch.linalg.solve_ex(point_blocks, rhs)
    if bool((point_info != 0).any()) or not bool(torch.isfinite(point_solution).all()):
        return None
    inverse_cross = point_solution[..., :-1]
    inverse_gradient = point_solution[..., -1]
    schur = camera_hessian - torch.einsum("nki,nij->kj", cross, inverse_cross)
    schur = 0.5 * (schur + schur.T)
    schur_rhs = -camera_gradient.reshape(-1) + torch.einsum("nki,ni->k", cross, inverse_gradient)
    camera_delta, camera_info = torch.linalg.solve_ex(schur, schur_rhs)
    if bool(camera_info != 0) or not bool(torch.isfinite(camera_delta).all()):
        return None
    point_delta = -inverse_gradient - torch.einsum("nik,k->ni", inverse_cross, camera_delta)
    if not bool(torch.isfinite(point_delta).all()):
        return None
    return camera_delta.reshape(-1, 6), point_delta


@torch.no_grad()
def solve_local_ba(
    Rts, points, observations, intrinsics, *, num_optimized_cameras,
    iterations=15, huber_delta=3.0, max_reprojection_error=8.0,
):
    """Jointly refine camera poses and landmarks with pixel-space robust LM.

    Inputs are Cx4x4 world-to-camera poses, Nx3 world points, NxCx2 pixels
    (negative coordinates mark absent observations), and Cx3x3 fixed intrinsics.
    The first ``num_optimized_cameras`` poses are variables. At least two
    noncoincident fixed boundary cameras are required to anchor monocular scale.
    The caller selects a connected, sufficiently triangulated local subgraph.

    Shape/configuration errors raise ValueError; unusable numerical input returns
    ``success=False`` without modifying the caller's tensors. Costs are mean
    Huber penalties in pixel units on a fixed observation set. ``inlier_mask``
    has shape NxC and reports final thresholded observations from that set.
    """
    if Rts.ndim != 3 or Rts.shape[1:] != (4, 4):
        raise ValueError("Rts must have shape (C, 4, 4)")
    cameras = Rts.shape[0]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if observations.shape != (len(points), cameras, 2):
        raise ValueError("observations must have shape (N, C, 2)")
    if intrinsics.shape != (cameras, 3, 3):
        raise ValueError("intrinsics must have shape (C, 3, 3)")
    nopt = int(num_optimized_cameras)
    if nopt != num_optimized_cameras or nopt < 1 or cameras - nopt < 2:
        raise ValueError("local BA requires optimized cameras and at least two fixed cameras")
    if int(iterations) != iterations or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not math.isfinite(float(huber_delta)) or huber_delta < 0:
        raise ValueError("huber_delta must be finite and nonnegative")
    if not math.isfinite(float(max_reprojection_error)) or max_reprojection_error <= 0:
        raise ValueError("max_reprojection_error must be finite and positive")
    tensors = (Rts, points, observations, intrinsics)
    if any(t.device != Rts.device or not t.is_floating_point() for t in tensors):
        raise ValueError("all inputs must be floating point tensors on the same device")

    result = {
        "Rts": Rts.clone(), "points": points.clone(),
        "inlier_mask": torch.zeros(observations.shape[:2], dtype=torch.bool, device=Rts.device),
        "initial_cost": float("inf"), "final_cost": float("inf"),
        "accepted_iterations": 0, "iterations": 0, "success": False,
        "reason": "invalid_input", "num_observations": 0,
    }
    if not all(bool(torch.isfinite(t).all()) for t in (Rts, points, intrinsics)):
        return result
    # Float64 keeps the Schur subtraction stable for weak but valid parallax.
    poses = Rts.to(dtype=torch.float64).clone()
    landmarks = points.to(dtype=torch.float64).clone()
    measurements = observations.to(dtype=torch.float64)
    calibration = intrinsics.to(dtype=torch.float64)
    rotations = poses[:, :3, :3]
    identity = torch.eye(3, device=poses.device, dtype=poses.dtype).expand(cameras, -1, -1)
    if (not torch.allclose(rotations.transpose(-1, -2) @ rotations, identity, atol=1e-3, rtol=0)
            or not torch.allclose(torch.linalg.det(rotations), torch.ones(cameras, device=poses.device, dtype=poses.dtype), atol=1e-3, rtol=0)
            or not torch.allclose(poses[:, 3], poses.new_tensor([0, 0, 0, 1]).expand(cameras, -1), atol=1e-6, rtol=0)):
        result["reason"] = "invalid_camera_poses"
        return result
    centers = -(poses[nopt:, :3, :3].transpose(-1, -2) @ poses[nopt:, :3, 3, None]).squeeze(-1)
    if float(torch.pdist(centers).max()) <= 1e-6:
        result["reason"] = "coincident_boundary_cameras"
        return result
    if bool((calibration[:, (0, 1), (0, 1)] <= 0).any()):
        result["reason"] = "invalid_calibration"
        return result

    pixels, depths, jcamera, jpoint = _project_with_jacobians(poses, landmarks, calibration)
    residual = pixels - measurements
    mask = torch.isfinite(measurements).all(-1) & (measurements >= 0).all(-1)
    mask &= (depths > 1e-6) & torch.isfinite(pixels).all(-1)
    mask &= torch.linalg.vector_norm(residual, dim=-1) <= max_reprojection_error
    usable_tracks = (mask.sum(-1) >= 2) & mask[:, :nopt].any(-1)
    mask &= usable_tracks[:, None]
    result["inlier_mask"] = mask.clone()
    result["num_observations"] = int(mask.sum())
    if not bool((mask[:, :nopt].sum(0) >= 6).all()) or not bool(mask[:, nopt:].any()):
        result["reason"] = "insufficient_observations"
        return result
    observed_boundary_centers = centers[mask[:, nopt:].any(0)]
    if len(observed_boundary_centers) < 2 or float(torch.pdist(observed_boundary_centers).max()) <= 1e-6:
        result["reason"] = "insufficient_boundary_observations"
        return result

    # Remove empty landmarks from the solve, retaining their exact input values
    # in the returned full point array.
    active_indices = usable_tracks.nonzero(as_tuple=True)[0]
    landmarks = landmarks[active_indices]
    measurements = measurements[active_indices]
    fixed_mask = mask[active_indices]
    pixels, depths, jcamera, jpoint = _project_with_jacobians(poses, landmarks, calibration)
    cost = _cost_on_fixed_mask(pixels, depths, measurements, fixed_mask, huber_delta)
    result["initial_cost"] = float(cost)
    damping = 1e-3
    accepted_costs = [float(cost)]
    reason = "iteration_limit"
    for iteration in range(int(iterations)):
        result["iterations"] = iteration + 1
        residual = pixels - measurements
        blocks = _normal_equations(jcamera, jpoint, residual, fixed_mask, nopt, huber_delta)
        step = _schur_step(blocks, damping)
        if step is None:
            damping *= 10
            if damping > 1e12:
                reason = "singular_system"
                break
            continue
        camera_delta, point_delta = step
        candidate_poses = _apply_pose_increment(poses, camera_delta, nopt)
        candidate_points = landmarks + point_delta
        candidate_pixels, candidate_depths, candidate_jcamera, candidate_jpoint = (
            _project_with_jacobians(candidate_poses, candidate_points, calibration)
        )
        candidate_cost = _cost_on_fixed_mask(
            candidate_pixels, candidate_depths, measurements, fixed_mask, huber_delta,
        )
        if bool(torch.isfinite(candidate_cost)) and bool(candidate_cost < cost):
            relative_gain = float((cost - candidate_cost) / cost.clamp_min(1e-12))
            poses, landmarks = candidate_poses, candidate_points
            pixels, depths = candidate_pixels, candidate_depths
            jcamera, jpoint = candidate_jcamera, candidate_jpoint
            cost = candidate_cost
            result["accepted_iterations"] += 1
            accepted_costs.append(float(cost))
            damping = max(damping / 3, 1e-9)
            tiny_step = (
                float(camera_delta.abs().max()) < 1e-10
                and float(point_delta.abs().max()) < 1e-10
            )
            if relative_gain < 1e-7 or tiny_step:
                reason = "converged"
                break
        else:
            damping *= 10
            if damping > 1e12:
                reason = "no_improvement"
                break

    result["Rts"] = poses.to(dtype=Rts.dtype)
    # Fixed boundary transforms must remain bit-for-bit unchanged.
    result["Rts"][nopt:] = Rts[nopt:]
    result["points"][active_indices] = landmarks.to(dtype=points.dtype)
    final_mask = fixed_mask & (depths > 1e-6)
    final_mask &= torch.linalg.vector_norm(pixels - measurements, dim=-1) <= max_reprojection_error
    result["inlier_mask"].zero_()
    result["inlier_mask"][active_indices] = final_mask
    result["final_cost"] = float(cost)
    result["success"] = result["accepted_iterations"] > 0
    result["reason"] = reason if result["success"] else "no_improvement"
    result["accepted_costs"] = accepted_costs
    return result
