import pytest
import torch

from poses.local_ba_solver import (
    _apply_pose_increment,
    _cost_on_fixed_mask,
    _project_with_jacobians,
    _se3_exp,
    solve_local_ba,
)


def _scene(dtype=torch.float64, device="cpu"):
    rng = torch.Generator().manual_seed(125)
    points = torch.rand(120, 3, generator=rng, dtype=torch.float64)
    points[:, :2] = (points[:, :2] - 0.5) * points.new_tensor([2.4, 1.6])
    points[:, 2] = 3.5 + points[:, 2] * 3
    poses = torch.eye(4, dtype=torch.float64)[None].repeat(5, 1, 1)
    rotation, _ = _se3_exp(points.new_tensor([
        [0.005, -0.02, 0.01, 0, 0, 0], [0.01, 0.01, -0.01, 0, 0, 0],
        [-0.01, 0.03, 0.005, 0, 0, 0], [0, -0.015, 0, 0, 0, 0],
        [0.005, 0.015, 0, 0, 0, 0],
    ]))
    centers = points.new_tensor([
        [-0.25, 0.02, 0.03], [0, -0.03, 0.04], [0.3, 0.01, 0],
        [-0.7, 0, 0], [0.8, 0.04, 0.03],
    ])
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = -(rotation @ centers[..., None]).squeeze(-1)
    intrinsics = torch.eye(3, dtype=torch.float64)[None].repeat(5, 1, 1)
    intrinsics[:, 0, 0] = points.new_tensor([510, 550, 530, 540, 500])
    intrinsics[:, 1, 1] = points.new_tensor([480, 500, 520, 510, 490])
    intrinsics[:, 0, 2] = points.new_tensor([320, 321, 318, 319, 322])
    intrinsics[:, 1, 2] = points.new_tensor([240, 241, 239, 242, 238])
    intrinsics[:, 0, 1] = 0.8  # Exercise the full projection Jacobian, including skew.
    pixels = _project_with_jacobians(poses, points, intrinsics)[0]
    pixels += torch.randn(pixels.shape, generator=rng, dtype=torch.float64) * 0.015
    perturbation = torch.randn(3, 6, generator=rng, dtype=torch.float64)
    perturbation[:, :3] *= 0.0025
    perturbation[:, 3:] *= 0.012
    initial_poses = _apply_pose_increment(poses, perturbation, 3)
    initial_points = points + torch.randn(points.shape, generator=rng, dtype=torch.float64) * 0.008
    return tuple(t.to(device=device, dtype=dtype) for t in (
        poses, points, initial_poses, initial_points, pixels, intrinsics,
    ))


def _camera_centers(poses):
    return -(poses[:, :3, :3].transpose(-1, -2) @ poses[:, :3, 3, None]).squeeze(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_bundle_adjustment_recovers_poses_and_shared_landmarks(dtype):
    truth, points, initial_poses, initial_points, pixels, intrinsics = _scene(dtype)
    pose_snapshot, point_snapshot, calibration_snapshot = (
        initial_poses.clone(), initial_points.clone(), intrinsics.clone()
    )
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3, iterations=20)
    assert result["success"], result["reason"]
    assert result["accepted_iterations"] > 2
    assert result["final_cost"] < result["initial_cost"] * 0.001
    assert result["inlier_mask"].shape == pixels.shape[:2]
    assert result["inlier_mask"].all()
    assert all(a > b for a, b in zip(result["accepted_costs"], result["accepted_costs"][1:]))
    initial_center_error = (_camera_centers(initial_poses[:3]) - _camera_centers(truth[:3])).norm()
    fitted_center_error = (_camera_centers(result["Rts"][:3]) - _camera_centers(truth[:3])).norm()
    assert fitted_center_error < initial_center_error * 0.05
    assert (result["Rts"][:3, :3, :3] - truth[:3, :3, :3]).norm() < (
        initial_poses[:3, :3, :3] - truth[:3, :3, :3]
    ).norm() * 0.05
    assert (result["points"] - points).norm() < (initial_points - points).norm() * 0.15
    assert torch.equal(result["Rts"][3:], initial_poses[3:])
    assert torch.equal(initial_poses, pose_snapshot)
    assert torch.equal(initial_points, point_snapshot)
    assert torch.equal(intrinsics, calibration_snapshot)
    assert result["Rts"].dtype == dtype and result["points"].dtype == dtype
    assert result["Rts"].device == initial_poses.device


def test_analytic_camera_and_point_jacobians_match_finite_differences():
    poses, points, _, _, _, intrinsics = _scene()
    points = points[:4]
    pixels, _, jcamera, jpoint = _project_with_jacobians(poses, points, intrinsics)
    epsilon = 1e-6
    for axis in range(6):
        delta = points.new_zeros(3, 6)
        delta[1, axis] = epsilon
        plus = _project_with_jacobians(_apply_pose_increment(poses, delta, 3), points, intrinsics)[0]
        minus = _project_with_jacobians(_apply_pose_increment(poses, -delta, 3), points, intrinsics)[0]
        torch.testing.assert_close((plus[:, 1] - minus[:, 1]) / (2 * epsilon),
                                   jcamera[:, 1, :, axis], atol=1e-6, rtol=1e-7)
    for axis in range(3):
        delta = torch.zeros_like(points)
        delta[:, axis] = epsilon
        plus = _project_with_jacobians(poses, points + delta, intrinsics)[0]
        minus = _project_with_jacobians(poses, points - delta, intrinsics)[0]
        torch.testing.assert_close((plus - minus) / (2 * epsilon), jpoint[..., axis],
                                   atol=1e-6, rtol=1e-7)
    assert torch.isfinite(pixels).all()


def test_outliers_missing_pixels_and_invalid_initial_depth_are_excluded_once():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene()
    pixels[0, 0] += 400
    pixels[1, 1] = -1
    pixels[2, 2] = float("nan")
    # This point has measured pixels but lies behind all cameras initially.
    initial_points[3, 2] = -4
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3, iterations=20)
    assert result["success"]
    assert result["final_cost"] < result["initial_cost"]
    assert not result["inlier_mask"][0, 0]
    assert not result["inlier_mask"][1, 1]
    assert not result["inlier_mask"][2, 2]
    assert not result["inlier_mask"][3].any()
    assert torch.equal(result["points"][3], initial_points[3])
    assert torch.isfinite(result["Rts"]).all() and torch.isfinite(result["points"]).all()
    assert torch.equal(result["Rts"][3:], initial_poses[3:])
    assert result["num_observations"] == 5 * len(initial_points) - 8


def test_trial_cannot_lower_cost_by_putting_an_existing_observation_behind_camera():
    poses, points, _, _, pixels, intrinsics = _scene()
    projection, depth, _, _ = _project_with_jacobians(poses, points, intrinsics)
    mask = torch.ones_like(depth, dtype=torch.bool)
    initial_cost = _cost_on_fixed_mask(projection, depth, pixels, mask, 3)
    bad_depth = depth.clone()
    bad_depth[0, 0] = -1
    assert torch.isfinite(initial_cost)
    assert torch.isinf(_cost_on_fixed_mask(projection, bad_depth, pixels, mask, 3))
    bad_projection = projection.clone()
    bad_projection[0, 0] = float("nan")
    assert torch.isinf(_cost_on_fixed_mask(bad_projection, depth, pixels, mask, 3))


def test_no_positive_depth_observations_returns_unchanged_failure():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene()
    initial_points[:, 2] = -5
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3)
    assert not result["success"]
    assert result["reason"] == "insufficient_observations"
    assert result["accepted_iterations"] == 0
    assert not result["inlier_mask"].any()
    assert torch.equal(result["Rts"], initial_poses)
    assert torch.equal(result["points"], initial_points)


def test_coincident_boundary_cameras_cannot_anchor_monocular_scale():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene()
    initial_poses[4] = initial_poses[3]
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3)
    assert not result["success"]
    assert result["reason"] == "coincident_boundary_cameras"
    assert torch.equal(result["Rts"], initial_poses)


def test_nonrigid_camera_pose_is_rejected_without_mutation():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene()
    initial_poses[0, 0, 0] = 2
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3)
    assert not result["success"] and result["reason"] == "invalid_camera_poses"
    assert torch.equal(result["Rts"], initial_poses)


def test_unobserved_second_boundary_camera_cannot_anchor_scale():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene()
    pixels[:, 4] = -1
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3)
    assert not result["success"]
    assert result["reason"] == "insufficient_boundary_observations"
    assert torch.equal(result["Rts"], initial_poses)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_joint_ba_preserves_device_and_improves_the_same_objective():
    _, _, initial_poses, initial_points, pixels, intrinsics = _scene(torch.float32, "cuda")
    result = solve_local_ba(initial_poses, initial_points, pixels, intrinsics,
                           num_optimized_cameras=3, iterations=15)
    assert result["success"]
    assert result["final_cost"] < result["initial_cost"] * 0.001
    assert result["Rts"].is_cuda and result["points"].is_cuda
    assert result["inlier_mask"].is_cuda
    assert torch.equal(result["Rts"][3:], initial_poses[3:])
