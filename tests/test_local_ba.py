from types import SimpleNamespace

import json

import pytest
import torch

from poses.local_ba import (
    CovisibilityLocalBA, _connected_observations, build_shared_tracks,
    select_local_window,
)
from slam_backend import SLAMBackend


def make_scene(num_cameras=12, num_points=80, noisy=False):
    generator = torch.Generator().manual_seed(17)
    points = torch.rand(num_points, 3, generator=generator, dtype=torch.float64)
    points = points * torch.tensor([2.0, 1.6, 4.0]) + torch.tensor([0.0, -0.8, 6.0])
    K = torch.tensor([[200., 0., 200.], [0., 220., 150.], [0., 0., 1.]], dtype=torch.float64)
    keyframes = []
    for i in range(num_cameras):
        Rt = torch.eye(4, dtype=torch.float64)
        Rt[0, 3] = -0.15 * i
        cam = points + Rt[:3, 3]
        uv = (cam @ K.T)[:, :2] / cam[:, 2:]
        desc = SimpleNamespace(kpts=uv, matches={}, has_pt3d=torch.ones(num_points, dtype=torch.bool),
                               pts_conf=torch.ones(num_points), pts3d=points.clone())
        estimated = Rt.clone()
        if noisy and i >= 2:
            estimated[1, 3] += 0.012 * (i % 3 + 1)
        kf = SimpleNamespace(index=i, width=400, height=300, info={"is_test": False},
                             desc_kpts=desc, K=K.clone(), Rt=estimated, mask_pyr=None)
        kf.get_Rt = lambda kf=kf: kf.Rt
        keyframes.append(kf)
    for i, kf in enumerate(keyframes):
        for j in range(max(0, i - 4), i):
            ids = torch.arange(num_points)
            kf.desc_kpts.matches[j] = SimpleNamespace(idx=ids, idx_other=ids)
            keyframes[j].desc_kpts.matches[i] = SimpleNamespace(idx=ids, idx_other=ids)
    scene = SimpleNamespace(keyframes=keyframes, applied=[])
    def apply(pose_updates, landmark_updates):
        scene.applied.append((pose_updates, landmark_updates))
        for i, pose in pose_updates.items():
            keyframes[i].Rt = pose
        return {"poses_updated": len(pose_updates), "landmarks_updated": len(landmark_updates)}
    scene.apply_local_ba_result = apply
    return scene, points


def test_window_uses_covisibility_and_separated_fixed_boundaries():
    scene, _ = make_scene(16)
    variable, fixed, reason = select_local_window(scene.keyframes, 15)
    assert reason is None
    assert len(variable) == 8 and len(fixed) == 2
    assert {11, 12, 13, 14, 15} <= set(variable)
    assert set(variable).isdisjoint(fixed)
    assert all(i < 11 for i in fixed)


@pytest.mark.parametrize("window_size,recent_keyframes,fixed_keyframes", [
    (7, 5, 2), (9, 5, 2), (9, 6, 3), (10, 6, 2), (10, 5, 3),
])
def test_valid_window_settings_preserve_recent_variables_and_total_limit(
    window_size, recent_keyframes, fixed_keyframes
):
    scene, _ = make_scene(16)
    variable, fixed, reason = select_local_window(
        scene.keyframes, 15, window_size=window_size,
        recent_keyframes=recent_keyframes, fixed_keyframes=fixed_keyframes,
    )
    assert reason is None
    assert len(variable) + len(fixed) == window_size
    assert len(fixed) == fixed_keyframes
    assert set(range(16 - recent_keyframes, 16)) <= set(variable)
    assert set(variable).isdisjoint(fixed)


@pytest.mark.parametrize("options", [
    {"window_size": 11}, {"window_size": 6}, {"window_size": 0},
    {"recent_keyframes": 4}, {"recent_keyframes": 9},
    {"fixed_keyframes": 1}, {"fixed_keyframes": 6},
    {"window_size": 7, "recent_keyframes": 6},
])
def test_window_rejects_conflicting_size_settings(options):
    scene, _ = make_scene(16)
    with pytest.raises(ValueError):
        select_local_window(scene.keyframes, 15, **options)


@pytest.mark.parametrize("name,integer_value", [
    ("window_size", 10), ("recent_keyframes", 5), ("fixed_keyframes", 2),
])
@pytest.mark.parametrize("value_kind", ["float", "string", "boolean", "none"])
def test_window_settings_require_integer_types(name, integer_value, value_kind):
    value = {"float": float(integer_value), "string": str(integer_value),
             "boolean": True, "none": None}[value_kind]
    scene, _ = make_scene(7)
    with pytest.raises(ValueError):
        select_local_window(scene.keyframes, 6, **{name: value})


@pytest.mark.parametrize("num_cameras,reason", [
    (1, "insufficient_recent_keyframes"), (4, "insufficient_recent_keyframes"),
    (5, "insufficient_fixed_keyframes"), (6, "insufficient_fixed_keyframes"),
])
def test_short_history_is_skipped_without_using_recent_frames_as_anchors(num_cameras, reason):
    scene, _ = make_scene(num_cameras)
    assert select_local_window(scene.keyframes, num_cameras - 1) == ([], [], reason)
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_every=1)).run(scene, num_cameras - 1)
    assert not event["accepted"] and event["reason"] == reason
    assert scene.applied == []


@pytest.mark.parametrize("missing", ["holdout", "keypoints"])
def test_ineligible_newest_keyframe_skips_local_window(missing):
    scene, _ = make_scene(16)
    if missing == "holdout":
        scene.keyframes[15].info["is_test"] = True
    else:
        scene.keyframes[15].desc_kpts = None
    assert select_local_window(scene.keyframes, 15)[2] == "test_or_missing_keypoints"


def test_recent_window_filters_holdout_missing_keypoints_and_future_frames():
    scene, _ = make_scene(16)
    scene.keyframes[11].info["is_test"] = True
    scene.keyframes[12].desc_kpts = None
    variable, fixed, reason = select_local_window(scene.keyframes, 13)
    assert reason is None
    assert {7, 8, 9, 10, 13} <= set(variable)
    selected = set(variable) | set(fixed)
    assert len(selected) == 10
    assert selected.isdisjoint({11, 12, 14, 15})
    assert set(fixed).isdisjoint({7, 8, 9, 10, 13})


def set_pair_matches(scene, first, second, count):
    ids = torch.arange(count)
    scene.keyframes[first].desc_kpts.matches[second] = SimpleNamespace(idx=ids, idx_other=ids)
    scene.keyframes[second].desc_kpts.matches[first] = SimpleNamespace(idx=ids, idx_other=ids)


def clear_matches(scene):
    for keyframe in scene.keyframes:
        keyframe.desc_kpts.matches.clear()


def test_other_candidates_keep_covisibility_total_score_and_newer_id_tiebreak():
    scene, _ = make_scene(12)
    clear_matches(scene)
    # Frame 0 wins by total support, although frame 1 has the strongest single
    # edge. Frames 2 and 3 tie for the remaining slot; the newer ID wins.
    for first, second, count in (
        (0, 7, 45), (0, 8, 45), (1, 7, 70),
        (2, 7, 40), (2, 8, 40), (3, 7, 40), (3, 8, 40),
    ):
        set_pair_matches(scene, first, second, count)
    # A large total cannot admit a frame whose every individual link is below
    # the existing minimum shared-point threshold of 30.
    for recent in range(7, 12):
        set_pair_matches(scene, 4, recent, 29)
    variable, fixed, reason = select_local_window(scene.keyframes, 11, window_size=7)
    assert reason is None
    assert variable == [7, 8, 9, 10, 11]
    assert fixed == [0, 3]


def test_covisibility_candidates_expand_from_frames_already_selected():
    scene, _ = make_scene(12)
    clear_matches(scene)
    # Frame 2 becomes eligible through frame 0 once frame 0 is selected.
    set_pair_matches(scene, 0, 7, 80)
    set_pair_matches(scene, 1, 7, 40)
    set_pair_matches(scene, 2, 0, 60)
    variable, fixed, reason = select_local_window(scene.keyframes, 11, window_size=7)
    assert reason is None
    assert variable == [7, 8, 9, 10, 11]
    assert fixed == [0, 2]


def test_weak_recent_frames_stay_selected_and_disconnected_ba_does_not_write(monkeypatch):
    import poses.local_ba as module
    scene, _ = make_scene(16)
    clear_matches(scene)
    # Plenty of strong old candidates must not replace recent frames 11--14,
    # even though those recent frames have no usable observations.
    for older in range(11):
        set_pair_matches(scene, older, 15, 80)
    variable, fixed, reason = select_local_window(scene.keyframes, 15)
    assert reason is None
    assert {11, 12, 13, 14, 15} <= set(variable)
    assert set(variable) | set(fixed) == set(range(6, 16))
    before = torch.stack([keyframe.Rt.clone() for keyframe in scene.keyframes])

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("Disconnected recent frames must be rejected before solving")

    monkeypatch.setattr(module, "solve_local_ba", unexpected_solver)
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_every=1)).run(scene, 15)
    assert not event["accepted"]
    assert event["reason"] == "disconnected_or_underconstrained_cameras"
    assert {11, 12, 13, 14, 15} <= set(event["optimized_keyframe_ids"])
    assert scene.applied == []
    assert torch.equal(torch.stack([keyframe.Rt for keyframe in scene.keyframes]), before)


def test_shared_tracks_have_single_landmark_and_reject_bad_observation():
    scene, expected_points = make_scene(6)
    # An otherwise consistent track has one gross mismatch; reliable existing
    # geometry should retain its other five measured observations.
    scene.keyframes[5].desc_kpts.kpts[0] += 70
    problem = build_shared_tracks(scene.keyframes, [2, 3, 4, 5], [0, 1])
    assert len(problem["points"]) == len(expected_points)
    index = next(n for n, track in enumerate(problem["tracks"]) if (0, 0) in track)
    assert (5, 0) not in problem["tracks"][index]
    assert torch.allclose(problem["points"][index], expected_points[0], atol=1e-8)
    for track in problem["tracks"]:
        assert len({i for i, _ in track}) == len(track)


def test_shared_tracks_filter_low_parallax_and_image_masks():
    scene, _ = make_scene(5)
    for kf in scene.keyframes:
        kf.mask_pyr = [torch.zeros(1, 300, 400, dtype=torch.bool)]
    assert len(build_shared_tracks(scene.keyframes, [2, 3, 4], [0, 1])["points"]) == 0
    scene, _ = make_scene(5)
    assert len(build_shared_tracks(scene.keyframes, [2, 3, 4], [0, 1], min_parallax_deg=80)["points"]) == 0


def test_two_fixed_poses_must_have_nonzero_baseline():
    scene, _ = make_scene(7)
    for kf in scene.keyframes:
        kf.Rt = torch.eye(4, dtype=torch.float64)
    assert select_local_window(scene.keyframes, 6)[2] == "degenerate_anchor_baseline"


def test_disconnected_observation_components_are_rejected():
    uv = torch.full((20, 4, 2), -1.)
    uv[:10, :2] = 10
    uv[10:, 2:] = 10
    assert not _connected_observations(uv)


def test_one_landmark_bridge_does_not_make_a_strongly_constrained_graph():
    uv = torch.full((21, 4, 2), -1.)
    uv[:10, :2] = 10
    uv[10:20, 2:] = 10
    uv[20] = 10
    assert not _connected_observations(uv)


def test_local_ba_jointly_updates_old_poses_and_shared_tracks():
    scene, _ = make_scene(7, noisy=True)
    # Dataset metadata must not become an optimization input.
    for kf in scene.keyframes:
        kf.info["Rt"] = torch.full((4, 4), float("nan"))
        kf.info["depth"] = torch.full((2, 2), float("nan"))
    ba = CovisibilityLocalBA(SimpleNamespace(local_ba_iterations=10, local_ba_every=1))
    event = ba.run(scene, 6)
    assert event["accepted"], event
    assert event["final_cost"] < event["initial_cost"]
    poses, tracks = scene.applied[0]
    assert set(poses) == {2, 3, 4, 5, 6}
    assert event["fixed_keyframe_ids"] == [0, 1]
    assert len(tracks) >= 50
    assert all(len(observations) >= 3 for observations, _ in tracks)
    assert torch.isfinite(torch.stack(list(poses.values()))).all()


def test_local_ba_disabled_and_cadence_do_not_mutate_scene():
    scene, _ = make_scene(7, noisy=True)
    assert CovisibilityLocalBA(SimpleNamespace(local_ba_disable=True)).run(scene, 6) is None
    assert CovisibilityLocalBA(SimpleNamespace(local_ba_every=4)).run(scene, 6) is None
    assert scene.applied == []


def test_failed_solver_diagnostics_are_json_safe(monkeypatch):
    import poses.local_ba as module
    scene, _ = make_scene(7, noisy=True)
    monkeypatch.setattr(module, "solve_local_ba", lambda *args, **kwargs: {
        "initial_cost": float("inf"), "final_cost": float("inf"),
        "accepted_iterations": 0, "iterations": 0, "success": False,
        "reason": "insufficient_observations",
    })
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_every=1)).run(scene, 6)
    assert not event["accepted"] and event["initial_cost"] is None
    json.dumps(event, allow_nan=False)
    assert scene.applied == []


def test_final_inliers_must_still_constrain_both_anchors(monkeypatch):
    import poses.local_ba as module
    scene, _ = make_scene(7, noisy=True)
    def lose_anchor(poses, points, observations, intrinsics, **kwargs):
        inliers = (observations >= 0).all(-1)
        inliers[:, -1] = False
        return {"Rts": poses, "points": points, "inlier_mask": inliers,
                "initial_cost": 1., "final_cost": 0.5, "accepted_iterations": 1,
                "iterations": 1, "success": True, "reason": "iteration_limit"}
    monkeypatch.setattr(module, "solve_local_ba", lose_anchor)
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_every=1)).run(scene, 6)
    assert not event["accepted"]
    assert event["reason"] == "insufficient_final_camera_constraints"
    assert scene.applied == []


def test_excessive_pose_update_is_rejected_before_map_mutation(monkeypatch):
    import poses.local_ba as module
    scene, _ = make_scene(7, noisy=True)
    def jump(poses, points, observations, intrinsics, **kwargs):
        updated = poses.clone()
        updated[0, 0, 3] += 1000
        return {"Rts": updated, "points": points, "inlier_mask": (observations >= 0).all(-1),
                "initial_cost": 1., "final_cost": 0.5, "accepted_iterations": 1,
                "iterations": 1, "success": True, "reason": "iteration_limit"}
    monkeypatch.setattr(module, "solve_local_ba", jump)
    event = CovisibilityLocalBA(SimpleNamespace(local_ba_every=1)).run(scene, 6)
    assert not event["accepted"] and event["reason"] == "excessive_pose_change"
    assert scene.applied == []


def test_backend_runs_local_ba_before_own_gaussian_insertion():
    backend = object.__new__(SLAMBackend)
    backend.args = SimpleNamespace()
    backend.scene_model = SimpleNamespace(keyframes=[SimpleNamespace(index=0, info={})])
    calls = []
    backend._timed = lambda func, *args: (func(*args), 1.0)
    backend.add_keyframe = lambda *args: backend.scene_model.keyframes[0]
    backend.run_local_ba = lambda i: (calls.append("ba") or {"phase": "local_bundle_adjustment"})
    backend.scene_model.add_new_gaussians = lambda i: calls.append("gaussians")
    backend.run_mapping_schedule = lambda: (calls.append("mapping") or [{"phase": "mapping"}])
    summary = backend.register_incremental_keyframe(9, None, {}, None, torch.eye(4), 100.)
    assert calls == ["ba", "gaussians", "mapping"]
    assert summary.phases[0]["phase"] == "local_bundle_adjustment"
    assert summary.timings_ms["local_ba"] == 1.0
