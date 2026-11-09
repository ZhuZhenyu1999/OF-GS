import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import calibration_preflight


CURRENT_MINIBA = """
class MiniBA:
    def __init__(self):
        n_cams = 8
        f_init = torch.ones(n_cams, 2, device='cuda')
        centre = torch.randn(n_cams, 2, device='cuda')
    def __call__(self, Rs6D, ts, f, xyz, centre, uv):
        f, centre, _ = _camera_intrinsics(f, centre, self.n_cams, xyz)
        return self.optimizer(Rs6D, ts, f, xyz, centre, uv)
"""

CURRENT_SCENE = """
class SceneModel:
    def __init__(self):
        self.gaussian_owner_keyframe_ids = []

    def _fit_gaussian_owner_keyframe_ids(self):
        self.gaussian_owner_keyframe_ids = owners
        return owners

    def _apply_gaussian_add_and_prune(self, extension_tensors, valid_mask,
                                     extension_observer_ids=None, extension_owner_ids=None):
        old_owners = self._fit_gaussian_owner_keyframe_ids()
        if extension_owner_ids is None:
            extension_owner_ids = extension_observer_ids[:, 0].clone()
        self.gaussian_owner_keyframe_ids = torch.cat([old_owners[valid_mask], extension_owner_ids])

    def add_new_gaussians(self, keyframe_id):
        extension_observer_ids = self._observer_ids_for_keyframe(new_pts.shape[0], keyframe.index)
        self._apply_gaussian_add_and_prune(extension_tensors, valid_mask,
                                          extension_observer_ids=extension_observer_ids)

    def apply_local_ba_result(self, pose_updates, landmark_updates):
        owners = self._fit_gaussian_owner_keyframe_ids()
"""

CURRENT_LOCAL_BA_SOURCES = {
    "poses/local_ba.py": """
class CovisibilityLocalBA:
    def run(self, scene_model, newest_id):
        result = solve_local_ba(Rts, points, observations, intrinsics)
        return scene_model.apply_local_ba_result(poses, points)
""",
    "poses/local_ba_solver.py": """
def solve_local_ba(Rts, points, observations, intrinsics, *, num_optimized_cameras):
    pass
""",
    "scene/scene_model.py": CURRENT_SCENE,
    "slam_backend.py": """
class SLAMBackend:
    def __init__(self):
        self.local_ba = CovisibilityLocalBA(args)
    def register_incremental_keyframe(self):
        self._timed(self.run_local_ba, keyframe.index)
""",
    "slam.py": "# Runtime entry point is hashed without importing its dependencies.\n",
}
CURRENT_TRAJECTORY_OUTPUTS = '''
import unavailable_runtime_dependency
TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping")
'''
CURRENT_TRAJECTORY_ENTRYPOINT = '''
import unavailable_runtime_dependency
def run_slam(args):
    return write_trajectory_stage_outputs(args.model_path, {
        "frontend_miniba": frontend_records,
        "online_mapping": online_records,
    })
'''


def make_sources(root, mini_ba=CURRENT_MINIBA):
    for relative in calibration_preflight.CALIBRATION_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(mini_ba if relative == "poses/mini_ba.py" else "", encoding="utf-8")


def make_local_ba_sources(root, scene_source=CURRENT_SCENE):
    make_sources(root)
    for relative, source in CURRENT_LOCAL_BA_SOURCES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(scene_source if relative == "scene/scene_model.py" else source, encoding="utf-8")


def make_trajectory_sources(root, outputs=CURRENT_TRAJECTORY_OUTPUTS,
                            entrypoint=CURRENT_TRAJECTORY_ENTRYPOINT):
    (root / "slam_outputs.py").write_text(outputs, encoding="utf-8")
    (root / "slam.py").write_text(entrypoint, encoding="utf-8")


def test_calibration_preflight_reports_deployed_path_and_hash(tmp_path, capsys):
    make_sources(tmp_path)
    report = calibration_preflight.check_calibration_sources(tmp_path)
    mini_ba = tmp_path / "poses" / "mini_ba.py"
    entry = report["files"]["poses/mini_ba.py"]
    assert entry["path"] == str(mini_ba.resolve())
    assert entry["sha256"] == hashlib.sha256(mini_ba.read_bytes()).hexdigest()
    assert report["errors"] == []
    assert entry["path"] in capsys.readouterr().out


@pytest.mark.parametrize(
    "stale, expected",
    [
        (CURRENT_MINIBA.replace("torch.ones(n_cams, 2", "torch.ones(1"), "old \\[1\\] focal"),
        (CURRENT_MINIBA.replace("torch.randn(n_cams, 2", "torch.randn(2"), "principal point"),
        (CURRENT_MINIBA.replace("_camera_intrinsics", "legacy_intrinsics"), "missing _camera_intrinsics"),
        (
            CURRENT_MINIBA.replace(
                "        f, centre, _ = _camera_intrinsics(f, centre, self.n_cams, xyz)\n"
                "        return self.optimizer(Rs6D, ts, f, xyz, centre, uv)",
                "        return self.optimizer(Rs6D, ts, f, xyz, centre, uv)\n"
                "        f, centre, _ = _camera_intrinsics(f, centre, self.n_cams, xyz)",
            ),
            "before graph replay",
        ),
    ],
)
def test_calibration_preflight_rejects_stale_interfaces(tmp_path, stale, expected):
    make_sources(tmp_path, stale)
    with pytest.raises(RuntimeError, match=expected):
        calibration_preflight.check_calibration_sources(tmp_path)


def test_calibration_preflight_rejects_missing_file(tmp_path):
    make_sources(tmp_path)
    (tmp_path / "utils.py").unlink()
    with pytest.raises(RuntimeError, match="Cannot read"):
        calibration_preflight.check_calibration_sources(tmp_path)


def test_calibration_preflight_cli_works_without_site_packages(tmp_path):
    make_sources(tmp_path)
    result = subprocess.run(
        [sys.executable, "-S", str(Path(calibration_preflight.__file__).resolve()), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "MiniBA source interface OK" in result.stdout


def test_calibration_preflight_cli_exits_nonzero_on_stale_sources(tmp_path):
    make_sources(tmp_path, CURRENT_MINIBA.replace("torch.ones(n_cams, 2", "torch.ones(1"))
    result = subprocess.run(
        [sys.executable, "-S", str(Path(calibration_preflight.__file__).resolve()), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "no SLAM sequence was started" in result.stderr


def test_local_ba_preflight_reports_every_runtime_source_hash(tmp_path, capsys):
    make_local_ba_sources(tmp_path)
    report = calibration_preflight.check_calibration_sources(tmp_path, check_local_ba=True)
    assert set(report["files"]) == set(calibration_preflight.CALIBRATION_FILES + calibration_preflight.LOCAL_BA_FILES)
    for relative in calibration_preflight.LOCAL_BA_FILES:
        path = tmp_path / relative
        assert report["files"][relative] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert report["errors"] == []
    assert "Local BA source interface OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "stale, expected",
    [
        (CURRENT_SCENE.replace("def apply_local_ba_result", "def old_commit"), "SceneModel.apply_local_ba_result"),
        (CURRENT_SCENE.replace("pose_updates, landmark_updates", "pose_updates"), "SceneModel.apply_local_ba_result"),
        (CURRENT_SCENE.replace("class SceneModel", "class UnusedSceneModel"), "SceneModel.apply_local_ba_result"),
        (CURRENT_SCENE.replace("self.gaussian_owner_keyframe_ids = []", "self.legacy_owners = []"), "creation-owner initialization"),
        (CURRENT_SCENE.replace("def _fit_gaussian_owner_keyframe_ids", "def legacy_fit"), "_fit_gaussian_owner_keyframe_ids is missing"),
        (CURRENT_SCENE.replace("extension_owner_ids=None", "legacy_owners=None"), "missing extension_owner_ids"),
        (CURRENT_SCENE.replace("old_owners[valid_mask]", "old_owners"), "owner preservation"),
        (CURRENT_SCENE.replace("extension_observer_ids[:, 0]", "extension_observer_ids[:, 1]"), "owner preservation"),
        (CURRENT_SCENE.replace("keyframe.index", "viewer.index"), "creation-owner path"),
        (CURRENT_SCENE.replace("extension_observer_ids=extension_observer_ids", "extension_observer_ids=None"), "creation-owner path"),
    ],
)
def test_local_ba_preflight_rejects_partial_scene_sync(tmp_path, stale, expected):
    make_local_ba_sources(tmp_path, stale)
    with pytest.raises(RuntimeError, match=expected):
        calibration_preflight.check_calibration_sources(tmp_path, check_local_ba=True)


def test_local_ba_preflight_cannot_be_satisfied_by_comment(tmp_path):
    stale = CURRENT_SCENE.replace("def apply_local_ba_result", "def old_commit")
    stale += "\n# def apply_local_ba_result(self, pose_updates, landmark_updates):\n"
    make_local_ba_sources(tmp_path, stale)
    with pytest.raises(RuntimeError, match="SceneModel.apply_local_ba_result"):
        calibration_preflight.check_calibration_sources(tmp_path, check_local_ba=True)


@pytest.mark.parametrize("relative", calibration_preflight.LOCAL_BA_FILES)
def test_local_ba_preflight_rejects_missing_runtime_source(tmp_path, relative):
    make_local_ba_sources(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(RuntimeError, match="Cannot read"):
        calibration_preflight.check_calibration_sources(tmp_path, check_local_ba=True)


@pytest.mark.parametrize(
    "relative, stale, expected",
    [
        ("poses/local_ba.py", "class CovisibilityLocalBA:\n    def run(self): pass\n", "solve/scene commit"),
        ("poses/local_ba_solver.py", "def solve_local_ba(Rts): pass\n", "joint solve_local_ba"),
        ("slam_backend.py", "class SLAMBackend: pass\n", "CovisibilityLocalBA initialization"),
        ("slam.py", "def (\n", "slam.py cannot be parsed"),
    ],
)
def test_local_ba_preflight_rejects_stale_runtime_source(tmp_path, relative, stale, expected):
    make_local_ba_sources(tmp_path)
    (tmp_path / relative).write_text(stale, encoding="utf-8")
    with pytest.raises(RuntimeError, match=expected):
        calibration_preflight.check_calibration_sources(tmp_path, check_local_ba=True)


def test_local_ba_preflight_cli_without_site_packages(tmp_path):
    make_local_ba_sources(tmp_path)
    result = subprocess.run(
        [sys.executable, "-S", str(Path(calibration_preflight.__file__).resolve()),
         "--root", str(tmp_path), "--check_local_ba"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Local BA source interface OK" in result.stdout


def test_local_ba_preflight_cli_fails_before_runtime_import(tmp_path):
    make_local_ba_sources(tmp_path, CURRENT_SCENE.replace("def apply_local_ba_result", "def old_commit"))
    result = subprocess.run(
        [sys.executable, "-S", str(Path(calibration_preflight.__file__).resolve()),
         "--root", str(tmp_path), "--check-local-ba"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "SceneModel.apply_local_ba_result" in result.stderr
    assert "no SLAM sequence was started" in result.stderr


def test_trajectory_preflight_reports_hashes_without_importing_sources(tmp_path):
    make_trajectory_sources(tmp_path)
    report = calibration_preflight.check_trajectory_sources(tmp_path)
    assert report["errors"] == []
    assert set(report["files"]) == {"slam.py", "slam_outputs.py"}
    for relative, entry in report["files"].items():
        path = tmp_path / relative
        assert entry == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_trajectory_preflight_accepts_reversed_caller_dictionary_order(tmp_path):
    entrypoint = '''
def run_slam(args):
    return write_trajectory_stage_outputs(args.model_path, {
        "online_mapping": online_records,
        "frontend_miniba": frontend_records,
    })
'''
    make_trajectory_sources(tmp_path, entrypoint=entrypoint)
    report = calibration_preflight.check_trajectory_sources(tmp_path)
    assert report["errors"] == []


@pytest.mark.parametrize("declaration", [
    'TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping", "final_refinement")',
    'TRAJECTORY_STAGES = ("online_mapping", "frontend_miniba")',
    'TRAJECTORY_STAGES = ("frontend_miniba",)',
    '# TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping")',
    'TRAJECTORY_STAGES = runtime_stages()',
])
def test_trajectory_preflight_rejects_stale_or_unverifiable_exporter(tmp_path, declaration):
    make_trajectory_sources(tmp_path, outputs=declaration)
    with pytest.raises(RuntimeError, match="slam_outputs.py TRAJECTORY_STAGES must be exactly"):
        calibration_preflight.check_trajectory_sources(tmp_path)


@pytest.mark.parametrize("entrypoint", [
    CURRENT_TRAJECTORY_ENTRYPOINT.replace('"online_mapping": online_records,',
                                         '"online_mapping": online_records, "final_refinement": final_records,'),
    CURRENT_TRAJECTORY_ENTRYPOINT.replace('"online_mapping": online_records,', ''),
    CURRENT_TRAJECTORY_ENTRYPOINT.replace('"online_mapping": online_records,',
                                         '"frontend_miniba": online_records,'),
    'def run_slam(args):\n    write_trajectory_stage_outputs(args.model_path, dynamic_records)\n',
    '# write_trajectory_stage_outputs(args.model_path, {"frontend_miniba": x, "online_mapping": y})\n',
])
def test_trajectory_preflight_rejects_stale_or_unverifiable_producer(tmp_path, entrypoint):
    make_trajectory_sources(tmp_path, entrypoint=entrypoint)
    with pytest.raises(RuntimeError, match="slam.py.*write_trajectory_stage_outputs"):
        calibration_preflight.check_trajectory_sources(tmp_path)


@pytest.mark.parametrize("relative", calibration_preflight.TRAJECTORY_FILES)
@pytest.mark.parametrize("failure", ["missing", "syntax"])
def test_trajectory_preflight_rejects_unreadable_sources(tmp_path, relative, failure):
    make_trajectory_sources(tmp_path)
    path = tmp_path / relative
    if failure == "missing":
        path.unlink()
    else:
        path.write_text("def (", encoding="utf-8")
    with pytest.raises(RuntimeError, match=relative):
        calibration_preflight.check_trajectory_sources(tmp_path)


def test_trajectory_preflight_is_opt_in_for_standalone_miniba_checks(tmp_path):
    make_sources(tmp_path)
    report = calibration_preflight.check_calibration_sources(tmp_path)
    assert "slam.py" not in report["files"]
    assert "slam_outputs.py" not in report["files"]
    make_trajectory_sources(tmp_path)
    report = calibration_preflight.check_calibration_sources(tmp_path, check_trajectory_stages=True)
    assert set(calibration_preflight.TRAJECTORY_FILES) <= report["files"].keys()


@pytest.mark.parametrize("stale", [False, True])
def test_trajectory_preflight_cli_without_site_packages(tmp_path, stale):
    make_sources(tmp_path)
    outputs = CURRENT_TRAJECTORY_OUTPUTS
    if stale:
        outputs = outputs.replace('"online_mapping")', '"online_mapping", "final_refinement")')
    make_trajectory_sources(tmp_path, outputs=outputs)
    result = subprocess.run(
        [sys.executable, "-S", str(Path(calibration_preflight.__file__).resolve()),
         "--root", str(tmp_path), "--check-trajectory-stages"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == int(stale), result.stderr
    if stale:
        assert "no SLAM sequence was started" in result.stderr
        assert "TRAJECTORY_STAGES" in result.stderr
        assert "restart Python" in result.stderr
    else:
        assert "Trajectory source interface OK" in result.stdout


def test_single_sequence_rejects_stale_exporter_before_cuda_or_dataset(monkeypatch, tmp_path):
    import slam

    make_trajectory_sources(tmp_path, outputs=CURRENT_TRAJECTORY_OUTPUTS.replace(
        '"online_mapping")', '"online_mapping", "final_refinement")'))
    monkeypatch.setattr(slam, "__file__", str(tmp_path / "slam.py"))
    monkeypatch.setattr(slam, "seed_everything", lambda *a: pytest.fail("CUDA seeding started"))
    monkeypatch.setattr(slam, "create_dataset", lambda *a: pytest.fail("Dataset loading started"))
    monkeypatch.setattr(slam, "SLAMFrontend", lambda *a: pytest.fail("Tracking initialized"))
    monkeypatch.setattr(slam, "SLAMBackend", lambda *a: pytest.fail("Mapping initialized"))
    with pytest.raises(RuntimeError, match="TRAJECTORY_STAGES"):
        slam.run_slam(SimpleNamespace())


@pytest.mark.parametrize("entrypoint", ["waymo", "kitti", "kitti_wrapper"])
@pytest.mark.parametrize("disable_local_ba", [False, True])
def test_batches_reject_mixed_trajectory_versions_before_first_sequence(
    monkeypatch, tmp_path, entrypoint, disable_local_ba
):
    from scripts import run_all_slam, run_kitti_slam

    make_local_ba_sources(tmp_path)
    make_trajectory_sources(tmp_path, outputs=CURRENT_TRAJECTORY_OUTPUTS.replace(
        '"online_mapping")', '"online_mapping", "final_refinement")'))
    monkeypatch.setattr(run_all_slam, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *a: [tmp_path / "sequence"])
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *a, **kw: pytest.fail("SLAM was launched"))
    options = ["--output_root", str(tmp_path / "results")]
    if disable_local_ba:
        options.append("--local_ba_disable")
    with pytest.raises(RuntimeError, match="TRAJECTORY_STAGES"):
        if entrypoint == "kitti_wrapper":
            run_kitti_slam.main(options)
        else:
            run_all_slam.main(["--dataset_type", entrypoint, *options])
    assert not (tmp_path / "results/slam_batch_summary.json").exists()
