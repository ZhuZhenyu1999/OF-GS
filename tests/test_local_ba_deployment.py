from pathlib import Path
import shutil

import pytest

from slam_backend import validate_local_ba_scene_interface


def test_runtime_guard_identifies_missing_map_commit_interface():
    class OldSceneModel:
        pass

    with pytest.raises(RuntimeError, match="apply_local_ba_result") as error:
        validate_local_ba_scene_interface(OldSceneModel)
    assert str(Path(__file__).resolve()) in str(error.value)
    assert "restart Python" in str(error.value)


def test_runtime_guard_requires_ownership_update_not_only_method_name():
    class IncompleteSceneModel:
        def apply_local_ba_result(self, poses, points):
            pass
        def _fit_gaussian_owner_keyframe_ids(self):
            pass
        def _rotate_gaussian_quaternions(self):
            pass
        def _apply_gaussian_add_and_prune(self, extensions, mask):
            pass

    with pytest.raises(RuntimeError, match="extension_owner_ids"):
        validate_local_ba_scene_interface(IncompleteSceneModel)


def test_runtime_guard_accepts_complete_interface():
    class CompleteSceneModel:
        def apply_local_ba_result(self, poses, points):
            pass
        def _fit_gaussian_owner_keyframe_ids(self):
            pass
        def _rotate_gaussian_quaternions(self):
            pass
        def _apply_gaussian_add_and_prune(self, extensions, mask, extension_owner_ids=None):
            pass

    validate_local_ba_scene_interface(CompleteSceneModel)


def test_batch_requests_no_local_ba_preflight_for_disabled_ablation(monkeypatch, tmp_path):
    from scripts import run_all_slam

    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [tmp_path / "100613"])
    checked = []
    monkeypatch.setattr(run_all_slam, "check_calibration_sources", lambda root, **kw: checked.append(kw))
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *args: {"returncode": 0, "sequence": "100613"})
    run_all_slam.main(["--output_root", str(tmp_path / "results"), "--local_ba_disable"])
    assert checked == [{"check_local_ba": False, "check_trajectory_stages": True}]


def test_partial_scene_sync_stops_batch_before_any_sequence_starts(monkeypatch, tmp_path):
    from scripts import calibration_preflight, run_all_slam

    root = Path(__file__).resolve().parents[1]
    for relative in (calibration_preflight.CALIBRATION_FILES + calibration_preflight.LOCAL_BA_FILES
                     + calibration_preflight.TRAJECTORY_FILES):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    scene_path = tmp_path / "scene/scene_model.py"
    scene_path.write_text(scene_path.read_text(encoding="utf-8").replace(
        "    def apply_local_ba_result(", "    def missing_apply_local_ba_result(", 1), encoding="utf-8")
    monkeypatch.setattr(run_all_slam, "ROOT", tmp_path)
    monkeypatch.setattr(run_all_slam, "discover_jobs", lambda *args: [tmp_path / "100613", tmp_path / "106762"])
    monkeypatch.setattr(run_all_slam, "run_sequence", lambda *args: pytest.fail("SLAM was launched"))
    with pytest.raises(RuntimeError, match="SceneModel.apply_local_ba_result"):
        run_all_slam.main(["--output_root", str(tmp_path / "results")])
