"""Deployment checks for the focused local-BA updater, using only stdlib."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.build_local_ba_update import LOCAL_BA_SOURCE_FILES, build_local_ba_update
from scripts.calibration_preflight import CALIBRATION_FILES


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "args.py", "slam.py", "slam_backend.py", "slam_outputs.py",
    "utils.py", "scene/keyframe.py",
    "depth_export.py", "scripts/export_da2_depth.py", "tests/test_depth_export.py",
    "dataloaders/kitti_dataset.py",
    "dataloaders/dl3dv_dataset.py",
    "visualization/__init__.py", "visualization/server.py",
    "visualization/snapshot.py", "visualization/index.html", "tests/test_live_viewer.py",
    "poses/local_ba.py", "poses/local_ba_solver.py", "scene/scene_model.py",
    "scripts/run_all_slam.py", "scripts/calibration_preflight.py",
    "README_TEST2_SLAM.md", "tests/test_local_ba.py",
    "tests/test_local_ba_solver.py", "tests/test_local_ba_map.py",
    "tests/test_calibration_preflight.py", "tests/test_local_ba_deployment.py",
    "tests/test_backend_intrinsics.py", "tests/test_slam_rules.py", "tests/test_gaussian_ablation.py",
}


def _load(path):
    spec = importlib.util.spec_from_file_location("generated_local_ba_update", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script, *arguments):
    return subprocess.run(
        [sys.executable, "-S", str(script), *(str(value) for value in arguments)],
        capture_output=True, text=True, check=False,
    )


@pytest.fixture
def update(tmp_path):
    script = build_local_ba_update(ROOT, tmp_path / "apply_local_ba_update.py")
    return script, _load(script)


def test_local_ba_manifest_contains_complete_feature_without_unrelated_changes(update):
    _, module = update
    assert set(LOCAL_BA_SOURCE_FILES) == EXPECTED_FILES
    assert len(LOCAL_BA_SOURCE_FILES) == len(EXPECTED_FILES)
    assert set(module.MANIFEST) == EXPECTED_FILES
    assert module.REMOVE_PATHS == ()
    forbidden = {
        "poses/feature_detector.py", "poses/matcher.py", "scene/dense_extractor.py",
        "poses/clidd_adapter.py", "poses/guided_mvs.py", "requirements.txt",
    }
    assert not forbidden.intersection(module.MANIFEST)
    assert not any(name.startswith(("CLIDD-main/", "submodules/"))
                   or Path(name).suffix in {".cu", ".cuh", ".h", ".pth"}
                   for name in module.MANIFEST)
    for name, digest in module.MANIFEST.items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_update_repairs_stale_scene_and_preserves_existing_pipeline(update, tmp_path):
    script, module = update
    checkout = tmp_path / "server checkout"
    checkout.mkdir()
    # The server log's calibration cohort already matches the local baseline.
    # Supply it unchanged so the post-update source preflight can check both
    # calibration and local BA without importing torch or CUDA.
    for name in CALIBRATION_FILES:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    originals = {
        "slam.py": b"# original server entry\n",
        "scene/scene_model.py": b"class SceneModel:\n    pass\n",
        "poses/feature_detector.py": b"# keep the server detector\n",
        "scene/dense_extractor.py": b"# keep the server dense extractor\n",
        "poses/guided_mvs.cu": b"// keep the compiled CUDA source\n",
        "models/cache/existing_model.bin": b"keep cached weights\x00\xff",
        "results/experiment.json": b'{"keep": true}\n',
    }
    for name, data in originals.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    before = _run(script, "--root", checkout, "--check-only")
    assert before.returncode == 1, before.stderr
    assert "DIFF scene/scene_model.py" in before.stdout
    assert not (checkout / ".calibration-backups").exists()
    assert (checkout / "scene/scene_model.py").read_bytes() == originals["scene/scene_model.py"]

    applied = _run(script, "--root", checkout)
    assert applied.returncode == 0, applied.stderr
    assert "UPDATE OK" in applied.stdout
    assert "pip install" not in applied.stdout
    for name, digest in module.MANIFEST.items():
        data = (checkout / name).read_bytes()
        assert data == (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
    for name, data in originals.items():
        if name not in module.MANIFEST:
            assert (checkout / name).read_bytes() == data

    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    journal = json.loads((backups[0] / "backup_manifest.json").read_text(encoding="utf-8"))
    assert journal["status"] == "complete"
    assert journal["removed"] == {}
    for name in ("slam.py", "scene/scene_model.py"):
        assert (backups[0] / name).read_bytes() == originals[name]
        assert journal["files"][name]["existed"] is True

    probe = _run(checkout / "scripts/calibration_preflight.py", "--root", checkout,
                 "--check_local_ba")
    assert probe.returncode == 0, probe.stderr
    assert "Local BA source interface OK" in probe.stdout
    assert str(checkout / "scene/scene_model.py") in probe.stdout

    checked = _run(script, "--root", checkout, "--check-only")
    assert checked.returncode == 0, checked.stderr
    assert "CHECK OK" in checked.stdout
    repeated = _run(script, "--root", checkout)
    assert repeated.returncode == 0, repeated.stderr
    assert "Already up to date" in repeated.stdout
    assert list((checkout / ".calibration-backups").iterdir()) == backups


def test_local_ba_builder_cli_works_without_site_packages(tmp_path):
    output = tmp_path / "offline_update.py"
    result = _run(ROOT / "scripts/build_local_ba_update.py", "--root", ROOT, "--output", output)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest() in result.stdout
    assert set(_load(output).MANIFEST) == EXPECTED_FILES
