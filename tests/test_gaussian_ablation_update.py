"""Exercise the Gaussian-ablation offline updater without site packages/CUDA."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.build_gaussian_ablation_update import (
    GAUSSIAN_ABLATION_SOURCE_FILES,
    build_gaussian_ablation_update,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = {
    "dataloaders/dl3dv_dataset.py",
    "args.py", "slam.py", "scene/scene_model.py", "scripts/run_all_slam.py",
    "utils.py", "scene/keyframe.py",
    "slam_backend.py", "slam_outputs.py",
    "scripts/calibration_preflight.py",
    "depth_export.py", "scripts/export_da2_depth.py", "tests/test_depth_export.py",
    "visualization/__init__.py", "visualization/server.py",
    "visualization/snapshot.py", "visualization/index.html", "tests/test_live_viewer.py",
    "README_TEST2_SLAM.md", "scripts/build_calibration_update.py",
    "scripts/build_gaussian_ablation_update.py", "tests/test_args.py",
    "tests/test_backend_intrinsics.py",
    "tests/test_gaussian_ablation.py", "tests/test_slam_rules.py",
    "tests/test_run_all_slam.py", "tests/test_gaussian_ablation_update.py",
}


def _load(path):
    spec = importlib.util.spec_from_file_location("generated_gaussian_ablation_update", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script, *arguments):
    return subprocess.run(
        [sys.executable, "-S", str(script), *(str(value) for value in arguments)],
        capture_output=True, text=True, check=False,
    )


def _parse_gaussian_switches(checkout, direct, gate):
    probe = (
        "import json, runpy, sys; "
        "source = runpy.run_path('args.py'); "
        "parsed = source['get_args'](); "
        "assert 'torch' not in sys.modules; "
        "print(json.dumps({'direct': parsed.direct_gaussian_sampling, "
        "'gate': parsed.pose_gaussian_gate}))"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", probe, "-s", "sequence", "-m", "output",
         "--direct_gaussian_sampling", direct, "--pose_gaussian_gate", gate],
        cwd=checkout, capture_output=True, text=True, check=False,
    )


@pytest.fixture
def update(tmp_path):
    script = build_gaussian_ablation_update(ROOT, tmp_path / "apply_gaussian_ablation_update.py")
    return script, _load(script)


def test_gaussian_update_manifest_contains_matching_python_feature(update):
    _, module = update
    assert set(GAUSSIAN_ABLATION_SOURCE_FILES) == EXPECTED_FILES
    assert len(GAUSSIAN_ABLATION_SOURCE_FILES) == len(EXPECTED_FILES)
    assert set(module.MANIFEST) == EXPECTED_FILES
    assert module.REMOVE_PATHS == ()
    assert not any(name.startswith(("CLIDD-main/", "submodules/", "models/"))
                   or Path(name).suffix in {".cu", ".cuh", ".h", ".pth"}
                   for name in module.MANIFEST)
    for name, digest in module.MANIFEST.items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_gaussian_update_repairs_old_parser_and_preserves_unrelated_assets(update, tmp_path):
    script, module = update
    checkout = tmp_path / "server checkout"
    checkout.mkdir()
    originals = {
        "args.py": (
            b"import argparse\ndef get_args():\n"
            b"    p = argparse.ArgumentParser()\n"
            b"    p.add_argument('-s')\n    p.add_argument('-m')\n"
            b"    return p.parse_args()\n"
        ),
        "slam.py": b"# original server entry\n",
        "scene/scene_model.py": b"class SceneModel:\n    pass\n",
        "poses/mini_ba.py": b"# keep deployed calibration baseline\n",
        "poses/local_ba.py": b"# keep deployed local BA\n",
        "slam_backend.py": b"# old deployed backend\n",
        "poses/feature_detector.py": b"# keep server detector\n",
        "scene/optimizers.py": b"# keep optimizer implementation\n",
        "submodules/diff-gaussian-rasterization/rasterize_points.cu": b"// keep CUDA source\n",
        "models/cache/existing_model.bin": b"cached weights\x00\xff",
        "results/experiment.json": b'{"keep": true}\n',
    }
    for name, data in originals.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    stale = _parse_gaussian_switches(checkout, "on", "on")
    assert stale.returncode == 2
    assert "unrecognized arguments: --direct_gaussian_sampling on --pose_gaussian_gate on" in stale.stderr

    before = _run(script, "--root", checkout, "--check-only")
    assert before.returncode == 1, before.stderr
    assert "DIFF args.py" in before.stdout
    assert not (checkout / ".calibration-backups").exists()
    assert (checkout / "args.py").read_bytes() == originals["args.py"]

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

    for direct in ("on", "off"):
        for gate in ("on", "off"):
            parsed = _parse_gaussian_switches(checkout, direct, gate)
            assert parsed.returncode == 0, parsed.stderr
            assert json.loads(parsed.stdout) == {"direct": direct == "on", "gate": gate == "on"}

    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    journal = json.loads((backups[0] / "backup_manifest.json").read_text(encoding="utf-8"))
    assert journal["status"] == "complete"
    assert journal["removed"] == {}
    for name in ("args.py", "slam.py", "scene/scene_model.py"):
        assert (backups[0] / name).read_bytes() == originals[name]
        assert journal["files"][name]["existed"] is True
    assert journal["files"]["scripts/run_all_slam.py"]["existed"] is False

    checked = _run(script, "--root", checkout, "--check-only")
    assert checked.returncode == 0, checked.stderr
    assert "CHECK OK" in checked.stdout
    repeated = _run(script, "--root", checkout)
    assert repeated.returncode == 0, repeated.stderr
    assert "Already up to date" in repeated.stdout
    assert list((checkout / ".calibration-backups").iterdir()) == backups


def test_gaussian_builder_cli_works_without_site_packages(tmp_path):
    output = tmp_path / "offline_update.py"
    result = _run(ROOT / "scripts/build_gaussian_ablation_update.py", "--root", ROOT, "--output", output)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert "SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest() in result.stdout
    assert set(_load(output).MANIFEST) == EXPECTED_FILES
