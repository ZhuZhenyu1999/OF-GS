"""Check that an offline update repairs a mixed two-/three-stage deployment."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.build_trajectory_update import TRAJECTORY_SOURCE_FILES, build_trajectory_update


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TRIO = {"slam.py", "slam_backend.py", "slam_outputs.py"}


def _load(path):
    spec = importlib.util.spec_from_file_location("generated_trajectory_update", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script, *arguments):
    return subprocess.run(
        [sys.executable, "-S", str(script), *(str(argument) for argument in arguments)],
        capture_output=True, text=True, check=False,
    )


def _runtime_python_sources(root):
    """Trace ordinary and lazy repository imports without loading CUDA/models."""
    packages = {"poses", "scene", "dataloaders", "visualization", "scripts"}
    pending = [root / name for name in (
        "slam.py", "scripts/run_all_slam.py", "scripts/run_kitti_slam.py",
        "scripts/calibration_preflight.py", "scripts/export_da2_depth.py",
    )]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        package = source.relative_to(root).parent.parts
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else ()
                base = ".".join((*prefix, *(() if node.module is None else node.module.split("."))))
                modules = [base, *(base + "." + alias.name for alias in node.names if alias.name != "*")]
            for name in modules:
                parts = name.split(".")
                if not name or (len(parts) > 1 and parts[0] not in packages):
                    continue
                candidate = root.joinpath(*parts)
                if candidate.with_suffix(".py").is_file():
                    imported = candidate.with_suffix(".py")
                elif parts[0] in packages and (candidate / "__init__.py").is_file():
                    imported = candidate / "__init__.py"
                else:
                    continue
                pending.append(imported)
                for parent in imported.parents:
                    if parent == root:
                        break
                    if (parent / "__init__.py").is_file():
                        pending.append(parent / "__init__.py")
    return {source.relative_to(root).as_posix() for source in visited}


@pytest.fixture
def update(tmp_path):
    script = build_trajectory_update(ROOT, tmp_path / "apply_trajectory_update.py")
    return script, _load(script)


def test_trajectory_manifest_has_complete_python_runtime_and_matching_stage_tests(update):
    _, module = update
    files = set(TRAJECTORY_SOURCE_FILES)
    assert len(files) == len(TRAJECTORY_SOURCE_FILES)
    assert set(module.MANIFEST) == files
    assert module.REMOVE_PATHS == ()
    assert RUNTIME_TRIO <= files
    assert _runtime_python_sources(ROOT) <= files
    assert {
        "visualization/index.html", "scripts/run_waymo_kitti_gpu2.sh",
        "scripts/build_trajectory_update.py", "scripts/build_calibration_update.py",
        "tests/test_outputs.py", "tests/test_trajectory_stages.py",
        "tests/test_slam_rules.py", "tests/test_calibration_preflight.py",
        "tests/test_backend_intrinsics.py",
        "tests/test_run_all_slam.py", "tests/test_run_kitti_slam.py",
        "CLIDD-main/clidd.py", "CLIDD-main/model/model.py",
        "CLIDD-main/model/modules.py", "CLIDD-main/model/triton_plugin.py",
    } <= files
    assert not any(
        name.startswith(("artifacts/", "models/", "results/", "submodules/"))
        or Path(name).suffix not in {".py", ".md", ".html", ".sh"}
        for name in files
    )
    for name, digest in module.MANIFEST.items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_trajectory_update_repairs_mixed_stage_sources_without_touching_results_or_cuda(update, tmp_path):
    script, module = update
    checkout = tmp_path / "server checkout"
    for name in TRAJECTORY_SOURCE_FILES:
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # Reproduce the reported deployment: current entry point, old output schema.
    output_path = checkout / "slam_outputs.py"
    source = output_path.read_text(encoding="utf-8")
    source = source.replace(
        'TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping")',
        'TRAJECTORY_STAGES = ("frontend_miniba", "online_mapping", "final_refinement")',
        1,
    )
    output_path.write_text(source, encoding="utf-8")
    old_output = output_path.read_bytes()
    assert old_output != (ROOT / "slam_outputs.py").read_bytes()
    untouched = {
        "poses/ransac.cu": b"// existing CUDA source\n",
        "submodules/renderer/renderer.pyd": b"existing compiled extension\x00",
        "CLIDD-main/weights/S64.pth": b"existing model weights\x00\xff",
        "models/cache/model.bin": b"existing cache",
        "results/waymo_gpu2_no_final/slam_metrics.json": b'{"existing": true}',
        "results/waymo_gpu2_no_final/slam_poses_final_refinement.json": b"historical result",
        "artifacts/old_update.py": b"# retain old updater\n",
    }
    for name, data in untouched.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    failed_preflight = _run(checkout / "scripts/calibration_preflight.py", "--root", checkout,
                            "--check_trajectory_stages")
    assert failed_preflight.returncode != 0, failed_preflight.stdout
    before = _run(script, "--root", checkout, "--check-only")
    assert before.returncode == 1, before.stdout + before.stderr
    assert "DIFF slam_outputs.py" in before.stdout
    assert output_path.read_bytes() == old_output
    assert not (checkout / ".calibration-backups").exists()

    applied = _run(script, "--root", checkout)
    assert applied.returncode == 0, applied.stdout + applied.stderr
    for name, digest in module.MANIFEST.items():
        assert hashlib.sha256((checkout / name).read_bytes()).hexdigest() == digest
    for name, data in untouched.items():
        assert (checkout / name).read_bytes() == data
    preflight = _run(checkout / "scripts/calibration_preflight.py", "--root", checkout,
                     "--check_local_ba", "--check_trajectory_stages")
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr

    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "slam_outputs.py").read_bytes() == old_output
    journal = json.loads((backups[0] / "backup_manifest.json").read_text(encoding="utf-8"))
    assert journal["status"] == "complete" and journal["removed"] == {}
    checked = _run(script, "--root", checkout, "--check-only")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    repeated = _run(script, "--root", checkout)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert "Already up to date" in repeated.stdout
    assert list((checkout / ".calibration-backups").iterdir()) == backups


def test_trajectory_builder_cli_runs_without_site_packages(tmp_path):
    output = tmp_path / "offline_trajectory_update.py"
    completed = _run(ROOT / "scripts/build_trajectory_update.py", "--root", ROOT, "--output", output)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHA256: " + hashlib.sha256(output.read_bytes()).hexdigest() in completed.stdout
    assert set(_load(output).MANIFEST) == set(TRAJECTORY_SOURCE_FILES)
