"""Check source updates against local imports without importing CUDA or models."""

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGES = {"poses", "scene", "dataloaders", "visualization"}
ENTRY_POINTS = ("slam.py", "scripts/run_all_slam.py")


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "source_package_dependency_builder", ROOT / "scripts/build_calibration_update.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_core_module(root, module_name):
    parts = module_name.split(".")
    if not module_name or (len(parts) > 1 and parts[0] not in CORE_PACKAGES):
        return None
    path = root.joinpath(*parts)
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if parts[0] in CORE_PACKAGES and (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def _imported_modules(root, source):
    # ast.walk also finds imports inside constructors and other lazy loaders.
    package_parts = source.relative_to(root).parent.parts
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package_parts):
                    continue
                prefix = package_parts[: len(package_parts) - node.level + 1]
                suffix = tuple(node.module.split(".")) if node.module else ()
                base = ".".join((*prefix, *suffix))
            else:
                base = node.module or ""
            if base:
                yield base
                # Handles both "from package import module" and relative imports.
                for alias in node.names:
                    if alias.name != "*":
                        yield base + "." + alias.name


def _runtime_source_closure(root, entry_points=ENTRY_POINTS):
    root = root.resolve()
    pending = [root / name for name in entry_points]
    visited = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        for module_name in _imported_modules(root, source):
            imported = _resolve_core_module(root, module_name)
            if imported is None:
                continue
            pending.append(imported)
            # Python also executes regular parent packages' __init__.py files.
            for directory in imported.parents:
                if directory == root:
                    break
                initializer = directory / "__init__.py"
                if initializer.is_file():
                    pending.append(initializer)
    return {source.relative_to(root).as_posix() for source in visited}


def test_update_contains_complete_runtime_source_dependencies():
    builder = _load_builder()
    closure = _runtime_source_closure(ROOT)
    assert {"poses/clidd_adapter.py", "scene/scene_model.py", "dataloaders/__init__.py"} <= closure
    missing = closure - set(builder.SOURCE_FILES)
    assert not missing, "Runtime files missing from the update plan: " + ", ".join(sorted(missing))


def test_omitting_clidd_adapter_is_detected():
    builder = _load_builder()
    planned = set(builder.SOURCE_FILES)
    assert "poses/clidd_adapter.py" in builder.SOURCE_FILES
    planned.remove("poses/clidd_adapter.py")
    assert _runtime_source_closure(ROOT) - planned == {"poses/clidd_adapter.py"}


def test_update_includes_dynamically_loaded_descriptor_sources():
    # These sources are loaded from a path at runtime, so the SLAM AST import
    # graph alone cannot find their dependencies.
    builder = _load_builder()
    expected = {
        "CLIDD-main/clidd.py", "CLIDD-main/model/model.py",
        "CLIDD-main/model/modules.py", "CLIDD-main/model/triton_plugin.py",
        "CLIDD-main/LICENSE", "CLIDD-main/requirements.txt", "CLIDD-main/README.md",
        "CLIDD-main/weights/S64.pth", "requirements.txt",
    }
    assert expected <= set(builder.SOURCE_FILES)
    assert all((ROOT / name).is_file() for name in expected)
    assert hashlib.sha256((ROOT / "CLIDD-main/weights/S64.pth").read_bytes()).hexdigest() == (
        "bcd378383a9c0193247181e4d8a1369a9b30342183d314e23fd55b869dd4e6fd"
    )


def test_updates_include_viewer_assets_and_optional_depth_export():
    from scripts.build_gaussian_ablation_update import GAUSSIAN_ABLATION_SOURCE_FILES
    from scripts.build_local_ba_update import LOCAL_BA_SOURCE_FILES
    from scripts.build_viewer_update import LIVE_VIEWER_SOURCE_FILES

    expected = {
        "visualization/__init__.py", "visualization/server.py",
        "visualization/snapshot.py", "visualization/index.html",
        "tests/test_live_viewer.py",
        "depth_export.py", "scripts/export_da2_depth.py", "tests/test_depth_export.py",
    }
    for files in (_load_builder().SOURCE_FILES, GAUSSIAN_ABLATION_SOURCE_FILES,
                  LOCAL_BA_SOURCE_FILES, LIVE_VIEWER_SOURCE_FILES):
        assert expected <= set(files)
        assert "args.py" in files
        assert {"slam.py", "slam_backend.py", "slam_outputs.py"} <= set(files)
        assert "scripts/calibration_preflight.py" in files
        assert "dataloaders/dl3dv_dataset.py" in files
    assert all((ROOT / name).is_file() for name in expected)


def test_full_update_contains_trajectory_repair_builder_tests_and_gpu2_runner():
    assert {
        "scripts/build_trajectory_update.py", "tests/test_trajectory_update.py",
        "scripts/run_waymo_kitti_gpu2.sh",
    } <= set(_load_builder().SOURCE_FILES)


def test_updates_keep_mapping_loss_api_and_keyframe_schedule_together():
    from scripts.build_gaussian_ablation_update import GAUSSIAN_ABLATION_SOURCE_FILES
    from scripts.build_local_ba_update import LOCAL_BA_SOURCE_FILES
    from scripts.build_viewer_update import LIVE_VIEWER_SOURCE_FILES

    # The scene and loss helper must agree on the monocular-depth arguments;
    # the keyframe owns the depth-weight and resolution schedules. Deploying
    # only one can leave an incompatible loss API or schedule on the server.
    required = {
        "utils.py", "scene/scene_model.py", "scene/keyframe.py", "args.py",
        "tests/test_backend_intrinsics.py", "tests/test_slam_rules.py",
        "tests/test_gaussian_ablation.py",
    }
    for files in (_load_builder().SOURCE_FILES, LOCAL_BA_SOURCE_FILES,
                  GAUSSIAN_ABLATION_SOURCE_FILES, LIVE_VIEWER_SOURCE_FILES):
        assert required <= set(files)


def test_standalone_depth_export_dependencies_are_packaged():
    from scripts.build_viewer_update import LIVE_VIEWER_SOURCE_FILES

    closure = _runtime_source_closure(ROOT, entry_points=("scripts/export_da2_depth.py",))
    assert "depth_export.py" in closure
    assert closure <= set(LIVE_VIEWER_SOURCE_FILES)


def test_live_viewer_update_applies_offline_and_preserves_existing_results(tmp_path):
    from scripts.build_viewer_update import LIVE_VIEWER_SOURCE_FILES, build_viewer_update

    script = build_viewer_update(ROOT, tmp_path / "apply_live_viewer.py")
    checkout = tmp_path / "server checkout"
    checkout.mkdir()
    (checkout / "slam.py").write_text("# old entry point\n", encoding="utf-8")
    (checkout / "results").mkdir()
    result_file = checkout / "results" / "preserve.txt"
    result_file.write_bytes(b"existing experiment")
    for extra in ([], ["--check-only"], []):
        completed = subprocess.run(
            [sys.executable, "-S", str(script), "--root", str(checkout), *extra],
            capture_output=True, text=True, check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    for name in LIVE_VIEWER_SOURCE_FILES:
        assert (checkout / name).read_bytes() == (ROOT / name).read_bytes()
    assert result_file.read_bytes() == b"existing experiment"
    backups = list((checkout / ".calibration-backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "slam.py").read_text(encoding="utf-8") == "# old entry point\n"


def test_closure_follows_lazy_relative_and_package_imports_without_vendor(tmp_path):
    files = {
        "slam.py": "import root_helper\nfrom poses.feature_detector import Detector\n",
        "scripts/run_all_slam.py": "import root_helper\n",
        "root_helper.py": "import torch\nimport vendor.backend\n",
        "poses/feature_detector.py": (
            "def lazy_loader():\n    from scene.dense_extractor import DenseExtractor\n"
        ),
        "scene/dense_extractor.py": "from dataloaders.reader import read\n",
        "dataloaders/__init__.py": "from . import metadata\n",
        "dataloaders/metadata.py": "VERSION = 1\n",
        "dataloaders/reader.py": "from .helpers import resolve\n",
        "dataloaders/helpers.py": "from poses.feature_detector import Detector\n",
        "vendor/backend.py": "raise RuntimeError('vendor must not be imported')\n",
    }
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    assert _runtime_source_closure(tmp_path) == set(files) - {"vendor/backend.py"}
