"""Build an offline, self-contained source updater (standard library only)."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


SOURCE_FILES = (
    "dataloaders/dl3dv_dataset.py", "tests/test_dl3dv_dataset.py",
    "args.py", "utils.py", "slam.py", "slam_frontend.py", "slam_backend.py",
    "depth_export.py", "scripts/export_da2_depth.py", "tests/test_depth_export.py",
    "visualization/__init__.py", "visualization/server.py",
    "visualization/snapshot.py", "visualization/index.html",
    "slam_outputs.py", "dataloaders/waymo_dataset.py", "dataloaders/tum_dataset.py",
    "dataloaders/kitti_dataset.py",
    "dataloaders/image_dataset.py", "dataloaders/read_write_model.py",
    "dataloaders/__init__.py", "poses/pose_initializer.py", "poses/mini_ba.py",
    "poses/local_ba.py", "poses/local_ba_solver.py",
    "poses/ransac.py", "poses/ransac.cu", "poses/triangulator.py", "poses/guided_mvs.py",
    "poses/clidd_adapter.py", "poses/feature_detector.py", "poses/matcher.py",
    "poses/guided_mvs.cu", "poses/cuda_utils.cuh",
    "scene/keyframe.py", "scene/scene_model.py", "scene/dense_extractor.py",
    "scene/anchor.py", "scene/mono_depth.py", "scene/optimizers.py",
    "requirements.txt", "CLIDD-main/clidd.py", "CLIDD-main/model/model.py",
    "CLIDD-main/model/modules.py", "CLIDD-main/model/triton_plugin.py",
    "CLIDD-main/LICENSE", "CLIDD-main/requirements.txt", "CLIDD-main/README.md",
    "CLIDD-main/weights/S64.pth",
    "tests/test_camera_intrinsics.py",
    "tests/test_geometry_intrinsics.py", "tests/test_backend_intrinsics.py",
    "tests/test_dataloaders.py", "tests/test_kitti_dataset.py", "tests/test_kitti_entrypoint.py",
    "tests/test_args.py", "tests/test_renderer_smoke.py", "README_TEST2_SLAM.md",
    "scripts/run_all_slam.py", "scripts/calibration_preflight.py",
    "scripts/check_bootstrap_calibration.py",
    "scripts/build_calibration_update.py", "tests/test_calibration_update.py",
    "scripts/build_local_ba_update.py", "tests/test_local_ba_update.py",
    "scripts/build_gaussian_ablation_update.py", "tests/test_gaussian_ablation_update.py",
    "scripts/build_viewer_update.py", "tests/test_live_viewer.py",
    "scripts/build_trajectory_update.py", "tests/test_trajectory_update.py",
    "scripts/run_waymo_kitti_gpu2.sh",
    "tests/test_local_ba_deployment.py",
    "tests/test_source_package_dependencies.py",
    "tests/test_local_ba.py", "tests/test_local_ba_solver.py", "tests/test_local_ba_map.py",
    "tests/test_slam_rules.py", "tests/test_gaussian_ablation.py",
    "tests/test_outputs.py", "tests/test_trajectory_stages.py",
    "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
    "submodules/diff-gaussian-rasterization/rasterize_points.h",
    "submodules/diff-gaussian-rasterization/rasterize_points.cu",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu",
    "submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu",
)
OPTIONAL_FILES = ("tests/test_run_all_slam.py", "tests/test_calibration_preflight.py")
# Package the complete core import graph and the dynamically loaded CLIDD
# implementation so server checkouts cannot retain an incompatible extractor.
REMOVE_PATHS = (
    "poses/vggt_batcher.py", "poses/vggt_runner.py", "vggt-main",
    "artifacts/apply_trajectory_update_20261013.py",
    "artifacts/apply_server_calibration_20261012.py",
    "artifacts/calibration_sync_20261012.zip", "artifacts/server_calibration_20261012",
    "artifacts/apply_clidd_import_fix_20261013.py",
    "artifacts/apply_source_cleanup_20261013.py",
    "models/cache", "accelerated_features", "accelerated_features-main",
    "submodules/accelerated_features", "poses/xfeat_adapter.py", "scene/extractor_model.py",
)


def _validate_name(name):
    if not isinstance(name, str):
        raise ValueError("Unsafe payload path: " + repr(name))
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe payload path: " + repr(name))
    return path


def _validate_plan(source_names, remove_names):
    sources, removals = tuple(source_names), tuple(remove_names)
    for names, label in ((sources, "update source"), (removals, "removal")):
        for name in names:
            _validate_name(name)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate " + label + " paths")
    paths = [(name, _validate_name(name)) for name in sources + removals]
    for index, (name, path) in enumerate(paths):
        if path.parts[0] == ".calibration-backups":
            raise ValueError("Update plan overlaps backup directory: " + name)
        for other_name, other in paths[:index]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Conflicting update/removal paths: " + other_name + " and " + name)


    removed_backups = [PurePosixPath("removed") / name for name in removals]
    for name in sources:
        source = PurePosixPath(name)
        if name == "backup_manifest.json" or any(
                source.is_relative_to(saved) or saved.is_relative_to(source) for saved in removed_backups):
            raise ValueError("Conflicting backup paths for update source: " + name)


_RUNTIME = r'''
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import sys
import tempfile
import zipfile


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _validate_name(name):
    if not isinstance(name, str):
        raise ValueError("Unsafe payload path: " + repr(name))
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Unsafe payload path: " + repr(name))
    return path


def _validate_plan(source_names, remove_names):
    sources, removals = tuple(source_names), tuple(remove_names)
    for names, label in ((sources, "update source"), (removals, "removal")):
        for name in names:
            _validate_name(name)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate " + label + " paths")
    paths = [(name, _validate_name(name)) for name in sources + removals]
    for index, (name, path) in enumerate(paths):
        if path.parts[0] == ".calibration-backups":
            raise ValueError("Update plan overlaps backup directory: " + name)
        for other_name, other in paths[:index]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError("Conflicting update/removal paths: " + other_name + " and " + name)


    removed_backups = [PurePosixPath("removed") / name for name in removals]
    for name in sources:
        source = PurePosixPath(name)
        if name == "backup_manifest.json" or any(
                source.is_relative_to(saved) or saved.is_relative_to(source) for saved in removed_backups):
            raise ValueError("Conflicting backup paths for update source: " + name)


def _unpack_payload():
    archive = base64.b64decode(PAYLOAD_B64, validate=True)
    if _digest(archive) != PAYLOAD_SHA256:
        raise ValueError("Embedded archive SHA256 mismatch; no files were changed")
    _validate_plan(MANIFEST, REMOVE_PATHS)
    files = {}
    with zipfile.ZipFile(io.BytesIO(archive)) as payload:
        names = payload.namelist()
        if len(names) != len(set(names)) or set(names) != set(MANIFEST):
            raise ValueError("Embedded manifest does not match archive members")
        for name in names:
            _validate_name(name)
            info = payload.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsupported archive member: " + name)
            data = payload.read(name)
            if _digest(data) != MANIFEST[name]:
                raise ValueError("Embedded file SHA256 mismatch: " + name)
            files[name] = data
    return files


def _safe_target(root, name, allow_directory=False):
    path = _validate_name(name)
    if root.is_symlink():
        raise ValueError("Refusing symlink repository root: " + str(root))
    target = root.joinpath(*path.parts)
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("Refusing symlink in update target: " + str(cursor))
        if cursor != target and cursor.exists() and not cursor.is_dir():
            raise ValueError("Update parent is not a directory: " + str(cursor))
    if not target.resolve().is_relative_to(root):
        raise ValueError("Target escapes repository: " + str(target))
    if target.exists() and not (target.is_file() or (allow_directory and target.is_dir())):
        raise ValueError("Update target is not a regular file: " + str(target))
    return target


def _atomic_write(target, data, mode=0o644):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".calibration-", dir=target.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _print_miniba(root, label):
    path = root / "poses/mini_ba.py"
    checksum = _digest(path.read_bytes()) if path.is_file() else "MISSING"
    print("MiniBA " + label + ": " + str(path))
    print("MiniBA " + label + " SHA256: " + checksum)


def apply_update(root, check_only=False):
    files = _unpack_payload()  # Validate everything before touching the checkout.
    requested_root = Path(root).expanduser()
    if requested_root.is_symlink():
        raise ValueError("Refusing symlink repository root: " + str(requested_root))
    root = requested_root.resolve(strict=True)
    if not root.is_dir() or not (root / "slam.py").is_file():
        raise ValueError("--root must name the TEST2-SLAM checkout containing slam.py")
    targets = {name: _safe_target(root, name) for name in files}
    removal_targets = {name: _safe_target(root, name, allow_directory=True) for name in REMOVE_PATHS}
    print("Repository root: " + str(root), flush=True)
    print("Python: " + sys.executable, flush=True)
    _print_miniba(root, "before")
    changed = [name for name, target in targets.items()
               if not target.exists() or _digest(target.read_bytes()) != MANIFEST[name]]
    existing_removals = [name for name, target in removal_targets.items() if target.exists()]
    print("Changed files: " + str(len(changed)) + " / " + str(len(files)), flush=True)
    print("Obsolete paths present: " + str(len(existing_removals)), flush=True)
    if check_only:
        for name in changed:
            print("DIFF " + name)
        for name in existing_removals:
            print("REMOVE " + name)
        differs = bool(changed or existing_removals)
        print("CHECK FAILED: source differs or obsolete paths remain" if differs
              else "CHECK OK: all embedded SHA256 values match and obsolete paths are absent")
        return 1 if differs else 0
    if not changed and not existing_removals:
        print("Already up to date; no backup needed")
        _print_miniba(root, "after")
        return 0

    backup_base = root / ".calibration-backups"
    if backup_base.is_symlink() or not backup_base.resolve().is_relative_to(root):
        raise ValueError("Unsafe backup directory: " + str(backup_base))
    backup_base.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-")
    backup = Path(tempfile.mkdtemp(prefix=timestamp, dir=backup_base))
    records = {}
    for name in changed:
        target = targets[name]
        existed = target.exists()
        records[name] = {"existed": existed, "new_sha256": MANIFEST[name]}
        if existed:
            records[name].update(sha256=_digest(target.read_bytes()), mode=stat.S_IMODE(target.stat().st_mode))
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
            if _digest(saved.read_bytes()) != records[name]["sha256"]:
                raise RuntimeError("Backup verification failed: " + name)
    removal_records = {
        name: {"existed": target.exists(),
               "kind": "directory" if target.is_dir() else "file",
               "backup_path": "removed/" + name}
        for name, target in removal_targets.items()
    }
    journal = {"root": str(root), "status": "prepared", "files": records, "removed": removal_records}
    journal_path = backup / "backup_manifest.json"
    journal_path.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    print("Backup: " + str(backup), flush=True)
    attempted = []
    moved = []
    created_dirs = []
    try:
        for name in existing_removals:
            target = _safe_target(root, name, allow_directory=True)
            saved = backup / removal_records[name]["backup_path"]
            saved.parent.mkdir(parents=True, exist_ok=True)
            # The backup is in this checkout. Rename without a copy fallback so
            # large model directories stay intact and cross-device moves fail safely.
            os.rename(target, saved)
            moved.append(name)
        for name in changed:
            target = _safe_target(root, name)
            absent_parents = []
            parent = target.parent
            while not parent.exists():
                absent_parents.append(parent)
                parent = parent.parent
            for parent in reversed(absent_parents):
                parent.mkdir()
                created_dirs.append(parent)
            attempted.append(name)
            _atomic_write(target, files[name], records[name].get("mode", 0o644))
        for name in targets:
            target = _safe_target(root, name)
            if _digest(target.read_bytes()) != MANIFEST[name]:
                raise RuntimeError("Post-update SHA256 mismatch: " + name)
        for name in removal_targets:
            if _safe_target(root, name, allow_directory=True).exists():
                raise RuntimeError("Obsolete update path remains: " + name)
        journal["status"] = "complete"
        journal_path.write_text(json.dumps(journal, indent=2), encoding="utf-8")
    except Exception as error:
        rollback_errors = []
        for name in reversed(attempted):
            try:
                target = _safe_target(root, name)
                record = records[name]
                if record["existed"]:
                    data = (backup / name).read_bytes()
                    if _digest(data) != record["sha256"]:
                        raise RuntimeError("Backup SHA256 mismatch")
                    _atomic_write(target, data, record["mode"])
                    if _digest(target.read_bytes()) != record["sha256"]:
                        raise RuntimeError("Rollback SHA256 mismatch")
                elif target.exists():
                    target.unlink()
            except Exception as rollback_error:
                rollback_errors.append(name + ": " + str(rollback_error))
        for name in reversed(moved):
            try:
                target = _safe_target(root, name, allow_directory=True)
                if target.exists():
                    raise RuntimeError("Rollback target already exists")
                os.rename(backup / removal_records[name]["backup_path"], target)
                if not target.exists():
                    raise RuntimeError("Removed path was not restored")
            except Exception as rollback_error:
                rollback_errors.append(name + ": " + str(rollback_error))
        for parent in reversed(created_dirs):
            try:
                parent.rmdir()
            except OSError:
                pass
        journal["status"] = "rollback_failed" if rollback_errors else "rolled_back"
        journal["error"] = str(error)
        journal["rollback_errors"] = rollback_errors
        journal_path.write_text(json.dumps(journal, indent=2), encoding="utf-8")
        if rollback_errors:
            raise RuntimeError("Update failed; rollback needs attention. Backup: " + str(backup)
                               + "; " + "; ".join(rollback_errors)) from error
        raise RuntimeError("Update failed; original files restored. Backup: " + str(backup)
                           + "; " + str(error)) from error
    _print_miniba(root, "after")
    print("UPDATE OK: all " + str(len(files)) + " source SHA256 values verified")
    print("Removed obsolete paths: " + str(len(existing_removals)) + "; originals are in backup/removed")
    rebuild_rasterizer = any(name.startswith("submodules/diff-gaussian-rasterization/") for name in files)
    print("Restart SLAM only after rebuilding the rasterizer and checking this checkout:"
          if rebuild_rasterizer else "Restart SLAM after checking this Python source update:")
    python = shlex.quote(sys.executable)
    print("  cd " + shlex.quote(str(root)))
    if rebuild_rasterizer:
        print("  " + python + " -m pip install --no-build-isolation --no-deps --force-reinstall "
              "./submodules/diff-gaussian-rasterization")
    print("  " + python + " " + shlex.quote(str(Path(__file__).resolve()))
          + " --root " + shlex.quote(str(root)) + " --check-only")
    if "poses/clidd_adapter.py" in files:
        print("  " + python + " -c " + shlex.quote(
            "from poses.clidd_adapter import clidd_descriptor_dim; "
            "assert clidd_descriptor_dim('S64') == 64; print('CLIDD adapter import OK')"))
    if "poses/feature_detector.py" in files:
        print("  " + python + " -m pip install -r CLIDD-main/requirements.txt")
        print("  " + python + " -c " + shlex.quote(
            "from poses.clidd_adapter import validate_clidd_install; "
            "print(validate_clidd_install(None, 'S64'))"))
    if rebuild_rasterizer:
        print("  " + python + " -c " + shlex.quote(
            "from slam_outputs import write_trajectory_stage_outputs; print('trajectory output import OK')"))
        print("  " + python + " -m pytest tests/test_renderer_smoke.py -q -rs")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Apply the offline TEST2-SLAM source cleanup update")
    parser.add_argument("--root", required=True, type=Path, help="Explicit TEST2-SLAM checkout directory")
    parser.add_argument("--check-only", action="store_true", help="Check source SHA256 and obsolete paths without modifying files")
    args = parser.parse_args()
    try:
        return apply_update(args.root, args.check_only)
    except Exception as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_update(root: Path, destination: Path, *, source_files=None,
                 optional_files=None, remove_paths=None) -> Path:
    """Embed the current source revision in a runnable, offline Python updater."""
    root = Path(root).resolve(strict=True)
    destination = Path(destination).resolve()
    sources = tuple(SOURCE_FILES if source_files is None else source_files)
    optional = tuple(OPTIONAL_FILES if optional_files is None else optional_files)
    removals = tuple(REMOVE_PATHS if remove_paths is None else remove_paths)
    paths = list(sources)
    _validate_plan(paths, removals)
    for name in sources:
        if not (root / name).is_file():
            raise FileNotFoundError("Required update source is missing: " + str(root / name))
    for name in optional:
        _validate_name(name)
        if (root / name).is_file():
            paths.append(name)
    _validate_plan(paths, removals)
    manifest = {}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as payload:
        for name in paths:
            path = root / name
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Update source escapes root: " + name)
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            payload.writestr(name, data)
    archive_bytes = archive.getvalue()
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    header = ("#!/usr/bin/env python3\n"
              '"""Offline source cleanup update. Run with an explicit --root checkout."""\n'
              "MANIFEST = " + repr(manifest) + "\n"
              "REMOVE_PATHS = " + repr(removals) + "\n"
              "PAYLOAD_SHA256 = " + repr(hashlib.sha256(archive_bytes).hexdigest()) + "\n"
              "PAYLOAD_B64 = (\n"
              + "".join("    " + repr(encoded[index:index + 100]) + "\n"
                        for index in range(0, len(encoded), 100)) + ")\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(header + _RUNTIME, encoding="utf-8", newline="\n")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    destination = args.output or args.root / "artifacts/apply_clidd_only_update_20261013.py"
    result = build_update(args.root, destination)
    print(result)
    print("SHA256: " + hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
