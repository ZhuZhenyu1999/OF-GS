"""Build a live-viewer/DA2-export update for the current calibration/CLIDD baseline."""

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_calibration_update import build_update


# The entry point and browser assets must move together. This focused update
# assumes the existing calibration, CLIDD, local-BA and renderer baseline.
LIVE_VIEWER_SOURCE_FILES = (
    "args.py", "slam.py", "slam_backend.py", "slam_outputs.py", "dataloaders/dl3dv_dataset.py",
    "utils.py", "scene/scene_model.py", "scene/keyframe.py",
    "depth_export.py", "scripts/export_da2_depth.py", "tests/test_depth_export.py",
    "visualization/__init__.py", "visualization/server.py",
    "visualization/snapshot.py", "visualization/index.html",
    "README_TEST2_SLAM.md",
    "scripts/build_viewer_update.py", "scripts/build_calibration_update.py",
    "scripts/calibration_preflight.py",
    "scripts/build_gaussian_ablation_update.py", "scripts/build_local_ba_update.py",
    "tests/test_args.py", "tests/test_live_viewer.py",
    "tests/test_backend_intrinsics.py", "tests/test_slam_rules.py", "tests/test_gaussian_ablation.py",
    "tests/test_source_package_dependencies.py",
    "tests/test_gaussian_ablation_update.py", "tests/test_local_ba_update.py",
)


def build_viewer_update(root: Path, destination: Path) -> Path:
    """Package visualization and depth export without deleting files or rebuilding CUDA."""
    return build_update(
        root, destination, source_files=LIVE_VIEWER_SOURCE_FILES,
        optional_files=(), remove_paths=(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    destination = args.output or args.root / "artifacts/apply_live_viewer_update_20261102.py"
    result = build_viewer_update(args.root, destination)
    print(result)
    print("SHA256: " + hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
