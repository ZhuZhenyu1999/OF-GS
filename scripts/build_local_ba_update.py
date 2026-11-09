"""Build a focused offline local-BA update without detector or CUDA changes."""

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_calibration_update import build_update


# These files form one local-BA revision. In particular, the scene commit and
# Gaussian ownership lifecycle must accompany the backend and solver changes.
LOCAL_BA_SOURCE_FILES = (
    "args.py",
    "utils.py",
    "slam.py",
    "slam_outputs.py",
    "depth_export.py",
    "scripts/export_da2_depth.py",
    "tests/test_depth_export.py",
    "visualization/__init__.py",
    "visualization/server.py",
    "visualization/snapshot.py",
    "visualization/index.html",
    # Current entry points import the KITTI reader even for Waymo runs.
    "dataloaders/kitti_dataset.py",
    "dataloaders/dl3dv_dataset.py",
    "slam_backend.py",
    "poses/local_ba.py",
    "poses/local_ba_solver.py",
    "scene/scene_model.py",
    "scene/keyframe.py",
    "scripts/run_all_slam.py",
    "scripts/calibration_preflight.py",
    "README_TEST2_SLAM.md",
    "tests/test_local_ba.py",
    "tests/test_live_viewer.py",
    "tests/test_local_ba_solver.py",
    "tests/test_local_ba_map.py",
    "tests/test_calibration_preflight.py",
    "tests/test_local_ba_deployment.py",
    "tests/test_backend_intrinsics.py",
    "tests/test_slam_rules.py",
    "tests/test_gaussian_ablation.py",
)


def build_local_ba_update(root: Path, destination: Path) -> Path:
    """Package the complete BA feature while preserving all unrelated assets."""
    return build_update(
        root,
        destination,
        source_files=LOCAL_BA_SOURCE_FILES,
        optional_files=(),
        remove_paths=(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    destination = args.output or args.root / "artifacts/apply_local_ba_update_20261013.py"
    result = build_local_ba_update(args.root, destination)
    print(result)
    print("SHA256: " + hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
