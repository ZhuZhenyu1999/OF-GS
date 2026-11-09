"""Build a focused offline Gaussian-ablation Python source update."""

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_calibration_update import build_update


# The parser, entry point and scene implementation must be deployed together.
# This update targets the calibration/local-BA baseline already deployed on the
# server. Clone/split densification uses the existing renderer's means2D
# gradients, so no rasterizer ABI, CUDA sources or compiled extensions change.
GAUSSIAN_ABLATION_SOURCE_FILES = (
    "dataloaders/dl3dv_dataset.py",
    "args.py",
    "utils.py",
    "slam.py",
    "slam_backend.py",
    "slam_outputs.py",
    "depth_export.py",
    "scripts/export_da2_depth.py",
    "tests/test_depth_export.py",
    "visualization/__init__.py",
    "visualization/server.py",
    "visualization/snapshot.py",
    "visualization/index.html",
    "scene/scene_model.py",
    "scene/keyframe.py",
    "scripts/run_all_slam.py",
    "scripts/calibration_preflight.py",
    "README_TEST2_SLAM.md",
    "scripts/build_calibration_update.py",
    "scripts/build_gaussian_ablation_update.py",
    "tests/test_args.py",
    "tests/test_backend_intrinsics.py",
    "tests/test_live_viewer.py",
    "tests/test_gaussian_ablation.py",
    "tests/test_slam_rules.py",
    "tests/test_run_all_slam.py",
    "tests/test_gaussian_ablation_update.py",
)


def build_gaussian_ablation_update(root: Path, destination: Path) -> Path:
    """Package matching Gaussian controls without replacing unrelated assets."""
    return build_update(
        root,
        destination,
        source_files=GAUSSIAN_ABLATION_SOURCE_FILES,
        optional_files=(),
        remove_paths=(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    destination = args.output or args.root / "artifacts/apply_gaussian_ablation_update_20261029.py"
    result = build_gaussian_ablation_update(args.root, destination)
    print(result)
    print("SHA256: " + hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
