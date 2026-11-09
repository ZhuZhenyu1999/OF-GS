"""Build an offline two-stage trajectory repair using the deployed CUDA/weights."""

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_calibration_update import build_update


# Synchronize the complete Python runtime: a new entry point must not retain an
# older backend, output schema, preflight, or batch runner. Existing CUDA source,
# compiled extensions, pretrained weights and experiment results stay in place.
TRAJECTORY_SOURCE_FILES = (
    "args.py", "utils.py", "slam.py", "slam_frontend.py", "slam_backend.py",
    "slam_outputs.py", "depth_export.py",
    "dataloaders/__init__.py", "dataloaders/image_dataset.py",
    "dataloaders/read_write_model.py", "dataloaders/waymo_dataset.py",
    "dataloaders/kitti_dataset.py", "dataloaders/tum_dataset.py",
    "dataloaders/dl3dv_dataset.py",
    "poses/clidd_adapter.py", "poses/feature_detector.py", "poses/matcher.py",
    "poses/pose_initializer.py", "poses/mini_ba.py", "poses/ransac.py",
    "poses/triangulator.py", "poses/guided_mvs.py", "poses/local_ba.py",
    "poses/local_ba_solver.py",
    "scene/anchor.py", "scene/dense_extractor.py", "scene/keyframe.py",
    "scene/mono_depth.py", "scene/optimizers.py", "scene/scene_model.py",
    "visualization/__init__.py", "visualization/server.py",
    "visualization/snapshot.py", "visualization/index.html",
    # The descriptor implementation is imported dynamically from these paths.
    "CLIDD-main/clidd.py", "CLIDD-main/model/model.py",
    "CLIDD-main/model/modules.py", "CLIDD-main/model/triton_plugin.py",
    "scripts/export_da2_depth.py", "scripts/run_all_slam.py",
    "scripts/run_kitti_slam.py", "scripts/run_waymo_kitti_gpu2.sh",
    "scripts/calibration_preflight.py", "scripts/build_calibration_update.py",
    "scripts/build_trajectory_update.py", "README_TEST2_SLAM.md",
    "tests/test_args.py", "tests/test_outputs.py", "tests/test_trajectory_stages.py",
    "tests/test_slam_rules.py", "tests/test_gaussian_ablation.py",
    "tests/test_backend_intrinsics.py",
    "tests/test_calibration_preflight.py", "tests/test_run_all_slam.py",
    "tests/test_run_kitti_slam.py", "tests/test_local_ba.py",
    "tests/test_local_ba_solver.py", "tests/test_local_ba_map.py",
    "tests/test_local_ba_deployment.py", "tests/test_live_viewer.py",
    "tests/test_depth_export.py", "tests/test_trajectory_update.py",
)


def build_trajectory_update(root: Path, destination: Path) -> Path:
    """Package matching runtime sources without optional payloads or removals."""
    return build_update(
        root, destination, source_files=TRAJECTORY_SOURCE_FILES,
        optional_files=(), remove_paths=(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    destination = args.output or args.root / "artifacts/apply_trajectory_update_20261108.py"
    result = build_trajectory_update(args.root, destination)
    print(result)
    print("SHA256: " + hashlib.sha256(result.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
