"""DL3DV exports: rgb/ plus normalized intrinsics.json and cameras.json."""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from dataloaders.kitti_dataset import IMAGE_EXTENSIONS, KITTIDataset
from utils import is_test_hold_frame


def discover_dl3dv_sequences(source_path):
    root = Path(source_path)

    def is_sequence(path):
        return (path / "rgb").is_dir() and (path / "intrinsics.json").is_file()

    if not root.is_dir():
        return []
    if is_sequence(root):
        return [root]
    children = [path for path in root.iterdir() if path.is_dir() and is_sequence(path)]
    return sorted(children, key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name))


def normalized_intrinsics(values, width, height):
    """Export values use width for fx/cx and height for fy/cy."""
    try:
        fx, fy, cx, cy = (float(values[key]) for key in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("DL3DV intrinsics require numeric fx, fy, cx, cy") from exc
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError("DL3DV intrinsics must be finite with positive focal lengths")
    return np.array([[fx * width, 0, cx * width],
                     [0, fy * height, cy * height], [0, 0, 1]], dtype=np.float64)


def camera_pose(camera, quaternion_order="xyzw"):
    """Build the exported rigid transform; direction is selected separately."""
    quaternion = np.asarray(camera["cam_quat"], dtype=np.float64)
    translation = np.asarray(camera["cam_trans"], dtype=np.float64)
    if (quaternion.shape != (4,) or translation.shape != (3,)
            or not np.isfinite(quaternion).all() or not np.isfinite(translation).all()):
        raise ValueError("DL3DV camera requires a finite quaternion (4) and translation (3)")
    if quaternion_order == "wxyz":
        quaternion = quaternion[[1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"Unknown DL3DV quaternion order: {quaternion_order}")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("DL3DV camera quaternion must be nonzero")
    x, y, z, w = quaternion / norm
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ]
    pose[:3, 3] = translation
    return pose


class DL3DVDataset(KITTIDataset):
    """Reuse the RGB loading/iteration interface, with DL3DV calibration/poses.

    Export pose defaults are xyzw/c2w in the OpenCV camera frame. These are
    explicit conventions, not inferred from the quaternion values. GT is used
    only for evaluation; missing cameras.json or unmatched entries omit GT.
    """

    def __init__(self, args):
        self.args = args
        sequences = discover_dl3dv_sequences(args.source_path)
        if not sequences:
            raise FileNotFoundError(f"No DL3DV sequence with rgb/ and intrinsics.json under {args.source_path}")
        self.root = self.sequence_root = sequences[0]
        if len(sequences) > 1:
            logging.warning("Multiple DL3DV sequences found; using %s", self.root)
        self.rgb_dir = self.root / "rgb"
        self.calib_path = self.root / "intrinsics.json"
        self.calibration = json.loads(self.calib_path.read_text(encoding="utf-8"))
        self.pose_format = getattr(args, "pose_format", "auto")
        if self.pose_format not in {"auto", "c2w", "w2c"}:
            raise ValueError(f"Unknown pose format: {self.pose_format}")
        self.quaternion_order = getattr(args, "dl3dv_quaternion_order", "xyzw")
        self.device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.downsampling = float(args.downsampling)
        if not np.isfinite(self.downsampling) or args.start_at < 0:
            raise ValueError("DL3DV requires finite downsampling and nonnegative start_at")
        paths = sorted(path for path in self.rgb_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        selected = list(enumerate(paths))[args.start_at:]
        if args.max_frames > 0:
            selected = selected[:args.max_frames]
        if not selected:
            raise FileNotFoundError(f"No DL3DV images remain in {self.rgb_dir} after start_at/max_frames filtering")
        self.source_indices, self.image_paths = map(list, zip(*selected))
        first = self._read_image(self.image_paths[0])
        self.original_height, self.original_width = first.shape[:2]
        self.source_K = normalized_intrinsics(self.calibration, self.original_width, self.original_height)
        resolution = self.original_width * self.original_height
        if self.downsampling <= 0 and resolution > 1_500_000:
            self.downsampling = (resolution / 1_500_000) ** 0.5
        scale = self.downsampling if self.downsampling > 0 else 1.0
        self.width = max(1, round(self.original_width / scale))
        self.height = max(1, round(self.original_height / scale))

        self.cameras_path = self.root / "cameras.json"
        self.cameras = {}
        if self.cameras_path.is_file():
            cameras = json.loads(self.cameras_path.read_text(encoding="utf-8"))
            if not isinstance(cameras, list):
                raise ValueError(f"Expected a camera list in {self.cameras_path}")
            for camera in cameras:
                name = camera["image_name"].replace("\\", "/").rsplit("/", 1)[-1]
                if name in self.cameras:
                    raise ValueError(f"Duplicate DL3DV camera image_name: {name}")
                self.cameras[name] = camera
        self.infos = [self._build_info(index, path) for index, path in enumerate(self.image_paths)]
        self.K = self.infos[0]["K_np"].copy()
        self.current_index = 0

    def _build_info(self, index, image_path):
        camera = self.cameras.get(image_path.name, {})
        calibration = {**self.calibration, **{key: camera[key] for key in ("fx", "fy", "cx", "cy") if key in camera}}
        source_K = normalized_intrinsics(calibration, self.original_width, self.original_height)
        K = source_K.copy()
        sx, sy = self.width / self.original_width, self.height / self.original_height
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        info = {
            "is_test": is_test_hold_frame(index, self.args.test_hold),
            "name": image_path.name, "frame_id": index,
            "source_index": self.source_indices[index], "frame_stem": image_path.stem,
            "dataset_type": "dl3dv", "image_path": str(image_path),
            "depth_path": None, "calib_path": str(self.calib_path),
            "gt_path": str(self.cameras_path) if camera else None,
            "K_np": K.astype(np.float32), "source_K_np": source_K,
            "focal_np": K[[0, 1], [0, 1]].astype(np.float32),
            "intrinsics_calibrated": True, "image_model": "pinhole",
        }
        if "cam_quat" in camera and "cam_trans" in camera:
            pose = camera_pose(camera, self.quaternion_order)
            fmt = "c2w" if self.pose_format == "auto" else self.pose_format
            c2w = pose if fmt == "c2w" else np.linalg.inv(pose)
            info.update(gt_c2w_np=c2w.astype(np.float32),
                        Rt_np=np.linalg.inv(c2w).astype(np.float32),
                        gt_pose_format=fmt, gt_pose_format_verified=False,
                        gt_quaternion_order=self.quaternion_order)
        return info
