"""KITTI rectified images with optional, stem-matched ground-truth poses."""

import copy
import logging
from pathlib import Path

import numpy as np
import torch

from utils import is_test_hold_frame


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _image_directories(root, camera=None):
    if camera is None:
        cameras = range(4)
    else:
        cameras = [int(camera)]
    candidates = []
    for index in cameras:
        candidates.extend([root / f"image_{index}", root / f"image_{index:02d}" / "data"])
    # An explicit camera folder identifies its calibration unambiguously. rgb/
    # is the exported-layout fallback when no selected camera folder exists.
    return candidates + [root / "rgb"]


def _is_kitti_sequence(root):
    return (root / "calib.txt").is_file() and any(path.is_dir() for path in _image_directories(root))


def discover_kitti_sequences(source_path):
    """Find a sequence, a directory of sequences, or KITTI/sequences/."""
    root = Path(source_path)
    if not root.is_dir():
        return []
    if _is_kitti_sequence(root):
        return [root]
    sequences = [child for child in sorted(root.iterdir()) if child.is_dir() and _is_kitti_sequence(child)]
    sequences_root = root / "sequences"
    if sequences_root.is_dir():
        sequences.extend(child for child in sorted(sequences_root.iterdir())
                         if child.is_dir() and _is_kitti_sequence(child))
    return sequences


def parse_kitti_calib(calib_path, camera="2"):
    """Read the selected rectified camera's K from KITTI P0/P1/P2/P3."""
    if str(camera) not in {"0", "1", "2", "3"}:
        raise ValueError(f"KITTI camera must be 0, 1, 2 or 3, got {camera!r}")
    path = Path(calib_path)
    projections = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, values = line.partition(":")
        if separator:
            projections[key.strip()] = values.split()
    labels = (f"P{camera}", f"P_rect_{int(camera):02d}")
    label = next((key for key in labels if key in projections), None)
    if label is None:
        raise ValueError(f"Missing KITTI camera projection {labels[0]} in {path}")
    try:
        projection = np.asarray(projections[label], dtype=np.float64).reshape(3, 4)
    except ValueError as error:
        raise ValueError(f"Expected 12 numeric values for {label} in {path}") from error
    K = projection[:, :3].copy()
    if (not np.isfinite(projection).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or not np.allclose([K[0, 1], K[1, 0]], 0)):
        raise ValueError(f"Invalid rectified pinhole projection {label} in {path}")
    # P[:, 3] encodes the camera baseline, not the intrinsic principal point.
    return K


def load_kitti_pose(pose_path):
    """Read a 3x4 or homogeneous 4x4 pose, without guessing its convention."""
    path = Path(pose_path)
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Could not parse KITTI pose in {path}") from error
    if values.size == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3] = values.reshape(3, 4)
    elif values.size == 16:
        matrix = values.reshape(4, 4)
    else:
        raise ValueError(f"Expected a 3x4 or 4x4 KITTI pose in {path}, got shape {values.shape}")
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=5e-3)):
        raise ValueError(f"Invalid rigid KITTI pose in {path}")
    return matrix


class KITTIDataset:
    """Load sequence/rgb, sequence/calib.txt and optional sequence/gt/*.txt.

    Rectified odometry image_0 through image_3 directories are also accepted.
    GT files are matched by image stem and assumed to describe that image's
    camera. By default they are c2w; --pose_format w2c overrides this exporter
    convention. No axis or camera-baseline transform is guessed from the poses.
    """

    def __init__(self, args):
        self.args = args
        self.camera = str(getattr(args, "kitti_camera", "2"))
        if self.camera not in {"0", "1", "2", "3"}:
            raise ValueError(f"KITTI camera must be 0, 1, 2 or 3, got {self.camera!r}")
        source = Path(args.source_path)
        sequences = discover_kitti_sequences(source)
        if not sequences:
            raise FileNotFoundError(f"No KITTI sequence with calib.txt and rgb/ or image_*/ found under {source}")
        self.root = self.sequence_root = sequences[0]
        if len(sequences) > 1:
            logging.warning("Multiple KITTI sequences found; using %s", self.root)
        self.rgb_dir = next((path for path in _image_directories(self.root, self.camera) if path.is_dir()), None)
        if self.rgb_dir is None:
            raise FileNotFoundError(f"No RGB folder for KITTI camera {self.camera} in {self.root}")
        self.gt_dir = self.root / "gt"
        self.calib_path = self.root / "calib.txt"
        self.source_K = parse_kitti_calib(self.calib_path, self.camera)
        self.pose_format = getattr(args, "pose_format", "auto")
        if self.pose_format not in {"auto", "c2w", "w2c"}:
            raise ValueError(f"Unknown pose format: {self.pose_format}")
        self.device = torch.device(args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.downsampling = float(args.downsampling)
        if not np.isfinite(self.downsampling):
            raise ValueError("KITTI downsampling must be finite")
        if args.start_at < 0:
            raise ValueError("KITTI start_at must be nonnegative")

        paths = sorted(path for path in self.rgb_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len({path.stem for path in paths}) != len(paths):
            raise ValueError(f"Ambiguous image filename stems in {self.rgb_dir}")
        selected = list(enumerate(paths))[args.start_at:]
        if args.max_frames > 0:
            selected = selected[:args.max_frames]
        if not selected:
            raise FileNotFoundError(f"No KITTI images remain in {self.rgb_dir} after start_at/max_frames filtering")
        self.source_indices, self.image_paths = map(list, zip(*selected))
        first = self._read_image(self.image_paths[0])
        self.original_height, self.original_width = first.shape[:2]
        resolution = self.original_width * self.original_height
        if self.downsampling <= 0 and resolution > 1_500_000:
            self.downsampling = (resolution / 1_500_000) ** 0.5
        scale = self.downsampling if self.downsampling > 0 else 1.0
        self.width = max(1, round(self.original_width / scale))
        self.height = max(1, round(self.original_height / scale))
        sx, sy = self.width / self.original_width, self.height / self.original_height
        self.K = self.source_K.copy()
        self.K[0, 0] *= sx
        self.K[1, 1] *= sy
        self.K[0, 2] = (self.K[0, 2] + 0.5) * sx - 0.5
        self.K[1, 2] = (self.K[1, 2] + 0.5) * sy - 0.5
        self.K = self.K.astype(np.float32)
        self.infos = [self._build_info(index, path) for index, path in enumerate(self.image_paths)]
        self.current_index = 0

    def __len__(self):
        return len(self.image_paths)

    def get_image_size(self):
        return self.height, self.width

    def __getitem__(self, index):
        return self.load_frame(index)

    def getnext(self):
        if self.current_index >= len(self):
            raise StopIteration
        item = self.load_frame(self.current_index)
        self.current_index += 1
        return item

    def _build_info(self, index, image_path):
        pose_path = self.gt_dir / f"{image_path.stem}.txt"
        info = {
            "is_test": is_test_hold_frame(index, self.args.test_hold),
            "name": image_path.name,
            "frame_id": index,
            "source_index": self.source_indices[index],
            "frame_stem": image_path.stem,
            "dataset_type": "kitti",
            "image_path": str(image_path),
            "depth_path": None,
            "calib_path": str(self.calib_path),
            "gt_path": str(pose_path) if pose_path.is_file() else None,
            "kitti_camera": self.camera,
            "K_np": self.K.copy(),
            "source_K_np": self.source_K.copy(),
            "focal_np": self.K[[0, 1], [0, 1]].copy(),
            "intrinsics_calibrated": True,
            "image_model": "pinhole",
        }
        if pose_path.is_file():
            matrix = load_kitti_pose(pose_path)
            fmt = "c2w" if self.pose_format == "auto" else self.pose_format
            c2w = matrix if fmt == "c2w" else np.linalg.inv(matrix)
            info["gt_c2w_np"] = c2w.astype(np.float32)
            info["Rt_np"] = np.linalg.inv(c2w).astype(np.float32)
            info["gt_pose_format"] = fmt
            info["gt_pose_format_verified"] = False
        return info

    @staticmethod
    def _read_image(path):
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Image at {path} could not be loaded")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected an 8-bit KITTI image at {path}, got {image.dtype}")
        return image

    def _load_image_cpu(self, path):
        import cv2

        image = self._read_image(path)
        if image.shape[:2] != (self.original_height, self.original_width):
            raise ValueError(f"Frame {path} has size {image.shape[:2]}, expected {(self.original_height, self.original_width)}")
        if (self.width, self.height) != (self.original_width, self.original_height):
            image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

    def load_frame(self, index):
        info = copy.deepcopy(self.infos[index])
        image = self._load_image_cpu(self.image_paths[index])
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].to(self.device)
            image = image[:3]
        for key in ("K", "focal", "Rt", "gt_c2w"):
            if f"{key}_np" in info:
                info[key] = torch.tensor(info[f"{key}_np"], dtype=torch.float32, device=self.device)
        return image.to(self.device), info
