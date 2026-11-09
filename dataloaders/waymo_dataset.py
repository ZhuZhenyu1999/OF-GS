import copy
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch

from utils import is_test_hold_frame


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
META_EXTENSIONS = (".txt", ".npy", ".json")
DEPTH_EXTENSIONS = (".png", ".npy", ".npz", ".exr", ".tiff", ".tif")
INTRINSIC_KEYS = ("fx", "fy", "cx", "cy")
DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3")


def _sorted_files(folder: Path, extensions):
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions],
        key=lambda p: p.name,
    )


def _files_by_stem(folder: Path, extensions):
    files = {}
    for path in _sorted_files(folder, extensions):
        if path.stem in files:
            raise ValueError(f"Ambiguous frame stem {path.stem!r} in {folder}")
        files[path.stem] = path
    return files


def _numbers_from_text(text: str):
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def parse_waymo_calib(calib_path: os.PathLike | str | None):
    if calib_path is None:
        return {}
    path = Path(calib_path)
    if not path.exists():
        return {}

    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
        if values.shape == (3, 3):
            return {"fx": float(values[0, 0]), "fy": float(values[1, 1]), "cx": float(values[0, 2]), "cy": float(values[1, 2])}
        flat = values.reshape(-1)
        if flat.size in (4, 9):
            return dict(zip(INTRINSIC_KEYS + DISTORTION_KEYS, map(float, flat)))
        raise ValueError(f"Expected a 3x3 K or a 4/9-element intrinsic vector in {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        flat = {}
        for key in INTRINSIC_KEYS + DISTORTION_KEYS:
            if key in data:
                flat[key] = float(data[key])
        if "K" in data:
            K = np.asarray(data["K"], dtype=np.float32).reshape(3, 3)
            flat.update({"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])})
        return flat

    text = path.read_text(encoding="utf-8", errors="ignore")
    calib = {}
    for key in INTRINSIC_KEYS + DISTORTION_KEYS:
        match = re.search(rf"\b{key}\b\s*[:=]?\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)", text, re.IGNORECASE)
        if match:
            calib[key.lower()] = float(match.group(1))
    if calib:
        # A partially labelled calibration must not consume unrelated extrinsic
        # numbers as a K matrix.
        return calib

    nums = _numbers_from_text(text)
    if len(nums) == 9:
        K = np.asarray(nums, dtype=np.float64).reshape(3, 3)
        if np.allclose(K[2], [0, 0, 1]) and K[0, 1] == 0 and K[1, 0] == 0:
            calib.update({"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])})
        else:
            calib.update(zip(INTRINSIC_KEYS + DISTORTION_KEYS, nums))
    elif len(nums) == 4:
        calib.update(zip(INTRINSIC_KEYS, nums))
    return calib


def load_pose_matrix(pose_path: os.PathLike | str | None):
    if pose_path is None:
        return None
    path = Path(pose_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.asarray(data.get("pose", data.get("matrix", data)), dtype=np.float32)
    else:
        matrix = np.loadtxt(path, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix in {path}, got shape {matrix.shape}")
    return matrix


def pose_matrix_to_w2c(matrix: np.ndarray, pose_format: str, default_format: str = "c2w"):
    fmt = default_format if pose_format == "auto" else pose_format
    if fmt == "c2w":
        return np.linalg.inv(matrix).astype(np.float32)
    if fmt == "w2c":
        return matrix.astype(np.float32)
    raise ValueError(f"Unknown pose format: {pose_format}")


def resolve_waymo_camera_root(source_path: os.PathLike | str, camera_name: str = "FRONT"):
    root = Path(source_path)
    candidates = [
        root / camera_name,
        root,
    ]
    for candidate in candidates:
        if (candidate / "rgb").exists():
            sequence_root = candidate.parent if candidate.name == camera_name else candidate
            return sequence_root, candidate

    for child in sorted(root.iterdir()) if root.exists() else []:
        if child.is_dir() and (child / camera_name / "rgb").exists():
            return child, child / camera_name
    raise FileNotFoundError(
        f"Could not find a Waymo camera folder with rgb/ under {root} for camera {camera_name}"
    )


def discover_waymo_sequences(source_path: os.PathLike | str, camera_name: str = "FRONT"):
    root = Path(source_path)
    if not root.exists():
        return []
    try:
        sequence_root, _ = resolve_waymo_camera_root(root, camera_name)
        if sequence_root == root or (root / camera_name / "rgb").exists() or (root / "rgb").exists():
            return [sequence_root]
    except FileNotFoundError:
        pass
    return [
        child
        for child in sorted(root.iterdir()) if child.is_dir() and (child / camera_name / "rgb").exists()
    ]


class WaymoDataset:
    """
    Reader for the local Waymo layout:

        sequence/FRONT/rgb
        sequence/FRONT/depth
        sequence/FRONT/calib
        sequence/FRONT/gt

    GT poses and depth paths are returned only as metadata for evaluation.
    Metadata is associated by identical filename stem, never sorted position.
    The input image model must be specified when distortion is nonzero; PNG
    files alone do not reveal whether the exporter already undistorted them.
    """
    def __init__(self, args):
        self.args = args
        self.sequence_root, self.camera_root = resolve_waymo_camera_root(args.source_path, args.camera_name)
        self.rgb_dir = self.camera_root / "rgb"
        self.calib_dir = self.camera_root / "calib"
        self.gt_dir = self.camera_root / "gt"
        self.depth_dir = self.camera_root / "depth"
        self.downsampling = args.downsampling
        self.input_image_model = getattr(args, "waymo_image_model", "auto")
        if self.input_image_model not in ("auto", "pinhole", "opencv"):
            raise ValueError(f"Unknown Waymo image model: {self.input_image_model}")
        self.device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

        image_paths = list(_files_by_stem(self.rgb_dir, IMAGE_EXTENSIONS).values())
        self.source_indices = list(range(len(image_paths)))
        if args.start_at > 0:
            image_paths = image_paths[args.start_at:]
            self.source_indices = self.source_indices[args.start_at:]
        if args.max_frames > 0:
            image_paths = image_paths[: args.max_frames]
            self.source_indices = self.source_indices[: args.max_frames]
        if not image_paths:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")
        self.image_paths = image_paths

        first = self._read_image(self.image_paths[0])
        self.original_height, self.original_width = first.shape[:2]
        res = self.original_width * self.original_height
        if self.downsampling <= 0.0 and res > 1_500_000:
            self.downsampling = (res / 1_500_000) ** 0.5
        scale = self.downsampling if self.downsampling > 0 else 1.0
        self.width = max(1, round(self.original_width / scale))
        self.height = max(1, round(self.original_height / scale))

        self.calib_files = _files_by_stem(self.calib_dir, META_EXTENSIONS)
        self.gt_files = _files_by_stem(self.gt_dir, META_EXTENSIONS)
        self.depth_files = _files_by_stem(self.depth_dir, DEPTH_EXTENSIONS)
        self.infos = [self._build_info(i, path) for i, path in enumerate(self.image_paths)]
        self.current_index = 0

    def __len__(self):
        return len(self.image_paths)

    def get_image_size(self):
        return self.height, self.width

    def __getitem__(self, index):
        return self.load_frame(index)

    def getnext(self):
        item = self.load_frame(self.current_index)
        self.current_index += 1
        return item

    def _scaled_calib(self, calib):
        if not calib:
            return calib
        sx = self.width / self.original_width
        sy = self.height / self.original_height
        scaled = dict(calib)
        for focal, principal, scale in (("fx", "cx", sx), ("fy", "cy", sy)):
            if focal in scaled:
                scaled[focal] *= scale
            if principal in scaled:
                # cv2.resize maps pixel centres with a half-pixel offset.
                scaled[principal] = (scaled[principal] + 0.5) * scale - 0.5
        return scaled

    def _build_info(self, index, image_path: Path):
        stem = image_path.stem
        calib_path = self.calib_files.get(stem)
        gt_path = self.gt_files.get(stem)
        depth_path = self.depth_files.get(stem)

        source_calib = parse_waymo_calib(calib_path)
        calibrated = set(INTRINSIC_KEYS).issubset(source_calib)
        if calibrated and (not np.isfinite([source_calib[key] for key in INTRINSIC_KEYS]).all()
                           or source_calib["fx"] <= 0 or source_calib["fy"] <= 0):
            raise ValueError(f"Invalid pinhole intrinsics in {calib_path}")
        distortion = np.array([source_calib.get(key, 0.0) for key in DISTORTION_KEYS], dtype=np.float64)
        if not np.isfinite(distortion).all():
            raise ValueError(f"Non-finite distortion in {calib_path}")
        has_distortion = bool(np.any(distortion != 0))
        if self.input_image_model == "auto" and has_distortion:
            raise ValueError(
                f"Nonzero distortion in {calib_path}; the PNG export model is unknown. "
                "Use --waymo_image_model pinhole only for already undistorted images with matching K, "
                "or --waymo_image_model opencv for raw images requiring undistortion."
            )
        undistort = self.input_image_model == "opencv" and has_distortion
        if undistort and not calibrated:
            raise ValueError(f"Cannot undistort {image_path} without complete fx/fy/cx/cy in {calib_path}")
        calib = self._scaled_calib(source_calib)
        for key in DISTORTION_KEYS:
            calib[key] = 0.0
        info = {
            "is_test": is_test_hold_frame(index, self.args.test_hold),
            "name": image_path.name,
            "frame_id": index,
            "source_index": self.source_indices[index],
            "frame_stem": stem,
            "dataset_type": "waymo",
            "image_path": str(image_path),
            "depth_path": str(depth_path) if depth_path else None,
            "calib_path": str(calib_path) if calib_path else None,
            "gt_path": str(gt_path) if gt_path else None,
            "calib": calib,
            "source_calib": source_calib,
            "intrinsics_calibrated": calibrated,
            "input_image_model": self.input_image_model,
            "image_model": "pinhole",
            "undistorted_by_loader": undistort,
            "source_distortion_np": distortion,
        }

        if calibrated:
            K = np.array([[calib["fx"], 0, calib["cx"]], [0, calib["fy"], calib["cy"]], [0, 0, 1]], dtype=np.float32)
            info["K_np"] = K
            info["source_K_np"] = np.array([[source_calib["fx"], 0, source_calib["cx"]], [0, source_calib["fy"], source_calib["cy"]], [0, 0, 1]], dtype=np.float64)
            info["focal_np"] = np.array([calib["fx"], calib["fy"]], dtype=np.float32)
        else:
            fallback_f = 0.7 * self.width
            info["K_np"] = np.array([[fallback_f, 0, (self.width - 1) * 0.5], [0, fallback_f, (self.height - 1) * 0.5], [0, 0, 1]], dtype=np.float32)
            info["focal_np"] = np.array([fallback_f, fallback_f], dtype=np.float32)
            logging.warning("No complete matching Waymo calibration for %s; using uncalibrated focal fallback.", image_path)

        matrix = load_pose_matrix(gt_path)
        if matrix is not None:
            info["Rt_np"] = pose_matrix_to_w2c(matrix, self.args.pose_format, default_format="c2w")
            info["gt_c2w_np"] = np.linalg.inv(info["Rt_np"]).astype(np.float32)
            info["gt_pose_format"] = "c2w" if self.args.pose_format == "auto" else self.args.pose_format
            info["gt_pose_format_verified"] = False
        return info

    @staticmethod
    def _read_image(image_path: Path, mode=None):
        import cv2

        if mode is None:
            mode = cv2.IMREAD_UNCHANGED
        image = cv2.imread(str(image_path), mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        return image

    def _load_image_cpu(self, image_path: Path, mode=None, info=None):
        import cv2

        image = self._read_image(image_path, mode)
        if image.shape[:2] != (self.original_height, self.original_width):
            raise ValueError(f"Frame {image_path} has size {image.shape[:2]}, expected {(self.original_height, self.original_width)}")
        if info is not None and info["undistorted_by_loader"]:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.shape[-1] == 3:
                # The alpha channel records valid rectification pixels so black
                # remap borders are excluded from photometric losses.
                alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
                image = np.concatenate((image, alpha), axis=-1)
            image = cv2.undistort(image, info["source_K_np"], info["source_distortion_np"], None, info["source_K_np"])
        if (self.width, self.height) != (self.original_width, self.original_height):
            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def load_frame(self, index):
        info = copy.deepcopy(self.infos[index])
        image = self._load_image_cpu(self.image_paths[index], info=info)
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].to(self.device)
            image = image[:3]
        info["K"] = torch.tensor(info["K_np"], dtype=torch.float32, device=self.device)
        info["focal"] = torch.tensor(info["focal_np"], dtype=torch.float32, device=self.device)
        if "Rt_np" in info:
            info["Rt"] = torch.tensor(info["Rt_np"], dtype=torch.float32, device=self.device)
            info["gt_c2w"] = torch.tensor(info["gt_c2w_np"], dtype=torch.float32, device=self.device)
        return image.to(self.device), info
