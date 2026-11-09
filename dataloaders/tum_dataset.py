import copy
import os
from pathlib import Path

import numpy as np
import torch

from utils import is_test_hold_frame


def read_tum_rgbd_list(path: os.PathLike | str):
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            entries.append((float(parts[0]), parts[1]))
    return entries


def read_tum_groundtruth(path: os.PathLike | str):
    entries = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            timestamp = float(parts[0])
            values = np.asarray([float(x) for x in parts[1:8]], dtype=np.float32)
            entries.append((timestamp, values))
    return entries


def associate_timestamps(first, second, max_dt=0.02):
    """
    Greedy nearest-neighbor timestamp association.

    Returns (idx_first, idx_second, abs_dt) tuples.
    """
    second_unused = set(range(len(second)))
    matches = []
    for i, (t0, _) in enumerate(first):
        best_j = None
        best_dt = float("inf")
        for j in list(second_unused):
            dt = abs(t0 - second[j][0])
            if dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j is not None and best_dt <= max_dt:
            second_unused.remove(best_j)
            matches.append((i, best_j, best_dt))
    return matches


def quat_xyzw_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def tum_pose_to_w2c(values):
    t = values[:3]
    q = values[3:7]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quat_xyzw_to_rotmat(q)
    c2w[:3, 3] = t
    return np.linalg.inv(c2w).astype(np.float32), c2w


def tum_default_intrinsics(source_path: os.PathLike | str):
    text = str(source_path).lower()
    if "freiburg2" in text:
        return 520.9, 521.0, 325.1, 249.7
    if "freiburg3" in text:
        return 535.4, 539.2, 320.1, 247.6
    return 517.3, 516.5, 318.6, 255.3


def discover_tum_sequences(source_path: os.PathLike | str):
    root = Path(source_path)
    if not root.exists():
        return []
    if (root / "rgb.txt").exists():
        return [root]
    return [child for child in sorted(root.iterdir()) if child.is_dir() and (child / "rgb.txt").exists()]


class TUMDataset:
    """
    Reader for TUM RGB-D sequences.

    Depth and GT are associated by timestamp and returned as metadata. Depth is
    not used for tracking or mapping.
    """
    def __init__(self, args):
        self.args = args
        self.root = Path(args.source_path)
        self.rgb_entries = read_tum_rgbd_list(self.root / "rgb.txt")
        self.depth_entries = read_tum_rgbd_list(self.root / "depth.txt") if (self.root / "depth.txt").exists() else []
        self.gt_entries = read_tum_groundtruth(self.root / "groundtruth.txt") if (self.root / "groundtruth.txt").exists() else []
        self.downsampling = args.downsampling
        self.device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

        if not self.rgb_entries:
            raise FileNotFoundError(f"No RGB entries found in {self.root / 'rgb.txt'}")

        self.rgb_to_depth = {i: j for i, j, _ in associate_timestamps(self.rgb_entries, self.depth_entries, args.tum_max_dt)}
        self.rgb_to_gt = {i: j for i, j, _ in associate_timestamps(self.rgb_entries, self.gt_entries, args.tum_max_dt)}

        first_index = args.start_at
        last_index = len(self.rgb_entries) if args.max_frames <= 0 else min(len(self.rgb_entries), first_index + args.max_frames)
        self.indices = list(range(first_index, last_index))
        if not self.indices:
            raise FileNotFoundError("No TUM RGB frames remain after start_at/max_frames filtering.")

        import cv2
        original = cv2.imread(str(self._rgb_path(self.indices[0])), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise FileNotFoundError(self._rgb_path(self.indices[0]))
        self.original_height, self.original_width = original.shape[:2]
        first = self._load_image_cpu(self._rgb_path(self.indices[0]))
        self.width = first.shape[2]
        self.height = first.shape[1]
        res = self.width * self.height
        if self.downsampling <= 0.0 and res > 1_500_000:
            self.downsampling = (res / 1_500_000) ** 0.5
            first = self._load_image_cpu(self._rgb_path(self.indices[0]))
            self.width = first.shape[2]
            self.height = first.shape[1]

        self.fx, self.fy, self.cx, self.cy = self._intrinsics()
        self.infos = [self._build_info(local_id, rgb_index) for local_id, rgb_index in enumerate(self.indices)]
        self.current_index = 0

    def __len__(self):
        return len(self.indices)

    def get_image_size(self):
        return self.height, self.width

    def __getitem__(self, index):
        return self.load_frame(index)

    def getnext(self):
        item = self.load_frame(self.current_index)
        self.current_index += 1
        return item

    def _intrinsics(self):
        fx, fy, cx, cy = tum_default_intrinsics(self.root)
        fx = self.args.tum_fx if self.args.tum_fx > 0 else fx
        fy = self.args.tum_fy if self.args.tum_fy > 0 else fy
        cx = self.args.tum_cx if self.args.tum_cx > 0 else cx
        cy = self.args.tum_cy if self.args.tum_cy > 0 else cy
        sx, sy = self.width / self.original_width, self.height / self.original_height
        return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5

    def _rgb_path(self, rgb_index):
        return self.root / self.rgb_entries[rgb_index][1]

    def _depth_path(self, rgb_index):
        depth_index = self.rgb_to_depth.get(rgb_index)
        return self.root / self.depth_entries[depth_index][1] if depth_index is not None else None

    def _build_info(self, local_index, rgb_index):
        timestamp, rel_path = self.rgb_entries[rgb_index]
        K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        info = {
            "is_test": is_test_hold_frame(local_index, self.args.test_hold),
            "name": Path(rel_path).name,
            "frame_id": local_index,
            "timestamp": timestamp,
            "dataset_type": "tum",
            "image_path": str(self.root / rel_path),
            "depth_path": str(self._depth_path(rgb_index)) if self._depth_path(rgb_index) else None,
            "K_np": K,
            "focal_np": np.array([self.fx, self.fy], dtype=np.float32),
            "intrinsics_calibrated": True,
        }
        gt_index = self.rgb_to_gt.get(rgb_index)
        if gt_index is not None:
            Rt, c2w = tum_pose_to_w2c(self.gt_entries[gt_index][1])
            info["Rt_np"] = Rt
            info["gt_c2w_np"] = c2w
            info["gt_timestamp"] = self.gt_entries[gt_index][0]
        return info

    def _load_image_cpu(self, image_path: Path, mode=None):
        import cv2

        if mode is None:
            mode = cv2.IMREAD_UNCHANGED
        image = cv2.imread(str(image_path), mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    def load_frame(self, index):
        rgb_index = self.indices[index]
        image = self._load_image_cpu(self._rgb_path(rgb_index))
        info = copy.deepcopy(self.infos[index])
        if image.shape[0] == 4:
            info["mask"] = image[-1][None]
            image = image[:3]
        info["K"] = torch.tensor(info["K_np"], dtype=torch.float32, device=self.device)
        info["focal"] = torch.tensor(info["focal_np"], dtype=torch.float32, device=self.device)
        if "Rt_np" in info:
            info["Rt"] = torch.tensor(info["Rt_np"], dtype=torch.float32, device=self.device)
            info["gt_c2w"] = torch.tensor(info["gt_c2w_np"], dtype=torch.float32, device=self.device)
        return image.to(self.device), info
