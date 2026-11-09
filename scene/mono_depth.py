#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import contextlib
import torch
import os
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch.nn.functional as F

from utils import sample

if TYPE_CHECKING:
    from poses.feature_detector import DescribedKeypoints

os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"

size = 518
encoder = "vitl"
DEFAULT_DA3_MODEL_ID = "depth-anything/DA3MONO-LARGE"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _append_sys_path_once(path: Path):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.append(path_str)


def _load_depth_anything_v2_class():
    _append_sys_path_once(REPO_ROOT / "submodules" / "Depth-Anything-V2")
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2


def _load_depth_anything3_class():
    da3_root = REPO_ROOT / "submodules" / "Depth-Anything-3"
    _append_sys_path_once(da3_root / "src")
    _append_sys_path_once(da3_root)
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 backend requires depth_anything_3. "
            "Install Depth-Anything-3 or place it under submodules/Depth-Anything-3, "
            "then rerun with --mono_depth_backend da3mono."
        ) from exc

    return DepthAnything3


def _looks_like_local_path(value: str) -> bool:
    return (
        os.path.isabs(value)
        or value.startswith(("/", "\\", "~", "."))
        or os.path.splitdrive(value)[0] != ""
        or "\\" in value
    )


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


def _sorted_existing_dirs(paths):
    existing_dirs = [path for path in paths if path.is_dir()]
    return sorted(
        existing_dirs,
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def _da3_hf_cache_snapshot_dirs(root: Path, model_name: str):
    cache_dir = f"models--depth-anything--{model_name}"
    for cache_root in (
        root,
        root / "hub",
        root / "huggingface" / "hub",
        root / ".cache" / "huggingface" / "hub",
    ):
        snapshots_root = cache_root / cache_dir / "snapshots"
        if snapshots_root.is_dir():
            yield from _sorted_existing_dirs(snapshots_root.iterdir())


def _find_da3_local_model_dir(model_path: Path):
    if _is_model_dir(model_path):
        return model_path

    search_roots = []
    if model_path.exists():
        search_roots.append(model_path)
    if model_path.parent.exists():
        search_roots.append(model_path.parent)

    seen = set()
    for root in search_roots:
        for candidate in (
            root / "snapshots" / "*",
            root / "hub" / f"models--depth-anything--{model_path.name}" / "snapshots" / "*",
            root / f"models--depth-anything--{model_path.name}" / "snapshots" / "*",
        ):
            for match in _sorted_existing_dirs(candidate.parent.glob(candidate.name)):
                resolved = match.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                if _is_model_dir(match):
                    return match

        for match in _da3_hf_cache_snapshot_dirs(root, model_path.name):
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if _is_model_dir(match):
                return match

    return None


def _da3_lookup_hint(model_path: Path) -> str:
    model_name = model_path.name
    roots = []
    if model_path.exists():
        roots.append(model_path)
    if model_path.parent.exists():
        roots.append(model_path.parent)

    looked_in = [model_path]
    for root in roots:
        looked_in.extend(
            (
                root / "snapshots" / "<revision>",
                root / f"models--depth-anything--{model_name}" / "snapshots" / "<revision>",
                root / "hub" / f"models--depth-anything--{model_name}" / "snapshots" / "<revision>",
                root / "huggingface" / "hub" / f"models--depth-anything--{model_name}" / "snapshots" / "<revision>",
                root / ".cache" / "huggingface" / "hub" / f"models--depth-anything--{model_name}" / "snapshots" / "<revision>",
            )
        )

    unique_paths = []
    seen = set()
    for path in looked_in:
        path_str = str(path)
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path_str)
    return "; ".join(unique_paths)


def _find_da3_repo_model_dir(model_id: str):
    if _looks_like_local_path(model_id):
        return None

    parts = model_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None

    local_model_alias = REPO_ROOT / "models" / parts[1]
    return _find_da3_local_model_dir(local_model_alias)


def _resolve_da3_model_reference(model_id) -> str:
    model_id_str = os.fspath(model_id)
    if not model_id_str:
        raise ValueError(
            "Depth Anything 3 model id/path is empty. "
            f"Use a Hugging Face repo id such as {DEFAULT_DA3_MODEL_ID!r} "
            "or pass a valid local model directory."
        )

    model_path = Path(os.path.expandvars(os.path.expanduser(model_id_str)))
    if model_path.exists():
        local_model_dir = _find_da3_local_model_dir(model_path)
        if local_model_dir is not None:
            return str(local_model_dir.resolve())
        return str(model_path.resolve())

    if _looks_like_local_path(model_id_str):
        local_model_dir = _find_da3_local_model_dir(model_path)
        if local_model_dir is not None:
            return str(local_model_dir.resolve())
        raise FileNotFoundError(
            f"Depth Anything 3 model path does not exist: {model_id_str}. "
            "Could not find a local model directory containing config.json. "
            f"Looked in: {_da3_lookup_hint(model_path)}. "
            f"Use an existing local model/snapshot directory, or a Hugging Face repo id such as {DEFAULT_DA3_MODEL_ID!r}."
        )

    local_model_dir = _find_da3_repo_model_dir(model_id_str)
    if local_model_dir is not None:
        return str(local_model_dir.resolve())

    return model_id_str


def _tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return np.rint(image_np * 255.0).astype(np.uint8)


def _get_prediction_field(prediction, *names):
    if isinstance(prediction, (list, tuple)):
        if len(prediction) == 0:
            return None
        first = prediction[0]
        if isinstance(first, dict) or any(hasattr(first, name) for name in names):
            return _get_prediction_field(first, *names)
        if len(names) == 1 and names[0] == "depth":
            return prediction[0]
        if len(names) == 1 and names[0] in {"conf", "confidence"} and len(prediction) > 1:
            return prediction[1]

    if isinstance(prediction, dict):
        for name in names:
            if name in prediction:
                return prediction[name]
        return None

    for name in names:
        if hasattr(prediction, name):
            return getattr(prediction, name)
    return None


def _first_prediction_item(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError("Depth Anything 3 returned an empty prediction list")
        return value[0]
    return value


def _as_depth_map_tensor(value, device, dtype=torch.float32) -> torch.Tensor:
    value = _first_prediction_item(value)
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a 2D depth/confidence map from Depth Anything 3, got shape {tuple(tensor.shape)}"
        )
    return tensor[None, None]


def _resize_to_depth_size(map_tensor: torch.Tensor) -> torch.Tensor:
    if map_tensor.shape[-2:] == (size, size):
        return map_tensor
    return F.interpolate(map_tensor, (size, size), mode="bilinear", align_corners=True)


def _sobel_confidence(depth: torch.Tensor) -> torch.Tensor:
    sobel_x = (
        torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            device=depth.device,
            dtype=depth.dtype,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    sobel_y = (
        torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            device=depth.device,
            dtype=depth.dtype,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    edges = torch.cat((grad_x, grad_y), dim=0)

    edges_sq_norm = (edges**2).sum(0, keepdim=True)
    var = 0.2
    return torch.exp(-edges_sq_norm / var)


class MonoDepthInternal(torch.nn.Module):
    def __init__(self, backend="da2", da3_model_id=DEFAULT_DA3_MODEL_ID):
        super(MonoDepthInternal, self).__init__()
        self.backend = backend
        if self.backend == "da2":
            self._init_depth_anything_v2()
        elif self.backend == "da3mono":
            self._init_depth_anything3_mono(da3_model_id)
        else:
            raise ValueError(f"Unsupported mono depth backend: {backend}")

    def _init_depth_anything_v2(self):
        model_path = REPO_ROOT / "models" / f"depth_anything_v2_{encoder}.pth"
        if not os.path.exists(model_path):
            print(f"Downloading Depth-Anything-V2 model for {encoder}, may take a few minutes...")
            model_sizes = {
                "vits": "Small",
                "vitb": "Base",
                "vitl": "Large",
                "vitg": "Giant",
            }
            url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_sizes[encoder]}/resolve/main/depth_anything_v2_{encoder}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(url, model_path)
        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }
        DepthAnythingV2 = _load_depth_anything_v2_class()
        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self.model = model.to("cuda").half().eval()
        self.sobel_x = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )
        self.sobel_y = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device="cuda", dtype=torch.half
            ).unsqueeze(0).unsqueeze(0)
        )

    def _init_depth_anything3_mono(self, model_id):
        DepthAnything3 = _load_depth_anything3_class()
        model_reference = _resolve_da3_model_reference(model_id)
        self.model = DepthAnything3.from_pretrained(model_reference).to("cuda").eval()

    def forward(self, image: torch.Tensor):
        if self.backend == "da3mono":
            return self._forward_depth_anything3_mono(image)
        return self._forward_depth_anything_v2(image)

    def _forward_depth_anything_v2(self, image: torch.Tensor):
        img = torch.nn.functional.interpolate(
            image[None].half(), (size, size), mode="bilinear", align_corners=True
        )
        depth = self.model(img)[None]
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        edges = torch.cat((grad_x, grad_y), dim=0)

        edges_sq_norm = (edges**2).sum(0, keepdim=True)
        var = 0.2
        confidence = torch.exp(-edges_sq_norm / var)
        return depth.float(), confidence.float()

    def _forward_depth_anything3_mono(self, image: torch.Tensor):
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stdout(devnull),
            contextlib.redirect_stderr(devnull),
        ):
            prediction = self.model.inference([_tensor_to_uint8_rgb(image)])
        depth_value = _get_prediction_field(prediction, "depth")
        if depth_value is None:
            raise ValueError("Depth Anything 3 prediction does not contain a depth field")

        depth = _resize_to_depth_size(_as_depth_map_tensor(depth_value, image.device))
        t, s = get_t_s(depth)
        depth = (depth - t) / s

        conf_value = _get_prediction_field(prediction, "conf", "confidence")
        if conf_value is not None:
            confidence = _resize_to_depth_size(
                _as_depth_map_tensor(conf_value, image.device)
            ).float()
            confidence = confidence.nan_to_num(0.0).clamp(0.0, 1.0)
        else:
            confidence = _sobel_confidence(depth)

        return depth.float(), confidence.float()


def get_t_s(d):
    t = d.median()
    s = (d - t).abs().median()
    return t, s


def align_samples(tri_idepth: torch.Tensor, mono_idepth: torch.Tensor):
    t_tri, s_tri = get_t_s(tri_idepth)
    t_mono, s_mono = get_t_s(mono_idepth)
    scale = s_tri / s_mono
    offset = t_tri - t_mono * scale
    return mono_idepth * scale + offset, scale, offset


def align_depth(
    mono_depth_map: torch.Tensor, desc_kpts: DescribedKeypoints, width: int, height: int
):
    """Aligns the mono depth map with the triangulated depth from keypoints by finding the best scale and offset."""
    mono_idepth = sample(
        mono_depth_map,
        desc_kpts.kpts[desc_kpts.has_pt3d].view(1, 1, -1, 2),
        width,
        height,
    )[0, 0, 0]
    tri_idepth = 1 / desc_kpts.depth[desc_kpts.has_pt3d]

    mono_idepth_aligned, scale, offset = align_samples(tri_idepth, mono_idepth)
    err = (mono_idepth_aligned - tri_idepth).abs()
    valid = err < 5 * err.median()
    mono_idepth_aligned, scale, offset = align_samples(
        tri_idepth[valid], mono_idepth[valid]
    )
    mono_depth_map_aligned = mono_depth_map * scale + offset

    return mono_depth_map_aligned


class MonoDepthEstimator:
    @torch.no_grad()
    def __init__(self, width: int, height: int, args=None):
        self.width = width
        self.height = height
        da3_model_id = getattr(args, "da3_model_id", None)
        self.backend = "da3mono" if da3_model_id else getattr(args, "mono_depth_backend", "da2")
        if self.backend == "da3mono" and not da3_model_id:
            da3_model_id = DEFAULT_DA3_MODEL_ID
        model = MonoDepthInternal(self.backend, da3_model_id)

        if self.backend == "da2":
            dummy = torch.zeros(3, height, width).cuda()
            self.model = torch.cuda.make_graphed_callables(model, [dummy])
        else:
            self.model = model

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth, conf = self.model(image)
        return depth.clone(), conf.clone()
