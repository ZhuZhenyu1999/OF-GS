"""Offline Depth Anything V2 relative-depth exports, independent of SLAM state.

This deliberately uses the upstream image preprocessing on the original PNG,
not the SLAM loader's rectified/resized image or its median/MAD-normalized prior.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "depth_anything_v2_vitl.pth"
INPUT_SIZE = 518


def select_requested_images(image_paths, names):
    """Resolve exact basenames; never silently substitute or overwrite a frame."""
    by_name = {}
    for value in image_paths:
        path = Path(value)
        by_name.setdefault(path.name, []).append(path)
    selected, seen_names, seen_stems = [], set(), set()
    for name in names:
        name = str(name)
        if not name or name in (".", "..") or "/" in name or "\\" in name or ":" in name:
            raise ValueError(f"Expected an image filename without directories: {name!r}")
        if name in seen_names:
            continue
        seen_names.add(name)
        matches = by_name.get(name, ())
        if not matches:
            raise FileNotFoundError(f"Requested DA2 image is not in the selected input sequence: {name}")
        if len(matches) != 1:
            raise ValueError(f"Requested DA2 image name is ambiguous: {name}")
        stem = matches[0].stem.casefold()
        if stem in seen_stems:
            raise ValueError(f"Requested DA2 images would overwrite the same output stem: {name}")
        seen_stems.add(stem)
        selected.append(matches[0])
    return selected


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_png(path, pixels):
    import cv2

    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"Could not encode depth preview: {path}")
    Path(path).write_bytes(encoded.tobytes())


def save_relative_depth(depth, source_image, output_dir, metadata=None):
    """Save unscaled float32 predictions and min/max-normalized PNG previews.

    Non-finite predictions are rejected instead of silently being replaced.
    A constant prediction is preserved in NPY and visualized as all-zero gray.
    """
    import cv2
    import numpy as np

    source = Path(source_image).resolve(strict=True)
    raw = np.asarray(depth, dtype=np.float32)
    if raw.ndim != 2 or not raw.size:
        raise ValueError("DA2 relative depth must be a nonempty 2D array")
    if not np.isfinite(raw).all():
        raise ValueError("DA2 prediction contains NaN or infinity; refusing a misleading depth export")
    minimum, maximum = float(raw.min()), float(raw.max())
    constant = maximum == minimum
    # Calculate the range in float64 so even extreme finite float32 values are
    # normalized without overflowing. The raw output remains unchanged.
    gray = np.zeros(raw.shape, dtype=np.uint8) if constant else np.rint(
        (raw.astype(np.float64) - minimum) / (maximum - minimum) * 255
    ).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_depth": destination / f"{source.stem}_da2_relative.npy",
        "gray_png": destination / f"{source.stem}_da2_relative.png",
        "color_png": destination / f"{source.stem}_da2_color.png",
        "metadata": destination / f"{source.stem}_da2_metadata.json",
    }
    document = dict(metadata or {})
    document.update({
        "source_image": str(source),
        "source_sha256": _sha256(source),
        "shape": list(raw.shape),
        "saved_dtype": "float32",
        "convention": "relative inverse depth: larger values indicate nearer surfaces; not metric depth",
        "units": "arbitrary; no metric calibration",
        "min": minimum, "max": maximum, "constant_map": constant,
        "visualization": {
            "normalization": "per-image min/max to [0,255]; constant map becomes zero",
            "gray": "nearer is brighter", "color": "OpenCV TURBO; near is warm, far is cool",
            "raw_npy_normalized": False,
        },
        "files": {key: path.name for key, path in paths.items()},
    })
    np.save(paths["raw_depth"], raw, allow_pickle=False)
    _write_png(paths["gray_png"], gray)
    _write_png(paths["color_png"], color)
    paths["metadata"].write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
                                 encoding="utf-8")
    return {"source_image": str(source), **{key: str(path) for key, path in paths.items()}}


def _load_components():
    vendor = REPO_ROOT / "submodules" / "Depth-Anything-V2"
    if not (vendor / "depth_anything_v2" / "dpt.py").is_file():
        raise FileNotFoundError(f"Depth Anything V2 source is missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from depth_anything_v2.dpt import DepthAnythingV2
    from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet

    return DepthAnythingV2, Resize, NormalizeImage, PrepareForNet


def _predict(model, bgr, device, dtype, transforms):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    sample = {"image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0}
    for transform in transforms:
        sample = transform(sample)
    image = torch.from_numpy(sample["image"])[None].to(device=device, dtype=dtype)
    with torch.inference_mode():
        prediction = model(image)
        raw = F.interpolate(prediction[:, None].float(), bgr.shape[:2],
                            mode="bilinear", align_corners=True)[0, 0]
    return raw.cpu().numpy().astype(np.float32, copy=False), list(image.shape[-2:])


def export_da2_images(image_paths, output_dir, device="auto", checkpoint=None):
    """Export original source pixels using one dedicated, local-only ViT-L model.

    CUDA uses float16 to limit memory; CPU uses float32. Model construction and
    inference run inside fork_rng, preserving the caller's CPU/CUDA RNG states.
    """
    import cv2
    import numpy as np
    import torch

    paths = [Path(value).resolve(strict=True) for value in image_paths]
    if not paths:
        return []
    stems = [path.stem.casefold() for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("DA2 inputs must have unique stems to avoid overwriting outputs")
    weights = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
    weights = weights.resolve(strict=True)
    if not weights.is_file():
        raise FileNotFoundError(f"DA2 checkpoint is not a file: {weights}")
    requested_device = str(device)
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(requested_device)
    if target.type not in ("cuda", "cpu"):
        raise ValueError("DA2 export device must be auto, cpu or cuda[:index]")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for DA2 export but is not available; use --device cpu")
    cuda_devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == "cuda" else []
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    checkpoint_hash = _sha256(weights)
    Model, Resize, NormalizeImage, PrepareForNet = _load_components()
    transforms = (
        Resize(width=INPUT_SIZE, height=INPUT_SIZE, resize_target=False,
               keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    )
    model = None
    results = []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model = Model(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
            state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            del state_dict
            model = model.to(device=target, dtype=dtype).eval()
            # Optional xFormers installations do not provide CPU kernels. Use
            # the upstream ordinary attention on this dedicated model only.
            if target.type == "cpu":
                for module in model.modules():
                    if type(module).__name__ == "MemEffAttention":
                        from depth_anything_v2.dinov2_layers.attention import Attention
                        module.forward = Attention.forward.__get__(module, type(module))
            for path in paths:
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode DA2 source image: {path}")
                start = time.perf_counter()
                depth, network_shape = _predict(model, image, target, dtype, transforms)
                metadata = {
                    "model": "Depth Anything V2", "encoder": "vitl",
                    "checkpoint": str(weights), "checkpoint_sha256": checkpoint_hash,
                    "input_size": INPUT_SIZE, "network_input_shape": network_shape,
                    "device": str(target), "inference_dtype": str(dtype).removeprefix("torch."),
                    "inference_seconds": time.perf_counter() - start,
                    "preprocessing": {
                        "source_pixels": "original image; no SLAM/Waymo undistortion or loader resize",
                        "color": "BGR decoded to RGB / 255",
                        "resize": "aspect-preserving lower_bound 518; multiple of 14; OpenCV cubic",
                        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                        "output_resize": "bilinear to original image dimensions, align_corners=True",
                        "slam_normalization_or_alignment": False,
                    },
                }
                results.append(save_relative_depth(depth, path, output_dir, metadata))
    finally:
        del model
        gc.collect()
        if target.type == "cuda":
            with torch.cuda.device(target):
                torch.cuda.empty_cache()
    return results
