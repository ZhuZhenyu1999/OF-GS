from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch


CLIDD_DESCRIPTOR_DIMS = {
    "A48": 48,
    "N64": 64,
    "T64": 64,
    "S64": 64,
    "M64": 64,
    "L64": 64,
    "G128": 128,
    "E128": 128,
    "U128": 128,
}


def default_clidd_root() -> str:
    return str(Path(__file__).resolve().parents[1] / "CLIDD-main")


def clidd_descriptor_dim(model_name: str) -> int:
    name = str(model_name)
    if name not in CLIDD_DESCRIPTOR_DIMS:
        choices = ", ".join(sorted(CLIDD_DESCRIPTOR_DIMS))
        raise ValueError(f"Unknown CLIDD model '{name}'. Expected one of: {choices}")
    return CLIDD_DESCRIPTOR_DIMS[name]


def resolve_clidd_root(root: str | os.PathLike | None) -> Path:
    path = Path(root or default_clidd_root()).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def validate_clidd_install(root: str | os.PathLike | None, model_name: str) -> Path:
    root_path = resolve_clidd_root(root)
    clidd_py = root_path / "clidd.py"
    model_dir = root_path / "model"
    weight_path = root_path / "weights" / f"{model_name}.pth"
    if not clidd_py.exists():
        raise FileNotFoundError(
            f"CLIDD source file not found: {clidd_py}. "
            "Place CLIDD-main under TEST2-SLAM or pass --clidd_root."
        )
    if not model_dir.exists():
        raise FileNotFoundError(
            f"CLIDD model directory not found: {model_dir}. "
            "Expected the official CLIDD-main/model directory."
        )
    if not weight_path.exists():
        raise FileNotFoundError(
            f"CLIDD {model_name} weights not found: {weight_path}. "
            f"Put {model_name}.pth in CLIDD-main/weights before running TEST2-SLAM."
        )
    return root_path


@contextmanager
def _prepended_sys_path(path: Path):
    text = str(path)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


@contextmanager
def _temporary_cwd(path: Path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _load_clidd_class(root: Path):
    module_path = root / "clidd.py"
    module_name = f"_test2_slam_clidd_{abs(hash(str(root)))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import CLIDD from {module_path}")
    module = importlib.util.module_from_spec(spec)
    with _prepended_sys_path(root):
        spec.loader.exec_module(module)
    return module.CLIDD


def load_clidd_model(
    *,
    root: str | os.PathLike | None,
    model_name: str,
    top_k: int | None,
    radius: int,
    score: float,
    device: str | torch.device,
):
    root_path = validate_clidd_install(root, model_name)
    clidd_cls = _load_clidd_class(root_path)
    # CLIDD loads ./weights/{cfg}.pth, so construct it from its own directory.
    with _temporary_cwd(root_path):
        model = clidd_cls(model_name, top_k, radius=radius, score=score)
    return model.to(device).eval()
