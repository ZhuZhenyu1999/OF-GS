"""Depth export correctness without model weights, CUDA, or SLAM execution."""

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from args import get_args
import depth_export


def source_png(tmp_path, name="000094.png"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.array([[[4, 30, 200], [17, 80, 240], [250, 120, 10]],
                      [[24, 60, 40], [19, 7, 66], [1, 2, 3]]], dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())
    return path, image


def read_png(path):
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def test_raw_prediction_is_preserved_and_png_maps_near_to_bright(tmp_path):
    source, _ = source_png(tmp_path)
    depth = np.array([[2, 4, 6], [8, 10, 12]], dtype=np.float32)
    result = depth_export.save_relative_depth(depth, source, tmp_path / "深度输出",
                                               metadata={"encoder": "vitl"})
    raw = np.load(result["raw_depth"], allow_pickle=False)
    assert raw.dtype == np.float32
    np.testing.assert_array_equal(raw, depth)
    gray = read_png(result["gray_png"])
    np.testing.assert_array_equal(gray, [[0, 51, 102], [153, 204, 255]])
    np.testing.assert_array_equal(read_png(result["color_png"]),
                                  cv2.applyColorMap(gray, cv2.COLORMAP_TURBO))
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert metadata["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["source_image"] == str(source.resolve())
    assert metadata["shape"] == [2, 3]
    assert metadata["encoder"] == "vitl"
    assert metadata["saved_dtype"] == "float32"
    assert metadata["min"] == 2 and metadata["max"] == 12
    assert metadata["visualization"]["raw_npy_normalized"] is False
    assert "not metric" in metadata["convention"]


def test_constant_map_keeps_raw_values_without_division_by_zero(tmp_path):
    source, _ = source_png(tmp_path)
    result = depth_export.save_relative_depth(np.full((2, 3), 7.25), source, tmp_path / "out")
    np.testing.assert_array_equal(np.load(result["raw_depth"]), np.full((2, 3), 7.25))
    assert not read_png(result["gray_png"]).any()
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["constant_map"] is True
    assert metadata["min"] == metadata["max"] == 7.25


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_fails_before_writing_misleading_outputs(tmp_path, invalid):
    source, _ = source_png(tmp_path)
    depth = np.ones((2, 3), dtype=np.float32)
    depth[0, 0] = invalid
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="NaN or infinity"):
        depth_export.save_relative_depth(depth, source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("depth", [np.empty((0, 2)), np.ones((1, 2, 3))])
def test_invalid_depth_shape_is_rejected(tmp_path, depth):
    source, _ = source_png(tmp_path)
    with pytest.raises(ValueError, match="nonempty 2D"):
        depth_export.save_relative_depth(depth, source, tmp_path / "out")


def test_selected_images_use_exact_names_and_preserve_request_order(tmp_path):
    first, _ = source_png(tmp_path, "000093.png")
    requested, _ = source_png(tmp_path)
    assert depth_export.select_requested_images([first, requested], [requested.name, first.name, requested.name]) == [requested, first]
    with pytest.raises(FileNotFoundError, match="000095.png"):
        depth_export.select_requested_images([first, requested], ["000095.png"])
    # A sequence limited by --start_at/--max_frames must not borrow an excluded frame.
    with pytest.raises(FileNotFoundError, match="000094.png"):
        depth_export.select_requested_images([first], [requested.name])


@pytest.mark.parametrize("name", ["../000094.png", "sub/000094.png", "sub\\000094.png", "C:000094.png", "", ".."])
def test_requested_names_cannot_contain_paths(name):
    with pytest.raises(ValueError, match="filename"):
        depth_export.select_requested_images([], [name])


def test_ambiguous_names_and_output_stem_collisions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="ambiguous"):
        depth_export.select_requested_images([tmp_path / "a/000094.png", tmp_path / "b/000094.png"], ["000094.png"])
    with pytest.raises(ValueError, match="same output stem"):
        depth_export.select_requested_images([tmp_path / "000094.png", tmp_path / "000094.jpg"], ["000094.png", "000094.jpg"])


def test_predict_converts_bgr_to_rgb_and_restores_original_dimensions():
    bgr = np.full((4, 6, 3), [10, 20, 200], dtype=np.uint8)
    captured = {}

    def transform(sample):
        captured["rgb"] = sample["image"].copy()
        return {"image": sample["image"].transpose(2, 0, 1).astype(np.float32)}

    def model(image):
        assert not torch.is_grad_enabled()
        assert image.dtype == torch.float32
        captured["input"] = image
        return torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    raw, shape = depth_export._predict(model, bgr, torch.device("cpu"), torch.float32, [transform])
    np.testing.assert_allclose(captured["rgb"][0, 0], [200 / 255, 20 / 255, 10 / 255])
    assert shape == [4, 6]
    assert raw.shape == bgr.shape[:2]
    assert raw.dtype == np.float32
    assert raw[0, 0] == 1 and raw[-1, -1] == 7


def mock_predictor(monkeypatch):
    class FakeModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            torch.rand(4)  # Model initialization must not perturb SLAM's RNG.

    class Transform:
        def __init__(self, **kwargs):
            pass

    fake_attention = ModuleType("depth_anything_v2.dinov2_layers.attention")
    fake_attention.Attention = type("Attention", (torch.nn.Module,), {})
    fake_attention.MemEffAttention = type("MemEffAttention", (torch.nn.Module,), {})
    monkeypatch.setitem(sys.modules, fake_attention.__name__, fake_attention)
    monkeypatch.setattr(depth_export, "_load_components", lambda: (FakeModel, Transform, Transform, Transform))
    calls = []

    def predict(model, bgr, device, dtype, transforms):
        calls.append(bgr.copy())
        torch.rand(2)
        assert device.type == "cpu" and dtype == torch.float32
        return np.arange(bgr.shape[0] * bgr.shape[1], dtype=np.float32).reshape(bgr.shape[:2]), [518, 784]

    monkeypatch.setattr(depth_export, "_predict", predict)
    return calls


def test_export_uses_original_pixels_and_preserves_torch_rng(monkeypatch, tmp_path):
    source, image = source_png(tmp_path)
    checkpoint = tmp_path / "model.pth"
    torch.save({}, checkpoint)
    calls = mock_predictor(monkeypatch)
    before = torch.get_rng_state().clone()
    result = depth_export.export_da2_images([source], tmp_path / "out", device="cpu", checkpoint=checkpoint)
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert len(calls) == len(result) == 1
    np.testing.assert_array_equal(calls[0], image)
    metadata = json.loads(Path(result[0]["metadata"]).read_text(encoding="utf-8"))
    assert metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata["preprocessing"]["slam_normalization_or_alignment"] is False
    assert metadata["network_input_shape"] == [518, 784]
    assert metadata["inference_dtype"] == "float32"
    assert metadata["device"] == "cpu"


def test_export_restores_rng_after_prediction_failure(monkeypatch, tmp_path):
    source, _ = source_png(tmp_path)
    checkpoint = tmp_path / "model.pth"
    torch.save({}, checkpoint)
    mock_predictor(monkeypatch)

    def fail(*args):
        torch.rand(3)
        raise RuntimeError("prediction failed")

    monkeypatch.setattr(depth_export, "_predict", fail)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="prediction failed"):
        depth_export.export_da2_images([source], tmp_path / "out", device="cpu", checkpoint=checkpoint)
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert not (tmp_path / "out").exists()


def test_empty_or_missing_inputs_do_not_load_models(monkeypatch, tmp_path):
    monkeypatch.setattr(depth_export, "_load_components", lambda: pytest.fail("model must not load"))
    assert depth_export.export_da2_images([], tmp_path / "out", checkpoint=tmp_path / "missing.pth") == []
    with pytest.raises(FileNotFoundError):
        depth_export.export_da2_images([tmp_path / "missing.png"], tmp_path / "out")
    source, _ = source_png(tmp_path)
    with pytest.raises(FileNotFoundError):
        depth_export.export_da2_images([source], tmp_path / "out", checkpoint=tmp_path / "missing.pth")


def test_export_cli_is_opt_in_and_accepts_multiple_filenames(monkeypatch, tmp_path):
    argv = ["slam.py", "-s", str(tmp_path), "-m", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    assert get_args().export_da2_frames == []
    monkeypatch.setattr(sys, "argv", argv + ["--export_da2_frames", "000094.png", "000095.png"])
    assert get_args().export_da2_frames == ["000094.png", "000095.png"]


def test_slam_exports_before_frontend_or_keyframe_processing(monkeypatch, tmp_path):
    import slam

    source, _ = source_png(tmp_path)
    args = SimpleNamespace(seed=0, device="cpu", model_path=str(tmp_path / "run"), export_da2_frames=[source.name])
    events = []
    class Dataset:
        image_paths = [source]

        def get_image_size(self):
            return 2, 3

        def __len__(self):
            return 1

    dataset = Dataset()
    monkeypatch.setattr(slam, "create_dataset", lambda args: (dataset, "waymo"))
    monkeypatch.setattr(slam, "maybe_seed_from_dataset", lambda *args: None)

    def export(paths, output_dir, device="auto", checkpoint=None):
        assert paths == [source]
        assert Path(output_dir) == Path(args.model_path) / "da2_depth"
        assert device == "cpu"
        events.append("export")
        return []

    def frontend(*args):
        events.append("frontend")
        raise RuntimeError("stop after export")

    monkeypatch.setattr(depth_export, "export_da2_images", export)
    monkeypatch.setattr(slam, "SLAMFrontend", frontend)
    with pytest.raises(RuntimeError, match="stop after export"):
        slam.run_slam(args)
    assert events == ["export", "frontend"]
