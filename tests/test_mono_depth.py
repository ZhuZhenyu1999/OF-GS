import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene import mono_depth


class _FakeDepthAnything3:
    loaded_model_ids = []

    @classmethod
    def from_pretrained(cls, model_id):
        cls.loaded_model_ids.append(model_id)
        return _FakeDa3Model()


class _FakeDa3Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def inference(self, images):
        assert len(images) == 1
        assert images[0].dtype == np.uint8
        assert images[0].shape[-1] == 3
        return SimpleNamespace(
            depth=np.arange(20, dtype=np.float32).reshape(1, 4, 5),
            conf=np.full((1, 4, 5), 0.75, dtype=np.float32),
        )


class _FakeDa3ModelWithoutConfidence(_FakeDa3Model):
    def inference(self, images):
        return SimpleNamespace(depth=np.arange(20, dtype=np.float32).reshape(1, 4, 5))


class _NoisyFakeDa3Model(_FakeDa3Model):
    def inference(self, images):
        print("noisy da3 stdout")
        print("noisy da3 stderr", file=sys.stderr)
        return super().inference(images)


class _FakeDa2Model:
    def __call__(self, image):
        return torch.zeros(1, mono_depth.size, mono_depth.size)


def test_estimator_defaults_to_da2_when_da3_model_id_is_not_set(monkeypatch):
    init_calls = []

    def fake_init_da2(self):
        init_calls.append("da2")
        self.model = _FakeDa2Model()

    def fake_forward_da2(self, image):
        depth = torch.zeros(1, 1, mono_depth.size, mono_depth.size)
        confidence = torch.ones(1, 1, mono_depth.size, mono_depth.size)
        return depth, confidence

    monkeypatch.setattr(mono_depth.MonoDepthInternal, "_init_depth_anything_v2", fake_init_da2)
    monkeypatch.setattr(mono_depth.MonoDepthInternal, "_forward_depth_anything_v2", fake_forward_da2)
    monkeypatch.setattr(torch.cuda, "make_graphed_callables", lambda model, inputs: model)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    estimator = mono_depth.MonoDepthEstimator(5, 4, SimpleNamespace())
    image = torch.linspace(0, 1, 3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    depth, confidence = estimator(image)

    assert estimator.backend == "da2"
    assert init_calls == ["da2"]
    assert depth.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert confidence.shape == (1, 1, mono_depth.size, mono_depth.size)


def test_da3mono_estimator_keeps_old_depth_contract(monkeypatch):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    _FakeDepthAnything3.loaded_model_ids.clear()

    estimator = mono_depth.MonoDepthEstimator(
        5,
        4,
        SimpleNamespace(mono_depth_backend="da3mono", da3_model_id="fake/DA3MONO-LARGE"),
    )
    image = torch.linspace(0, 1, 3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)

    depth, confidence = estimator(image)

    assert _FakeDepthAnything3.loaded_model_ids == ["fake/DA3MONO-LARGE"]
    assert depth.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert confidence.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert torch.isfinite(depth).all()
    assert torch.allclose(confidence.mean(), torch.tensor(0.75), atol=1e-4)


def test_da3_model_id_forces_da3mono_backend(monkeypatch):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    _FakeDepthAnything3.loaded_model_ids.clear()

    estimator = mono_depth.MonoDepthEstimator(
        5,
        4,
        SimpleNamespace(mono_depth_backend="da2", da3_model_id="fake/DA3MONO-LARGE"),
    )

    assert estimator.backend == "da3mono"
    assert _FakeDepthAnything3.loaded_model_ids == ["fake/DA3MONO-LARGE"]


def test_da3mono_suppresses_model_inference_logs(monkeypatch, capsys):
    class NoisyFakeDepthAnything3(_FakeDepthAnything3):
        @classmethod
        def from_pretrained(cls, model_id):
            return _NoisyFakeDa3Model()

    monkeypatch.setattr(
        mono_depth,
        "_load_depth_anything3_class",
        lambda: NoisyFakeDepthAnything3,
    )

    model = mono_depth.MonoDepthInternal("da3mono", "fake/noisy")
    image = torch.linspace(0, 1, 3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)

    depth, confidence = model(image)
    captured = capsys.readouterr()

    assert "noisy da3" not in captured.out
    assert "noisy da3" not in captured.err
    assert depth.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert confidence.shape == (1, 1, mono_depth.size, mono_depth.size)


def test_da3mono_loads_existing_local_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    _FakeDepthAnything3.loaded_model_ids.clear()
    model_dir = tmp_path / "DA3MONO-LARGE"
    model_dir.mkdir()

    mono_depth.MonoDepthInternal("da3mono", str(model_dir))

    assert _FakeDepthAnything3.loaded_model_ids == [str(model_dir.resolve())]


def test_da3mono_resolves_hf_cache_snapshot_for_missing_local_alias(monkeypatch, tmp_path):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    _FakeDepthAnything3.loaded_model_ids.clear()
    requested_model_dir = tmp_path / "DA3MONO-LARGE"
    snapshot_dir = (
        tmp_path
        / "hub"
        / "models--depth-anything--DA3MONO-LARGE"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    mono_depth.MonoDepthInternal("da3mono", str(requested_model_dir))

    assert _FakeDepthAnything3.loaded_model_ids == [str(snapshot_dir.resolve())]


def test_da3mono_explicit_repo_id_prefers_repo_local_model(monkeypatch, tmp_path):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    monkeypatch.setattr(mono_depth, "REPO_ROOT", tmp_path)
    _FakeDepthAnything3.loaded_model_ids.clear()
    model_dir = tmp_path / "models" / "DA3MONO-LARGE"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    mono_depth.MonoDepthInternal("da3mono", mono_depth.DEFAULT_DA3_MODEL_ID)

    assert _FakeDepthAnything3.loaded_model_ids == [str(model_dir.resolve())]


def test_da3mono_reports_missing_explicit_local_model_path(monkeypatch, tmp_path):
    monkeypatch.setattr(mono_depth, "_load_depth_anything3_class", lambda: _FakeDepthAnything3)
    _FakeDepthAnything3.loaded_model_ids.clear()
    missing_model_dir = tmp_path / "missing" / "DA3MONO-LARGE"

    with pytest.raises(FileNotFoundError, match="Depth Anything 3 model path does not exist"):
        mono_depth.MonoDepthInternal("da3mono", str(missing_model_dir))

    assert _FakeDepthAnything3.loaded_model_ids == []


def test_da3mono_falls_back_to_sobel_confidence_when_model_conf_missing(monkeypatch):
    class FakeDepthAnything3NoConfidence(_FakeDepthAnything3):
        @classmethod
        def from_pretrained(cls, model_id):
            return _FakeDa3ModelWithoutConfidence()

    monkeypatch.setattr(
        mono_depth,
        "_load_depth_anything3_class",
        lambda: FakeDepthAnything3NoConfidence,
    )

    model = mono_depth.MonoDepthInternal("da3mono", "fake/no-conf")
    image = torch.linspace(0, 1, 3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)

    depth, confidence = model(image)

    assert depth.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert confidence.shape == (1, 1, mono_depth.size, mono_depth.size)
    assert torch.isfinite(confidence).all()
    assert confidence.min() >= 0
    assert confidence.max() <= 1
