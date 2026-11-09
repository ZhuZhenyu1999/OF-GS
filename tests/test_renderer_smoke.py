import importlib

import pytest
import torch

from utils import getProjectionMatrixFromIntrinsics


def _installed_renderer():
    try:
        module = importlib.import_module("diff_gaussian_rasterization")
    except ImportError as exc:
        pytest.skip(f"diff_gaussian_rasterization extension is not installed or built: {exc}")
    assert len(module.GaussianRasterizationSettings._fields) == 11, (
        "Installed rasterizer has a stale interface; rebuild submodules/diff-gaussian-rasterization"
    )
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for rasterizer smoke test")
def test_renderer_returns_images_and_supports_geometry_and_pose_gradients():
    module = _installed_renderer()
    settings = module.GaussianRasterizationSettings(
        16,
        16,
        1.0,
        1.0,
        torch.zeros(3, device="cuda"),
        1.0,
        torch.eye(4, device="cuda"),
        0,
        torch.zeros(3, device="cuda"),
        False,
        False,
    )
    rasterizer = module.GaussianRasterizer(settings)
    means3d = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    means2d = torch.zeros(1, 3, device="cuda", requires_grad=True)
    viewmatrix = torch.eye(4, device="cuda", requires_grad=True)
    out = rasterizer(
        means3d,
        means2d,
        torch.ones(1, 1, device="cuda") * 0.5,
        torch.ones(1, 1, 3, device="cuda") * 0.2,
        torch.empty(1, 0, 3, device="cuda"),
        torch.ones(1, 3, device="cuda") * 0.1,
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        viewmatrix,
    )
    assert len(out) == 4
    color, invdepth, main_gaussian_ids, radii = out
    assert color.shape == (3, 16, 16)
    assert invdepth.shape == (1, 16, 16)
    assert main_gaussian_ids.shape == (2, 16, 16)
    assert radii.shape == (1,)
    assert radii[0].item() > 0
    # A uniform image sum can cancel the screen-space position gradient of a
    # symmetric Gaussian. Weight pixels asymmetrically to exercise the signal
    # used by gradient-based densification.
    rows, columns = torch.meshgrid(
        torch.arange(16, device="cuda"), torch.arange(16, device="cuda"), indexing="ij"
    )
    spatial_weight = 1.0 + columns / 16.0 + rows / 32.0
    ((color * spatial_weight).sum() + invdepth.sum()).backward()
    for parameter in (means3d, means2d, viewmatrix):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert means3d.grad.abs().sum().item() > 0
    assert means2d.grad[:, :2].abs().sum().item() > 0
    assert viewmatrix.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for rasterizer smoke test")
def test_renderer_places_optical_axis_at_calibrated_principal_point():
    module = _installed_renderer()
    width, height = 80, 64
    K = torch.tensor([[60.0, 0.0, 30.25], [0.0, 90.0, 25.75], [0.0, 0.0, 1.0]], device="cuda")
    projection = getProjectionMatrixFromIntrinsics(0.01, 100.0, K, width, height).T.contiguous()
    settings = module.GaussianRasterizationSettings(
        height, width, width / (2 * K[0, 0].item()), height / (2 * K[1, 1].item()),
        torch.zeros(3, device="cuda"), 1.0, projection, 0, torch.zeros(3, device="cuda"),
        False, False,
    )
    image, *_ = module.GaussianRasterizer(settings)(
        torch.tensor([[0.0, 0.0, 2.0]], device="cuda"),
        torch.zeros(1, 3, device="cuda"),
        torch.full((1, 1), 0.5, device="cuda"),
        torch.full((1, 1, 3), 0.2, device="cuda"),
        torch.empty(1, 0, 3, device="cuda"),
        torch.full((1, 3), 0.05, device="cuda"),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
        torch.eye(4, device="cuda"),
    )
    weights = image.sum(dim=0)
    rows, columns = torch.meshgrid(
        torch.arange(height, device="cuda"), torch.arange(width, device="cuda"), indexing="ij"
    )
    assert weights.sum().item() > 0
    centroid = torch.stack([(weights * columns).sum(), (weights * rows).sum()]) / weights.sum()
    assert torch.allclose(centroid, K[:2, 2], atol=0.05)
