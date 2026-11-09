"""Exercise the deployed eight-camera MiniBA CUDA graph without pytest.

Run with the same Python and CUDA_VISIBLE_DEVICES as SLAM. For example:
    python scripts/check_bootstrap_calibration.py
    python scripts/check_bootstrap_calibration.py --npts 2000 --iters 200
"""

import argparse
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def project_points(torch, points, rotations, translations, focals, centres):
    """Compute pixels directly, independent of MiniBA's projection helper."""
    first = torch.nn.functional.normalize(rotations[..., 0], dim=-1)
    second = rotations[..., 1]
    second = torch.nn.functional.normalize(
        second - (first * second).sum(-1, keepdim=True) * first, dim=-1
    )
    rotation_matrices = torch.stack(
        (first, second, torch.cross(first, second, dim=-1)), dim=-1
    )
    local = torch.einsum("cij,pj->pci", rotation_matrices, points) + translations[None]
    return local[..., :2] / local[..., 2:] * focals[None] + centres[None]


def run_check(npts, iters):
    import torch
    import poses.mini_ba as mini_ba

    module_path = Path(mini_ba.__file__).resolve()
    print(f"Python: {sys.executable}", flush=True)
    print(f"Project root: {ROOT}", flush=True)
    print(f"MiniBA source: {module_path}", flush=True)
    print(f"MiniBA SHA256: {hashlib.sha256(module_path.read_bytes()).hexdigest()}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable; this check requires a CUDA GPU.", file=sys.stderr)
        return 2
    print(f"CUDA device: {torch.cuda.current_device()} ({torch.cuda.get_device_name()})", flush=True)
    print(f"Capturing bootstrap graph: 8 cameras, {npts} points, {iters} iterations", flush=True)

    # Vary both focal axes and the principal point across all eight cameras.
    generator = torch.Generator().manual_seed(61)
    points = torch.rand(npts, 3, generator=generator, dtype=torch.float32)
    points[:, :2] = points[:, :2] * 3 - 1.5
    points[:, 2] = points[:, 2] * 4 + 3
    points = points.cuda()
    rotations = torch.eye(3, device="cuda")[None, :, :2].repeat(8, 1, 1)
    offsets = torch.arange(8, device="cuda", dtype=points.dtype)
    translations = points.new_tensor([0.18, -0.08, 0.04]).repeat(8, 1)
    translations[:, 0] -= offsets * 0.1
    focals = torch.stack((800 + offsets * 2, 1100 + offsets * 3), dim=-1)
    centres = torch.stack((949.57 + offsets * 0.2, 633.24 - offsets * 0.1), dim=-1)
    initial_t = translations.clone()
    initial_t[0] += points.new_tensor([0.02, -0.01, 0.01])

    optimizer = mini_ba.MiniBA(
        1, 8, 0, npts, optimize_focal=False, optimize_3Dpts=True,
        make_cuda_graph=True, huber_delta=0, outlier_mad_scale=4,
        ep=1e-5, iters=iters,
    )
    with torch.no_grad():
        for replay in range(2):
            replay_focals = focals * (1 + replay * 0.05)
            replay_centres = centres + points.new_tensor([3.0, -2.0]) * replay
            uv = project_points(torch, points, rotations, translations, replay_focals, replay_centres)
            # Match PoseInitializer: no singleton batch axis on poses/points/uv.
            result = optimizer(
                rotations, initial_t, replay_focals, points, replay_centres, uv.reshape(-1)
            )
            if result[2].shape != (8, 2):
                raise AssertionError(f"focal output shape {tuple(result[2].shape)}; expected (8, 2)")
            torch.testing.assert_close(result[2], replay_focals, atol=0, rtol=0)
            reprojection = project_points(
                torch, result[3], result[0], result[1], replay_focals, replay_centres
            ) - uv
            residual = result[4].reshape_as(reprojection)
            torch.testing.assert_close(residual, reprojection, atol=5e-4, rtol=0)
            maximum_error = torch.maximum(residual.abs().max(), reprojection.abs().max()).item()
            if not torch.isfinite(reprojection).all().item() or maximum_error >= 2e-3:
                raise AssertionError(f"replay {replay + 1}: maximum reprojection error {maximum_error:.8f}px; expected < 0.002px")
            print(
                f"PASS replay {replay + 1}/2: fx/fy preserved for 8 cameras; "
                f"maximum reprojection error {maximum_error:.8f}px",
                flush=True,
            )
    torch.cuda.synchronize()
    print("PASS: deployed MiniBA supports eight-camera calibration and changing intrinsics across CUDA graph replays.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=positive_int, default=24, help="number of bootstrap points (default: 24)")
    parser.add_argument("--iters", type=positive_int, default=30, help="MiniBA iterations (default: 30)")
    args = parser.parse_args()
    if args.npts < 8:
        parser.error("--npts must be at least 8")
    try:
        return run_check(args.npts, args.iters)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
