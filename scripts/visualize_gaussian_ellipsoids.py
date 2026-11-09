"""Render saved SLAM Gaussian iso-surfaces from a named keyframe with VTK.

Dependencies: numpy, plyfile, Pillow, vtk>=9.5. No CUDA SLAM extensions needed.
The output is shaded, opaque ellipsoid geometry, not alpha-composited 3DGS RGB.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from plyfile import PlyData

# Keep the optional desktop visualization dependency separate from SLAM's env.
_local_deps = Path(__file__).resolve().parents[1] / ".codex_tmp/gaussian_viz_deps"
if _local_deps.is_dir():
    sys.path.insert(0, str(_local_deps))


def quaternion_matrices(q):
    """Normalized wxyz quaternions -> local-to-world rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
    ], axis=1).reshape(-1, 3, 3)


def load_map(model_path, image_name):
    metadata_path = model_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matches = [(i, k) for i, k in enumerate(metadata["keyframes"])
               if Path(k.get("info", {}).get("name", "")).name == image_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered keyframe named {image_name!r}, found {len(matches)}")
    keyframe_id, keyframe = matches[0]
    config = metadata["config"]
    width, height = int(config["width"]), int(config["height"])
    K = np.asarray(keyframe.get("K", config.get("K")), dtype=np.float64)
    Rt = np.asarray(keyframe["Rt"], dtype=np.float64)
    if K.shape != (3, 3) or Rt.shape != (4, 4):
        raise ValueError("Expected saved 3x3 K and 4x4 world-to-camera Rt")
    if not np.isfinite(K).all() or not np.isfinite(Rt).all():
        raise ValueError("Camera contains non-finite values")
    if not np.allclose(K[2], [0, 0, 1]) or not np.isclose(K[0, 1], 0):
        raise ValueError("Only zero-skew pinhole K is supported")
    if not np.allclose(Rt[:3, :3] @ Rt[:3, :3].T, np.eye(3), atol=1e-4):
        raise ValueError("Saved Rt does not contain an orthonormal camera rotation")
    chunks, sources = [], []
    for i in range(len(metadata["anchors"])):
        path = model_path / "point_clouds" / f"anchor_{i}.ply"
        data = PlyData.read(path)["vertex"].data
        def columns(names):
            return np.column_stack([data[n] for n in names]).astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            scales = np.exp(columns([f"scale_{j}" for j in range(3)]))
        chunks.append({
            "xyz": columns(["x", "y", "z"]), "scales": scales,
            "q": columns([f"rot_{j}" for j in range(4)]),
            "alpha": 1 / (1 + np.exp(-np.clip(columns(["opacity"])[:, 0], -80, 80))),
            "rgb": np.clip(.5 + .28209479177387814 * columns([f"f_dc_{j}" for j in range(3)]), 0, 1),
        })
        sources.append({"path": str(path.resolve()), "count": len(data), "bytes": path.stat().st_size})
    if not chunks:
        raise ValueError("The saved map contains no anchors")
    arrays = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    return arrays, K, Rt, width, height, keyframe_id, sources


def select_gaussians(arrays, K, Rt, width, height, args):
    count = len(arrays["xyz"])
    mask = np.ones(count, dtype=bool)
    for value in arrays.values():
        mask &= np.isfinite(value.reshape(count, -1)).all(axis=1)
    mask &= np.linalg.norm(arrays["q"], axis=1) > 1e-12
    mask &= (arrays["scales"] > 0).all(axis=1)
    stats = {"total": count, "finite_valid": int(mask.sum())}
    mask &= arrays["alpha"] >= args.min_opacity
    stats["after_opacity"] = int(mask.sum())
    mask &= arrays["scales"].max(axis=1) <= args.max_scale
    stats["after_max_scale"] = int(mask.sum())
    camera_xyz = arrays["xyz"] @ Rt[:3, :3].T + Rt[:3, 3]
    mask &= (camera_xyz[:, 2] >= args.near) & (camera_xyz[:, 2] <= args.far)
    stats["after_center_depth"] = int(mask.sum())
    ids = np.flatnonzero(mask)
    # Exact support of each ellipsoid along each frustum plane normal.
    axes = np.einsum("ij,njk->nik", Rt[:3, :3], quaternion_matrices(arrays["q"][ids]))
    axes *= (args.sigma * arrays["scales"][ids])[:, None, :]
    # Pinhole pixel centers are integers; image edges are at -.5 and size-.5.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2]+.5, K[1, 2]+.5
    normals = np.array([[fx, 0, cx], [-fx, 0, width-cx],
                        [0, fy, cy], [0, -fy, height-cy]], dtype=float)
    support = np.linalg.norm(np.einsum("pj,njk->npk", normals, axes), axis=2)
    intersects = (camera_xyz[ids] @ normals.T + support >= 0).all(axis=1)
    ids = ids[intersects]
    stats["frustum_intersections"] = len(ids)
    if args.max_gaussians and len(ids) > args.max_gaussians:
        ids = np.sort(np.random.default_rng(args.seed).choice(ids, args.max_gaussians, replace=False))
    stats["drawn"] = len(ids)
    if not len(ids):
        raise ValueError("No Gaussians passed the visualization filters")
    return ids, stats, camera_xyz


def vtk_matrix(vtk, values):
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, float(values[i, j]))
    return matrix


def projection_matrix(K, width, height, near, far):
    """OpenCV pixels (x right, y down, z forward) -> VTK OpenGL clip space."""
    return np.array([
        [2*K[0, 0]/width, 0, 1-2*(K[0, 2]+.5)/width, 0],
        [0, 2*K[1, 1]/height, 2*(K[1, 2]+.5)/height-1, 0],
        [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
        [0, 0, -1, 0],
    ])


def render(arrays, ids, K, Rt, width, height, args, path, interactive=False):
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise SystemExit('Install the optional renderer: python -m pip install "vtk>=9.5"') from exc
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(arrays["xyz"][ids], dtype=np.float32), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    q = arrays["q"][ids]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    for name, values in [
        ("scale", (arrays["scales"][ids] * args.sigma).astype(np.float32)),
        ("orientation", q.astype(np.float32)),
        ("rgb", np.round(arrays["rgb"][ids] * 255).astype(np.uint8)),
    ]:
        data = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
        data.SetName(name)
        poly.GetPointData().AddArray(data)
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)
    sphere.SetThetaResolution(16)
    sphere.SetPhiResolution(12)
    mapper = vtk.vtkGlyph3DMapper()
    mapper.SetInputData(poly)
    mapper.SetSourceConnection(sphere.GetOutputPort())
    mapper.SetScaleArray("scale")
    mapper.SetScaleModeToScaleByVectorComponents()
    mapper.SetScaleFactor(1.0)
    mapper.SetOrientationArray("orientation")
    mapper.SetOrientationModeToQuaternion()
    mapper.OrientOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("rgb")
    mapper.SetColorModeToDirectScalars()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetUserMatrix(vtk_matrix(vtk, Rt))
    actor.GetProperty().SetAmbient(.42)
    actor.GetProperty().SetDiffuse(.58)
    actor.GetProperty().SetSpecular(.15)
    actor.GetProperty().SetSpecularPower(18)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.055, .071, .098)
    renderer.AddActor(actor)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 0)
    camera.SetFocalPoint(0, 0, 1)
    camera.SetViewUp(0, -1, 0)
    clip_far = args.far + args.sigma * args.max_scale * 2
    camera.SetClippingRange(args.near * .1, clip_far)
    camera.SetExplicitProjectionTransformMatrix(vtk_matrix(
        vtk, projection_matrix(K, width, height, args.near*.1, clip_far)))
    camera.UseExplicitProjectionTransformMatrixOn()
    window = vtk.vtkRenderWindow()
    window.SetWindowName(f"Gaussian ellipsoids | {args.image_name}")
    window.AddRenderer(renderer)
    window.SetSize(width, height)
    window.SetMultiSamples(4)
    if not interactive:
        window.SetOffScreenRendering(1)
    window.Render()
    # Verify the actual VTK camera against pinhole projection, including y flip.
    camera_points = np.array([[0., 0., 2.], [.3, .2, 3.], [-.7, -.4, 4.]])
    errors = []
    for p in camera_points:
        renderer.SetWorldPoint(*p, 1.)
        renderer.WorldToDisplay()
        screen = renderer.GetDisplayPoint()
        expected = K @ p
        expected = expected[:2] / expected[2]
        errors.append(float(np.linalg.norm(np.array([screen[0]-.5, height-screen[1]-.5]) - expected)))
    if max(errors) > 1e-4:
        window.Finalize()
        raise RuntimeError(f"Camera projection validation failed: {errors}")
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.SetInputBufferTypeToRGB()
    capture.ReadFrontBufferOff()
    capture.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    if interactive:
        # Orbit about the scene rather than about a point one unit from camera.
        median_depth = float(np.median((arrays["xyz"][ids] @ Rt[:3, :3].T + Rt[:3, 3])[:, 2]))
        camera.SetFocalPoint(0, 0, median_depth)
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        interactor.Initialize()
        interactor.Start()
    window.Finalize()
    return {"vtk_version": vtk.vtkVersion.GetVTKVersion(), "projection_max_error_px": max(errors)}


def comparison(original_path, rendered_path, output_path, name, sigma, count, crop=None):
    rendered = Image.open(rendered_path).convert("RGB")
    original = Image.open(original_path).convert("RGB")
    if crop is not None:
        x0, y0, x1, y1 = crop
        original = original.crop((round(x0*original.width), round(y0*original.height),
                                  round(x1*original.width), round(y1*original.height)))
    original = original.resize(rendered.size, Image.Resampling.LANCZOS)
    panel_w = 900
    panel_h = round(panel_w * rendered.height / rendered.width)
    canvas = Image.new("RGB", (panel_w*2+72, panel_h+160), "#111822")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()
    draw.text((24, 16), f"Gaussian map  /  {name}" + ("  /  detail" if crop else ""), fill="#f5f7fa", font=font(27))
    draw.text((24, 58), "Original PNG", fill="#bac9d8", font=font(21))
    draw.text((panel_w+48, 58), f"Gaussian ellipsoids  /  {sigma:g} sigma", fill="#bac9d8", font=font(21))
    canvas.paste(original.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (24, 94))
    canvas.paste(rendered.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (panel_w+48, 94))
    draw.text((24, panel_h+110), f"Saved final map | {count:,} ellipsoids | Opaque shaded surfaces; SH DC color | Same camera pose", fill="#9eafc1", font=font(18))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--image_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--original_image", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Optional normalized image crop, e.g. .55 .4 .95 .9")
    parser.add_argument("--sigma", type=float, default=1.)
    parser.add_argument("--min_opacity", type=float, default=.05)
    parser.add_argument("--near", type=float, default=.05)
    parser.add_argument("--far", type=float, default=100.)
    parser.add_argument("--max_scale", type=float, default=1.)
    parser.add_argument("--max_gaussians", type=int, default=0, help="0 draws all filtered Gaussians")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interactive", action="store_true", help="Open an orbitable VTK view after saving")
    args = parser.parse_args()
    if not (args.width > 0 and args.sigma > 0 and 0 <= args.min_opacity <= 1
            and 0 < args.near < args.far and args.max_scale > 0 and args.max_gaussians >= 0):
        parser.error("Invalid dimensions, sigma, opacity, depth range, scale or sample limit")
    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            parser.error("Crop must satisfy 0 <= X0 < X1 <= 1 and 0 <= Y0 < Y1 <= 1")
    arrays, K, Rt, width, height, kf_id, sources = load_map(args.model_path, args.image_name)
    if args.crop is not None:
        x0, y0, x1, y1 = args.crop
        left, top, right, bottom = round(x0*width), round(y0*height), round(x1*width), round(y1*height)
        if right <= left or bottom <= top:
            parser.error("Crop has zero area at the saved image resolution")
        args.crop = [left/width, top/height, right/width, bottom/height]
        K = K.copy()
        K[0, 2] -= left
        K[1, 2] -= top
        width, height = right-left, bottom-top
    selected, stats, _ = select_gaussians(arrays, K, Rt, width, height, args)
    output_height = round(height * args.width / width)
    K_output = K.copy()
    K_output[0] *= args.width / width
    K_output[1] *= output_height / height
    K_output[0, 2] = (K[0, 2]+.5) * args.width / width - .5
    K_output[1, 2] = (K[1, 2]+.5) * output_height / height - .5
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image_name).stem
    path = args.output_dir / f"{stem}_ellipsoids.png"
    validation = render(arrays, selected, K_output, Rt, args.width, output_height, args, path, args.interactive)
    report = {
        "description": "Saved final Gaussian map, viewed from the named keyframe; opaque ellipsoid iso-surfaces, not 3DGS splatting.",
        "parameters": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "keyframe_index": kf_id, "K": K_output.tolist(), "Rt_world_to_camera": Rt.tolist(),
        "output_size": [args.width, output_height], "sources": sources, "counts": stats,
        "coordinate_units": "SLAM map units, not asserted metric meters",
        "surface": {"sigma": args.sigma, "opacity": 1.0, "color": "SH DC, without exposure or view-dependent SH"},
        "selection": "Opacity and maximum 1-sigma semiaxis filter, center depth range, conservative ellipsoid/frustum culling, optional seeded uniform sampling; no occlusion culling before drawing.",
        "comparison_image": "Original PNG is only cropped/resized, not additionally undistorted. Saved metadata does not specify the original input image model; comparison is not asserted pixel-aligned.",
        "validation": validation,
    }
    (args.output_dir / f"{stem}_ellipsoids.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(args.output_dir / f"{stem}_ellipsoids.npz", xyz=arrays["xyz"][selected],
                        scales=arrays["scales"][selected], rotation_wxyz=arrays["q"][selected],
                        opacity=arrays["alpha"][selected], rgb=arrays["rgb"][selected],
                        source_indices=selected, K=K_output, Rt=Rt)
    if args.original_image:
        comparison(args.original_image, path, args.output_dir / f"{stem}_comparison.png", args.image_name, args.sigma, len(selected), args.crop)
    print(json.dumps({"image": str(path.resolve()), "counts": stats, "validation": validation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
