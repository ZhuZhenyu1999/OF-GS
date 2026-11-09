"""Capture bounded CPU snapshots on the SLAM thread, never on HTTP threads."""

from __future__ import annotations

import base64
import time
import warnings

import cv2
import numpy as np
import torch

from utils import resize_intrinsics, resize_longest_side


def encode_image(image, max_size):
    """Encode an RGB CHW float tensor after resizing, before copying off GPU."""
    image = resize_longest_side(image.detach(), max_size)
    pixels = (image.nan_to_num().clamp(0, 1) * 255).to(torch.uint8)
    pixels = pixels.permute(1, 2, 0).contiguous().cpu().numpy()
    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Could not encode the live viewer image")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


class SnapshotPublisher:
    def __init__(self, viewer, max_points=20000, max_size=640, interval=0.5):
        self.viewer = viewer
        self.max_points = max_points
        self.max_size = max_size
        self.interval = interval
        self.last_update = float("-inf")
        self.elapsed_seconds = 0.0
        self.render_error = None

    def wait_if_paused(self):
        start = time.perf_counter()
        try:
            self.viewer.wait_if_paused()
        finally:
            self.elapsed_seconds += time.perf_counter() - start

    @torch.no_grad()
    def update(self, scene, frame_id, total_frames, image, info, Rt=None,
               force=False, status="running"):
        if not force and time.monotonic() - self.last_update < self.interval:
            return False
        start = time.perf_counter()
        try:
            scene.join_optimization_thread()
            snapshot = self._capture(scene, frame_id, total_frames, image, info, Rt)
            snapshot["status"] = status
            self.viewer.publish(snapshot)
            self.last_update = time.monotonic()
            return True
        finally:
            self.elapsed_seconds += time.perf_counter() - start

    def _points(self, scene):
        # Old anchors can live on CPU; never move an anchor or concatenate the
        # full map onto the GPU just to visualize it. Exclude the active alias.
        sources = [(scene.xyz, scene.f_dc)]
        active = getattr(scene, "active_anchor", None)
        for anchor in getattr(scene, "anchors", ()):
            if anchor is active:
                continue
            params = anchor.gaussian_params
            xyz = params["xyz"]["val"]
            if xyz is scene.xyz:
                continue
            sources.append((xyz, params["f_dc"]["val"]))
        count = sum(xyz.shape[0] for xyz, _ in sources)
        # Deterministic global indices keep RNG state and SLAM sampling intact.
        selected = np.linspace(0, count - 1, min(count, self.max_points), dtype=np.int64)
        points, colors, offset = [], [], 0
        for xyz, dc in sources:
            ids = selected[(selected >= offset) & (selected < offset + len(xyz))] - offset
            offset += len(xyz)
            if not len(ids):
                continue
            ids = torch.as_tensor(ids, device=xyz.device)
            p = xyz.detach()[ids].cpu().float().numpy()
            c = (dc.detach()[ids, 0, :] * 0.28209479177387814 + 0.5).clamp(0, 1).cpu().float().numpy()
            valid = np.isfinite(p).all(axis=1) & np.isfinite(c).all(axis=1)
            points.extend(p[valid].tolist())
            colors.extend(c[valid].tolist())
        return points, colors, count

    def _capture(self, scene, frame_id, total_frames, image, info, Rt):
        # render() acquires the same non-reentrant scene lock, so leave this
        # block before invoking it. No GPU work is dispatched by HTTP handlers.
        with scene.lock:
            points, colors, count = self._points(scene)
            trajectory = []
            render_keyframe = None
            for keyframe in scene.keyframes:
                pose = keyframe.get_Rt().detach().cpu().float()
                if torch.isfinite(pose).all():
                    trajectory.append(torch.linalg.inv(pose)[:3, 3].tolist())
                if getattr(keyframe, "info", {}).get("source_frame_id") == frame_id:
                    render_keyframe = keyframe

        snapshot = {
            "frame_id": frame_id, "total_frames": total_frames,
            "num_keyframes": len(scene.keyframes), "num_gaussians": count,
            "points": points, "colors": colors, "trajectory": trajectory,
            "input_image": encode_image(image, self.max_size),
            "camera_pose": None, "camera_frame_id": None,
            "message": "", "render_error": self.render_error,
        }
        # Ordinary non-keyframes are deliberately not localized by this SLAM.
        # Show the most recent keyframe's render with its own frame label.
        if Rt is None and render_keyframe is None and scene.keyframes:
            render_keyframe = scene.keyframes[-1]
        render_frame_id = frame_id
        render_width, render_height = image.shape[-1], image.shape[-2]
        if render_keyframe is not None:
            Rt = render_keyframe.get_Rt().detach()
            K = render_keyframe.get_intrinsics()
            render_frame_id = getattr(render_keyframe, "info", {}).get("source_frame_id", frame_id)
            render_width, render_height = render_keyframe.width, render_keyframe.height
        else:
            K = info.get("K", info.get("K_np", scene.K))
        if Rt is None:
            snapshot.update(render_image=None, render_frame_id=None)
            return snapshot
        Rt = torch.as_tensor(Rt, device=scene.xyz.device, dtype=scene.xyz.dtype).detach()
        if not torch.isfinite(Rt).all():
            snapshot.update(render_image=None, render_frame_id=None)
            return snapshot
        snapshot["camera_pose"] = torch.linalg.inv(Rt).cpu().tolist()
        snapshot["camera_frame_id"] = render_frame_id
        snapshot["render_frame_id"] = render_frame_id
        if self.render_error:
            snapshot["render_image"] = None
            return snapshot
        try:
            scale = min(1.0, self.max_size / max(render_width, render_height))
            width, height = max(1, round(render_width * scale)), max(1, round(render_height * scale))
            K = torch.as_tensor(K, device=Rt.device, dtype=Rt.dtype)
            K = resize_intrinsics(K, width / render_width, height / render_height)
            rendered = scene.render(width, height, Rt.T.contiguous(), 1.0,
                                    intrinsics=K, detach_gaussians=True)["render"]
            if render_keyframe is not None:
                exposure = getattr(render_keyframe, "exposure", None)
                if exposure is not None:
                    exposure = exposure.detach().to(rendered)
                    rendered = (exposure[:3, :3] @ rendered.reshape(3, -1)
                                + exposure[:3, 3, None]).reshape(3, height, width)
            snapshot["render_image"] = encode_image(rendered, self.max_size)
        except (RuntimeError, ValueError, TypeError) as exc:
            # A preview failure must not discard a reconstruction. Retain the
            # input, trajectory and map; report once and disable extra renders.
            self.render_error = "Gaussian preview disabled: " + str(exc)
            warnings.warn(self.render_error, RuntimeWarning)
            snapshot.update(render_image=None, render_error=self.render_error)
        return snapshot
