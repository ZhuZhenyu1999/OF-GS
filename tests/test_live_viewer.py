"""Real HTTP and CPU snapshot checks; no CUDA extensions or dataset required."""

import base64
import json
import socket
import sys
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from args import get_args
from visualization.server import LiveViewer


def request(viewer, path="/api/state", payload=None, headers=None, raw=None):
    headers = dict(headers or {})
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    query = Request(viewer.url + path, data=data, headers=headers)
    try:
        response = urlopen(query, timeout=3)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, response.headers, response.read()


def state(viewer):
    status, _, body = request(viewer)
    assert status == 200
    return json.loads(body)


@pytest.fixture
def viewer():
    live = LiveViewer("127.0.0.1", 0).start()
    try:
        yield live
    finally:
        live.close()


def test_http_serves_offline_ui_and_atomic_partial_snapshots(viewer):
    status, headers, body = request(viewer, "/")
    assert status == 200
    assert headers.get_content_type() == "text/html"
    assert headers["Cache-Control"] == "no-store"
    assert b"/api/state" in body
    assert b"/api/control" in body
    assert b"https://cdn" not in body
    assert "评测与保存" in body.decode("utf-8")
    assert "最终优化" not in body.decode("utf-8")
    assert viewer.port > 0

    points = [[1, 2, 3]]
    viewer.publish({"points": points, "colors": [[1, 0, 0]], "frame_id": 2})
    first = state(viewer)
    points[0][0] = 999
    viewer.publish({"frame_id": 3, "message": "地图更新"})
    second = state(viewer)
    assert second["points"] == [[1, 2, 3]]
    assert second["colors"] == [[1, 0, 0]]
    assert second["frame_id"] == 3
    assert second["message"] == "地图更新"
    assert second["revision"] > first["revision"]
    with pytest.raises(ValueError):
        viewer.publish({"points": [[float("nan"), 0, 0]]})
    assert state(viewer) == second


def test_controls_pause_resume_and_do_not_lose_updates(viewer):
    viewer.set_status("running")
    assert request(viewer, "/api/control", {"action": "pause"})[0] == 200
    assert state(viewer)["paused"] is True
    finished = threading.Event()
    worker = threading.Thread(target=lambda: (viewer.wait_if_paused(), finished.set()))
    worker.start()
    try:
        assert not finished.wait(0.05)
        viewer.publish({"frame_id": 4})
        assert state(viewer)["paused"] is True
        assert request(viewer, "/api/control", {"action": "resume"})[0] == 200
        assert finished.wait(2)
        assert state(viewer)["paused"] is False
    finally:
        viewer.close()
        worker.join(timeout=2)
    assert not worker.is_alive()


def test_close_wakes_paused_frame_boundary_and_is_idempotent(viewer):
    request(viewer, "/api/control", {"action": "pause"})
    finished = threading.Event()
    worker = threading.Thread(target=lambda: (viewer.wait_if_paused(), finished.set()))
    worker.start()
    assert not finished.wait(0.05)
    viewer.close()
    viewer.close()
    assert finished.wait(2)
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize("final_status", ["finalizing", "complete", "error", "stopped"])
def test_terminal_state_clears_pause_and_rejects_new_controls(viewer, final_status):
    request(viewer, "/api/control", {"action": "pause"})
    viewer.set_status(final_status, "Finished")
    snapshot = state(viewer)
    assert snapshot["status"] == final_status
    assert snapshot["paused"] is False
    assert request(viewer, "/api/control", {"action": "pause"})[0] == 409


@pytest.mark.parametrize("path", ["/missing", "/../args.py", "/api/state/missing"])
def test_unknown_http_paths_never_serve_workspace_files(viewer, path):
    assert request(viewer, path)[0] == 404


@pytest.mark.parametrize("payload", [None, [], {"action": "delete"}, {"action": 1}])
def test_control_rejects_invalid_json_shapes(viewer, payload):
    code, _, _ = request(viewer, "/api/control", raw=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"})
    assert code == 400
    assert state(viewer)["paused"] is False


def test_control_rejects_cross_origin_and_bad_bodies(viewer):
    pause = {"action": "pause"}
    assert request(viewer, "/api/control", pause,
                   headers={"Origin": "http://unrelated.example"})[0] == 403
    assert request(viewer, "/api/control", pause,
                   headers={"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert request(viewer, "/api/control", raw=b"action=pause",
                   headers={"Content-Type": "text/plain"})[0] == 415
    assert request(viewer, "/api/control", raw=b"{broken",
                   headers={"Content-Type": "application/json"})[0] == 400
    assert request(viewer, "/api/control", raw=b" " * 1025,
                   headers={"Content-Type": "application/json"})[0] == 413
    assert state(viewer)["paused"] is False
    assert request(viewer, "/api/control", pause,
                   headers={"Origin": viewer.url})[0] == 200


def test_port_conflict_has_actionable_error():
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        live = LiveViewer("127.0.0.1", occupied.getsockname()[1])
        try:
            with pytest.raises(RuntimeError, match="--port"):
                live.start()
        finally:
            live.close()


def test_viewer_cli_accepts_live_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path),
        "--viewer_mode", "web", "--ip", "127.0.0.1", "--port", "6010",
        "--viewer_interval", "0.25", "--viewer_max_points", "10000",
        "--viewer_image_size", "320", "--viewer_keep_alive",
    ])
    args = get_args()
    assert args.viewer_mode == "web"
    assert args.ip == "127.0.0.1"
    assert args.port == 6010
    assert args.viewer_interval == 0.25
    assert args.viewer_max_points == 10000
    assert args.viewer_image_size == 320
    assert args.viewer_keep_alive is True


@pytest.mark.parametrize("option,value", [
    ("viewer_interval", "0"), ("viewer_interval", "nan"),
    ("viewer_interval", "inf"), ("viewer_interval", "-1"),
    ("viewer_max_points", "0"), ("viewer_max_points", "-1"),
    ("viewer_image_size", "63"), ("port", "0"), ("port", "65536"),
])
def test_viewer_cli_rejects_invalid_limits(monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(sys, "argv", [
        "slam.py", "-s", str(tmp_path), "-m", str(tmp_path), "--" + option, value,
    ])
    with pytest.raises(SystemExit):
        get_args()


class RecordingViewer:
    def __init__(self):
        self.snapshot = {}
        self.publications = 0

    def publish(self, snapshot):
        self.snapshot.update(snapshot)
        self.publications += 1


class FakeScene:
    def __init__(self, count=20):
        self.lock = threading.Lock()
        self.xyz = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)
        self.f_dc = torch.zeros(count, 1, 3)
        self.opacity = torch.ones(count, 1)
        self.keyframe_pose = torch.eye(4)
        self.width, self.height = 200, 120
        self.K = torch.tensor([[200.0, 0, 99.5], [0, 240.0, 59.5], [0, 0, 1]])
        self.keyframes = [SimpleNamespace(
            get_Rt=lambda: self.keyframe_pose.clone(),
            get_intrinsics=lambda: self.K.clone(),
            width=self.width, height=self.height, info={"source_frame_id": 0},
        )]
        self.join_count = 0
        self.render_calls = []

    @property
    def n_active_gaussians(self):
        return len(self.xyz)

    def join_optimization_thread(self):
        self.join_count += 1

    def render(self, width, height, view_matrix, scaling_modifier, **kwargs):
        assert not torch.is_grad_enabled()
        # render() owns this same non-reentrant lock in the real scene.
        assert self.lock.acquire(blocking=False), "Snapshot must not render under the scene lock"
        self.lock.release()
        self.render_calls.append((width, height, view_matrix.clone(), kwargs))
        image = torch.zeros(3, height, width)
        image[0] = 1.0
        return {"render": image}


def input_image():
    image = torch.zeros(3, 120, 200)
    image[1] = 1.0
    return image


def decode_jpeg(data_url):
    import cv2

    assert data_url.startswith("data:image/jpeg;base64,")
    data = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def test_snapshot_uses_calibrated_render_pose_and_rgb_encoding():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene()
    publisher = SnapshotPublisher(sink, max_points=7, max_size=100, interval=0.5)
    Rt = torch.eye(4)
    Rt[0, 3] = -3.0
    rng_before = torch.random.get_rng_state().clone()
    assert publisher.update(scene, 4, 199, input_image(), {"K": scene.K}, Rt=Rt, force=True)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    snapshot = sink.snapshot
    assert snapshot["frame_id"] == 4
    assert snapshot["total_frames"] == 199
    assert snapshot["num_gaussians"] == 20
    assert snapshot["num_keyframes"] == 1
    assert len(snapshot["points"]) <= 7
    assert len(snapshot["colors"]) == len(snapshot["points"])
    np.testing.assert_allclose(snapshot["colors"], 0.5, atol=1e-4)
    np.testing.assert_allclose(snapshot["camera_pose"], torch.linalg.inv(Rt))
    assert scene.join_count >= 1
    width, height, view_matrix, render_options = scene.render_calls[-1]
    assert (width, height) == (100, 60)
    torch.testing.assert_close(view_matrix, Rt.T)
    expected_K = torch.tensor([[100.0, 0, 49.5], [0, 120.0, 29.5], [0, 0, 1]])
    torch.testing.assert_close(render_options["intrinsics"], expected_K)
    assert render_options.get("detach_gaussians") is True
    assert decode_jpeg(snapshot["input_image"]).shape == (60, 100, 3)
    assert decode_jpeg(snapshot["input_image"])[30, 50, 1] > 240
    rendered = decode_jpeg(snapshot["render_image"])
    assert rendered[30, 50, 2] > 240  # RGB red encoded as JPEG, decoded to BGR.
    assert rendered[30, 50, 0] < 10
    assert publisher.elapsed_seconds >= 0


def test_snapshot_filters_invalid_map_values_and_refreshes_adjusted_trajectory():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene(6)
    scene.xyz[0, 0] = float("nan")
    scene.xyz[1, 2] = float("inf")
    scene.f_dc[2, 0, 0] = float("nan")
    publisher = SnapshotPublisher(sink, max_points=6, max_size=100, interval=0.5)
    publisher.update(scene, 4, 199, input_image(), {"K": scene.K}, force=True)
    assert len(sink.snapshot["points"]) > 0
    assert np.isfinite(sink.snapshot["points"]).all()
    assert np.isfinite(sink.snapshot["colors"]).all()
    np.testing.assert_allclose(sink.snapshot["trajectory"], [[0, 0, 0]])
    scene.keyframe_pose[0, 3] = -5.0
    publisher.update(scene, 5, 199, input_image(), {"K": scene.K}, force=True)
    np.testing.assert_allclose(sink.snapshot["trajectory"], [[5, 0, 0]])


def test_snapshot_without_estimated_pose_labels_latest_keyframe_and_ignores_gt():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene()
    publisher = SnapshotPublisher(sink, max_points=7, max_size=100, interval=0.5)
    publisher.update(scene, 4, 199, input_image(), {"K": scene.K}, Rt=torch.eye(4), force=True)
    assert sink.snapshot["render_image"] is not None
    # info.Rt is GT; the viewer must never use it as an estimated camera pose.
    gt_pose = torch.eye(4)
    gt_pose[0, 3] = 999
    publisher.update(scene, 5, 199, input_image(), {"K": scene.K, "Rt": gt_pose}, force=True)
    assert sink.snapshot["camera_frame_id"] == 0
    assert sink.snapshot["render_frame_id"] == 0
    np.testing.assert_allclose(sink.snapshot["camera_pose"], torch.eye(4))
    assert sink.snapshot["input_image"] is not None
    assert sink.snapshot["render_image"] is not None

    scene.keyframes = []
    calls = len(scene.render_calls)
    publisher.update(scene, 6, 199, input_image(), {"K": scene.K, "Rt": gt_pose}, force=True)
    assert sink.snapshot["camera_pose"] is None
    assert sink.snapshot["render_image"] is None
    assert sink.snapshot["render_frame_id"] is None
    assert len(scene.render_calls) == calls


def test_snapshot_limits_update_work_and_force_refreshes_final_state():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene()
    publisher = SnapshotPublisher(sink, max_points=7, max_size=100, interval=3600)
    assert publisher.update(scene, 0, 199, input_image(), {"K": scene.K})
    calls = (sink.publications, scene.join_count, len(scene.render_calls))
    assert not publisher.update(scene, 1, 199, input_image(), {"K": scene.K})
    assert (sink.publications, scene.join_count, len(scene.render_calls)) == calls
    assert publisher.update(scene, 198, 199, input_image(), {"K": scene.K}, force=True, status="complete")
    assert sink.snapshot["frame_id"] == 198
    assert sink.snapshot["status"] == "complete"


def test_preview_failure_keeps_map_and_stops_retrying_extra_renders():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene()
    publisher = SnapshotPublisher(sink, max_points=7, max_size=100, interval=0.5)
    attempts = []

    def failing_render(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("preview allocation failed")

    scene.render = failing_render
    with pytest.warns(RuntimeWarning, match="preview allocation failed"):
        publisher.update(scene, 4, 199, input_image(), {"K": scene.K}, Rt=torch.eye(4), force=True)
    assert sink.snapshot["points"]
    assert sink.snapshot["input_image"] is not None
    assert sink.snapshot["render_image"] is None
    assert "preview allocation failed" in sink.snapshot["render_error"]
    publisher.update(scene, 5, 199, input_image(), {"K": scene.K}, Rt=torch.eye(4), force=True)
    assert len(attempts) == 1


def test_snapshot_samples_inactive_anchors_without_moving_or_counting_active_alias():
    from visualization.snapshot import SnapshotPublisher

    sink = RecordingViewer()
    scene = FakeScene(3)
    old_points = torch.tensor([[100.0, 0, 0], [101.0, 0, 0]])
    old_anchor = SimpleNamespace(gaussian_params={
        "xyz": {"val": old_points}, "f_dc": {"val": torch.zeros(2, 1, 3)},
    })
    active_anchor = SimpleNamespace(gaussian_params={
        "xyz": {"val": scene.xyz}, "f_dc": {"val": scene.f_dc},
    })
    scene.active_anchor = active_anchor
    scene.anchors = [old_anchor, active_anchor]
    publisher = SnapshotPublisher(sink, max_points=5, max_size=100, interval=0.5)
    publisher.update(scene, 4, 199, input_image(), {"K": scene.K}, force=True)
    assert sink.snapshot["num_gaussians"] == 5
    assert len(sink.snapshot["points"]) == 5
    assert [100.0, 0.0, 0.0] in sink.snapshot["points"]
    assert old_points.device.type == "cpu"


def test_disabled_viewer_does_not_import_or_start_visualization(monkeypatch):
    import builtins
    import slam

    args = SimpleNamespace(viewer_mode="none")
    monkeypatch.setattr(slam, "get_args", lambda: args)
    called = []
    monkeypatch.setattr(slam, "run_slam", lambda value: called.append(value))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("visualization"), "Disabled viewer must stay lazy"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    slam.main()
    assert called == [args]


@pytest.mark.parametrize("mode,keep_alive,failure", [
    ("web", False, None), ("local", True, None), ("server", False, None),
    ("web", False, RuntimeError("SLAM failed")), ("web", False, KeyboardInterrupt()),
])
def test_main_starts_completes_and_closes_viewer(monkeypatch, tmp_path, mode, keep_alive, failure):
    import slam
    import visualization.server

    events = []
    args = SimpleNamespace(viewer_mode=mode, viewer_keep_alive=keep_alive,
                           ip="127.0.0.1", port=6009, model_path=str(tmp_path))

    class Viewer:
        def __init__(self, host, port, open_browser=False):
            events.append(("create", host, port, open_browser))
            self.port = port
            self.url = f"http://{host}:{port}"

        def start(self):
            events.append(("start",))
            return self

        def set_status(self, status, message=""):
            events.append(("status", status))

        def run_until_closed(self):
            events.append(("keep_alive",))
            raise KeyboardInterrupt()

        def close(self):
            events.append(("close",))

    def run(value, viewer):
        assert value is args
        assert isinstance(viewer, Viewer)
        events.append(("run",))
        if failure is not None:
            raise failure

    monkeypatch.setattr(slam, "get_args", lambda: args)
    monkeypatch.setattr(slam, "run_slam", run)
    monkeypatch.setattr(visualization.server, "LiveViewer", Viewer)
    if failure is None:
        slam.main()
    else:
        with pytest.raises(type(failure)):
            slam.main()
    assert events[0] == ("create", "127.0.0.1", 6009, mode == "local")
    assert events[1:3] == [("start",), ("run",)]
    assert events[-1] == ("close",)
    if failure is None:
        assert ("status", "complete") in events
        assert (("keep_alive",) in events) is keep_alive
    elif isinstance(failure, KeyboardInterrupt):
        assert ("status", "stopped") in events
    else:
        assert ("status", "error") in events


def test_slam_publishes_bootstrap_skipped_failed_and_final_frames_before_saving(monkeypatch, tmp_path):
    import slam
    import visualization.snapshot

    events = []
    snapshots = []
    args = SimpleNamespace(
        seed=0, device="cpu", model_path=str(tmp_path), viewer_mode="web",
        viewer_max_points=10, viewer_image_size=100, viewer_interval=0.5,
        num_keyframes_miniba_bootstrap=2, eval_lpips_max_size=512, eval_lpips_device="cpu",
    )

    def pose(frame_id):
        result = torch.eye(4)
        result[0, 3] = -float(frame_id)
        return result

    class Dataset:
        def __init__(self):
            self.frame_id = 0

        def __len__(self):
            return 6

        def get_image_size(self):
            return 120, 200

        def getnext(self):
            frame_id = self.frame_id
            self.frame_id += 1
            return input_image(), {"name": str(frame_id), "is_test": frame_id == 3}

    class Frontend:
        def __init__(self, *args):
            self.matcher = self.triangulator = None
            self.focal_init = torch.tensor([200.0])

        def extract(self, image):
            return None

        def decide_keyframe(self, frame_id, desc):
            return SimpleNamespace(selected=frame_id in (0, 1, 4, 5), reason="test")

        def accept_keyframe(self, *args):
            pass

        def initialize_bootstrap(self, *args, **kwargs):
            return [pose(0), pose(1)], self.focal_init, 0.0

        def initialize_incremental(self, scene, desc, frame_id, *args, **kwargs):
            return None if frame_id == 4 else pose(frame_id)

    class Scene:
        def __init__(self):
            self.keyframes = []
            self.n_active_gaussians = 10

        def join_optimization_thread(self):
            pass

        def enable_inference_mode(self):
            events.append("inference")

        def save(self, *args):
            events.append("save")

    class Backend:
        def __init__(self, *args):
            self.scene_model = Scene()
            self.keyframe_to_frame_id = {}

        def add(self, frame_id):
            index = len(self.scene_model.keyframes)
            self.scene_model.keyframes.append(SimpleNamespace(get_Rt=lambda: pose(frame_id)))
            self.keyframe_to_frame_id[index] = frame_id

        def register_bootstrap(self, packets, *args):
            for packet in packets:
                self.add(packet["frame_id"])
            return SimpleNamespace(timings_ms={})

        def register_incremental_keyframe(self, frame_id, *args):
            self.add(frame_id)
            return SimpleNamespace(timings_ms={})

        def finalize_mapping(self):
            events.append("finalize")
            return SimpleNamespace(timings_ms={})

    class Publisher:
        elapsed_seconds = 0.0

        def __init__(self, *args, **kwargs):
            pass

        def wait_if_paused(self):
            events.append("pause_boundary")

        def update(self, scene, frame_id, total_frames, image, info, Rt=None,
                   force=False, status="running"):
            events.append("snapshot")
            snapshots.append((frame_id, Rt, force, status))
            assert "inference" not in events
            assert "save" not in events

    class Progress:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix_str(self, value):
            pass

    viewer = RecordingViewer()
    viewer.set_status = lambda status, message="": events.append(status)
    monkeypatch.setattr(slam, "create_dataset", lambda args: (Dataset(), "waymo"))
    monkeypatch.setattr(slam, "maybe_seed_from_dataset", lambda *args: None)
    monkeypatch.setattr(slam, "SLAMFrontend", Frontend)
    monkeypatch.setattr(slam, "SLAMBackend", Backend)
    monkeypatch.setattr(slam, "tqdm", Progress)
    monkeypatch.setattr(visualization.snapshot, "SnapshotPublisher", Publisher)
    monkeypatch.setattr(slam, "write_trajectory_stage_outputs", lambda path, stages: {
        key: {"RMSE ATE [m]": None} for key in stages
    })
    monkeypatch.setattr(slam, "compute_trajectory_outputs", lambda records: ([], None, {}))
    monkeypatch.setattr(slam, "compute_render_metrics", lambda *args, **kwargs: ([], {}))
    metrics = {}
    monkeypatch.setattr(slam, "write_outputs", lambda path, values, *args: metrics.update(values))
    slam.run_slam(args, viewer)

    assert [item[0] for item in snapshots] == [0, 1, 2, 3, 4, 5, 5]
    assert snapshots[0][1] is None  # Bootstrap still accumulating.
    assert snapshots[1][2] is True  # Completed bootstrap is immediately visible.
    assert snapshots[2][1] is None  # Skipped ordinary frame has no invented pose.
    torch.testing.assert_close(snapshots[3][1], pose(3))  # Held-out tracked frame.
    assert snapshots[4][1] is None  # Failed localization still publishes input.
    assert snapshots[-1][2:] == (True, "finalizing")
    torch.testing.assert_close(snapshots[-1][1], pose(5))
    assert events.count("pause_boundary") == 6
    last_snapshot = len(events) - 1 - events[::-1].index("snapshot")
    assert events.index("finalize") < last_snapshot < events.index("inference")
    assert last_snapshot < events.index("save")
    assert metrics["num_frames"] == 6
    assert metrics["viewer_mode"] == "web"
    assert "pose_refinement" not in metrics["avg_keyframe_stage_ms"]
    assert "RMSE ATE final refinement [m]" not in metrics
