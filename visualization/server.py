"""A small, offline HTTP viewer; this module never touches CUDA or scene state.

The SLAM thread publishes plain JSON snapshots at safe frame boundaries. Request
threads only serve immutable bytes and set a pause event, so a browser cannot
race the optimizer or invoke rendering from another thread.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


class LiveViewer:
    def __init__(self, host="127.0.0.1", port=6009, open_browser=False):
        self.host = host
        self.port = int(port)
        self.open_browser = open_browser
        self._lock = threading.RLock()
        self._resume = threading.Event()
        self._resume.set()
        self._closed = threading.Event()
        self._httpd = None
        self._thread = None
        self._html = Path(__file__).with_name("index.html").read_bytes()
        self._state = {
            "frame_id": -1, "total_frames": 0, "num_keyframes": 0,
            "num_gaussians": 0, "status": "initializing", "message": "",
            "points": [], "colors": [], "trajectory": [],
            "camera_pose": None, "input_image": None, "render_image": None,
            "paused": False, "revision": 0,
        }
        self._payload = self._encode(self._state)

    @staticmethod
    def _encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")

    @property
    def url(self):
        host = "127.0.0.1" if self.host in ("", "0.0.0.0", "::") else self.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    @property
    def paused(self):
        return not self._resume.is_set()

    def start(self):
        if self._httpd is not None:
            return self
        if self._closed.is_set():
            raise RuntimeError("A closed LiveViewer cannot be restarted")
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _reply(self, code, body, content_type="application/json; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                path = urlsplit(self.path).path
                if path == "/":
                    self._reply(200, viewer._html, "text/html; charset=utf-8")
                elif path == "/api/state":
                    with viewer._lock:
                        payload = viewer._payload
                    self._reply(200, payload)
                elif path == "/favicon.ico":
                    self._reply(204, b"")
                else:
                    self._reply(404, b'{"error":"Not found"}')

            def do_POST(self):
                # Consume the small, bounded control body before rejecting a
                # request. Closing with unread bytes can reset the socket on
                # Windows before the client receives our HTTP error response.
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if not 0 < length <= 1024:
                    self._reply(413, b'{"error":"Invalid control body length"}')
                    return
                try:
                    self.connection.settimeout(2)
                    raw_body = self.rfile.read(length)
                except OSError:
                    self._reply(400, b'{"error":"Could not read control body"}')
                    return
                if urlsplit(self.path).path != "/api/control":
                    self._reply(404, b'{"error":"Not found"}')
                    return
                origin = self.headers.get("Origin")
                if origin:
                    try:
                        source = urlsplit(origin)
                        same_origin = source.scheme == "http" and source.netloc.lower() == self.headers.get("Host", "").lower()
                    except ValueError:
                        same_origin = False
                    if not same_origin:
                        self._reply(403, b'{"error":"Cross-origin control is disabled"}')
                        return
                if self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none"):
                    self._reply(403, b'{"error":"Cross-site control is disabled"}')
                    return
                if self.headers.get_content_type() != "application/json":
                    self._reply(415, b'{"error":"Expected application/json"}')
                    return
                try:
                    request = json.loads(raw_body)
                except (ValueError, UnicodeError, OSError):
                    self._reply(400, b'{"error":"Invalid JSON"}')
                    return
                if not isinstance(request, dict) or request.get("action") not in ("pause", "resume"):
                    self._reply(400, b'{"error":"Expected pause or resume"}')
                    return
                with viewer._lock:
                    if viewer._closed.is_set() or viewer._state["status"] in ("finalizing", "complete", "error", "stopped"):
                        self._reply(409, b'{"error":"SLAM is no longer running"}')
                        return
                    if request["action"] == "pause":
                        viewer._resume.clear()
                    else:
                        viewer._resume.set()
                    viewer._update_locked({})
                self._reply(200, b'{"ok":true}')

        server_type = ThreadingHTTPServer
        if ":" in self.host:
            class IPv6Server(ThreadingHTTPServer):
                address_family = socket.AF_INET6
            server_type = IPv6Server
        try:
            self._httpd = server_type((self.host, self.port), Handler)
        except OSError as exc:
            raise RuntimeError(f"Cannot start SLAM viewer on {self.host}:{self.port}: {exc}. Choose another --port.") from exc
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.1}, daemon=True,
                                        name="slam-viewer-http")
        self._thread.start()
        if self.open_browser:
            try:
                webbrowser.open(self.url)
            except webbrowser.Error:
                pass
        return self

    def _update_locked(self, update):
        # Round-trip caller-owned objects: mutation after publication must not
        # change a future snapshot unexpectedly.
        merged = dict(self._state)
        merged.update(update)
        merged["paused"] = self.paused
        merged["revision"] = self._state["revision"] + 1
        payload = self._encode(merged)
        self._state = json.loads(payload)
        self._payload = payload

    def publish(self, snapshot):
        with self._lock:
            self._update_locked(snapshot)

    def set_status(self, status, message=""):
        with self._lock:
            if status in ("finalizing", "complete", "error", "stopped"):
                self._resume.set()
            self._update_locked({"status": status, "message": message})

    def wait_if_paused(self):
        while not self._closed.is_set() and not self._resume.wait(timeout=0.2):
            pass

    def run_until_closed(self):
        """Keep the final view available; Ctrl+C is handled by the caller."""
        while not self._closed.wait(timeout=0.2):
            pass

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self._resume.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
