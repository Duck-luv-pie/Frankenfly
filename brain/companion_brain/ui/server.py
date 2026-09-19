"""Live dashboard: a tiny HTTP server (stdlib only) that serves the single-page UI, a
server-sent-events stream of the brain state at the body rate, the latest camera frame as
JPEG, and the circuit layout (neuron positions / types) once."""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

UI_DIR = Path(__file__).parent
HTML = UI_DIR / "index.html"
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".glb": "model/gltf-binary", ".json": "application/json"}


class Dashboard:
    def __init__(self, circuit, cfg, port: int = 8600, open_browser: bool = False):
        self.port = port
        self.cond = threading.Condition()
        self.version = 0
        self.latest = b"{}"
        self.jpeg: bytes | None = None
        self.circuit_json = self._circuit_json(circuit, cfg)
        self.type_ids = self.type_names = None
        if circuit.cell_type is not None:
            self.type_names, self.type_ids = np.unique(circuit.cell_type, return_inverse=True)
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", HTML.read_bytes())
                elif self.path == "/circuit.json":
                    self._send(200, "application/json", dash.circuit_json)
                elif self.path.startswith("/frame.jpg"):
                    if dash.jpeg is None:
                        self._send(204, "image/jpeg", b"")
                    else:
                        self._send(200, "image/jpeg", dash.jpeg, cache=False)
                elif self.path == "/events":
                    self._events()
                else:
                    self._static()

            def _static(self):
                rel = self.path.split("?")[0].lstrip("/")
                target = (UI_DIR / rel).resolve()
                if UI_DIR.resolve() not in target.parents or not target.is_file() or target.suffix not in STATIC_TYPES:
                    self._send(404, "text/plain", b"not found")
                    return
                self._send(200, STATIC_TYPES[target.suffix], target.read_bytes())

            def _send(self, code, ctype, body, cache=True):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        with dash.cond:
                            dash.cond.wait_for(lambda: dash.version != seen, timeout=5.0)
                            seen = dash.version
                            payload = dash.latest
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        print(f"[ui] dashboard at http://localhost:{port}", flush=True)
        if open_browser:
            webbrowser.open(f"http://localhost:{port}")

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _circuit_json(circuit, cfg) -> bytes:
        n = circuit.n
        pos = circuit.pos if circuit.pos is not None else np.zeros((n, 3), np.float32)
        # frontal view: x lateral, y dorsal-ventral. Normalize to 0..1 using neurons with a position.
        has = (pos[:, 0] > 0) & (pos[:, 1] > 0)
        lo = pos[has].min(0) if has.any() else np.zeros(3)
        hi = pos[has].max(0) if has.any() else np.ones(3)
        nx = np.where(has, (pos[:, 0] - lo[0]) / max(hi[0] - lo[0], 1), -1)
        ny = np.where(has, (pos[:, 1] - lo[1]) / max(hi[1] - lo[1], 1), -1)
        role = np.zeros(n, np.int8)
        for g in cfg.inputs:
            role[circuit.idx(g)] = 1
        for g in cfg.readouts:
            role[circuit.idx(g)] = 2
        group_of = [""] * n
        for g in list(cfg.inputs) + list(cfg.readouts):
            for i in circuit.idx(g):
                group_of[i] = g
        return json.dumps({
            "n": int(n),
            "x": np.round(nx, 4).tolist(),
            "y": np.round(ny, 4).tolist(),
            "role": role.tolist(),
            "group": group_of,
            "type": (circuit.cell_type.tolist() if circuit.cell_type is not None else []),
            "inputs": list(cfg.inputs), "readouts": list(cfg.readouts),
            "behaviors": list(cfg.decode.behaviors.keys()),
            "connections": int(circuit.m),
        }, separators=(",", ":")).encode()

    def top_types(self, counts: np.ndarray, k: int = 10) -> list:
        if self.type_ids is None:
            return []
        per_type = np.bincount(self.type_ids, weights=counts, minlength=len(self.type_names))
        order = np.argsort(-per_type)[:k]
        return [[str(self.type_names[i]), int(per_type[i])] for i in order if per_type[i] > 0]

    def publish(self, state: dict, frame=None) -> None:
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.jpeg = buf.tobytes()
        payload = json.dumps(state, separators=(",", ":"), default=_default).encode()
        with self.cond:
            self.latest = payload
            self.version += 1
            self.cond.notify_all()


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
