"""Live dashboard: a tiny HTTP server (stdlib only) that serves the single-page UI, a
server-sent-events stream of the brain state at the body rate, the latest camera frame as
JPEG, and the circuit layout (neuron positions / types) once."""
from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

UI_DIR = Path(__file__).parent
HTML = UI_DIR / "index.html"
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css", ".glb": "model/gltf-binary", ".json": "application/json", ".gz": "application/gzip"}


class Dashboard:
    def __init__(self, circuit, cfg, port: int = 8600, open_browser: bool = False):
        self.port = port
        self.cond = threading.Condition()
        self.version = 0
        self.latest = b"{}"
        self.jpeg: bytes | None = None
        self.retina: np.ndarray | None = None      # latest frame from the fly's own eyes (gray uint8 HxW)
        self.retina_seq = 0
        self.retina_at = 0.0
        self.world: dict = {}                       # latest world senses posted by the page
        self.world_at = 0.0
        self.atlas_dir = cfg.path("data.atlas_dir") if cfg.dotted("data.atlas_dir") else None
        self.circuit_json = self._circuit_json(circuit, cfg, self.atlas_dir)
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
                elif self.path.startswith("/atlas/") and dash.atlas_dir is not None:
                    self._static(dash.atlas_dir, self.path[len("/atlas/"):])
                else:
                    self._static()

            def _static(self, base=None, rel=None):
                base = (base or UI_DIR).resolve()
                rel = (rel if rel is not None else self.path).split("?")[0].lstrip("/")
                target = (base / rel).resolve()
                if base not in target.parents or not target.is_file() or target.suffix not in STATIC_TYPES:
                    self._send(404, "text/plain", b"not found")
                    return
                self._send(200, STATIC_TYPES[target.suffix], target.read_bytes())

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n) if n else b""
                if self.path == "/retina" and n >= 4:
                    w, h = int.from_bytes(body[0:2], "little"), int.from_bytes(body[2:4], "little")
                    if w * h == n - 4 and 0 < w <= 512 and 0 < h <= 512:
                        dash.retina = np.frombuffer(body[4:], dtype=np.uint8).reshape(h, w).copy()
                        dash.retina_seq += 1
                        dash.retina_at = time.time()
                elif self.path == "/world":
                    try:
                        dash.world = json.loads(body.decode())
                        dash.world_at = time.time()
                    except ValueError:
                        pass
                self._send(204, "text/plain", b"")

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
                        if payload == b"{}":
                            self.wfile.write(b": waiting for the brain\n\n")   # SSE comment = keep-alive
                        else:
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
    def _circuit_json(circuit, cfg, atlas_dir=None) -> bytes:
        n = circuit.n
        atlas = None
        if atlas_dir is not None and (Path(atlas_dir) / "catalog.json.gz").exists():
            from .atlas_map import build_mapping
            from ..data.annotations import load_annotations
            from ..data.download import raw_paths
            import pandas as pd
            ann = load_annotations(raw_paths(cfg)["annotations"]).set_index("root_id")
            sides = ann.reindex(circuit.root_ids)["side"].fillna("").to_numpy()
            slots, stats = build_mapping(circuit, sides, Path(atlas_dir))
            atlas = {"slot": slots.tolist(), "stats": stats}
            print(f"[atlas] {stats['mapped']:,}/{stats['n']:,} simulated neurons mapped onto {stats['dataset']} "
                  f"({stats['how']}); {stats['skeletons']:,} full skeletons available", flush=True)
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
            "atlas": atlas,
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
