"""The spiking hunter in the real world: a camera (and the PIR, if the body is on) instead of the arena,
the RoboMaster S1 as the legs. `companion hunt-brain --real --camera SRC --s1 PORT`.

Each tick (hunt.tick_s): newest frame -> senses/people.py -> the arena's observation layout
[heat 2, vision bins x 2, body 4] -> the spiking hunter -> drive [forward, turn] -> the S1 (as the
fly's would-be speed and yaw rate, which the driver converts to sticks). A small page shows the
frame with the detections, the columns, the drive and the neurons at http://localhost:8601.
"""
from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

from ..senses.camera import CameraStream
from ..senses.people import PeopleSense
from ..ui.server import STATIC_TYPES, UI_DIR
from .arena import BatchArena

HTML = UI_DIR / "hunt_real.html"


class RealHunt:
    def __init__(self, cfg, spiking, camera: str, cam_fov_deg: float = 62.0, body=None, pir=None):
        self.cfg, self.spiking, self.body, self.pir = cfg, spiking, body, pir
        g = {**dict(cfg.hunt_gpu), "obs_noise": 0.0}
        self.arena = BatchArena(cfg.hunt, g, 1, "cpu", seed=0)      # only for its sensor geometry and observation layout
        spiking.bind_arena(self.arena)
        self.cam = CameraStream(camera, 320, 240)
        self.sense = PeopleSense(self.arena.spec.bins, math.degrees(self.arena.fov), cam_fov_deg, tick_s=self.arena.dt)
        self.vmax, self.wmax_dps = float(self.arena.vmax), math.degrees(float(self.arena.wmax))
        self.prev_drive = np.zeros(2, dtype=np.float32)
        self.speed = 0.0
        self.t0 = time.time()
        self.t = 0.0
        self.episode_s = float(self.arena.episode_s)
        self.cell_type = spiking.hc.circuit.cell_type
        self.paused = False
        self.rover_on = body is not None
        self.lock = threading.Lock()
        self.last: dict = {}
        self.frame_jpg = b""
        self.heat_on = True

    def tick(self) -> dict:
        gray = self.cam.read()
        fly_moving = abs(self.speed) > 0.05
        vis, boxes = self.sense.observe(gray, fly_moving, color=self.cam.preview if gray is not None else None)
        heat = np.zeros(2, dtype=np.float32)
        if self.heat_on and self.pir is not None:
            self.pir.poll()
            if int(getattr(self.pir, "pir", 0)):
                heat[:] = min(1.0, float(self.arena.heat_gain))             # the HC-SR501: one bit into both hot cells (as in the arena's pir model)
        body = np.array([self.speed / self.vmax, self.prev_drive[0], self.prev_drive[1], (self.t % self.episode_s) / self.episode_s], dtype=np.float32)
        obs = torch.from_numpy(np.concatenate([heat, vis.reshape(-1), body]))[None]
        drive = self.spiking(obs)
        f, tu = float(drive[0, 0]), float(drive[0, 1])
        # the fly's would-be motion, as the arena would integrate it (the S1 driver turns these into sticks)
        target = self.vmax * max(0.0, f)
        self.speed += (target - self.speed) * min(1.0, self.arena.dt / 0.3)
        yaw_dps = -self.wmax_dps * float(self.arena.turn_sign) * tu * -1.0   # arena: turn +ve = right; yaw_dps +ve = clockwise
        self.prev_drive[:] = (f, tu)
        self.t += self.arena.dt
        if self.body is not None and self.rover_on:
            self.body.send({"t": int((time.time() - self.t0) * 1000), "state": "hunt",
                            "motor": {"forward": max(0.0, f), "backward": 0.0, "turn": tu, "yaw_dps": yaw_dps, "speed_mps": self.speed}})
        spk = self.spiking.spikes
        lit = spk.nonzero()[0]
        types: dict[str, int] = {}
        for i in lit:
            t = str(self.cell_type[i]); types[t] = types.get(t, 0) + int(spk[i])
        rates = self.spiking.runner.rates(300)
        st = {"t": round(self.t, 2), "heat": [round(float(heat[0]), 3), round(float(heat[1]), 3)],
              "vis": [round(float(v), 2) for v in vis[:, 0]], "motion": [round(float(v), 2) for v in vis[:, 1]],
              "boxes": [[int(v) for v in b] for b in boxes], "drive": [round(f, 3), round(tu, 3)],
              "speed_mps": round(self.speed, 2), "yaw_dps": round(yaw_dps, 1), "fov_deg": self.sense.cam_fov,
              "rover": (self.body.status() if self.body is not None and hasattr(self.body, "status") else None), "rover_on": self.rover_on,
              "paused": self.paused, "heat_on": self.heat_on, "camera_ok": gray is not None,
              "brain": {"n_spikes": int(spk.sum()), "top": sorted(types.items(), key=lambda kv: -kv[1])[:8],
                        "rates": {g: [round(rates[g]["left"], 1), round(rates[g]["right"], 1)] for g in ("DNa02", "DNa01", "DN_all", "MDN", "DNp09")}}}
        if self.cam.preview is not None:
            img = self.cam.preview.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (80, 220, 120), 2)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                self.frame_jpg = buf.tobytes()
        with self.lock:
            self.last = st
        return st

    def close(self) -> None:
        self.cam.close()
        if self.body is not None:
            self.body.close()


def serve(cfg, spiking, camera: str, cam_fov_deg: float = 62.0, body=None, pir=None, port: int = 8601, open_browser: bool = False) -> None:
    hunt = RealHunt(cfg, spiking, camera, cam_fov_deg, body=body, pir=pir)
    cond = threading.Condition()
    box = {"version": 0, "payload": b"{}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/hunt_real.html"):
                self._send(200, "text/html; charset=utf-8", HTML.read_bytes())
            elif path == "/events":
                self._events()
            elif path == "/control":
                self._send(204, "text/plain", b"")
            elif path == "/frame.jpg":
                self._send(200, "image/jpeg", hunt.frame_jpg or b"")
            else:
                rel = path.lstrip("/")
                target = (UI_DIR / rel).resolve()
                if UI_DIR.resolve() not in target.parents or not target.is_file() or target.suffix not in STATIC_TYPES:
                    self._send(404, "text/plain", b"not found")
                else:
                    self._send(200, STATIC_TYPES[target.suffix], target.read_bytes())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            if "paused" in req:
                hunt.paused = bool(req["paused"])
            if "rover" in req:
                hunt.rover_on = bool(req["rover"]) and body is not None
                print(f"[hunt-real] rover {'on' if hunt.rover_on else 'off (the fly only watches)'}", flush=True)
            if "heat" in req:
                hunt.heat_on = bool(req["heat"])
            self._send(204, "text/plain", b"")

        def _send(self, code, ctype, payload):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: box["version"] != seen, timeout=1.0)
                        seen, payload = box["version"], box["payload"]
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"[hunt-real] the spiking fly ({spiking.hc.n:,} neurons) hunting through {camera} at {url}; rover {'on' if hunt.rover_on else 'not attached'} (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    dt = hunt.arena.dt
    nxt = time.perf_counter()
    try:
        while True:
            if not hunt.paused:
                st = hunt.tick()
                payload = json.dumps(st, separators=(",", ":")).encode()
                with cond:
                    box["payload"] = payload
                    box["version"] += 1
                    cond.notify_all()
            nxt += dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 1.0:
                nxt = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        hunt.close()
        server.shutdown()
