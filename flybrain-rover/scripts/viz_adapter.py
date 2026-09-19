"""
viz_adapter.py -- drive Ducks's 3-D viewer (hunting-fly/brain/companion_brain/ui/hunt_gpu.html) with OUR brain.

    python scripts/viz_adapter.py --ui ../hunting-fly/brain/companion_brain/ui --replay replay/episode_7_trained.json
    python scripts/viz_adapter.py --ui ../hunting-fly/brain/companion_brain/ui --live ws://localhost:8765   # from demo.py

Serves his page and speaks his protocol: GET / (the page), /events (server-sent events, one JSON state per
tick), /brain.json (neuron map: normalised x,y per neuron, class, type, hunter group), /who (label),
static files from --ui (and --assets for the .glb models), POST /control (accepted; "lobotomy" is echoed
into the state so the HUD follows). Two sources:
  --replay  one of our replay JSONs (50 Hz, full rover/human poses) played back in a loop;
  --live    the demo bridge's websocket feed (no world pose on the robot: the pose is dead-reckoned from the
            motor command, forward * --v-max and turn * --w-max integrated the way env/arena.py steps the rover,
            starting at the origin facing +y and clamped to the arena walls; people are placed from the detector
            boxes by bearing and apparent width relative to that estimate; the state says pose_estimated: true).
Coordinate mapping: his fly = [x, z, heading, speed] with heading 0 = +z and x = sin(heading); ours is
x right, y up, yaw CCW from +x, so his x = -our x, his z = our y, his heading = yaw - pi/2 (a rotation, not a
reflection: a right turn is a right turn in both, and a person on the fly's right is drawn on its right).
Neuron map: data/neuron_positions.json (neuPrint somaLocation of our 15,000 neurons), frontal view
(x = medial-lateral, y = dorsal-ventral); spikes arrive as indices into the bridge's 512-neuron sample.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATIC_TYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".glb": "model/gltf-binary",
                ".gltf": "model/gltf+json", ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
                ".html": "text/html; charset=utf-8", ".svg": "image/svg+xml", ".ico": "image/x-icon", ".bin": "application/octet-stream"}
GROUP_OF = {"LC10a": 1, "LC11": 1, "LC12": 1, "LC15": 1, "LC4": 1, "LPLC2": 1, "THERMO": 2, "DNa02": 3, "DNa01": 3, "DNp09": 3, "GF": 3,
            "PAM": 4, "PPL1": 4, "KC": 5, "MBON": 5}
CLASS_OF = {"visual_projection": 1, "descending_neuron": 2, "cb_intrinsic": 3, "ol_intrinsic": 4, "cb_sensory": 5, "vnc_intrinsic": 6}
V_MAX, W_MAX_DEG = 0.7, 90.0          # the bridge's defaults: m/s at forward = 1, deg/s at turn = 1 (live dead reckoning)
LIVE_ARENA_L = 12.0                   # live feed has no meta.arena_L: top of env/arena.py's size range, inside the viewer's rails
WHO_DEFAULT = "the male-CNS spiking fly (flybrain-rover): 15,000 neurons, live"


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def build_brain_json(brain_npz: str, positions_json: str, viz_neurons: list[int] | None) -> tuple[bytes, dict]:
    """His /brain.json: n, x, y (normalised, -1 = no position), cls, type, group. Also returns a map from
    'viz index' (the bridge's 512-sample index) to neuron index."""
    import numpy as np
    d = np.load(brain_npz, allow_pickle=False)
    ids = [int(v) for v in d["ids"]]; types = [str(t) for t in d["types"]]; sup = [str(s) for s in d["superclass"]]
    pos = json.load(open(positions_json)) if os.path.exists(positions_json) else {}
    xyz = np.full((len(ids), 3), np.nan)
    for i, b in enumerate(ids):
        p = pos.get(str(b), {}).get("pos")
        if p:
            xyz[i] = p[:3]
    has = ~np.isnan(xyz[:, 0])
    lo, hi = np.nanpercentile(xyz[has], 1, axis=0), np.nanpercentile(xyz[has], 99, axis=0)
    x = np.where(has, np.clip((xyz[:, 0] - lo[0]) / max(hi[0] - lo[0], 1), 0, 1), -1)
    y = np.where(has, np.clip((xyz[:, 1] - lo[1]) / max(hi[1] - lo[1], 1), 0, 1), -1)
    group = []
    for t in types:
        g = 0
        for k, v in GROUP_OF.items():
            if t.startswith(k) or (k == "THERMO" and (t.startswith("TRN_VP3") or ("VP3" in t and t.endswith("PN")))) or (k == "GF" and t == "DNp01"):
                g = v; break
        group.append(g)
    payload = json.dumps({"n": len(ids), "x": np.round(x, 4).tolist(), "y": np.round(y, 4).tolist(),
                          "cls": [CLASS_OF.get(s, 0) for s in sup], "type": types, "group": group},
                         separators=(",", ":")).encode()
    return payload, {"viz": viz_neurons or []}


class Source:
    """Yields our frames (dicts with the replay/FORMAT.md fields) at their own rate."""
    def __init__(self, replay: str | None, live: str | None, loop: bool = True):
        self.replay, self.live, self.loop = replay, live, loop
        self.meta = {}
        if replay:
            d = json.load(open(replay)); self.frames = d["frames"]; self.meta = d.get("meta", {})
        self.latest = None

    def run(self, on_frame):
        if self.replay:
            dt = float(self.meta.get("dt", 0.02))
            while True:
                nxt = time.perf_counter()
                for f in self.frames:
                    on_frame(f)
                    nxt += dt; time.sleep(max(0.0, nxt - time.perf_counter()))
                if not self.loop:
                    return
        else:
            import asyncio
            import websockets

            async def main():
                while True:
                    try:
                        async with websockets.connect(self.live) as ws:
                            async for msg in ws:
                                on_frame(json.loads(msg))
                    except Exception as e:  # noqa: BLE001
                        print("live feed:", type(e).__name__, "retrying", flush=True); await asyncio.sleep(2)
            asyncio.run(main())


class Adapter:
    def __init__(self, src: Source, viz_neurons: list[int], fov_deg: float = 98.43, v_max: float = V_MAX, w_max_deg: float = W_MAX_DEG):
        self.src, self.viz, self.fov = src, viz_neurons, fov_deg
        self.lock = threading.Condition(); self.version = 0; self.payload = b"{}"
        self.total = 0.0; self.touches = 0; self.locked = False; self.t_first = None; self.track_steps = 0; self.steps_since_lock = 0
        self.lobotomized = False; self.heat_on = True; self.episode = 0; self.seed = int(src.meta.get("seed", 0)); self.last_t = -1.0
        self.live = not src.replay                                     # live: no world pose in the frames, dead-reckon one
        self.v_max, self.w_max = float(v_max), math.radians(w_max_deg)
        self.arena_half = float(src.meta.get("arena_L", LIVE_ARENA_L)) / 2
        self.pose = None                                               # [x, y, yaw] in our frame, live mode only

    def dead_reckon(self, t: float, fwd: float, turn: float, restart: bool) -> tuple[list[float], float]:
        """Integrate the motor command the way env/arena.py steps the rover: yaw -= turn * w_max * dt, then
        x += v cos(yaw) dt with v = forward * v_max, walls clamp (no wrap). dt = frame clock delta clamped to
        [0, 0.1] s. Restart (clock went backwards) or first frame: origin, facing +y, no motion this frame."""
        if restart or self.pose is None:
            self.pose = [0.0, 0.0, math.pi / 2]; dt = 0.0
        else:
            dt = min(max(t - self.last_t, 0.0), 0.1)
        fwd, turn = max(-1.0, min(1.0, fwd)), max(-1.0, min(1.0, turn))
        x, y, yaw = self.pose
        yaw = _wrap(yaw - turn * self.w_max * dt)                      # turn > 0 = clockwise, yaw is CCW-positive
        v, h = fwd * self.v_max, self.arena_half
        x = min(max(x + v * math.cos(yaw) * dt, -h), h); y = min(max(y + v * math.sin(yaw) * dt, -h), h)
        self.pose = [x, y, yaw]
        return [x, y], yaw

    def state(self, f: dict) -> dict:
        t = float(f.get("t", 0.0))
        restarted = t < self.last_t                                    # replay looped / demo restarted
        if restarted:
            self.episode += 1; self.total = 0.0; self.touches = 0; self.locked = False; self.t_first = None; self.track_steps = 0; self.steps_since_lock = 0
        fwd, turn = float(f.get("forward", 0.0)), float(f.get("turn", 0.0))
        if self.live:                                                  # no world pose on the robot: estimate it
            rxy, yaw = self.dead_reckon(t, fwd, turn, restarted)
        else:
            rxy = f.get("rxy", [0.0, 0.0]); yaw = float(f.get("ryaw", 0.0))
        self.last_t = t
        speed = abs(fwd) * 1.2
        fly = [round(-rxy[0], 3), round(rxy[1], 3), round(_wrap(yaw - math.pi / 2), 4), round(speed, 3)]
        people = []
        if "hxy" in f:                                                  # replay: true poses
            for (hx, hy) in f["hxy"]:
                people.append([round(-hx, 3), round(hy, 3), 1.0, 0.0, 1.0, 0, 1, 0])
        else:                                                           # live: place people from detector boxes
            for x0, y0, x1, y1 in f.get("boxes", []):
                az0 = math.atan((x0 - 160) / 138.1); az1 = math.atan((x1 - 160) / 138.1)
                width = max(az1 - az0, 1e-3); dist = min(6.0, 0.5 / width)      # 0.5 m shoulders
                bearing = 0.5 * (az0 + az1)                                          # + = right
                ang = yaw - bearing
                people.append([round(-(rxy[0] + dist * math.cos(ang)), 3), round(rxy[1] + dist * math.sin(ang), 3), 0.0, 0.0, 1.0, 0, 1, 0])
        reward = float(f.get("reward", 0.0)); self.total += reward
        touching = reward >= 0.5
        if touching:
            if not self.locked:
                self.locked = True; self.t_first = t; self.touches += 1
            self.track_steps += 1
        if self.locked:
            self.steps_since_lock += 1
        rates = f.get("rates", {})
        r = lambda k: float(rates.get(k, 0.0))
        lit = [[self.viz[i], 1] for i in f.get("spikes", []) if i < len(self.viz)]
        top = sorted(((k, round(v)) for k, v in rates.items() if v > 1 and k not in ("KC", "MBON", "DN_ALL")), key=lambda kv: -kv[1])[:8]
        brain = {"lit": lit, "n_spikes": len(lit), "top": top,
                 "rates": {"DNa02": [round(r("DNa02_L"), 1), round(r("DNa02_R"), 1)], "DNa01": [round(r("DNa01_L"), 1), round(r("DNa01_R"), 1)],
                           "DN_all": [round(r("DN_ALL"), 2), round(r("DN_ALL"), 2)], "MDN": [0.0, 0.0], "DNp09": [round(r("DNp09"), 1), round(r("DNp09"), 1)]},
                 "motor": {"forward": round(fwd, 2), "turn": round(turn, 2)}}
        heat = f.get("heat", [0.0, 0.0])
        st = {"t": round(t, 2), "seed": self.seed, "episode": self.episode, "fly": fly, "people": people,
              "humans": "people walking about (male-CNS spiking fly, flybrain-rover)", "target": 0, "locked": self.locked, "touches": self.touches,
              "t_first": round(self.t_first, 2) if self.t_first is not None else None,
              "track": round(self.track_steps / max(1, self.steps_since_lock), 3) if self.locked else None,
              "heat": [round(float(heat[0]), 3), round(float(heat[1]), 3)] if self.heat_on else [0.0, 0.0],
              "vis": [round(float(v), 2) for v in f.get("pres", [0.0] * 24)], "motion": [round(abs(float(v)), 2) for v in f.get("mot", [0.0] * 24)],
              "drive": [round(fwd, 3), round(turn, 3)], "reward": round(reward, 3), "total": round(self.total, 2), "done": False,
              "lobotomized": bool(f.get("lobotomy", self.lobotomized)), "heat_on": self.heat_on, "see_m": 6.0, "fov_deg": round(self.fov, 1),
              "brain": brain, "stats": {}, "history": []}
        if self.live:
            st["pose_estimated"] = True                                # dead-reckoned from the motor command, not measured
        return st

    def on_frame(self, f: dict):
        st = self.state(f)
        payload = json.dumps(st, separators=(",", ":")).encode()
        with self.lock:
            self.payload = payload; self.version += 1; self.lock.notify_all()


def serve(ad: Adapter, brain_json: bytes, ui_dir: Path, asset_dirs: list[Path], port: int, who: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/hunt_gpu.html"):
                self._send(200, "text/html; charset=utf-8", (ui_dir / "hunt_gpu.html").read_bytes())
            elif path == "/events":
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-store"); self.end_headers()
                seen = -1
                try:
                    while True:
                        with ad.lock:
                            ad.lock.wait_for(lambda: ad.version != seen, timeout=5.0)
                            seen, payload = ad.version, ad.payload
                        self.wfile.write(b"data: " + payload + b"\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            elif path == "/who":
                self._send(200, "text/plain; charset=utf-8", who.encode())
            elif path == "/brain.json":
                self._send(200, "application/json", brain_json)
            else:
                rel = path.lstrip("/")
                for base in [ui_dir, *asset_dirs]:
                    tgt = (base / rel).resolve()
                    if base.resolve() in tgt.parents and tgt.is_file() and tgt.suffix in STATIC_TYPES:
                        return self._send(200, STATIC_TYPES[tgt.suffix], tgt.read_bytes())
                if rel == "fly.js":                                     # his fly model module, if his push is still missing it
                    return self._send(200, "text/javascript", FLY_STUB.encode())
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            if "lobotomy" in body:
                ad.lobotomized = bool(body["lobotomy"])
                if ad.live:                                   # press the demo's own hotkey; replay is a recording
                    try:
                        import socket as _s
                        _s.socket(_s.AF_INET, _s.SOCK_DGRAM).sendto(b"4" if ad.lobotomized else b"5", ("127.0.0.1", 9600))
                    except OSError:
                        pass
            if "heat" in body:
                ad.heat_on = bool(body["heat"])
            self._send(204, "text/plain", b"")

    srv = ThreadingHTTPServer(("0.0.0.0", port), H); srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[viz-adapter] serving Ducks's viewer with our brain at http://localhost:{port}  (ui {ui_dir})", flush=True)


FLY_STUB = """// stand-in for fly.js until Ducks pushes the real module. Same interface: createFly() -> { root, setPose },
// demonstrationPose(fly, name, t, fps) -> pose. Loads the Body Lab fly rig if the adapter serves it, else a simple body.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
export function createFly() {
  const root = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(.8, 2.2, 4, 12), new THREE.MeshStandardMaterial({ color: '#6b4a2b', roughness: .6 }));
  body.rotation.z = Math.PI / 2; body.position.y = 1.2; body.castShadow = true; root.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(.7, 16, 16), new THREE.MeshStandardMaterial({ color: '#8a2b2b' })); head.position.set(2, 1.4, 0); root.add(head);
  new GLTFLoader().load('/drosophila-male-rig.glb', g => { root.remove(body); root.remove(head); g.scene.traverse(o => { if (o.isMesh) o.castShadow = true; }); root.add(g.scene); }, undefined, () => {});
  return { root, setPose() {} };
}
export function demonstrationPose() { return {}; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", default="../hunting-fly/brain/companion_brain/ui", help="directory holding hunt_gpu.html (+ fly.js, glb)")
    ap.add_argument("--assets", default="replay/assets", help="extra static dir for .glb models")
    ap.add_argument("--replay", default=None); ap.add_argument("--live", default=None)
    ap.add_argument("--brain", default="data/brain.npz"); ap.add_argument("--positions", default="data/neuron_positions.json")
    ap.add_argument("--port", type=int, default=8601)
    ap.add_argument("--who", default=WHO_DEFAULT)
    ap.add_argument("--no-loop", action="store_true")
    ap.add_argument("--v-max", type=float, default=V_MAX, help="m/s at forward = 1 (live pose dead reckoning; the bridge's default)")
    ap.add_argument("--w-max", type=float, default=W_MAX_DEG, help="deg/s at turn = 1 (live pose dead reckoning; the bridge's default)")
    a = ap.parse_args()
    if not (a.replay or a.live):
        sys.exit("give --replay file.json or --live ws://host:port")
    src = Source(a.replay, a.live, loop=not a.no_loop)
    viz = src.meta.get("viz_neurons")
    if viz is None:                                                    # live feed: same fixed sample as the bridge
        import torch
        import numpy as np
        N = int(np.load(a.brain, allow_pickle=False)["N"])
        viz = torch.randperm(N, generator=torch.Generator().manual_seed(0))[:512].tolist()
    brain_json, _ = build_brain_json(a.brain, a.positions, viz)
    ad = Adapter(src, viz, v_max=a.v_max, w_max_deg=a.w_max)
    who = a.who + (" (pose dead-reckoned from the motor command)" if ad.live and a.who == WHO_DEFAULT else "")
    serve(ad, brain_json, Path(a.ui), [Path(a.assets)], a.port, who)
    src.run(ad.on_frame)


if __name__ == "__main__":
    main()
