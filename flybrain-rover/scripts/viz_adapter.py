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
        self.speed = 1.0          # the viewer's slider; replay only, live is real time
        self.paused = False       # the viewer's pause; replay only, live freezes the fly instead
        self.restart = False      # the viewer's next episode; replay only, jumps back to frame 0
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
                    if self.restart:                         # next episode: back to the top of the recording
                        self.restart = False; break
                    while self.paused:                       # hold on the current frame, do not drop it
                        time.sleep(0.05); nxt = time.perf_counter()
                    on_frame(f)
                    nxt += dt / max(0.05, self.speed)        # the slider is 0.25x to 4x
                    time.sleep(max(0.0, nxt - time.perf_counter()))
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



def brighten(path) -> bytes:
    """Serve Ducks's viewer with resting neurons visible.

    His brain panel draws a neuron that is not firing at 35% alpha in a 1.5 pixel square, which on a
    projector or a bright room is invisible: you see the spikes appear out of nothing instead of seeing
    15,000 neurons with some of them lighting up. This rewrites those two constants as the page is served,
    so his file on his branch is never touched and a later pull of his work still just works. If he changes
    that line, the substitution silently does nothing and you get his original.
    """
    html = path.read_text()
    old = "ctx.globalAlpha = .35; ctx.fillRect(px, py, 1.5, 1.5);"
    new = "ctx.globalAlpha = .62; ctx.fillRect(px, py, 2.4, 2.4);"
    if old in html:
        html = html.replace(old, new)
    old_fb = "CLASS_COLOR[BRAIN.cls[i]] || '#6b7385'"
    if old_fb in html:
        html = html.replace(old_fb, "CLASS_COLOR[BRAIN.cls[i]] || '#8f9bb3'")
    if "</body>" in html:
        html = html.replace("</body>", OPERATOR_PANEL + "</body>", 1)
    else:
        html += OPERATOR_PANEL
    return html.encode()


# Two controls the presenter actually uses, appended to Ducks's page as it is served so his file is never
# touched. They POST the same /control endpoint his own buttons use, and the state comes back on every
# frame, so two people watching two browsers see the same thing.
OPERATOR_PANEL = """
<div id="opbar" style="position:fixed;left:14px;bottom:14px;z-index:9999;display:flex;gap:8px;
     font:600 12px ui-monospace,Menlo,monospace">
  <button id="op-lobo" style="padding:10px 14px;border-radius:6px;cursor:pointer;border:1px solid #47403a;
    background:#221d19;color:#ece7e1">LOBOTOMIZE</button>
  <button id="op-stop" style="padding:10px 14px;border-radius:6px;cursor:pointer;border:1px solid #47403a;
    background:#221d19;color:#ece7e1">STOP</button>
  <span id="op-say" style="align-self:center;color:#8e867d;font-weight:400;max-width:34ch"></span>
</div>
<script>
(function(){
  var lobo=false, halt=false, ro=false;
  var bL=document.getElementById('op-lobo'), bS=document.getElementById('op-stop'),
      say=document.getElementById('op-say');
  function paint(){
    bL.textContent = lobo ? 'WAKE IT UP' : 'LOBOTOMIZE';
    bL.style.background = lobo ? '#5a1a14' : '#221d19';
    bL.style.borderColor = lobo ? '#FF3B22' : '#47403a';
    bS.textContent = halt ? 'RESUME' : 'STOP';
    bS.style.background = halt ? '#4a3a10' : '#221d19';
    bS.style.borderColor = halt ? '#FFC24A' : '#47403a';
    say.textContent = halt ? 'wheels stopped, brain still running'
                    : lobo ? 'LC10a, LC4 and LPLC2 disconnected: it is searching for you'
                    : '';
  }
  function post(o){ fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},
                    body:JSON.stringify(o)}).catch(function(){}); }
  bL.onclick=function(){ if(ro)return; lobo=!lobo; paint(); post({lobotomy:lobo}); };
  bS.onclick=function(){ if(ro)return; halt=!halt; paint(); post({halted:halt}); };
  document.addEventListener('keydown',function(e){
    if(e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) return;
    if(e.key==='8'||e.key==='9'){ lobo=(e.key==='8'); paint(); post({lobotomy:lobo}); }
    if(e.key==='0'){ halt=!halt; paint(); post({halted:halt}); }
  });
  // a spectator browser cannot drive the robot; the operator's can
  fetch('/config').then(function(r){return r.json()}).then(function(c){
    if(c && c.read_only){ ro=true; bL.disabled=bS.disabled=true;
      bL.style.opacity=bS.style.opacity=.4; bL.style.cursor=bS.style.cursor='not-allowed';
      say.textContent='watching only'; }
  }).catch(function(){});
  // follow the truth from the feed, so the buttons never lie about the robot
  var es=new EventSource('/events');
  es.onmessage=function(ev){ try{ var f=JSON.parse(ev.data);
    if(typeof f.lobotomized==='boolean' && f.lobotomized!==lobo){ lobo=f.lobotomized; paint(); }
    if(typeof f.halted==='boolean' && f.halted!==halt){ halt=f.halted; paint(); }
  }catch(e){} };
  paint();
})();
</script>
"""


class Adapter:
    def __init__(self, src: Source, viz_neurons: list[int], fov_deg: float = 98.43, v_max: float = V_MAX, w_max_deg: float = W_MAX_DEG):
        self.src, self.viz, self.fov = src, viz_neurons, fov_deg
        self.fov_rad = math.radians(fov_deg)
        self.lock = threading.Condition(); self.version = 0; self.payload = b"{}"
        self.total = 0.0; self.touches = 0; self.locked = False; self.t_first = None; self.track_steps = 0; self.steps_since_lock = 0
        self.lobotomized = False; self.silenced = False; self.halted = False; self._halt_sent = False; self.heat_on = True; self.episode = 0; self.seed = int(src.meta.get("seed", 0)); self.last_t = -1.0
        self.live = not src.replay                                     # live: no world pose in the frames, dead-reckon one
        self.v_max, self.w_max = float(v_max), math.radians(w_max_deg)
        self.fly_speed = 1.0      # the viewer's slider, live only: scales the fly and nothing else
        self.fly_frozen = False   # the viewer's pause, live only: holds the fly still, brain untouched
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
        if self.fly_frozen:                                            # paused: the body holds, the brain does not
            return (x, y), yaw
        yaw = _wrap(yaw - turn * self.w_max * self.fly_speed * dt)                      # turn > 0 = clockwise, yaw is CCW-positive
        v, h = fwd * self.v_max * self.fly_speed, self.arena_half
        x = min(max(x + v * math.cos(yaw) * dt, -h), h); y = min(max(y + v * math.sin(yaw) * dt, -h), h)
        self.pose = [x, y, yaw]
        return [x, y], yaw

    def state(self, f: dict) -> dict:
        t = float(f.get("t", 0.0))
        restarted = t < self.last_t                                    # replay looped / demo restarted
        if restarted:
            self.episode += 1; self.total = 0.0; self.touches = 0; self.locked = False; self.t_first = None; self.track_steps = 0; self.steps_since_lock = 0
        fwd, turn = float(f.get("forward", 0.0)), float(f.get("turn", 0.0))
        if self.live:                                                  # the robot reports no pose, so estimate it
            rxy, yaw = self.dead_reckon(t, fwd, turn, restarted)        # from the command, wheels live or not
        else:
            rxy = f.get("rxy", [0.0, 0.0]); yaw = float(f.get("ryaw", 0.0))
        self.last_t = t
        speed = 0.0 if (self.live and self.fly_frozen) else abs(fwd) * 1.2   # frozen: stop the model animating too
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
                mot = f.get("mot") or []
                c0, c1 = int(max(0, min(len(mot) - 1, (az0 + self.fov_rad / 2) / self.fov_rad * len(mot)))), int(max(0, min(len(mot) - 1, (az1 + self.fov_rad / 2) / self.fov_rad * len(mot)))) if mot else (0, 0)
                walk = min(1.7, 6.0 * (sum(abs(m) for m in mot[c0:c1 + 1]) / max(1, c1 - c0 + 1))) if mot else 0.0
                # A detection only gives a bearing and an apparent size, so the world position is the fly's
                # estimate plus that ray. Both halves drift, so clamp into the arena or people walk through
                # the wall and the viewer draws them in the void outside the floor.
                h = self.arena_half - 0.3
                px = min(max(rxy[0] + dist * math.cos(ang), -h), h)
                py = min(max(rxy[1] + dist * math.sin(ang), -h), h)
                people.append([round(-px, 3), round(py, 3), round(walk, 2), 0.0, 1.0, 0, 1, 0])
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
              "lobotomized": bool(f.get("blind", self.lobotomized)), "halted": bool(f.get("halted", self.halted)), "heat_on": self.heat_on, "see_m": 6.0, "fov_deg": round(self.fov, 1),
              "brain": brain, "stats": {}, "history": []}
        if self.live:
            st["heading_deg"] = round(math.degrees(_wrap(yaw - math.pi / 2)), 1)   # HUD: heading 0 = his +z
            st["yaw_dps"] = round(math.degrees(turn * self.w_max), 1)                # + = clockwise, like his S-Bus driver
            st["pose_estimated"] = True                                              # dead-reckoned from the command, not measured
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
                self._send(200, "text/html; charset=utf-8", brighten(ui_dir / "hunt_gpu.html"))
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
            elif path == "/config":
                # Senthil's viewer (branch sim-mirror-webapp) asks for this on load and, if it 404s, falls
                # back to the local feed AND calls applyReadOnly(), which greys out pause, speed, next and
                # lobotomize. Served locally beside our own brain, the operator is the presenter, so this
                # says read_only false; a public build overrides it with window.__RELAY_CONFIG__ anyway.
                self._send(200, "application/json", json.dumps({
                    "read_only": False, "feed": "local", "relay_url": None, "s1_rejections": 0,
                }, separators=(",", ":")).encode())
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
                        # 8/9, not 4/5: 4 wipes learning, which the untrained brain barely notices. 8 cuts
                        # the visual pathway, which is the change an audience can actually see.
                        _s.socket(_s.AF_INET, _s.SOCK_DGRAM).sendto(b"8" if ad.lobotomized else b"9", ("127.0.0.1", 9600))
                    except OSError:
                        pass
            if "halted" in body:
                ad.halted = bool(body["halted"])
                if ad.live:
                    try:
                        import socket as _s
                        if ad.halted != ad._halt_sent:        # hotkey 0 is a toggle, so only send on a change
                            _s.socket(_s.AF_INET, _s.SOCK_DGRAM).sendto(b"0", ("127.0.0.1", 9600))
                            ad._halt_sent = ad.halted
                    except OSError:
                        pass
                else:
                    ad.src.paused = ad.halted
            if "heat" in body:
                ad.heat_on = bool(body["heat"])
            if "speed" in body:
                try:
                    v = max(0.05, min(8.0, float(body["speed"])))
                except (TypeError, ValueError):
                    v = None
                if v is not None:
                    if ad.live:
                        ad.fly_speed = v          # only the fly: the people come from the camera, not a clock
                    else:
                        ad.src.speed = v          # a recording's fly and people share one clock, so it is playback
            if not ad.live and "paused" in body:
                ad.src.paused = bool(body["paused"])          # replay: one clock, so it holds everything
            if ad.live and "paused" in body:
                ad.fly_frozen = bool(body["paused"])          # live: hold the fly, leave the brain running
            if not ad.live and body.get("next"):
                ad.src.restart = True                         # replay: play the recording again from the start
            if ad.live and body.get("next"):
                # the page's pause silences the brain (hotkey l, a toggle); next episode is a fresh fly: baseline
                # brain (hotkey 1) and the dead-reckoned pose back at the origin
                try:
                    import socket as _s
                    sk = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
                    if body.get("next"):
                        sk.sendto(b"1", ("127.0.0.1", 9600))
                        if ad.halted:
                            sk.sendto(b"0", ("127.0.0.1", 9600))   # a fresh fly is not a stopped one
                        ad.pose = None; ad.episode += 1
                        ad.lobotomized = ad.silenced = ad.halted = ad._halt_sent = False
                except OSError:
                    pass
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
    ap.add_argument("--ui", default="../hunting-fly-mirror/brain/companion_brain/ui",
                    help="directory holding hunt_gpu.html (+ fly.js, vendor/, models/). The copy on the team's main "
                         "branch is only the HTML and renders black; use a worktree of sim-mirror-webapp (newest, has "
                         "camera modes and the spectator config) or s1-fly-brain.")
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
    who = a.who + (" (live; the fly's pose is dead-reckoned from the motor command when the wheels are live, drawn at the origin in dry run)" if ad.live and a.who == WHO_DEFAULT else "")
    serve(ad, brain_json, Path(a.ui), [Path(a.assets)], a.port, who)
    src.run(ad.on_frame)


if __name__ == "__main__":
    main()
