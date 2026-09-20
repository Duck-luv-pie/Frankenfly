"""A small live view of the GPU fly hunting: the trained network runs in one room of the batch arena
at real time and a page (ui/hunt_gpu.html, the dashboard's 3-D room with its walking humans) shows
the chase, the fly's senses (heat left / right, the retinal columns) and its drive, over server-sent
events. `companion hunt-gpu --watch --load data/cache/hunter_gpu.pt`."""
from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from ..ui.server import STATIC_TYPES, UI_DIR
from .arena import BatchArena, scripted_drive, wrap
from .brain import EvaderNet, HunterNet, build
from .mirror import Mirror
from .ppo import summarize

HTML = UI_DIR / "hunt_gpu.html"


class Watch:
    """One room, one fly, stepped at the arena's tick rate; `state()` is what the page draws."""

    def __init__(self, cfg, net: HunterNet | None, seed: int = 0, speed: float = 1.0, device: str = "cpu", humans: EvaderNet | None = None, spiking=None, body=None,
                 v_max=None, w_max=None, sign_yaw=None):
        self.cfg, self.speed = cfg, float(speed)
        self.body = body                                         # a real body (the RoboMaster S1) mirroring the fly's drive
        # the Mirror is the one gate between the fly's drive and the S1's sticks (scaling, clamping, caps).
        self.mirror = Mirror(body, v_max=v_max, w_max=w_max, sign_yaw=sign_yaw) if body is not None else None
        self.estop = False                                       # the operator's emergency stop (local machine only, R6.4)
        self._paused = False                                     # the sim is paused: the mirror centres every tick (R7.3)
        self.halt_unconfirmed = False                            # an E-stop centred write could not be confirmed (R6.5)
        self.t0 = time.time()
        self.spiking = spiking                                   # a SpikingHunter: the connectome's neurons drive the fly
        g = {**dict(cfg.hunt_gpu), "obs_noise": 0.0}
        if dict(g.get("humans", {})).get("mode", "learn") == "learn" and humans is None:
            g = {**g, "humans": {**dict(g.get("humans", {})), "mode": "flee"}}
        self.arena = BatchArena(cfg.hunt, g, 1, device, seed=seed)
        self.humans = humans.to(device).eval() if humans is not None else None
        self.net = net.to(device).eval() if net is not None else None
        # the lobotomy: a second brain kept beside the trained one (a network fly that never learned anything, or the
        # spiking circuit with its senses severed); nothing is deleted, the switch just decides which drives the fly
        self.smart = self.net
        torch.manual_seed(1)
        self.dumb = build(self.arena.spec, dict(cfg.hunt_gpu).get("net", {})).to(device).eval() if self.net is not None else None
        self.lobotomized = False
        self.heat_on = True                                      # the heat / PIR sense can be switched off: the fly then has only its eyes
        self.see_m_default = float(self.arena.see_m)             # the eye's range can be made unlimited (see_m 0)
        self.h = self.net.initial_state(1, device) if self.net is not None else None
        self.dumb_spiking = None
        if spiking is not None:
            from .brain_train import SpikingHunter
            spiking.bind_arena(self.arena)
            # the lobotomized fly: the same neurons with every synapse out of the sensory cells cut (the raw connectome
            # would still chase people on its innate pursuit wiring, which is no lobotomy at all)
            self.dumb_spiking = SpikingHunter(cfg, spiking.hc, None, verbose=False)
            self.dumb_spiking.bind_arena(self.arena)
            cut = self.dumb_spiking.sever_senses()
            print(f"[hunt-gpu] lobotomy ready: {cut:,} synapses out of the sensory neurons cut in the spare brain", flush=True)
            c = spiking.hc.circuit
            self.cell_type = c.cell_type
            self.brain_json = self._brain_json(c)
        self.seed, self.episode, self.total = int(seed), 0, 0.0
        self.results: list[dict] = []
        self.last: dict = {}
        self.lock = threading.Lock()
        self._start(self.seed)
        self.last = {"humans": {"patrol": "patrol walkers", "wander": "people walking about", "flee": "scripted runners", "learn": "trained runners"}[self.arena.human_mode]}

    def _start(self, seed: int) -> None:
        self.arena.reset(torch.arange(1, device=self.arena.device), [seed & 0xFFFFFFFF])
        self.episode += 1
        self.total = 0.0
        if self.h is not None:
            self.h.zero_()
        if self.spiking is not None:
            self.spiking.reset()
            self.dumb_spiking.reset()

    @staticmethod
    def _brain_json(c) -> bytes:
        """The subcircuit's neurons for the page: anatomical position (frontal view, normalized), class, type, hunter group."""
        import numpy as np
        pos = c.pos if c.pos is not None else np.zeros((c.n, 3), np.float32)
        has = (pos[:, 0] > 0) & (pos[:, 1] > 0)
        lo = pos[has].min(0) if has.any() else np.zeros(3)
        hi = pos[has].max(0) if has.any() else np.ones(3)
        x = np.where(has, (pos[:, 0] - lo[0]) / max(hi[0] - lo[0], 1), -1)
        y = np.where(has, (pos[:, 1] - lo[1]) / max(hi[1] - lo[1], 1), -1)
        group = [""] * c.n
        for g in ("LC10a", "LC11", "LC12", "LC15", "TRN_hot", "DNa01", "DNa02", "DNp09", "MDN"):
            for i in c.groups.get(g, {}).get("all", []):
                group[int(i)] = g
        for i, sc in enumerate(c.super_class if c.super_class is not None else [""] * c.n):
            if not group[i] and sc == "descending":
                group[i] = "DN"
        return json.dumps({"n": int(c.n), "x": np.round(x, 4).tolist(), "y": np.round(y, 4).tolist(), "cls": (c.super_class.tolist() if c.super_class is not None else []),
                           "type": c.cell_type.tolist(), "group": group}, separators=(",", ":")).encode()

    @torch.no_grad()
    def tick(self) -> dict:
        a = self.arena
        obs = a.observe(noise=False)
        if not self.heat_on:
            obs[:, :2] = 0.0                                     # the hot cells see nothing: vision alone
        brain = None
        if self.spiking is not None:
            hunter = self.dumb_spiking if self.lobotomized else self.spiking
            drive, d = hunter(obs)
            spk = hunter.spikes
            lit = spk.nonzero()[0]
            rates = hunter.runner.rates(300)
            types = {}
            for i in lit:
                t = str(self.cell_type[i])
                types[t] = types.get(t, 0) + int(spk[i])
            brain = {"lit": [[int(i), int(spk[i])] for i in lit], "n_spikes": int(spk.sum()), "top": sorted(types.items(), key=lambda kv: -kv[1])[:8],
                     "rates": {g: [round(rates[g]["left"], 1), round(rates[g]["right"], 1)] for g in ("DNa02", "DNa01", "DN_all", "MDN", "DNp09")},
                     "motor": {k: round(float(v), 2) for k, v in d.motor.items() if k in ("forward", "backward", "turn")}}
        elif self.net is None:
            drive = scripted_drive(obs, a.spec)
        else:
            drive, _, _, self.h = self.net.act(obs, self.h, deterministic=not self.lobotomized)   # the lobotomized fly twitches at random
        hdrive = self.humans.act(a.observe_humans(), deterministic=True)[0] if (self.humans is not None and a.human_mode == "learn") else None
        h0 = float(a.heading[0])
        r, done, info = a.step(drive, hdrive)
        # the fly's real motion this tick, in the robot's units: yaw +ve = clockwise from above (the arena's heading
        # grows to the left), speed along its nose. A real body copies these, not the drive channels.
        yaw_dps = 0.0 if bool(done[0]) else -math.degrees(float(wrap(a.heading[0:1] - h0)[0])) / a.dt * self.speed
        speed_mps = float(a.speed[0]) * self.speed
        if self.mirror is not None:
            # the gate: E-stop and pause both send the neutral centre regardless of the drive (E-stop dominates),
            # otherwise the scaled drive is sent before the next tick (R6.2, R7.3, R4.2).
            centered = self.estop or self._paused
            ok = self.mirror.send(float(drive[0, 0]), float(drive[0, 1]), centered=centered)
            # R6.5: while the E-stop is active, if the centred write cannot be confirmed (the S-Bus link is
            # down), keep the E-stop state, keep sending centred, and surface an "unconfirmed halt".
            self.halt_unconfirmed = self.estop and not (ok and getattr(self.body, "link_ok", True))
        self.total += float(r[0])
        heat, vis, _ = a.spec.split(obs)
        tgt = int(a.target[0])
        st = {"t": round(float(a.t[0]), 2), "seed": int(a.env_seed[0]), "episode": self.episode, "fly": [round(float(a.x[0]), 3), round(float(a.z[0]), 3),
              round(float(a.heading[0]), 4), round(float(a.speed[0]), 3)],
              "people": [[round(float(a.px[0, i]), 3), round(float(a.pz[0, i]), 3), round(float(a.pspeed[0, i]) / a.dt, 2), round(float(a.pyaw[0, i]), 3), round(float(a.stamina[0, i]), 2), int(a.resting[0, i]), int(info["sensed"][0, i]), int(a.alert[0, i])] for i in range(int(a.alive[0].sum()))],
              "humans": {"patrol": "patrol walkers", "wander": "people walking about", "flee": "scripted runners", "learn": "trained runners"}[a.human_mode],
              "target": tgt, "locked": bool(a.locked[0]), "touches": int(a.touches[0]), "t_first": round(float(a.t_first[0]), 2) if bool(a.locked[0]) else None,
              "track": round(float(a.track_steps[0]) / max(1, int(a.steps[0] - a.lock_step[0])), 3) if bool(a.locked[0]) else None,
              "heat": [round(float(heat[0, 0]), 3), round(float(heat[0, 1]), 3)], "vis": [round(float(v), 2) for v in vis[0, :, 0]],
              "motion": [round(float(v), 2) for v in vis[0, :, 1]], "drive": [round(float(drive[0, 0]), 3), round(float(drive[0, 1]), 3)],
              "reward": round(float(r[0]), 3), "total": round(self.total, 2), "done": bool(done[0]), "lobotomized": self.lobotomized, "heat_on": self.heat_on,
              "see_m": a.see_m, "fov_deg": round(math.degrees(a.fov), 1), "brain": brain,
              "heading_deg": round((-math.degrees(float(a.heading[0]))) % 360.0, 1), "yaw_dps": round(yaw_dps, 1), "speed_mps": round(speed_mps, 2),
              "estop": self.estop, "halt_unconfirmed": self.halt_unconfirmed,
              "body": (self.body.status() if self.body is not None and hasattr(self.body, "status") else None)}
        if bool(done[0]):
            ep = info["episodes"][0]
            ep["total_reward"] = round(self.total, 2)
            self.results.append(ep)
            self._start(self.seed + self.episode)
        st["stats"] = summarize(self.results[-50:], a.episode_s, a.side_reward)
        st["history"] = [{k: e[k] for k in ("seed", "touched", "t_touch", "track", "touches", "total_reward")} for e in self.results[-8:]]
        with self.lock:
            self.last = st
        return st

    def next_episode(self) -> None:
        with self.lock:
            self._start(self.seed + self.episode)

    def set_estop(self, on: bool) -> None:
        """The operator's emergency stop, from the local machine only (R6.4): while active the mirror
        sends only the neutral centre (R6.2); clearing it resumes the drive on the next tick (R6.3)."""
        self.estop = bool(on)
        if not self.estop:
            self.halt_unconfirmed = False

    def lobotomize(self, on: bool) -> None:
        """Swap the untrained brain in (on) or the trained one back (off); the fly starts afresh either way."""
        with self.lock:
            if self.spiking is not None:
                self.lobotomized = bool(on)
                return
            if self.smart is None:
                return
            self.lobotomized = bool(on)
            self.net = self.dumb if self.lobotomized else self.smart
            self.h = self.net.initial_state(1, self.h.device)


def serve(cfg, net: HunterNet | None, port: int = 8601, open_browser: bool = False, seed: int = 0, speed: float = 1.0, net_label: str = "trained fly",
          humans: EvaderNet | None = None, spiking=None, body=None, v_max=None, w_max=None, sign_yaw=None,
          read_only: bool = True, feed: str = "local", relay_url=None, bind: str = "0.0.0.0",
          publish_url=None, publish_token=None) -> None:
    # R9.1 / R9.5: the port is configurable but must be a usable TCP port (1024-65535); refuse to start on
    # an out-of-range port with a clear error that names the offending port rather than binding somewhere odd.
    if not isinstance(port, int) or not (1024 <= port <= 65535):
        raise SystemExit(f"[hunt-gpu] invalid port {port!r}: choose a port in the range 1024-65535")
    watch = Watch(cfg, net, seed=seed, speed=speed, humans=humans, spiking=spiking, body=body, v_max=v_max, w_max=w_max, sign_yaw=sign_yaw)
    who = (f"({spiking.hc.n:,} spiking neurons of the connectome, {net_label})" if spiking is not None else "(the scripted hunter)" if net is None else f"({net_label})") + f" vs {watch.last.get('humans', '')}"
    cond = threading.Condition()
    # `s1_rejections` counts inbound network messages discarded as ignored-for-S1 so the rejection is
    # externally observable (R8.5); it is surfaced on /config alongside the read-only / feed configuration.
    box = {"version": 0, "payload": b"{}", "paused": False, "s1_rejections": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html", "/hunt_gpu.html"):
                self._send(200, "text/html; charset=utf-8", HTML.read_bytes())
            elif path == "/events":
                self._events()
            elif path == "/config":
                # what the front-end needs to decide read-only UI (R1.1) and its feed source (R11): the
                # spectator flag, the feed ("local" same-origin SSE / "remote" relay), and the relay url.
                self._send(200, "application/json", json.dumps({
                    "read_only": bool(read_only),
                    "feed": feed,
                    "relay_url": relay_url,
                    "s1_rejections": box["s1_rejections"],
                }, separators=(",", ":")).encode())
            elif path == "/who":
                self._send(200, "text/plain; charset=utf-8", who.encode())
            elif path == "/brain.json":
                self._send(200, "application/json", watch.brain_json if spiking is not None else b"null")
            else:
                rel = path.lstrip("/")
                target = (UI_DIR / rel).resolve()
                if UI_DIR.resolve() not in target.parents or not target.is_file() or target.suffix not in STATIC_TYPES:
                    self._send(404, "text/plain", b"not found")
                else:
                    self._send(200, STATIC_TYPES[target.suffix], target.read_bytes())

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            except (json.JSONDecodeError, ValueError):
                body = {}
            if not isinstance(body, dict):
                body = {}
            # R8.2/R8.3: this POST handler is the only inbound network surface and it has NO branch that
            # writes to S1Body -- the sim keys below (next/lobotomy/heat/unlimited/paused/speed) never reach
            # the S-Bus. Any inbound message that looks like a robot command (stick / motion / velocity /
            # actuator fields) is discarded without translating any part of it into an S-Bus frame, leaving
            # the S1 command stream unchanged, and the rejection is recorded so it is observable (R8.5).
            if any(k in body for k in ("motor", "sticks", "stick", "motion", "velocity", "actuator", "channels", "throttle", "drive")):
                box["s1_rejections"] += 1
                print(f"[hunt-gpu] inbound message ignored for the S1 (never reaches S1Body): keys={sorted(body)}"
                      f" | total ignored-for-S1: {box['s1_rejections']}", flush=True)
            # R1.2 / R9.6: while Spectator_Mode is enabled every web client is read-only -- a control POST is
            # rejected with 403 and nothing in the simulation or robot state changes. The local operator does
            # not rely on POST (they use terminal / keyboard hotkeys, R1.4), so this does not lock them out.
            if read_only:
                self._send(403, "application/json", b'{"error":"read-only: controls are disabled for spectators"}')
                return
            if body.get("next"):
                watch.next_episode()
            if "lobotomy" in body:
                watch.lobotomize(bool(body["lobotomy"]))
                print(f"[hunt-gpu] {'lobotomized: the untrained brain drives now' if watch.lobotomized else 'the trained brain is back'}", flush=True)
            if "heat" in body:
                watch.heat_on = bool(body["heat"])
                print(f"[hunt-gpu] heat sense {'on' if watch.heat_on else 'off: the fly has only its eyes'}", flush=True)
            if "unlimited" in body:
                watch.arena.see_m = 0.0 if body["unlimited"] else watch.see_m_default
                print(f"[hunt-gpu] eye range {'unlimited' if watch.arena.see_m == 0 else f'{watch.arena.see_m:g} m'}", flush=True)
            if "paused" in body:
                box["paused"] = bool(body["paused"])
                watch._paused = box["paused"]
            if "speed" in body:
                watch.speed = max(0.1, min(8.0, float(body["speed"])))
            self._send(204, "text/plain", b"")

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with cond:
                        cond.wait_for(lambda: box["version"] != seen, timeout=5.0)
                        seen, payload = box["version"], box["payload"]
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    # R9.2: bind to all interfaces ("0.0.0.0") so LAN devices can reach the view; deployed mode passes
    # "127.0.0.1" to keep the machine off the network (R14.3). R9.5: an already-in-use (or otherwise
    # unbindable) port raises OSError here -- turn it into a clear refuse-to-start error naming the port.
    try:
        server = ThreadingHTTPServer((bind, port), Handler)
    except OSError as e:
        raise SystemExit(f"[hunt-gpu] cannot serve on {bind}:{port} ({e.strerror or e}); "
                         f"the port may already be in use -- pass --port with a free port in 1024-65535")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"[hunt-gpu] watching {'the spiking connectome fly' if spiking is not None else 'the scripted hunter' if net is None else 'the trained fly'} at {url} (Ctrl-C to stop)", flush=True)
    # R11.5 / R14.1: when a publish URL is configured, start the outbound-only Publisher on the SAME shared
    # box + Condition so it pushes each tick's payload to the relay. It holds no reference to S1Body, opens
    # no listening socket, and its failures are isolated so the local SSE serving and the mirror carry on.
    publisher = None
    if publish_url:
        try:
            from .publish import Publisher
            publisher = Publisher(box, cond, publish_url, token=publish_token)
            publisher.start()
            print(f"[hunt-gpu] publishing state outbound to the relay at {publish_url}", flush=True)
        except Exception as e:
            print(f"[hunt-gpu] could not start the publish client, local serving continues: {e}", flush=True)
            publisher = None
    if open_browser:
        webbrowser.open(url)
    dt = watch.arena.dt
    nxt = time.perf_counter()
    try:
        while True:
            watch._paused = bool(box["paused"])
            if not watch._paused:
                st = watch.tick()
                payload = json.dumps(st, separators=(",", ":")).encode()
                with cond:
                    box["payload"] = payload
                    box["version"] += 1
                    cond.notify_all()
                if st["done"]:
                    e = watch.results[-1]
                    print(f"[hunt-gpu] episode {watch.episode - 1} seed {e['seed']}: "
                          f"{('first touch at %.1f s, tracked %.0f%% of the rest, %d contacts' % (e['t_touch'], 100 * e['track'], e['touches'])) if e['touched'] else 'no touch'}"
                          f" | reward {e['total_reward']:.1f}", flush=True)
            elif watch.mirror is not None:
                watch.mirror.send(0.0, 0.0, centered=True)   # paused-centering: keep the S1 stopped every tick (R7.3)
            nxt += dt / watch.speed
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 1.0:          # fell behind (sleeping laptop): do not race to catch up
                nxt = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[hunt-gpu] stopped", flush=True)
    finally:
        if publisher is not None:
            publisher.stop()
        server.shutdown()
