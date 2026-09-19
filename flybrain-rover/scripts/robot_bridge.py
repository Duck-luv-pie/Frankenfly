"""
robot_bridge.py -- camera -> person detector -> retina -> fly brain -> RoboMaster S1 wheels.
Runs on the laptop next to the robot. The control loop never leaves the LAN (CLAUDE.md rule 5).

    python scripts/robot_bridge.py --bench                        # brain-only timing, no camera
    python scripts/robot_bridge.py --source 0 --show              # laptop webcam, print commands, no robot
    python scripts/robot_bridge.py --source clip.mp4 --show
    python scripts/robot_bridge.py --robot --checkpoint checkpoints/tfA_both_latest.pt
    python scripts/robot_bridge.py --fake-boxes --no-camera --dry-run --max-frames 300   # latency test, no OpenCV needed

Per frame (budget < 100 ms camera-to-wheels):
    frame -> detector (ultralytics yolo11n, class person; or --fake-box) -> Retina.from_boxes
    -> Senses.inject(columns, heat) -> N x LIF.step (1 ms each) -> Motor.decode
    -> chassis.drive_speed(x = forward * v_max m/s, y = 0, z = turn * w_max deg/s)
Conventions: turn > 0 = turn RIGHT = clockwise = positive z in the RoboMaster SDK. Azimuth + = image right.
Keys in the --show window: q quit, r reward (PAM burst), p punish (PPL1 burst), l toggle lobotomy (brain
silenced: robot stops), space emergency stop. Without a window, pressing Enter in the terminal is the E-stop.
Safety: --dry-run prints commands instead of sending them; a watchdog sends zero velocity whenever no
frame has arrived for --watchdog-ms (300). Latency (camera timestamp -> command timestamp) is logged per
frame to --latency-log (logs/bridge_latency.csv) and summarised at exit.
Heat: optional serial line "L R" floats in [0,1] (--heat-serial /dev/tty.usbmodem...), else zeros.
The brain dt is 1 ms but a camera frame arrives every ~33-66 ms; --substeps sets how many 1 ms steps
run per frame (default 20, the same as training). Measure with --bench on the demo laptop first.
"""
import argparse
import csv
import json
import os
import random
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain, pick_device  # noqa: E402
from brain.motor import Motor  # noqa: E402
from brain.retina import Camera, Retina  # noqa: E402
from brain.retinotopy import column_assignment  # noqa: E402
from brain.explore import Exploratory  # noqa: E402
from brain.senses import Senses  # noqa: E402


# ---------------------------------------------------------------- brain
class Brain:
    def __init__(self, checkpoint=None, gain=0.05, device=None, engine="auto", k_t=0.02, k_f=0.3,
                 amps=None, substeps=20, forward_source="dn_all", retinotopy="rank", explore=False,
                 shuffle=False, brain_path="data/brain.npz"):
        self.device = pick_device(device)
        d, N, groups = load_brain(brain_path)
        self.lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1,
                       device=self.device, engine=engine, g=gain)
        amps = dict(amps or {})
        if checkpoint:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.lif.set_w(ck["w"].to(self.device))
            if "gain" in ck:
                self.lif.set_gain(float(ck["gain"]))
            amps = {**(ck.get("amps") or {}), **amps}
            k_t, k_f = ck.get("k_t", k_t), ck.get("k_f", k_f)
            forward_source = ck.get("forward_source", "dna01_dnp09")
            retinotopy = ck.get("retinotopy", "index")
            print(f"loaded {checkpoint} (run {ck.get('run')}, gen {ck.get('gen')}, stage {ck.get('stage')}, forward {forward_source})")
        col_of = None
        if retinotopy != "index" and os.path.exists("data/lc_columns.json"):
            col_of = column_assignment(groups, d["ids"], mode=retinotopy, device=self.device)
        self.groups = {k: v.to(self.device) for k, v in groups.items()}
        self.cam = Camera()
        self.retina = Retina(self.cam, device=self.device)
        self.senses = Senses(self.groups, N, device=self.device, col_of=col_of, **amps)
        self.motor = Motor(self.groups, k_t=k_t, k_f=k_f, device=self.device, forward_source=forward_source)
        # exploratory state, internal drive, not sensory (brain/explore.py); dt = wall-clock frame interval
        self.explorer = Exploratory(self.groups, N, 1, device=self.device, dt=1 / 30) if explore else None
        self.substeps = substeps
        # the control that answers "is it the wiring, or just the neurons?": same cells, same in- and
        # out-degrees, same signed weights, randomized targets. Built eagerly so the demo never stalls.
        self.lif_real, self.lif_shuffled, self.shuffled = self.lif, None, False
        if shuffle:
            self.build_shuffled()
        self.lobotomy = False
        self.pending_reward = 0.0
        self.last_t = None
        self.viz, self.last_r, self.last_heat, self.last_exploring = None, None, None, False

    @torch.no_grad()
    def build_shuffled(self, seed=0):
        """Permute the post (target) column of every synapse. Each neuron keeps its exact in-degree and
        out-degree and every synapse keeps its sign and weight; only who-connects-to-whom is destroyed."""
        if self.lif_shuffled is not None:
            return self.lif_shuffled
        real = self.lif_real
        post, pre = real.post_idx.cpu(), real.pre_idx.cpu()
        perm = torch.randperm(post.numel(), generator=torch.Generator().manual_seed(seed))
        t0 = time.time()
        self.lif_shuffled = LIF(torch.stack([post[perm], pre]), real.w.detach().cpu(), real.N, 1,
                                device=self.device, engine=real.engine, g=real.g)
        print(f"shuffled-wiring control built in {time.time() - t0:.1f} s ({real.nnz:,} synapses rewired)", flush=True)
        return self.lif_shuffled

    @torch.no_grad()
    def use_shuffled(self, on):
        """Swap the brain the robot is driven by. Neuron identity is unchanged, so groups, senses and
        motor read-outs stay valid; only the connectivity differs."""
        if on and self.lif_shuffled is None:
            self.build_shuffled()
        self.shuffled = bool(on) and self.lif_shuffled is not None
        self.lif = self.lif_shuffled if self.shuffled else self.lif_real
        self.lif.reset()
        return self.shuffled

    @torch.no_grad()
    def step(self, boxes_px, heat_LR, now):
        """boxes_px: list of (x0, y0, x1, y1) in a 320x240 frame. -> (forward, turn) floats."""
        dt = 1 / 30 if self.last_t is None else max(1e-3, now - self.last_t)
        self.last_t = now
        K = max(1, len(boxes_px))
        b = torch.zeros(1, K, 4, device=self.device)
        v = torch.zeros(1, K, dtype=torch.bool, device=self.device)
        for i, bb in enumerate(boxes_px):
            b[0, i] = torch.tensor(bb, dtype=torch.float32); v[0, i] = True
        r = self.retina.from_boxes(b, v, dt)
        heat = torch.tensor([heat_LR], device=self.device, dtype=torch.float32)
        I = self.senses.inject(r, heat)
        if self.explorer is not None:
            self.explorer.dt = dt
            self.explorer.inject(I, r, heat)
        if self.pending_reward:
            self.senses.dopamine(I, torch.tensor([self.pending_reward], device=self.device))
            self.pending_reward = 0.0
        if self.lobotomy:
            I.zero_()
        for _ in range(self.substeps):
            spk = self.lif.step(I)
            if self.viz is not None:
                self.viz.accumulate_substep(spk)
        fwd, turn = self.motor.decode(self.lif)
        self.last_r, self.last_heat = r, heat
        self.last_exploring = bool(self.explorer.exploring[0].item()) if self.explorer is not None else False
        if self.lobotomy:
            return 0.0, 0.0
        return fwd.item(), turn.item()

    def rates(self, keys=("LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF")):
        return {k: round(self.lif.rates(self.groups[k]).item(), 1) for k in keys}


# ---------------------------------------------------------------- detectors
class YoloPerson:
    def __init__(self, weights="yolo11n.pt", conf=0.4, imgsz=320, device=None):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("ultralytics is not installed: run `pip install ultralytics` (downloads yolo11n.pt on "
                              "first use), or test the plumbing with --fake-boxes") from e
        if device is None:   # detector on the GPU if there is one (MPS: 29 ms vs 83 ms CPU on the M2 Pro); brain stays on CPU
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, self.conf, self.imgsz, self.device = YOLO(weights), conf, imgsz, device
        print(f"yolo11n person detector on {device}", flush=True)

    def __call__(self, frame_bgr):
        res = self.model.predict(frame_bgr, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False,
                                 device=self.device)[0]
        return [tuple(map(float, xyxy)) for xyxy in res.boxes.xyxy.cpu().numpy()]


class FakeBox:
    """A person-sized box sweeping left-right across the frame, for plumbing tests without a detector."""
    def __init__(self, W=320, H=240, period=6.0):
        self.W, self.H, self.period, self.t0 = W, H, period, time.time()

    def __call__(self, frame_bgr=None):
        ph = ((time.time() - self.t0) % self.period) / self.period
        cx = self.W * (0.15 + 0.7 * (0.5 - 0.5 * np.cos(2 * np.pi * ph)))
        w = 40
        return [(cx - w / 2, 20.0, cx + w / 2, float(self.H))]


def companion_path(explicit=None):
    """Locate Ducks's `companion_brain` package (hunting-fly) so we can reuse his S1 and camera drivers."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for c in ([explicit] if explicit else []) + [os.path.join(here, "..", "hunting-fly-s1", "brain"),
                                                 os.path.join(here, "..", "hunting-fly", "brain"),
                                                 os.path.join(here, "..", "brain")]:
        if c and os.path.isfile(os.path.join(c, "companion_brain", "body", "s1.py")):
            c = os.path.abspath(c)
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    return None


class StreamEye:
    """A network camera (ESP32-CAM MJPEG, phone IP-camera app, RTSP) as the fly's eye.

    Uses Ducks's `companion_brain.senses.camera.CameraStream` when his repo is on disk: it resolves
    mDNS names that OpenCV cannot, keeps the capture buffer at one frame, reads in a background thread
    that holds only the newest frame (so the brain never falls behind the camera), and reconnects after
    a stall. Falls back to a plain OpenCV capture when his package is missing. read() returns a 320x240
    BGR frame, or None when nothing new has arrived yet."""

    def __init__(self, url, companion_dir=None):
        import cv2
        self.url, self.cv2, self.src = url, cv2, None
        if companion_path(companion_dir):
            try:
                from companion_brain.senses.camera import CameraStream
                self.src = CameraStream(url)
                print(f"eye: {url} via Ducks's CameraStream (mDNS, newest-frame thread, auto-reconnect)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"his CameraStream unavailable ({type(e).__name__}); plain OpenCV capture", flush=True)
        if self.src is None:
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise SystemExit(f"cannot open stream {url!r}  (open it in a browser first to check)")
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"eye: {url} (OpenCV capture)", flush=True)

    def read(self):
        if self.src is not None:
            self.src.read()                      # his read() downsamples for his optic lobe ...
            return self.src.preview              # ... and keeps the 320x240 BGR frame here, which is our retina's input
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.src is not None:
            self.src.close()
        elif getattr(self, "cap", None) is not None:
            self.cap.release()


def source_kind(src):
    """How to treat --source: a digit is a local webcam, anything with :// is a network stream
    (ESP32-CAM, phone IP-camera app, RTSP), otherwise a video file that ends."""
    if str(src).isdigit():
        return "webcam"
    if "://" in str(src):
        return "stream"
    return "file"


def scale_boxes(boxes, W, H, cam):
    sx, sy = cam.W / W, cam.H / H
    return [(x0 * sx, y0 * sy, x1 * sx, y1 * sy) for x0, y0, x1, y1 in boxes]


# ---------------------------------------------------------------- robot / heat
class Rover:
    def __init__(self, conn="ap", v_max=0.7, w_max_deg=90.0):
        from robomaster import robot
        self.ep = robot.Robot()
        self.ep.initialize(conn_type=conn)
        self.ep.gimbal.recenter().wait_for_completed()          # turret locked at neutral: camera is body-fixed
        self.ep.camera.start_video_stream(display=False, resolution="360p")
        self.v_max, self.w_max = v_max, w_max_deg

    def frame(self):
        return self.ep.camera.read_cv2_image(strategy="newest", timeout=1.0)

    def drive(self, forward, turn):
        self.ep.chassis.drive_speed(x=forward * self.v_max, y=0, z=turn * self.w_max, timeout=0.5)

    def stop(self):
        self.ep.chassis.drive_speed(x=0, y=0, z=0)

    def close(self):
        self.stop()
        self.ep.camera.stop_video_stream()
        self.ep.close()


class VizFeed:
    """Live brain feed for the Three.js viz: one JSON object per camera frame, same fields as replay/FORMAT.md
    (t, forward, turn, pres, size, mot, loom, heat, rates, spikes; plus boxes and exploring). Two outlets:
    a JSONL file (tail -f friendly) and a websocket broadcast (ws://localhost:<port>, latest frame per client)."""
    VIZ_KEYS = ["LC10a_L", "LC10a_R", "LC11_L", "LC11_R", "LC12_L", "LC12_R", "LC15_L", "LC15_R", "LC4_L", "LC4_R",
                "LPLC2_L", "LPLC2_R", "THERMO_L", "THERMO_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09",
                "GF", "PAM", "PPL1", "KC", "MBON", "DN_ALL"]

    def __init__(self, brain, jsonl=None, ws_port=None, n_viz=512):
        self.brain, self.f = brain, (open(jsonl, "a") if jsonl else None)
        g = torch.Generator().manual_seed(0)
        self.viz_idx = torch.randperm(brain.lif.N, generator=g)[:n_viz].to(brain.device)
        self.spk_sum = torch.zeros(n_viz, device=brain.device)
        self.latest, self.clients, self.loop = None, set(), None
        self.t0 = time.time()
        if ws_port:
            threading.Thread(target=self._serve, args=(ws_port,), daemon=True).start()
        header = dict(meta=dict(N=brain.lif.N, viz_neurons=self.viz_idx.tolist(), hz="camera frame rate",
                                keys=self.VIZ_KEYS, format="replay/FORMAT.md"))
        if self.f:
            self.f.write(json.dumps(header) + "\n"); self.f.flush()

    def accumulate_substep(self, spikes):
        self.spk_sum += spikes[0, self.viz_idx].float()

    def frame(self, t, boxes, fwd, turn, r, heat, exploring):
        b = self.brain
        msg = dict(t=round(t - self.t0, 3), forward=round(fwd, 3), turn=round(turn, 3),
                   boxes=[[round(v, 1) for v in bb] for bb in boxes], heat=[round(float(x), 3) for x in heat],
                   pres=[round(x, 3) for x in r["pres"][0].tolist()], size=[round(x, 3) for x in r["size"][0].tolist()],
                   mot=[round(x, 3) for x in r["mot"][0].tolist()],
                   loom=[round(r["loom_L"][0].item(), 3), round(r["loom_R"][0].item(), 3)],
                   rates={k: round(b.lif.rates(b.groups[k])[0].item(), 1) for k in self.VIZ_KEYS if k in b.groups},
                   spikes=self.spk_sum.nonzero().flatten().tolist(), exploring=bool(exploring), lobotomy=bool(b.lobotomy))
        self.spk_sum.zero_()
        line = json.dumps(msg)
        self.latest = line
        if self.f:
            self.f.write(line + "\n"); self.f.flush()
        if self.loop is not None and self.clients:
            self.loop.call_soon_threadsafe(self._broadcast, line)

    def _broadcast(self, line):
        import asyncio
        for c in list(self.clients):
            asyncio.ensure_future(c.send(line))

    def _serve(self, port):
        import asyncio
        try:
            import websockets
        except ImportError:
            print("websockets not installed (pip install websockets); viz websocket disabled", flush=True); return

        async def handler(ws):
            self.clients.add(ws)
            try:
                if self.latest:
                    await ws.send(self.latest)
                async for _ in ws:      # clients may send anything; we ignore it
                    pass
            finally:
                self.clients.discard(ws)

        async def main():
            self.loop = asyncio.get_running_loop()
            async with websockets.serve(handler, "0.0.0.0", port):
                print(f"viz websocket on ws://localhost:{port}", flush=True)
                await asyncio.Future()
        asyncio.run(main())


class UdpS1Rover:
    """The S1 driven through the Raspberry Pi that rides it: one JSON packet per frame over UDP to
    scripts/pi_s1_relay.py, which hands it to Ducks's S-Bus driver on the Pi's UART. Wireless, and his
    Pi is the body as he designed it. The relay centres the sticks itself if packets stop for 0.5 s, so a
    dead laptop or a dropped network is a stopped robot, not a runaway."""
    def __init__(self, target, v_max=0.7, w_max_deg=90.0, sign_yaw=1):
        import socket
        host, _, port = target.rpartition(":")
        self.addr = (host or "127.0.0.1", int(port or 4310))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sign_yaw = sign_yaw
        self.v_max, self.w_max = v_max, w_max_deg
        self.n = 0
        print(f"wheels: RoboMaster S1 via the Pi relay at udp://{self.addr[0]}:{self.addr[1]} "
              f"(forward/turn in [-1,1]; the Pi's driver maps them to sticks)", flush=True)

    def drive(self, forward, turn):
        msg = json.dumps({"forward": float(max(-1.0, min(1.0, forward))),
                          "turn": float(max(-1.0, min(1.0, self.sign_yaw * turn)))}, separators=(",", ":"))
        try:
            self.sock.sendto(msg.encode(), self.addr)
            self.n += 1
        except OSError:
            pass

    def stop(self):
        for _ in range(3):
            try:
                self.sock.sendto(b'{"stop":true}', self.addr)
            except OSError:
                pass

    def close(self):
        self.stop()


class SbusRover:
    """The S1 driven through its S-Bus receiver pins with Ducks's driver (hunting-fly, brain/companion_brain/body/s1.py):
    Mac USB -> ESP32 'sbus-bridge' (inverter) -> S1 motion controller. No SDK, no gimbal, nothing comes back.
    forward/turn in [-1, 1] map to his stick gains; his defaults were measured on this S1 (slow preset: 0.85 m/s,
    90 deg/s at full stick). We override rotation_only=False so the rover advances."""
    def __init__(self, port, companion_dir=None, v_max=0.7, w_max_deg=90.0, stick_forward=None, sign_yaw=1, sign_forward=1):
        import importlib
        if not companion_path(companion_dir):
            raise SystemExit("cannot find Ducks's companion_brain package; pass --companion-dir <hunting-fly>/brain")
        s1 = importlib.import_module("companion_brain.body.s1")
        cfg = dict(rotation_only=False, speed="slow", free_mode=True, timeout_s=0.5,
                   stick_forward=(stick_forward if stick_forward is not None else min(1.0, v_max / s1.DEFAULTS["speed_mps_full"])),
                   stick_yaw=min(1.0, w_max_deg / s1.DEFAULTS["yaw_dps_full"]), sign_yaw=sign_yaw, sign_forward=sign_forward)
        self.cfg, self.port = cfg, port
        self.body = s1.S1Body(cfg, port=port)
        self.v_max, self.w_max = v_max, w_max_deg
        print(f"S-Bus rover on {port}: full forward stick {cfg['stick_forward']:.2f}, full yaw stick {cfg['stick_yaw']:.2f} (turn>0 = right)", flush=True)

    def frame(self):
        return None                                       # no camera on this path: use --local-eye --source ...

    def drive(self, forward, turn):
        self.body.send({"motor": {"forward": float(forward), "turn": float(turn)}})

    def stop(self):
        self.body.send({"motor": {"forward": 0.0, "turn": 0.0}})

    def reconnect(self):
        """The USB-serial link dropped (a knocked cable at a demo table). Rebuild it."""
        try:
            self.body.close()
        except Exception:  # noqa: BLE001
            pass
        import importlib
        s1 = importlib.import_module("companion_brain.body.s1")
        self.body = s1.S1Body(self.cfg, port=self.port)

    def close(self):
        try:
            self.stop(); time.sleep(0.05); self.body.close()
        except Exception:  # noqa: BLE001
            pass


class RemoteRover:
    """Client for scripts/robot_daemon.py (the SDK lives in a Python 3.8 x86_64 process). Frames arrive as
    JPEG over localhost TCP; commands go back as 'V x y z' lines. Same interface as Rover."""
    def __init__(self, addr="127.0.0.1:9500", v_max=0.7, w_max_deg=90.0):
        self.addr = addr
        host, port = addr.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=5.0)
        self.sock.settimeout(1.0)
        self.buf = b""
        self.v_max, self.w_max = v_max, w_max_deg
        print(f"connected to robot daemon at {addr}", flush=True)

    def frame(self):
        import cv2
        import numpy as _np
        deadline = time.time() + 1.0
        while time.time() < deadline:
            while len(self.buf) >= 5 and self.buf[:1] == b"F":
                n = struct.unpack("<I", self.buf[1:5])[0]
                if len(self.buf) < 5 + n:
                    break
                jpg, self.buf = self.buf[5:5 + n], self.buf[5 + n:]
                if n == 0:
                    continue                                       # heartbeat (daemon without camera)
                if not (len(self.buf) >= 5 and self.buf[:1] == b"F" and len(self.buf) >= 5 + struct.unpack("<I", self.buf[1:5])[0]):
                    return cv2.imdecode(_np.frombuffer(jpg, _np.uint8), cv2.IMREAD_COLOR)   # newest complete frame
            try:
                chunk = self.sock.recv(1 << 16)
            except socket.timeout:
                return None
            if not chunk:
                return None
            self.buf += chunk
        return None

    def drive(self, forward, turn):
        self.sock.sendall(f"V {forward * self.v_max:.3f} 0 {turn * self.w_max:.2f}\n".encode())

    def reconnect(self):
        try:
            self.sock.close()
        except OSError:
            pass
        host, port = self.addr.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=5.0)
        self.sock.settimeout(1.0); self.buf = b""

    def stop(self):
        self.sock.sendall(b"S\n")

    def close(self):
        try:
            self.stop(); self.sock.sendall(b"Q\n")
        except OSError:
            pass
        self.sock.close()


class HeatSerial:
    def __init__(self, port, baud=115200):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.last = (0.0, 0.0)

    def read(self):
        try:
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                l, r = line.replace(",", " ").split()[:2]
                self.last = (min(1.0, max(0.0, float(l))), min(1.0, max(0.0, float(r))))
        except Exception:  # noqa: BLE001
            pass
        return self.last


# ---------------------------------------------------------------- main
def main(argv=None, hotkeys=None):
    hotkeys = hotkeys or {}
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index, video path, or 'robot'")
    ap.add_argument("--robot", action="store_true", help="drive the RoboMaster via the SDK in THIS process (needs Python 3.8)")
    ap.add_argument("--local-eye", action="store_true", help="with --robot-daemon: use --source (laptop webcam) as the eye, robot for wheels only")
    ap.add_argument("--s1-udp", default=None, metavar="HOST:PORT",
                    help="drive the S1 through the Pi relay (scripts/pi_s1_relay.py), e.g. companion-pi.local:4310")
    ap.add_argument("--s1-sbus", default=None, metavar="PORT",
                    help="drive the S1 over S-Bus via the ESP32 bridge on this serial port (Ducks's driver); use with --local-eye --source 0")
    ap.add_argument("--companion-dir", default=None, help="hunting-fly brain/ dir holding companion_brain (auto-detected)")
    ap.add_argument("--sign-yaw", type=int, default=1, choices=[1, -1], help="flip if the S1 turns the wrong way over S-Bus")
    ap.add_argument("--robot-daemon", default=None, metavar="HOST:PORT",
                    help="drive the RoboMaster through scripts/robot_daemon.py (recommended: SDK in its own py3.8 process)")
    ap.add_argument("--conn", default="ap", choices=["ap", "sta", "rndis"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--brain", default="data/brain.npz",
                    help="which circuit to run, e.g. data/brain_tiny.npz for the 2,211-neuron version")
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", default="event", help="event engine is the fast one at batch 1")
    ap.add_argument("--fake-box", "--fake-boxes", dest="fake_box", action="store_true",
                    help="no detector: synthetic sweeping person box")
    ap.add_argument("--no-camera", action="store_true", help="synthetic 320x240 frames at 30 fps (no OpenCV needed)")
    ap.add_argument("--dry-run", action="store_true", help="print wheel commands instead of sending them")
    ap.add_argument("--watchdog-ms", type=float, default=300.0)
    ap.add_argument("--chaos", type=float, default=0.0, metavar="P",
                    help="rehearsal: fail the camera, detector, brain and wheels at probability P per frame")
    ap.add_argument("--latency-log", default="logs/bridge_latency.csv")
    ap.add_argument("--explore", action="store_true", help="search when nobody is in view (exploratory internal state)")
    ap.add_argument("--shuffle", action="store_true",
                    help="also build the degree-preserving wiring-shuffle control, toggled live with key 6")
    ap.add_argument("--viz-jsonl", default=None, help="append one JSON frame per camera frame (replay/FORMAT.md fields)")
    ap.add_argument("--viz-ws", type=int, default=None, help="websocket port broadcasting the same frames (ws://localhost:PORT)")
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--yolo-device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--heat-serial", default=None)
    ap.add_argument("--v-max", type=float, default=0.7)
    ap.add_argument("--w-max", type=float, default=90.0, help="deg/s at turn = 1")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--bench", action="store_true", help="time the brain loop and exit")
    ap.add_argument("--cmd-port", type=int, default=9600, help="UDP port that accepts hotkeys (0 disables)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args(argv)

    brain = Brain(a.checkpoint, gain=a.gain, device=a.device, engine=a.engine, substeps=a.substeps, explore=a.explore,
                  shuffle=a.shuffle, brain_path=a.brain)
    if a.viz_jsonl or a.viz_ws:
        brain.viz = VizFeed(brain, jsonl=a.viz_jsonl, ws_port=a.viz_ws)
    print(f"brain on {brain.device} ({brain.lif.engine}), {a.substeps} x 1 ms per frame")
    if a.bench:
        det = FakeBox()
        for _ in range(5):
            brain.step(det(), (0.0, 0.0), time.time())
        t0 = time.time(); n = 30
        for _ in range(n):
            brain.step(det(), (0.0, 0.0), time.time())
        ms = (time.time() - t0) / n * 1000
        print(f"brain: {ms:.1f} ms per frame ({ms / a.substeps:.2f} ms per 1 ms step); "
              f"{'OK' if ms < 70 else 'TOO SLOW'} for the 100 ms budget (detector needs ~20-30 ms)")
        print("rates", brain.rates())
        return

    detector = FakeBox() if a.fake_box else None
    if detector is None:
        try:
            detector = YoloPerson(a.yolo, device=a.yolo_device)
        except ImportError as e:
            sys.exit(str(e))
    heat = HeatSerial(a.heat_serial) if a.heat_serial else None
    if a.s1_udp:
        rover = UdpS1Rover(a.s1_udp, a.v_max, a.w_max, sign_yaw=a.sign_yaw)
    elif a.s1_sbus:
        try:
            rover = SbusRover(a.s1_sbus, a.companion_dir, a.v_max, a.w_max, sign_yaw=a.sign_yaw)
        except Exception as e:  # noqa: BLE001
            # rehearsing before the ESP32 arrives is the normal case, so a dry run carries on without it
            if not a.dry_run:
                raise SystemExit(f"cannot open the S-Bus bridge on {a.s1_sbus}: {str(e).splitlines()[0][:100]}\n"
                                 f"  is the ESP32 plugged in?  ls /dev/cu.usb*   (or rehearse with --dry-run)")
            print(f"no S-Bus bridge on {a.s1_sbus} ({type(e).__name__}); dry run continues, wheel commands print only",
                  flush=True)
            rover = None
    else:
        rover = Rover(a.conn, a.v_max, a.w_max) if a.robot else (RemoteRover(a.robot_daemon, a.v_max, a.w_max) if a.robot_daemon else None)

    cap = eye = None
    local_eye = (rover is None) or (a.robot_daemon is not None and a.local_eye) or (a.s1_sbus is not None) or (a.s1_udp is not None)
    kind = source_kind(a.source)
    if local_eye and not a.no_camera:
        import cv2
        if kind == "stream":
            eye = StreamEye(a.source, a.companion_dir)
        else:
            src = int(a.source) if kind == "webcam" else a.source
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                sys.exit(f"cannot open {kind} source {a.source!r}"
                         + ("  (webcam: try --source 1; macOS asks for camera permission on first use)" if kind == "webcam" else ""))
            print(f"eye: {kind} {a.source!r}", flush=True)
        if rover is not None:
            print(f"eye = local camera {a.source}; wheels = robot daemon", flush=True)
    if a.show:
        import cv2

    # terminal keys: a line with a hotkey digit runs it; an empty line, "q" or space stops the wheels and exits
    estop = threading.Event()
    pending_keys = []

    def _stdin_watch():
        try:
            for line in sys.stdin:
                k = line.strip()
                if k in hotkeys:
                    pending_keys.append(k)
                else:
                    estop.set()
                    break
        except Exception:  # noqa: BLE001
            pass

    def run_hotkey(k):
        label, fn = hotkeys[k]
        try:
            fn(brain, rover)
            print(f"[{k}] {label}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{k}] {label} FAILED: {e}", flush=True)
    if sys.stdin and not sys.stdin.closed:
        threading.Thread(target=_stdin_watch, daemon=True).start()

    # The same hotkeys over UDP, so another process can press them: scripts/talk.py lets a judge say
    # "remove its eye" and the voice agent sends "2" here. Loopback only; a datagram is one key.
    def _udp_watch(port):
        import socket as _socket
        sk = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        try:
            sk.bind(("127.0.0.1", port))
        except OSError as e:
            print(f"hotkey port {port} unavailable ({e}); voice control off", flush=True)
            return
        print(f"hotkeys also on udp://127.0.0.1:{port}", flush=True)
        while True:
            data, _ = sk.recvfrom(16)
            k = data.decode(errors="ignore").strip()
            if k in hotkeys:
                pending_keys.append(k)
            elif k == " ":
                estop.set()
    if a.cmd_port:
        threading.Thread(target=_udp_watch, args=(a.cmd_port,), daemon=True).start()

    os.makedirs(os.path.dirname(a.latency_log) or ".", exist_ok=True)
    lat_f = open(a.latency_log, "a", newline="")
    lat_w = csv.writer(lat_f)
    if lat_f.tell() == 0:
        lat_w.writerow(["frame", "t_cam", "det_ms", "brain_ms", "cmd_ms", "total_ms", "boxes", "forward", "turn", "watchdog"])
    lats = []

    fails = {}

    def guard(label, fn, default=None, recover=None):
        """A demo degrades, it does not exit. Any failure in the camera, the detector, the brain, the
        wheels or the viewer is logged (rate limited), optionally recovered from, and the loop carries on."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            c = fails[label] = fails.get(label, 0) + 1
            if c <= 3 or c % 50 == 0:
                print(f"[{label}] {type(e).__name__}: {str(e)[:90]} (failure {c})", flush=True)
            if recover is not None and c % 10 == 0:
                try:
                    recover(); print(f"[{label}] reconnected", flush=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"[{label}] reconnect failed: {type(e2).__name__}", flush=True)
            return default

    def chaos(label):
        """--chaos p injects failures at probability p so a rehearsal proves the guards work."""
        if a.chaos and random.random() < a.chaos:
            raise RuntimeError(f"injected {label} failure")

    def send(fwd, turn, why=""):
        def _send():
            chaos("wheels")
            if rover and not a.dry_run:
                rover.drive(fwd, turn)
            elif a.dry_run and why:
                print(f"[dry-run] drive_speed(x={fwd * a.v_max:+.2f} m/s, y=0, z={turn * a.w_max:+.1f} deg/s) {why}")
        guard("wheels", _send, recover=getattr(rover, "reconnect", None))

    n = 0
    last_boxes = []
    last_frame_t = time.time()
    fps_period = 1 / 30
    try:
        while not estop.is_set():
            if a.no_camera:
                t_cam = time.time(); frame = None; W, H = 320, 240
            else:
                frame = guard("camera", lambda: (chaos("camera"), eye.read() if eye is not None
                                                 else (cap.read()[1] if cap is not None else rover.frame()))[1])
                t_cam = time.time()
                if frame is None:
                    if eye is not None:
                        time.sleep(0.002)          # the threaded reader has nothing new yet; do not spin a core
                    if (time.time() - last_frame_t) * 1000 > a.watchdog_ms:
                        send(0.0, 0.0, "WATCHDOG: no frame"); print("WATCHDOG: no frame, wheels zeroed", flush=True)
                        last_frame_t = time.time()
                    if cap is not None and kind == "file":
                        break                      # end of the video file; a webcam or stream just hiccuped
                    continue
                H, W = frame.shape[:2]
            wd = (t_cam - last_frame_t) * 1000 > a.watchdog_ms and n > 0
            last_frame_t = t_cam
            while pending_keys:
                run_hotkey(pending_keys.pop(0))
            raw = guard("detector", lambda: (chaos("detector"), detector(frame))[1])
            if raw is None:                      # a detector hiccup: reuse the last boxes briefly, then give up on them
                stale = fails.get("_stale", 0) + 1; fails["_stale"] = stale
                raw = last_boxes if stale <= 5 else []
            else:
                fails["_stale"] = 0
            last_boxes = raw
            boxes = scale_boxes(raw, W, H, brain.cam)
            t_det = time.time()
            h = heat.read() if heat else (0.0, 0.0)
            fwd, turn = guard("brain", lambda: (chaos("brain"), brain.step(boxes, h, t_cam))[1], (0.0, 0.0))
            t_brain = time.time()
            if wd:
                send(0.0, 0.0, "WATCHDOG: frame gap"); fwd, turn = 0.0, 0.0
            else:
                send(fwd, turn, "" if n % 30 else f"frame {n}")
            t_cmd = time.time()
            n += 1
            if brain.viz is not None and brain.last_r is not None:
                guard("viz", lambda: brain.viz.frame(t_cam, boxes, fwd, turn, brain.last_r,
                                                     brain.last_heat[0].tolist(), brain.last_exploring))
            total = (t_cmd - t_cam) * 1000
            lats.append(total)
            lat_w.writerow([n, f"{t_cam:.4f}", f"{1000 * (t_det - t_cam):.1f}", f"{1000 * (t_brain - t_det):.1f}",
                            f"{1000 * (t_cmd - t_brain):.1f}", f"{total:.1f}", len(boxes), f"{fwd:+.3f}", f"{turn:+.3f}", int(wd)])
            if n % 30 == 0:
                print(f"f{n} boxes {len(boxes)} fwd {fwd:+.2f} turn {turn:+.2f} | det {1000 * (t_det - t_cam):.0f} ms "
                      f"brain {1000 * (t_brain - t_det):.0f} ms total {total:.0f} ms | {brain.rates()}", flush=True)
            if a.show and frame is not None:
                vis = frame.copy()
                for x0, y0, x1, y1 in (detector(frame) if a.fake_box else []):
                    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                cv2.putText(vis, f"fwd {fwd:+.2f} turn {turn:+.2f} {total:.0f}ms{' LOBOTOMY' if brain.lobotomy else ''}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 113, 91), 2)
                cv2.imshow("flybrain", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("r"):
                    brain.pending_reward = 5.0
                elif k == ord("p"):
                    brain.pending_reward = -5.0
                elif k == ord("l"):
                    brain.lobotomy = not brain.lobotomy
                elif k == ord(" "):
                    send(0.0, 0.0, "E-STOP"); print("E-STOP", flush=True); break
                elif 32 <= k < 127 and chr(k) in hotkeys:
                    run_hotkey(chr(k))
            if a.max_frames and n >= a.max_frames:
                break
            if a.no_camera:
                time.sleep(max(0.0, fps_period - (time.time() - t_cam)))
    finally:
        if estop.is_set():
            print("E-STOP from terminal", flush=True)
        send(0.0, 0.0, "stop")
        if rover:
            rover.close()
        if cap is not None:
            cap.release()
        if eye is not None:
            eye.close()
        lat_f.close()
        if len(lats) > 12:
            lats = lats[5:]                    # drop startup frames (opening a stream can take seconds)
        if lats:
            lats.sort()
            print(f"latency camera->command over {len(lats)} frames: median {lats[len(lats) // 2]:.1f} ms, "
                  f"p95 {lats[int(0.95 * len(lats))]:.1f} ms, max {lats[-1]:.1f} ms "
                  f"({'OK' if lats[int(0.95 * len(lats))] < 100 else 'OVER'} the 100 ms budget) -> {a.latency_log}")


if __name__ == "__main__":
    main()
