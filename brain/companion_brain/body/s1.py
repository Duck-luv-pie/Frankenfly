"""The DJI RoboMaster S1 as the fly's legs: the brain's motor channels become S-Bus sticks on the
S1 motion controller (see docs/wiring.md, "RoboMaster S1"). No firmware hack, no SDK: the S1's
S-Bus port takes a standard receiver signal and the Raspberry Pi is the receiver.

S-Bus: 100000 baud, 8E2, inverted (the Pi's TX needs one inverter), 25 bytes every 14 ms:
0x0F, 16 x 11-bit channels LSB-first, flags, 0x00. S1 sticks are 1024 +/- 672 (user manual v1.8):
  1 strafe   2 forward/back   3 gimbal pitch   4 yaw (gimbal in chassis-lead, chassis in free mode)
  5 speed preset (3-pos)   6 chassis-lead (+) / free (-) mode   7 chassis set (+) / release (-)
Nothing comes back over S-Bus (no odometry, hits or video); the ESP32 body keeps eyes, sound, PIR.

This module imports nothing heavy so tools/s1_sbus.py can use it before the brain is installed.
"""
from __future__ import annotations

import math
import threading
import time

CENTER, SPAN = 1024, 672
LOW, HIGH = CENTER - SPAN, CENTER + SPAN
FRAME_S = 0.014
SPEEDS = {"slow": LOW, "medium": CENTER, "fast": HIGH}
DEFAULTS = {
    "port": "/dev/ttyAMA0",
    "stick_forward": 0.5,    # full forward drive = this fraction of stick throw (the S1 is fast; start low)
    "stick_strafe": 0.5,
    "stick_yaw": 1.0,        # full turn drive = this fraction of yaw stick; the S1 turns slowly per stick, so keep this high
    "free_mode": True,       # channel 4 turns the chassis (the hunt vehicle), not the gimbal
    "speed": "slow",         # S1 speed preset, channel 5
    "sign_forward": 1, "sign_strafe": 1, "sign_yaw": 1,   # flip after the bench test if an axis is mirrored
    "yaw_dps_full": 90.0,    # measured: degrees per second at full yaw stick (slow preset, 2026-09-19)
    "speed_mps_full": 0.85,  # measured: metres per second at full forward stick (slow preset)
    # False, because chase mode has to approach a person and not just spin on the spot. The two branches
    # disagreed here (s1-fly-brain False, sim-mirror-webapp True) and the merge kept False deliberately.
    # Nothing is lost: Mirror sets rotation_only = False on itself and on the body regardless
    # (hunt_gpu/mirror.py), so this default only governs the heading-only bench path. For a first
    # hardware run, prop the wheels off the ground rather than relying on this being True.
    "rotation_only": False,  # True: copy the fly's rotation only; forward and strafe stay centred
    "heading_gain": 3.0,     # heading tracking: yaw rate = gain x heading error (deg/s per deg), capped at yaw_dps_full
    "heading_deadband_deg": 2.0,
    "gimbal_pitch": 0.0,     # -1..1 held on channel 3
    "timeout_s": 0.2,        # failsafe: no per-tick packet for this long -> sticks centred (R7.1, 0.05..2.0)
    "release_asleep": True,  # the sleeping fly goes limp: chassis released on channel 7
}

TIMEOUT_MIN, TIMEOUT_MAX = 0.05, 2.0     # the failsafe timeout is configurable within this range (R7.1)


def encode(ch: list[int], failsafe: bool = False, lost: bool = False) -> bytes:
    """16 channels of 11 bits -> 25-byte S-Bus frame (missing channels centred)."""
    buf = bytearray(25)
    buf[0] = 0x0F
    bits = n = 0
    i = 1
    for v in (list(ch) + [CENTER] * 16)[:16]:
        bits |= (max(0, min(2047, int(v))) & 0x7FF) << n
        n += 11
        while n >= 8:
            buf[i] = bits & 0xFF
            i += 1
            bits >>= 8
            n -= 8
    buf[23] = (0x04 if lost else 0) | (0x08 if failsafe else 0)
    buf[24] = 0x00
    return bytes(buf)


def decode(frame: bytes) -> list[int]:
    """The inverse of `encode`, for tests and for reading a real receiver."""
    bits = int.from_bytes(frame[1:23], "little")
    return [(bits >> (11 * i)) & 0x7FF for i in range(16)]


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def stick(x: float) -> int:
    """-1..1 -> S1 stick value (1024 +/- 672)."""
    return int(round(CENTER + SPAN * max(-1.0, min(1.0, float(x)))))


def motor_to_sticks(motor: dict, cfg: dict, clip_axes: bool = True) -> dict:
    """The hunt vehicle's mapping (sim/hunt_arena.py) as stick deflections in -1..1:
    forward = forward - backward, yaw = the turn channel (DNa right minus left, +ve = right).
    With `clip_axes=False` the final axes are returned unclipped, so a caller can see whether a
    command falls outside the S1 stick range before it is squeezed back in (S1Body.send, R7.4)."""
    gf, gs, gy = (float(cfg.get(k, DEFAULTS[k])) for k in ("stick_forward", "stick_strafe", "stick_yaw"))
    clip = lambda x: max(-1.0, min(1.0, x))
    # physical units when the sender knows them (the hunt viewer: the fly's real yaw rate and speed) ...
    if "yaw_dps" in motor:
        yaw = float(motor["yaw_dps"]) / float(cfg.get("yaw_dps_full", DEFAULTS["yaw_dps_full"]))
    else:                                                    # ... else the brain's drive channels through the stick gains
        yaw = gy * float(motor.get("turn", 0.0))
    if "speed_mps" in motor:
        fwd = float(motor["speed_mps"]) / float(cfg.get("speed_mps_full", DEFAULTS["speed_mps_full"]))
    else:
        fwd = gf * clip(float(motor.get("forward", 0.0)) - float(motor.get("backward", 0.0)))
    strafe = gs * float(motor.get("strafe", 0.0))
    if bool(cfg.get("rotation_only", DEFAULTS["rotation_only"])):
        fwd = strafe = 0.0
    axis = clip if clip_axes else (lambda x: x)
    return {
        "forward": axis(float(cfg.get("sign_forward", 1)) * fwd),
        "strafe": axis(float(cfg.get("sign_strafe", 1)) * strafe),
        "yaw": axis(float(cfg.get("sign_yaw", 1)) * yaw),
    }


def channels(sticks: dict, cfg: dict, released: bool = False) -> list[int]:
    return [
        stick(sticks.get("strafe", 0.0)),
        stick(sticks.get("forward", 0.0)),
        stick(float(cfg.get("gimbal_pitch", 0.0))),
        stick(sticks.get("yaw", 0.0)),
        SPEEDS[str(cfg.get("speed", "slow"))],
        LOW if bool(cfg.get("free_mode", True)) else HIGH,
        LOW if released else HIGH,
    ]


def open_serial(port: str):
    try:
        import serial
    except ImportError as e:
        raise SystemExit("S1 driver needs pyserial: uv sync --extra s1  (or pip install pyserial)") from e
    return serial.Serial(port, baudrate=100000, bytesize=8, parity=serial.PARITY_EVEN, stopbits=2, write_timeout=0.05)


class S1Body:
    """A body for the brain loop (same duck type as BodyLink): `send(packet)` takes the motor
    channels, a background thread streams S-Bus frames at ~70 Hz, sticks centre if packets stop."""

    def __init__(self, cfg: dict | None = None, ser=None, port: str | None = None, start: bool = True, udp: bool = False):
        """`udp=True`: no serial port here; `send()` adds the S-Bus channels to the brain packet ("s1": {"ch": [...]})
        and the body ESP32 (firmware/body, GPIO 4) streams them to the S1. Put this body first in a MultiBody so the
        BodyLink after it sends the annotated packet."""
        self.cfg = {**DEFAULTS, **(dict(cfg) if cfg else {})}
        if port:
            self.cfg["port"] = port
        self.udp = udp
        # the failsafe timeout is clamped to its valid range (R7.1: 0.05..2.0 s)
        self.cfg["timeout_s"] = max(TIMEOUT_MIN, min(TIMEOUT_MAX, float(self.cfg.get("timeout_s", DEFAULTS["timeout_s"]))))
        if udp:
            self.ser = None       # the body ESP32 owns the port; nothing to open here
        else:
            self.ser = ser if ser is not None else open_serial(str(self.cfg["port"]))
        self.sticks = {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}     # the last neutral-safe state (kept on a rejected packet, R7.4)
        self.released = False
        self.last_packet = 0.0
        self.rejects = 0                   # out-of-range packets rejected by send (R7.4)
        self.saturations = 0               # physical-unit packets asking for more than the S1 can do
        self.last_valid = True             # False after the most recent send was rejected
        self.link_ok = True                # the last S-Bus write succeeded (an S1 link check for the E-stop, R6.5)
        # heading tracking (the hunt viewer sends the fly's absolute heading): the rover's heading is estimated from what
        # it was told (no odometry comes back), and it steers toward the fly's heading. A wobbling fly nets to nothing;
        # a real turn is followed to the degree, even when the fly out-turns the rover for a while.
        self.target_heading: float | None = None
        self.est_heading: float | None = None
        self.episode = None
        self.frames = 0
        self.pir = 0            # BodyLink attributes the loop reads; S-Bus carries nothing back
        self.busy = 0
        self.last_rx = 0.0
        self.addr = ("body-udp" if udp else str(self.cfg["port"]), 0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True, name="s1-sbus")
        if start and not udp:
            self._t.start()

    # --- the brain loop's interface
    def send(self, packet: dict) -> bool:
        """Update the target sticks from a motor packet. Returns True when the packet is accepted,
        False when it is rejected for being out of the valid stick range (R7.4): a rejected packet
        leaves the last neutral-safe sticks untouched and raises the validation-failure signal."""
        motor = packet.get("motor") or {}
        # R7.4: reject a command whose stick axes fall outside [-1, 1] (or are non-finite) before it is
        # clamped in, keeping the last good state, so a bad per-tick command cannot move the robot.
        #
        # But only on the unitless drive path. The two branches that met here had different producers:
        # Mirror pre-clamps and converts back through speed_mps_full / yaw_dps_full, so it never exceeds
        # 1.0 and the reject is a no-op for it. hunt_gpu/real.py deliberately sends the fly's *would-be*
        # physical motion, up to 1.6 m/s and 200 deg/s against an S1 calibrated at 0.85 and 90, and has
        # always relied on motor_to_sticks saturating -- configs/default.yaml calls that "the robot
        # traces the fly's path at ~45% of its pace". Rejecting those packets stops the chase demo dead
        # the moment the fly commits to a turn, and silently: send()'s bool has no caller, so the only
        # symptom is the failsafe centring the sticks half a second later.
        #
        # So: a physical-unit packet saturates, as it always did. A unitless one is still rejected, and
        # a non-finite value is rejected either way.
        raw = motor_to_sticks(motor, self.cfg, clip_axes=False)
        physical = ("speed_mps" in motor) or ("yaw_dps" in motor)
        if not all(math.isfinite(v) for v in raw.values()):
            with self._lock:
                self.rejects += 1
                self.last_valid = False
            print(f"[s1] rejected non-finite stick command {raw}, holding last safe state", flush=True)
            return False
        if not all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in raw.values()):
            if not physical:
                with self._lock:
                    self.rejects += 1
                    self.last_valid = False
                print(f"[s1] rejected out-of-range stick command {raw}, holding last safe state", flush=True)
                return False
            self.saturations += 1
            if self.saturations == 1 or self.saturations % 200 == 0:
                print(f"[s1] saturating: the fly asked for more than the S1 can do {raw} "
                      f"({self.saturations} so far)", flush=True)
        st = motor_to_sticks(motor, self.cfg)
        asleep = packet.get("state") == "sleep" and bool(self.cfg.get("release_asleep", True))
        with self._lock:
            self.last_valid = True
            if "heading_deg" in motor:
                h = float(motor["heading_deg"])
                ep = packet.get("episode")
                if self.est_heading is None or ep != self.episode:     # first packet / a new episode: the fly teleports, the rover is "aligned" with it
                    self.est_heading, self.episode = h, ep
                self.target_heading = h
            self.sticks = st
            self.released = asleep
            self.last_packet = time.monotonic()
        if self.udp:                                              # the body ESP32 does the S-Bus: channels ride in the packet
            packet["s1"] = {"ch": channels(st, self.cfg, asleep)}
            self.frames += 1
        return True

    def poll(self) -> dict | None:
        return None

    def status(self) -> dict:
        """What the robot is being told right now (for the viewer): sticks in -1..1 and the yaw rate they ask for."""
        with self._lock:
            st = dict(self.sticks)
        full = float(self.cfg.get("yaw_dps_full", DEFAULTS["yaw_dps_full"]))
        out = {"sticks": {k: round(v, 3) for k, v in st.items()}, "yaw_dps": round(st["yaw"] * full, 1),
               "saturated": abs(st["yaw"]) >= 0.999,
               "rotation_only": bool(self.cfg.get("rotation_only", DEFAULTS["rotation_only"])), "frames": self.frames,
               "valid": self.last_valid, "rejects": self.rejects, "saturations": self.saturations,
               "link_ok": self.link_ok}
        if self.est_heading is not None and self.target_heading is not None:
            out["heading_deg"] = round(self.est_heading, 1)
            out["error_deg"] = round(wrap_deg(self.target_heading - self.est_heading), 1)
        return out

    def close(self) -> None:
        with self._lock:
            self.sticks = {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}
        if self.udp:
            return
        if self._t.is_alive():
            time.sleep(5 * FRAME_S)          # a few centred frames before the signal disappears
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=0.5)
        try:
            self.ser.close()
        except Exception:
            pass

    # --- the S-Bus side
    def frame(self, now: float | None = None) -> bytes:
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = (now - self.last_packet) > float(self.cfg["timeout_s"])
            st = {"forward": 0.0, "strafe": 0.0, "yaw": 0.0} if stale else dict(self.sticks)
            if not stale and self.target_heading is not None:
                full = float(self.cfg.get("yaw_dps_full", DEFAULTS["yaw_dps_full"]))
                err = wrap_deg(self.target_heading - self.est_heading)
                cmd = 0.0 if abs(err) < float(self.cfg.get("heading_deadband_deg", 2.0)) else max(-full, min(full, float(self.cfg.get("heading_gain", 3.0)) * err))
                self.est_heading = (self.est_heading + cmd * FRAME_S) % 360.0    # the rover is assumed to do what it is told
                st["yaw"] = max(-1.0, min(1.0, float(self.cfg.get("sign_yaw", 1)) * cmd / full))
                self.sticks["yaw"] = st["yaw"]
            return encode(channels(st, self.cfg, self.released), lost=stale)

    def _loop(self) -> None:
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.ser.write(self.frame())
                self.frames += 1
                self.link_ok = True
            except Exception:
                self.link_ok = False        # the S-Bus write failed: the halt cannot be confirmed (R6.5)
            nxt += FRAME_S
            time.sleep(max(0.0, nxt - time.perf_counter()))


class MultiBody:
    """Several bodies as one: the ESP32 (eyes, sound, PIR) and the S1 (legs) both get every packet."""

    def __init__(self, *bodies):
        self.bodies = [b for b in bodies if b is not None]
        self.addr = self.bodies[0].addr if self.bodies else None

    @property
    def cfg(self) -> dict | None:
        """The S1's config, so callers that configure the body reach the real one.

        Mirror does `if getattr(body, "cfg", None) is not None: body.cfg["rotation_only"] = False; ...`
        to push its calibration down. Without this property a MultiBody answered None, Mirror silently
        skipped the whole block, and --s1 udp / --s1 usb ran on DEFAULTS with v_max, w_max and sign_yaw
        never reaching the S1 at all. It failed quietly, which is the worst way for a calibration to
        fail: the robot moves, just not at the speed or in the direction you asked for.
        """
        for b in self.bodies:
            c = getattr(b, "cfg", None)
            if c is not None:
                return c
        return None

    @property
    def pir(self) -> int:
        return max((int(getattr(b, "pir", 0)) for b in self.bodies), default=0)

    @property
    def busy(self) -> int:
        return max((int(getattr(b, "busy", 0)) for b in self.bodies), default=0)

    @property
    def last_rx(self) -> float:
        return max((float(getattr(b, "last_rx", 0.0)) for b in self.bodies), default=0.0)

    def status(self) -> dict | None:
        for b in self.bodies:
            if hasattr(b, "status"):
                return b.status()
        return None

    def send(self, packet: dict) -> bool:
        """Every body gets the packet. False if any of them rejected it, so a caller that checks the
        bool (S1Body.send's R7.4 contract) is not told True when the S1 in fact refused."""
        ok = True
        for b in self.bodies:
            if b.send(packet) is False:
                ok = False
        return ok

    def poll(self) -> dict | None:
        last = None
        for b in self.bodies:
            r = b.poll()
            last = r if r is not None else last
        return last

    def close(self) -> None:
        for b in self.bodies:
            b.close()
