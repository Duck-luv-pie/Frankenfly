"""The Mirror: the simulated fly's per-tick drive becomes S-Bus stick commands on the physical
RoboMaster S1 (see the sim-mirror-webapp design, "Mirror"). It is deliberately brain-agnostic --
the scripted hunter, a trained HunterNet and the spiking connectome all emit the same
drive = [forward, turn] in [-1, 1], and the Mirror maps them to sticks identically (R4.5).

Scaling and clamping reuse body/s1.py's `motor_to_sticks`, `stick` and `DEFAULTS` as the single
source of truth for the calibrated drive constants (stick_forward = 0.5, yaw_dps_full = 90 deg/s,
speed_mps_full = 0.85 m/s) so the robot and the sim agree by construction (R5.1). The Mirror caps
the forward and yaw sticks to the configured speed / turn limits, final-clamps to the S1 stick range
of [-1, 1], and hands the result to S1Body; a send failure is surfaced but never stops the sim (R4.6).
"""
from __future__ import annotations

import math

from ..body.s1 import DEFAULTS, motor_to_sticks, stick


def _finite(x) -> float:
    """A drive component the arithmetic can trust: NaN / inf collapse to 0 (neutral)."""
    x = float(x)
    return x if math.isfinite(x) else 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _sign_yaw(v) -> int:
    """`sign_yaw` from the config: 'normal' / 1 keep the yaw sense, 'inverted' / -1 reverse it (R5.4)."""
    if isinstance(v, str):
        return -1 if v.strip().lower() == "inverted" else 1
    return -1 if float(v) < 0 else 1


class Mirror:
    """Pure, brain-agnostic conversion from a drive `[forward, turn]` to S-Bus sticks, then hand-off to
    an S1Body. Construct with the S1 body (or None when no port is configured) and the optional speed /
    turn / yaw-sign limits; call `send(forward, turn)` each tick."""

    def __init__(self, body=None, cfg: dict | None = None, v_max=None, w_max=None, sign_yaw=None):
        self.body = body
        base = dict(cfg) if cfg else (dict(body.cfg) if body is not None and getattr(body, "cfg", None) else {})
        self.cfg = {**DEFAULTS, **base}
        # the Mirror copies the fly's forward *and* turn (unlike the heading-only bench default)
        self.cfg["rotation_only"] = False
        self.speed_mps_full = float(self.cfg.get("speed_mps_full", DEFAULTS["speed_mps_full"]))
        self.yaw_dps_full = float(self.cfg.get("yaw_dps_full", DEFAULTS["yaw_dps_full"]))
        # yaw sign lives in the Mirror's scaling; the body reproduces the final stick unchanged (sign_yaw = 1 below)
        self.cfg["sign_yaw"] = _sign_yaw(sign_yaw) if sign_yaw is not None else int(self.cfg.get("sign_yaw", 1))
        self.cfg["sign_forward"] = int(self.cfg.get("sign_forward", 1))
        # speed / turn caps, clamped to their valid ranges (R5.2: 0..0.85 m/s, R5.3: 0..90 deg/s)
        self.v_max_mps = _clamp(_finite(v_max), 0.0, self.speed_mps_full) if v_max is not None else self.speed_mps_full
        self.w_max_dps = _clamp(_finite(w_max), 0.0, self.yaw_dps_full) if w_max is not None else self.yaw_dps_full
        # the body streams exactly the sticks the Mirror computes: sign and gains already applied here (below)
        if body is not None and getattr(body, "cfg", None) is not None:
            body.cfg["rotation_only"] = False
            body.cfg["sign_forward"] = 1
            body.cfg["sign_yaw"] = 1
            body.cfg["speed_mps_full"] = self.speed_mps_full
            body.cfg["yaw_dps_full"] = self.yaw_dps_full

    def sticks(self, forward, turn) -> dict:
        """The drive `[forward, turn]` as final S1 stick deflections in [-1, 1]. Out-of-range components
        clamp to the nearest boundary before mapping (R4.4); `[0, 0]` maps to the neutral centre (R4.3);
        the forward / yaw sticks are capped to the speed / turn limits then final-clamped (R5.5-5.7)."""
        f = _clamp(_finite(forward), -1.0, 1.0)                    # R4.4
        t = _clamp(_finite(turn), -1.0, 1.0)
        # motor_to_sticks applies the calibrated stick_forward / stick_yaw gains and the yaw sign (R5.1)
        base = motor_to_sticks({"forward": max(0.0, f), "backward": max(0.0, -f), "turn": t}, self.cfg)
        fs, ys = base["forward"], base["yaw"]
        # cap in physical units: full stick = speed_mps_full / yaw_dps_full, so the cap is a stick fraction (R5.5, R5.6)
        max_fs = (self.v_max_mps / self.speed_mps_full) if self.speed_mps_full else 0.0
        max_ys = (self.w_max_dps / self.yaw_dps_full) if self.yaw_dps_full else 0.0
        fs = _clamp(fs, -max_fs, max_fs)
        ys = _clamp(ys, -max_ys, max_ys)
        # final clamp to the S1 stick range (R4.4, R5.7)
        return {"forward": _clamp(fs, -1.0, 1.0), "strafe": 0.0, "yaw": _clamp(ys, -1.0, 1.0)}

    def channels(self, forward, turn) -> dict:
        """The same sticks as integer S-Bus channel values (1024 +/- 672), via `stick` -- for tests / logging."""
        return {k: stick(v) for k, v in self.sticks(forward, turn).items()}

    def send(self, forward, turn, centered: bool = False) -> bool:
        """Convert `[forward, turn]` and send the sticks to the S1 through S1Body before the next tick (R4.2).
        `centered=True` sends the neutral centre regardless of the drive (E-stop / paused / failsafe, R6.2/R7.3).
        With no S1 configured nothing is sent (R4.7). A send failure is logged and swallowed so the sim keeps
        running (R4.6). Returns True when a command was sent, False otherwise."""
        if self.body is None:                                      # R4.7: no port -> no S-Bus commands
            return False
        if centered:
            packet = {"motor": {"speed_mps": 0.0, "yaw_dps": 0.0}}
        else:
            s = self.sticks(forward, turn)
            # send the final sticks as physical units; the body's motor_to_sticks reproduces them 1:1 (sign / gains
            # already applied here, so the body is configured with sign = 1 and rotation_only = False)
            packet = {"motor": {"speed_mps": s["forward"] * self.speed_mps_full,
                                "yaw_dps": s["yaw"] * self.yaw_dps_full}}
        try:
            self.body.send(packet)                                 # R4.2
            return True
        except Exception as e:                                     # R4.6: surface the failure, keep running
            print(f"[mirror] S1 send failed, continuing: {e}", flush=True)
            return False
