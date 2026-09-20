"""Turn readout-group firing rates into (a) continuous motor channels, (b) a behavior state label
and (c) modulators. Everything is a z-score against the brain's own resting activity
(see BrainRunner.calibrate), so a spontaneous, noisy brain still gives a quiet fly until
something really changes."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Decoded:
    state: str = "idle"
    scores: dict = field(default_factory=dict)      # behavior -> 0..1
    lateral: dict = field(default_factory=dict)     # behavior -> -1 (left) .. 1 (right)
    motor: dict = field(default_factory=dict)       # channel -> 0..1 (turn: -1..1), smoothed
    z: dict = field(default_factory=dict)           # group -> z-score of the current window (all sides)
    valence: float = 0.0
    arousal: float = 0.0
    reward: float = 0.0


def baseline_path(cfg, circuit_key: str) -> Path:
    key = hashlib.sha1(json.dumps({"c": circuit_key, "lif": dict(cfg.lif), "w": decode_windows(cfg)}, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return cfg.path("data.cache_dir") / f"baseline_{key}.json"


def decode_windows(cfg) -> list[float]:
    d = cfg.decode
    ws = {float(d.window_ms)}
    for spec in list(d.behaviors.values()) + list(d.motor.channels.values()) + list(d.modulators.values()):
        ws.add(float(spec.get("window_ms", d.window_ms)))
    return sorted(ws)


def load_or_calibrate(runner, cfg, circuit_key: str, rebuild: bool = False) -> dict:
    path = baseline_path(cfg, circuit_key)
    windows = decode_windows(cfg)
    if path.exists() and not rebuild:
        base = json.loads(path.read_text())
        if all(str(int(w)) in base and set(base[str(int(w))]) >= set(runner.readouts) for w in windows):
            print(f"[calibrate] loaded resting baseline {path.name}")
            return base
    print(f"[calibrate] measuring resting activity for {cfg.decode.calibrate_s} s of brain time ...")
    base = runner.calibrate(float(cfg.decode.calibrate_s), windows, warmup_s=float(cfg.decode.get("warmup_s", 2.0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=1))
    return base


class Decoder:
    """`baseline` is {window_ms: {group: {side: {mean, std}}}} from BrainRunner.calibrate. `decode`
    takes {window_ms: rates} for every window in `windows`."""

    def __init__(self, cfg, baseline: dict | None = None):
        self.cfg = cfg.decode
        self.baseline = baseline or {}
        self.windows = decode_windows(cfg)
        self.drift_tau = float(self.cfg.get("baseline_drift_s", 0.0))   # slow online re-centering of the resting mean
        self.min_std = float(self.cfg.get("min_std_hz", 1.0))
        self.z0 = float(self.cfg.get("z0", 1.0))
        self.hold_until: dict[str, float] = {}
        self.motor_state: dict[str, float] = {}
        self.last_t: float | None = None

    # ---- z-scores --------------------------------------------------------------------------
    def _win(self, spec: dict | None) -> str:
        return str(int(spec.get("window_ms", self.cfg.window_ms) if spec else self.cfg.window_ms))

    def z(self, rates_by_window: dict, group: str, side: str = "all", spec: dict | None = None) -> float:
        w = self._win(spec)
        rates = rates_by_window.get(w) or {}
        if group not in rates:
            return 0.0
        b = self.baseline.get(w, {}).get(group, {}).get(side, {"mean": 0.0, "std": 0.0})
        floor = float(spec.get("min_std_hz", self.min_std)) if spec else self.min_std   # big populations may use a lower floor
        return (rates[group][side] - b["mean"]) / max(b["std"], floor)

    def zgroup(self, rbw: dict, groups: list[str], side: str = "all", spec: dict | None = None) -> float:
        vals = [self.z(rbw, g, side, spec) for g in groups]
        return float(np.mean(vals)) if vals else 0.0

    def score(self, rbw: dict, spec: dict, side: str = "all") -> float:
        """0..1: z-score mapped through a dead zone (z0, default 1 sigma) up to z_ref."""
        z0 = float(spec.get("z0", self.z0))
        return float(np.clip((self.zgroup(rbw, spec["groups"], side, spec) - z0) / max(float(spec["z_ref"]) - z0, 1e-6), 0, 1))

    def lateral(self, rbw: dict, spec: dict) -> float:
        """-1..1: right minus left z-score, with the same dead zone. With `contrast` the difference is normalized by
        the two sides' activity (plus z_ref): a target straight ahead that drives both sides hard is a small turn, one
        side alone a full one, whatever the absolute rates; z0 is then a dead zone in contrast units (0..1)."""
        z0 = float(spec.get("z0", self.z0))
        r, l = self.zgroup(rbw, spec["groups"], "right", spec), self.zgroup(rbw, spec["groups"], "left", spec)
        if spec.get("contrast"):
            c = (r - l) / (max(r, 0.0) + max(l, 0.0) + max(float(spec["z_ref"]), 1e-6))
            mag = max(0.0, abs(c) - z0) / max(1.0 - z0, 1e-6)
            return float(np.clip(np.sign(c) * mag, -1, 1))
        d = r - l
        mag = max(0.0, abs(d) - z0) / max(float(spec["z_ref"]) - z0, 1e-6)
        return float(np.clip(np.sign(d) * mag, -1, 1))

    def drift(self, rates_by_window: dict, dt: float) -> None:
        """Slowly re-center resting means on what the brain is actually doing (homeostasis)."""
        if self.drift_tau <= 0 or dt <= 0:
            return
        k = min(1.0, dt / self.drift_tau)
        for w, groups in self.baseline.items():
            rates = rates_by_window.get(w) or {}
            for g, sides in groups.items():
                if g in rates:
                    for side, b in sides.items():
                        b["mean"] += k * (rates[g][side] - b["mean"])

    # ---- main ------------------------------------------------------------------------------
    def decode(self, rates: dict, asleep: bool = False, now: float | None = None) -> Decoded:
        """`rates` is {window_ms: rates_dict} (or a plain rates dict, used for every window)."""
        now = time.time() if now is None else now
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        def is_window_key(k):
            return isinstance(k, (int, float)) or str(k).replace(".", "", 1).isdigit()
        if rates and not all(is_window_key(k) for k in rates):
            rates = {str(int(w)): rates for w in self.windows}          # plain rates dict: use for every window
        rates = {str(int(float(k))): v for k, v in rates.items()}
        self.drift(rates, dt)
        d = Decoded()
        d.z = {g: round(self.z(rates, g), 2) for g in rates[self._win(None)]}

        # continuous motor channels, exponentially smoothed
        alpha = 1.0 - np.exp(-dt / (float(self.cfg.motor.smoothing_ms) * 1e-3)) if dt > 0 else 1.0
        for name, spec in self.cfg.motor.channels.items():
            raw = self.lateral(rates, spec) if spec.get("lateral") else self.score(rates, spec)
            prev = self.motor_state.get(name, 0.0)
            self.motor_state[name] = prev + alpha * (raw - prev)
        d.motor = {k: round(v, 3) for k, v in self.motor_state.items()}

        # discrete state label with priority + hold
        best, best_pri = None, -1
        for name, spec in self.cfg.behaviors.items():
            s = self.score(rates, spec)
            d.scores[name] = s
            d.lateral[name] = self.lateral(rates, spec)
            if s >= float(spec["min_score"]):
                self.hold_until[name] = now + float(spec["hold_ms"]) * 1e-3
            if self.hold_until.get(name, 0.0) > now and int(spec["priority"]) > best_pri:
                best, best_pri = name, int(spec["priority"])
        d.state = best or ("sleep" if asleep else "idle")

        m = self.cfg.modulators
        d.valence = self._bipolar(rates, m["valence"])
        d.arousal = float(np.clip(self.z(rates, m["arousal"]["plus"], "all", m["arousal"]) / float(m["arousal"]["z_ref"]), 0, 1))
        d.reward = self._bipolar(rates, m["reward"])
        return d

    def _bipolar(self, rates: dict, spec: dict) -> float:
        plus = self.z(rates, spec["plus"], "all", spec) if "plus" in spec else 0.0
        minus = self.z(rates, spec["minus"], "all", spec) if "minus" in spec else 0.0
        return float(np.clip((plus - minus) / float(spec["z_ref"]), -1, 1))
