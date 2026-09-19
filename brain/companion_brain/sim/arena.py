"""Headless two-odor arena: the dashboard's choice assay without a browser, so many flies can
be trained and tested in parallel. The kinematics are a port of the world in ui/fly3d.js:
motor channels from the brain set walking speed and turning, learned MBON valence steers toward
or away from the stronger-smelling antenna, and walls turn the fly back. Grape scent (odor A)
comes from the left end, lemon scent (odor B) from the right; during training the fly is confined
to the grape half, which is coated with sugar."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

AX, AZ = 12.0, 4.5
ODOR_SCALE = 12.0
CONFINE_X = -5.0


@dataclass
class ArenaState:
    x: float = 0.0
    z: float = 0.0
    heading: float = math.pi / 2
    satiety: float = 0.0
    odor_a: tuple = (0.0, 0.0)
    odor_b: tuple = (0.0, 0.0)
    sugar: float = 0.0
    feeding: float = 0.0
    log: list = field(default_factory=list)


class Arena:
    def __init__(self, valence_steering: float = 1.0, seed: int = 0):
        self.s = ArenaState()
        self.vs = valence_steering
        self.rng = np.random.default_rng(seed)
        self.s.heading = math.pi / 2 * (1 if self.rng.random() < 0.5 else -1)
        self.last_val = None
        self.turn_bias = 1.0

    # ---- senses ----------------------------------------------------------------------------
    def _antenna(self, side: float) -> tuple[float, float]:
        th = self.s.heading
        lx, lz = side * 0.12, 0.75
        return (self.s.x + lx * math.cos(th) + lz * math.sin(th), self.s.z - lx * math.sin(th) + lz * math.cos(th))

    @staticmethod
    def _conc(px: float, pz: float, sx: float) -> float:
        d = math.hypot(px - sx, pz)
        return 1.0 / (1.0 + (d / ODOR_SCALE) ** 2)

    def sense(self, training: bool) -> None:
        s = self.s
        hunger = 1.0 - s.satiety
        (lx, lz), (rx, rz) = self._antenna(1), self._antenna(-1)
        s.odor_a = (min(1.0, self._conc(lx, lz, -AX) * hunger), min(1.0, self._conc(rx, rz, -AX) * hunger))
        s.odor_b = (min(1.0, self._conc(lx, lz, AX) * hunger), min(1.0, self._conc(rx, rz, AX) * hunger))
        s.sugar = hunger if (training and s.x <= CONFINE_X) else 0.0

    # ---- movement ---------------------------------------------------------------------------
    def step(self, motor: dict, valence: float, dt: float, training: bool) -> None:
        s = self.s
        s.feeding = motor.get("feed", 0.0) if s.sugar > 0 else 0.0
        s.satiety = min(1.0, s.satiety + s.feeding * dt * 0.06)
        s.satiety = max(0.0, s.satiety - dt / 240.0)
        still = max(motor.get("freeze", 0), motor.get("groom", 0) * .8, motor.get("threat", 0) * .6, motor.get("song", 0) * .5, s.feeding)
        speed = (6.0 * motor.get("forward", 0) - 3.0 * motor.get("backward", 0)) * (1 - still)
        turn = 3.0 * motor.get("turn", 0) * (1 - still * .5)
        grad = (s.odor_a[1] - s.odor_a[0]) + (s.odor_b[1] - s.odor_b[0])
        turn += -self.vs * valence * grad * 6.0 * (1 - still)
        # klinotaxis on the learned value itself: the mushroom-body valence rises toward smells the fly has
        # learned to like. If it is falling, turn (with a persistent random bias); if it is rising, run.
        v = valence if self.last_val is None else self.last_val + 0.3 * (valence - self.last_val)
        dv = (v - (self.last_val if self.last_val is not None else v)) / max(dt, 1e-3)
        self.last_val = v
        trend = max(-1.0, min(1.0, dv * 4))
        if trend > 0.05:
            self.turn_bias = 1.0 if self.rng.random() < 0.5 else -1.0
        turn += self.vs * max(0.0, -trend) * 2.5 * self.turn_bias * (1 - still)
        speed += self.vs * (max(0.0, trend) * 3.0 + max(0.0, valence) * 1.5) * (1 - still)
        # walls: steer back toward the middle near the edge
        to_center = math.atan2(-s.x, -s.z)
        edge = max(abs(s.x) - (AX - 1.2), abs(s.z) - (AZ - 1.2))
        if edge > 0 and abs(speed) > 0.01:
            want = to_center if speed > 0 else to_center + math.pi
            d = math.atan2(math.sin(want - s.heading), math.cos(want - s.heading))
            turn += d * (2 + 6 * edge)
        s.heading = math.atan2(math.sin(s.heading + turn * dt), math.cos(s.heading + turn * dt))
        s.x += math.sin(s.heading) * speed * dt
        s.z += math.cos(s.heading) * speed * dt
        s.x = min(AX, max(-AX, s.x)); s.z = min(AZ, max(-AZ, s.z))
        if training and s.x > CONFINE_X:
            s.x = CONFINE_X
