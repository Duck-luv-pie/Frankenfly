"""The hunt arena: the fly as a ground vehicle in a room of walking humans.

A meter-scale port of the Fly / People Lab arena (fly project, `src/follow-world.js`): people walk
seeded patrol ellipses inside a 16 x 14 m room. The fly is a 32 cm vehicle steered only by the
brain's forward / backward / turn channels (an RC car: no walking, no flight). It touches a person
-> reward (sugar + reward dopamine, delivered while it stands still); it runs out of time ->
punishment (bitter + punishment dopamine). Everything the brain gets comes from two senses:

  heat    two heat sensors on the head looking 45 degrees left and right (cosine lobes), each
          summing every person's warmth 1 / (1 + (d / scale)^2) -> the arista's hot cells, per side
  vision  each person inside the 150 degree field of view is a dark moving object and a vertical
          bar on the hemifield it is seen in -> LC11 / LC10a and LC12 / LC15, the same optic-lobe
          features the retina produces, computed analytically here so thousands of episodes can run
          without rendering

The random layout uses the same mulberry32 generator as the JavaScript arena, so a training seed
reproduces the same people, patrols and start pose in the dashboard."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..senses.optic_lobe import Features

PEOPLE_X, PEOPLE_Z = 7.1, 5.8          # people are clamped to this box (as in follow-world.js)
ANCHOR_RX, ANCHOR_RZ = 1.14, 1.05      # patrol ellipse radii
MASK = 0xFFFFFFFF


def mulberry32(seed: int):
    """Bit-exact port of `seededRandom` in the fly project's human.js."""
    state = int(seed) & MASK

    def imul(a: int, b: int) -> int:
        return (a * b) & MASK

    def rnd() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & MASK
        t = state
        t = imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & MASK)) & MASK
        return ((t ^ (t >> 14)) & MASK) / 4294967296
    return rnd


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class Person:
    slot: int
    anchor_x: float
    anchor_z: float
    offset: float
    pace: float
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0


@dataclass
class Senses:
    heat: tuple = (0.0, 0.0)          # left, right, 0..1
    feats: Features = field(default_factory=Features)
    sugar: float = 0.0                # reward being delivered
    bitter: float = 0.0               # punishment being delivered
    self_motion: float = 0.0
    nearest_m: float = 0.0
    nearest_bearing: float = 0.0      # radians, +ve = on the fly's left


class HuntArena:
    def __init__(self, hunt_cfg: dict, seed: int = 0):
        self.h = dict(hunt_cfg)
        h = self.h
        self.ax, self.az = float(h["arena"]["x"]), float(h["arena"]["z"])
        self.count = max(1, min(8, int(h.get("people", 4))))
        self.episode_s = float(h.get("episode_s", 45.0))
        self.reward_s, self.punish_s = float(h.get("reward_s", 2.0)), float(h.get("punish_s", 2.0))
        # contact: a person is a solid cylinder of radius person_r; a touch is the vehicle's nose (nose_m ahead of
        # its centre) within touch_m of that skin. Head-on contact (|bearing| < frontal_deg) earns the full reward,
        # a glancing one earns side_reward. Brushing past with the body does not count.
        self.person_r, self.nose, self.touch_m = float(h.get("person_r", 0.20)), float(h.get("nose_m", 0.16)), float(h.get("touch_m", 0.06))
        self.body_solid = bool(h.get("body_solid", False))    # the vehicle's body stops at the skin (nose_m from its centre), not a 10 cm buffer
        self.frontal = math.radians(float(h.get("frontal_deg", 30)))
        self.side_reward = float(h.get("side_reward", 0.5))
        self.freeze_brakes = bool(h.get("freeze_brakes", False))
        self.vmax, self.vrev, self.wmax = float(h.get("speed_max", 1.6)), float(h.get("speed_reverse", 0.6)), float(h.get("turn_max", 2.5))
        self.turn_gain = float(h.get("turn_gain", 1.0))
        # The decoded turn channel is DNa01/02 right minus left, and DNa02 drives an ipsilateral turn
        # (Rayshubskiy et al. 2020), so +ve = turn right. Heading here increases to the left: sign -1.
        self.turn_sign = float(h.get("turn_sign", -1.0))
        self.fov = math.radians(float(h.get("fov_deg", 95)))
        self.heat_scale, self.heat_lobe = float(h.get("heat_scale_m", 3.0)), math.radians(float(h.get("heat_lobe_deg", 45)))
        self.heat_gain = float(h.get("heat_gain", 1.0))
        self.valence_steer = float(h.get("valence_steer", 6.0))
        self.people: list[Person] = []
        self.reset(seed)

    # ---- episodes ------------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        self.seed = int(seed) & MASK
        rng = mulberry32(self.seed)
        for _ in range(7):                       # the JS arena shuffles 8 shirt colours first
            rng()
        self.people = []
        for i in range(self.count):
            p = Person(i, ((i % 4) - 1.5) * 3.25, (i // 4) * 4 - 1.8, rng() * math.pi * 2, 0.66 + rng() * 0.17)
            rng()                                # walk phase (animation only)
            self.people.append(p)
        rng()                                    # the JS arena's follow target
        self.t = 0.0
        self._place_people()
        for _ in range(20):                      # a random start pose, not on top of anyone
            self.x = (rng() * 2 - 1) * (self.ax - 1.0)
            self.z = (rng() * 2 - 1) * (self.az - 1.0)
            self.heading = rng() * math.pi * 2 - math.pi
            if self.nearest()[0] > 2.0:
                break
        self.speed = 0.0
        self.phase = "hunt"                      # hunt -> reward | punish -> done
        self.phase_t = 0.0
        self.done = False
        self.touched = False
        self.t_touch = None
        self.ticks = self.facing_ticks = 0
        self.min_dist = self.nearest()[0]
        self.turn_bias = 1.0
        self.last_val = None
        self.senses = Senses()
        self.path_len = 0.0
        self.start_dist = self.min_dist
        self.contact = None                      # (bearing_deg, frontal) of the touch
        self.reward_amount = 0.0

    def _place_people(self) -> None:
        for p in self.people:
            th = self.t * p.pace + p.offset
            ox, oz = p.x, p.z
            p.x = max(-PEOPLE_X, min(PEOPLE_X, p.anchor_x + ANCHOR_RX * math.cos(th)))
            p.z = max(-PEOPLE_Z, min(PEOPLE_Z, p.anchor_z + ANCHOR_RZ * math.sin(th)))
            p.yaw = math.atan2(-ANCHOR_RX * math.sin(th), ANCHOR_RZ * math.cos(th))
            p.speed = math.hypot(p.x - ox, p.z - oz)

    # ---- geometry --------------------------------------------------------------------------
    def bearing(self, px: float, pz: float) -> tuple[float, float]:
        """(distance, bearing) of a point; bearing is +ve on the fly's left (heading 0 faces +z, left is +x)."""
        dx, dz = px - self.x, pz - self.z
        fwd = dx * math.sin(self.heading) + dz * math.cos(self.heading)
        left = dx * math.cos(self.heading) - dz * math.sin(self.heading)
        return math.hypot(dx, dz), math.atan2(left, fwd)

    def nearest(self) -> tuple[float, float]:
        best = (1e9, 0.0)
        for p in self.people:
            d, b = self.bearing(p.x, p.z)
            if d < best[0]:
                best = (d, b)
        return best

    # ---- senses ----------------------------------------------------------------------------
    def sense(self) -> Senses:
        s = Senses()
        s.sugar = self.reward_amount if self.phase == "reward" else 0.0
        s.bitter = 1.0 if self.phase == "punish" else 0.0
        hl = hr = 0.0
        half = self.fov / 2
        best = [0.0, 0.0]
        for p in self.people:
            d, b = self.bearing(p.x, p.z)
            warmth = 1.0 / (1.0 + (d / self.heat_scale) ** 2)
            hl += warmth * max(0.0, math.cos(b - self.heat_lobe))
            hr += warmth * max(0.0, math.cos(b + self.heat_lobe))
            if abs(b) < half:
                width = 2 * math.atan2(0.25, max(d, 0.3))           # shoulders ~0.5 m across
                strength = min(1.0, width / math.radians(25))
                moving = 1.0 if (p.speed > 1e-3 or abs(self.speed) > 0.05) else 0.4
                hf = s.feats.left if b >= 0 else s.feats.right
                if strength > getattr(hf, "small_object"):
                    hf.small_object = strength
                    hf.object_x = (1 - 2 * b / half) if b >= 0 else (-1 - 2 * b / half)
                    hf.object_y = 0.0
                hf.bar = max(hf.bar, 0.8 * strength * moving)
                if strength > best[1]:
                    best = [-b / half, strength]
        s.heat = (min(1.0, hl * self.heat_gain), min(1.0, hr * self.heat_gain))
        s.feats.object_x, s.feats.object_strength = best[0], best[1]
        s.feats.motion_energy = min(1.0, s.feats.left.bar + s.feats.right.bar)
        s.self_motion = min(1.0, abs(self.speed) / self.vmax)
        s.nearest_m, s.nearest_bearing = self.nearest()
        self.senses = s
        return s

    # ---- movement --------------------------------------------------------------------------
    def step(self, motor: dict, valence: float, dt: float) -> None:
        if self.done:
            return
        self.t += dt
        self._place_people()
        s = self.senses
        if self.phase in ("reward", "punish"):
            self.phase_t += dt
            self.speed = 0.0
            if self.phase_t >= (self.reward_s if self.phase == "reward" else self.punish_s):
                self.done = True
            return
        self.ticks += 1
        if s.nearest_m < 1e8 and abs(s.nearest_bearing) < math.radians(30):
            self.facing_ticks += 1
        brake = (1.0 - float(motor.get("freeze", 0.0))) if self.freeze_brakes else 1.0
        speed = (self.vmax * float(motor.get("forward", 0.0)) - self.vrev * float(motor.get("backward", 0.0))) * brake
        turn = self.turn_sign * self.wmax * self.turn_gain * float(motor.get("turn", 0.0))   # heading rate, +ve = left
        # learned steering (mushroom-body valence, as in the other worlds): approach turns toward the warmer side
        vs = float(self.h.get("_valence_steering", 1.0))
        grad = s.heat[1] - s.heat[0]
        turn += -vs * valence * grad * self.valence_steer
        v = valence if self.last_val is None else self.last_val + 0.3 * (valence - self.last_val)
        dv = (v - (self.last_val if self.last_val is not None else v)) / max(dt, 1e-3)
        self.last_val = v
        trend = max(-1.0, min(1.0, dv * 4))
        if trend > 0.05:
            self.turn_bias = 1.0 if (self.ticks % 2) else -1.0
        turn += vs * max(0.0, -trend) * 2.5 * self.turn_bias
        speed += vs * (max(0.0, trend) * 0.5 + max(0.0, valence) * 0.3) * self.vmax * brake
        speed = max(-self.vrev, min(self.vmax, speed))
        # walls: as in the other worlds, the edge turns the vehicle back toward the middle
        to_center = math.atan2(-self.x, -self.z)
        edge = max(abs(self.x) - (self.ax - 1.2), abs(self.z) - (self.az - 1.2))
        if edge > 0 and abs(speed) > 0.01:
            want = to_center if speed > 0 else to_center + math.pi
            turn += wrap(want - self.heading) * (2 + 6 * edge)
        self.heading = wrap(self.heading + turn * dt)
        nx, nz = self.x + math.sin(self.heading) * speed * dt, self.z + math.cos(self.heading) * speed * dt
        nx, nz = max(-self.ax, min(self.ax, nx)), max(-self.az, min(self.az, nz))
        solid = self.person_r + (self.nose if self.body_solid else 0.10)    # people are solid: the vehicle stops at their skin
        for p in self.people:
            d = math.hypot(p.x - nx, p.z - nz)
            if d < solid and d > 1e-6:
                nx, nz = p.x + (nx - p.x) / d * solid, p.z + (nz - p.z) / d * solid
        self.path_len += math.hypot(nx - self.x, nz - self.z)
        self.x, self.z, self.speed = nx, nz, speed
        d, _ = self.nearest()
        self.min_dist = min(self.min_dist, d)
        nose_x, nose_z = self.x + math.sin(self.heading) * self.nose, self.z + math.cos(self.heading) * self.nose
        hit = None
        for p in self.people:
            if math.hypot(p.x - nose_x, p.z - nose_z) <= self.person_r + self.touch_m:
                _, b = self.bearing(p.x, p.z)
                if hit is None or abs(b) < abs(hit):
                    hit = b
        if hit is not None:
            frontal = abs(hit) < self.frontal
            self.contact = (round(math.degrees(hit), 1), frontal)
            self.reward_amount = 1.0 if frontal else self.side_reward
            self.touched, self.t_touch, self.phase, self.phase_t, self.speed = True, self.t, "reward", 0.0, 0.0
        elif self.t >= self.episode_s:
            self.phase, self.phase_t, self.speed = "punish", 0.0, 0.0

    def result(self) -> dict:
        return {"seed": self.seed, "touched": self.touched, "t_touch": None if self.t_touch is None else round(self.t_touch, 2),
                "frontal": bool(self.contact and self.contact[1]), "contact_deg": self.contact[0] if self.contact else None,
                "facing": round(self.facing_ticks / max(1, self.ticks), 3), "min_dist": round(self.min_dist, 2),
                "efficiency": round(min(1.0, self.start_dist / max(self.path_len, 1e-6)), 3) if self.touched else 0.0,
                "final_dist": round(self.nearest()[0], 2)}


def scripted_policy(s: Senses) -> dict:
    """A hand-written hunter in the brain's motor convention (turn +ve = right): turn toward the warmer
    side and the seen object, drive forward. Shows the arena is solvable and is the reference the
    brain is measured against."""
    turn = 3.0 * (s.heat[1] - s.heat[0])
    if s.feats.object_strength > 0:
        turn += 1.5 * s.feats.object_x * s.feats.object_strength
    return {"forward": 1.0, "backward": 0.0, "turn": max(-1.0, min(1.0, turn)), "freeze": 0.0}
