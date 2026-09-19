"""The hunt arena as a batch of tensors: B independent rooms stepped together on one device.

A vectorized port of `sim/hunt_arena.py` with the same physics, the same seeded layouts (mulberry32,
so a seed shows the same people and start pose on the dashboard) and the same motor convention: the
fly is an RC car (no walking, no flight) driven by a forward drive in [-1, 1] (negative = reverse,
usually disabled) and a turn drive in [-1, 1] (+ve = right, as decoded from the descending neurons).
The 2 s sugar / bitter phases of the connectome arena are not needed here, the reward is a number.
By default (`track.enabled`) an episode runs the whole clock: the first touch locks the fly onto
that person and from then on it is paid for *tracking* them as they walk away, staying close and
keeping them ahead, with a bonus for every renewed contact. With tracking off the episode ends at
the first touch, as on the CPU.

The humans (`hunt_gpu.humans`) come in four kinds: `patrol` walkers on the seeded ellipses of the CPU
arena, `wander` (people going about their business: strolling between random waypoints, pausing now and
then, never minding the fly), a scripted `flee` reflex (run straight away from the fly, keep off the walls) and `learn`, an
adversary: every person is a runner driven by its own policy (`EvaderNet`, one network shared by all
people) that is trained by PPO against the fly, paid for keeping its distance and punished when caught
or tracked. Runners are a touch slower than the fly (`humans.speed_max` 1.5 m/s vs 1.6), so a fly that
keeps its line catches them, and the humans have to learn to use the room. Runners have stamina (`humans.stamina`):
sprinting drains it in `sprint_s` seconds, and an exhausted runner must stop and rest until it has
recovered (`rest_s`), so the fly that keeps tracking gets its chance to catch up. The rest is part of
what the runner senses, so a learned runner can plan its sprints. On top of that every runner is made
to stop now and then (`humans.pause`: every ~8 s for 2.5 s) whatever its policy says, the "player
stops and lets the fly catch up" that gives the fly its tracking practice. Runners only mind the fly
when it is close (`humans.alarm_m`, with hysteresis to `calm_m`); otherwise they stroll between random
waypoints of the room, so they do not end up huddled in the corners furthest from the fly.

The fly's senses have a range (`hunt_gpu.senses`): a person is seen only within `see_m` and felt only
within `heat_m`. When nobody is in range the fly is paid for covering new ground (`reward.explore`,
per new 1 m cell of the room) and charged for standing still (`reward.idle`): it searches, it does
not freeze. The progress and facing rewards apply only to a person it can sense.

What the fly senses (the observation vector, `Observation`):

  heat      two hot-cell sensors on the head looking `heat_lobe_deg` left and right (cosine lobes),
            each summing every person's warmth 1 / (1 + (d / heat_scale)^2); or, with `senses.pir`, an
            HC-SR501 PIR: one bit that is high for `hold_s` after a warm body moved inside its cone
  vision    `vision_bins` retinotopic columns across the field of view, like the LC columns of the
            optic lobe. Two channels per column: the strength of a small dark object (its angular
            width) and its motion (bar). A person covers the columns spanned by its shoulders.
  body      speed / speed_max, the previous forward and turn drives, and the fraction of the
            episode elapsed (urgency)

The reward ("the faster it picks a target and hits it, the better"): a touch pays `touch` (x
`side_reward` for a glancing contact) plus `fast` x the fraction of the episode still left; every
step pays `progress` x the metres closed on the chosen target minus `time` x dt; a timeout costs
`timeout`. The chosen target is the person nearest at the start; it changes only when another is
`switch_margin_m` closer, and each switch costs `switch`. After the first touch the target is locked:
every tick with the person's skin within `track.follow_m` of the nose and within `track.follow_deg`
of straight ahead pays `track.reward`; letting them get further than `track.lost_m` away costs
`track.lost` per tick; touching them again pays `track.retouch` (at most once per `track.cooldown_s`); and every
locked tick pays `track.facing` x cos(bearing to them), so keeping them in front is paid and losing them behind costs.
Sustained physical contact is the point: every tick the nose touches the locked person head-on pays `track.contact`,
growing to `track.contact_max` x that over `track.contact_ramp_s` of unbroken contact (results: `contact`, the
fraction of the ticks after the first touch spent touching, and `contact_s`, the longest unbroken contact). Trailing
is paid the same way: with `track.ramp_s` the per-tick tracking reward grows to `track.ramp_max` x over that long of
unbroken tracking (`track_s` in the results is the longest unbroken trail). Losing
them right after a touch (behind `track.lose_deg` or out of the senses within `track.hold_s` of the last touch,
without touching again) costs `track.lose` once per touch (results: `spins`).
With `track.start_locked` > 0 that fraction of the episodes begins at the first touch (the fly standing nose
to skin, already locked on), so a trainer can practise the tracking without hunting first; those episodes
are marked `started_locked` in the results and left out of the first-touch statistics.
The CPU arena's hunting `score` is computed from the same per-episode statistics (first touch) so the
two trainers can be compared."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..sim.hunt_arena import ANCHOR_RX, ANCHOR_RZ, MASK, PEOPLE_X, PEOPLE_Z, mulberry32

MAX_PEOPLE = 8


def pick_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class ObsSpec:
    bins: int

    @property
    def size(self) -> int:
        return 2 + 2 * self.bins + 4

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> heat [..., 2], vision [..., bins, 2], body [..., 4]"""
        heat = obs[..., :2]
        vis = obs[..., 2:2 + 2 * self.bins].reshape(*obs.shape[:-1], self.bins, 2)
        return heat, vis, obs[..., 2 + 2 * self.bins:]


@dataclass(frozen=True)
class HumanObs:
    """A runner's observation, in its own frame: the fly (distance / 10, sin and cos of its bearing, its speed,
    sin and cos of its heading relative to mine), my speed, the four walls (distance / 5, capped), my own yaw
    (sin, cos), the nearest other person (distance / 10, sin, cos), whether the fly is locked on me, and the
    fraction of the episode elapsed, my stamina (0..1) and whether I am resting (exhausted, forced to stand)."""
    size: int = 20


def layout(seed: int, count: int) -> dict:
    """The seeded layout, in the exact random-number order of the CPU arena (and the JavaScript one):
    people (anchor, patrol offset, pace), then a start pose not within 2 m of anyone."""
    rng = mulberry32(int(seed) & MASK)
    for _ in range(7):
        rng()
    people = []
    for i in range(count):
        ax, az = ((i % 4) - 1.5) * 3.25, (i // 4) * 4 - 1.8
        offset, pace = rng() * math.pi * 2, 0.66 + rng() * 0.17
        rng()
        people.append((ax, az, offset, pace))
    rng()
    px = [max(-PEOPLE_X, min(PEOPLE_X, ax + ANCHOR_RX * math.cos(off))) for ax, az, off, pace in people]
    pz = [max(-PEOPLE_Z, min(PEOPLE_Z, az + ANCHOR_RZ * math.sin(off))) for ax, az, off, pace in people]
    return {"people": people, "px": px, "pz": pz, "rng": rng}


class BatchArena:
    def __init__(self, hunt_cfg: dict, gpu_cfg: dict, n_envs: int, device: torch.device | str = "cpu", seed: int = 0):
        h, g = dict(hunt_cfg), dict(gpu_cfg)
        self.h, self.g = h, g
        self.B, self.device = int(n_envs), torch.device(device)
        self.dt = float(h.get("tick_s", 0.05))
        self.ax, self.az = float(h["arena"]["x"]), float(h["arena"]["z"])
        self.people_min = max(1, min(MAX_PEOPLE, int(g.get("people_min", h.get("people", 4)))))
        self.people_max = max(self.people_min, min(MAX_PEOPLE, int(g.get("people_max", h.get("people", 4)))))
        self.episode_s = float(h.get("episode_s", 45.0))
        self.max_steps = int(round(self.episode_s / self.dt))
        self.person_r, self.nose, self.touch_m = float(h.get("person_r", 0.20)), float(h.get("nose_m", 0.16)), float(h.get("touch_m", 0.06))
        self.body_solid = bool(h.get("body_solid", False))    # the vehicle's body stops at the skin (its nose can just touch), not a 10 cm buffer
        self.frontal = math.radians(float(h.get("frontal_deg", 30)))
        self.side_reward = float(h.get("side_reward", 0.5))
        self.vmax, self.vrev, self.wmax = float(h.get("speed_max", 1.6)), float(h.get("speed_reverse", 0.0)), float(h.get("turn_max", 2.5))
        self.turn_gain, self.turn_sign = float(h.get("turn_gain", 1.0)), float(h.get("turn_sign", -1.0))
        self.fov = math.radians(float(h.get("fov_deg", 150)))
        self.heat_scale, self.heat_lobe = float(h.get("heat_scale_m", 3.0)), math.radians(float(h.get("heat_lobe_deg", 45)))
        self.heat_gain = float(h.get("heat_gain", 1.0))
        self.spec = ObsSpec(int(g.get("vision_bins", 24)))
        self.obs_noise = float(g.get("obs_noise", 0.0))
        r = dict(g.get("reward", {}))
        self.r_touch, self.r_fast, self.r_timeout = float(r.get("touch", 10.0)), float(r.get("fast", 10.0)), float(r.get("timeout", 5.0))
        self.r_progress, self.r_time, self.r_facing = float(r.get("progress", 1.0)), float(r.get("time", 0.01)), float(r.get("facing", 0.002))
        self.r_switch, self.switch_margin = float(r.get("switch", 0.5)), float(r.get("switch_margin_m", 1.0))
        tr = dict(g.get("track", {}))
        self.track = bool(tr.get("enabled", True))
        self.follow_m, self.follow_deg = float(tr.get("follow_m", 0.6)), math.radians(float(tr.get("follow_deg", 45)))
        self.r_track, self.r_lost, self.lost_m = float(tr.get("reward", 0.05)), float(tr.get("lost", 0.02)), float(tr.get("lost_m", 2.5))
        self.r_retouch, self.cooldown_steps = float(tr.get("retouch", 2.0)), int(round(float(tr.get("cooldown_s", 2.0)) / self.dt))
        self.start_locked = float(tr.get("start_locked", 0.0))    # this fraction of episodes begins at the first touch, locked on: tracking practice
        self.r_track_facing = float(tr.get("facing", 0.0))        # per tick x cos(bearing to the locked person): keep them in front, not behind
        # sustained frontal contact: every tick the nose touches the locked person head-on pays `contact`, growing
        # linearly to `contact_max` x that over `contact_ramp_s` of unbroken contact (the streak resets when it breaks)
        self.r_contact, self.contact_ramp_s, self.contact_max = float(tr.get("contact", 0.0)), float(tr.get("contact_ramp_s", 3.0)), float(tr.get("contact_max", 3.0))
        # trailing: the per-tick tracking reward grows to `ramp_max` x over `ramp_s` of unbroken tracking (close and facing)
        self.track_ramp_s, self.track_max = float(tr.get("ramp_s", 0.0)), float(tr.get("ramp_max", 1.0))
        # spinning away: losing the locked person (behind `lose_deg`, or out of the senses) within `hold_s` of the last touch
        # without touching them again costs `lose`, once per touch
        self.r_lose, self.hold_steps, self.lose_rad = float(tr.get("lose", 0.0)), int(round(float(tr.get("hold_s", 2.0)) / self.dt)), math.radians(float(tr.get("lose_deg", 90)))
        hm = dict(g.get("humans", {}))
        self.human_mode = str(hm.get("mode", "learn"))          # patrol | flee | learn | wander
        assert self.human_mode in ("patrol", "flee", "learn", "wander"), self.human_mode
        self.hmax, self.hturn = float(hm.get("speed_max", 2.0)), float(hm.get("turn_max", 4.0))
        self.flee_m, self.wall_m = float(hm.get("flee_m", 6.0)), float(hm.get("wall_m", 2.0))
        hr = dict(hm.get("reward", {}))
        self.hr_caught, self.hr_tracked = float(hr.get("caught", 10.0)), float(hr.get("tracked", 0.05))
        self.hr_retouch, self.hr_distance = float(hr.get("retouch", 2.0)), float(hr.get("distance", 0.01))
        self.hr_contact = float(hr.get("contact", 0.0))          # per tick the locked person is touched head-on (mirrors track.contact)
        st = dict(hm.get("stamina", {}))
        self.sprint_s, self.rest_s = float(st.get("sprint_s", 4.0)), float(st.get("rest_s", 3.0))
        pz = dict(hm.get("pause", {}))
        self.pause_every, self.pause_for = float(pz.get("every_s", 8.0)), float(pz.get("for_s", 2.5))
        self.alarm_m, self.calm_m = float(hm.get("alarm_m", 4.0)), float(hm.get("calm_m", 6.0))
        self.walk_speed = float(hm.get("walk_speed", 0.8))
        sn = dict(g.get("senses", {}))
        self.see_m, self.heat_m = float(sn.get("see_m", 6.0)), float(sn.get("heat_m", 5.0))     # 0 = unlimited
        # an HC-SR501 PIR in place of the warmth lobes: a digital output that goes high when a warm body MOVES inside its
        # cone (or the vehicle's own motion sweeps one through it) and stays high for hold_s (retriggerable). One sensor
        # feeds both hot cells the same bit; two (count 2) look lobe_deg left and right.
        pir = dict(sn.get("pir", {}))
        self.pir = bool(pir.get("enabled", False))
        self.pir_n = 2 if int(pir.get("count", 1)) >= 2 else 1
        self.pir_fov, self.pir_range = math.radians(float(pir.get("fov_deg", 110))), float(pir.get("range_m", 6.0))
        self.pir_hold_steps, self.pir_motion = int(round(float(pir.get("hold_s", 2.0)) / self.dt)), float(pir.get("motion_mps", 0.05))
        self.pir_lobe = math.radians(float(pir.get("lobe_deg", 45)))
        self.r_explore, self.r_idle, self.cell_m = float(r.get("explore", 0.2)), float(r.get("idle", 0.01)), float(r.get("cell_m", 1.0))
        self.hspec = HumanObs()
        self.seed, self.n_reset = int(seed), 0
        self.count_rng = np.random.default_rng(seed)
        B, P, dev = self.B, MAX_PEOPLE, self.device
        f = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float32)  # noqa: E731
        self.x, self.z, self.heading, self.speed, self.t = f(B), f(B), f(B), f(B), f(B)
        self.anchor_x, self.anchor_z, self.offset, self.pace = f(B, P), f(B, P), f(B, P), f(B, P)
        self.alive = torch.zeros(B, P, device=dev, dtype=torch.bool)
        self.px, self.pz, self.pspeed, self.pyaw = f(B, P), f(B, P), f(B, P), f(B, P)
        self.stamina = torch.ones(B, P, device=dev)
        self.resting = torch.zeros(B, P, device=dev, dtype=torch.bool)
        self.rest_left, self.pause_in = f(B, P), f(B, P)              # seconds of forced standing left; seconds to the next scheduled stop
        self.rest_steps = torch.zeros(B, device=dev, dtype=torch.long)     # ticks the fly's target spent resting (a statistic)
        self.prev_drive = f(B, 2)
        self.steps = torch.zeros(B, device=dev, dtype=torch.long)
        self.facing_steps = torch.zeros(B, device=dev, dtype=torch.long)
        self.min_dist, self.path_len, self.start_dist, self.dist_sum = f(B), f(B), f(B), f(B)
        self.gx, self.gz = int(math.ceil(2 * self.ax / self.cell_m)) + 1, int(math.ceil(2 * self.az / self.cell_m)) + 1
        self.visited = torch.zeros(B, self.gx * self.gz, device=dev, dtype=torch.bool)      # the fly's coverage of the room this episode
        self.blind_steps, self.idle_steps = torch.zeros(B, device=dev, dtype=torch.long), torch.zeros(B, device=dev, dtype=torch.long)
        self.alert = torch.zeros(B, P, device=dev, dtype=torch.bool)                          # a runner minding the fly
        self.pir_hold = torch.zeros(B, 2, device=dev, dtype=torch.long)                      # ticks each PIR output stays high
        self.wx, self.wz = f(B, P), f(B, P)                                                    # a strolling runner's waypoint
        self.target = torch.zeros(B, device=dev, dtype=torch.long)
        self.target_d = f(B)
        self.switches = torch.zeros(B, device=dev, dtype=torch.long)
        self.locked = torch.zeros(B, device=dev, dtype=torch.bool)          # touched someone: the target is fixed
        self.lock_step, self.last_touch_step = torch.zeros(B, device=dev, dtype=torch.long), torch.zeros(B, device=dev, dtype=torch.long)
        self.touches, self.track_steps = torch.zeros(B, device=dev, dtype=torch.long), torch.zeros(B, device=dev, dtype=torch.long)
        self.contact_steps, self.contact_streak, self.contact_best = (torch.zeros(B, device=dev, dtype=torch.long) for _ in range(3))
        self.track_streak, self.track_best = torch.zeros(B, device=dev, dtype=torch.long), torch.zeros(B, device=dev, dtype=torch.long)
        self.last_contact_step = torch.zeros(B, device=dev, dtype=torch.long)      # any tick the nose was on the locked person
        self.spun = torch.zeros(B, device=dev, dtype=torch.bool)                   # already charged for losing them since the last touch
        self.spins = torch.zeros(B, device=dev, dtype=torch.long)
        self.t_first, self.path_first, self.first_b = f(B), f(B), f(B)
        self.first_frontal = torch.zeros(B, device=dev, dtype=torch.bool)
        self.started_locked = torch.zeros(B, device=dev, dtype=torch.bool)    # an episode that began at contact (see _start_at_contact)
        self.env_seed = torch.zeros(B, device=dev, dtype=torch.long)
        half = self.fov / 2
        self.bin_w = self.fov / self.spec.bins
        # column centres from the fly's left (+ve bearing) to its right, like a retina read left to right
        self.bin_c = (half - self.bin_w * (torch.arange(self.spec.bins, device=dev, dtype=torch.float32) + 0.5))
        self.reset(torch.arange(B, device=dev))

    # ---- episodes ----------------------------------------------------------------------------
    def next_seeds(self, n: int) -> list[int]:
        s = [(self.seed * 7919 + (self.n_reset + i) * 104729) & MASK for i in range(n)]
        self.n_reset += n
        return s

    def reset(self, idx: torch.Tensor, seeds: list[int] | None = None) -> None:
        """Start fresh episodes in the environments `idx` (seeded layouts, one seed per environment)."""
        idx = idx.to(self.device)
        n = int(idx.numel())
        if n == 0:
            return
        seeds = list(seeds) if seeds is not None else self.next_seeds(n)
        P = MAX_PEOPLE
        A = np.zeros((n, 4, P), dtype=np.float32)          # anchor_x, anchor_z, offset, pace
        alive = np.zeros((n, P), dtype=bool)
        pose = np.zeros((n, 3), dtype=np.float32)
        for k, seed in enumerate(seeds):
            count = int(self.count_rng.integers(self.people_min, self.people_max + 1)) if self.people_max > self.people_min else self.people_min
            lay = layout(seed, count)
            for i, (ax, az, off, pace) in enumerate(lay["people"]):
                A[k, :, i] = (ax, az, off, pace)
            alive[k, :count] = True
            rng = lay["rng"]
            for _ in range(20):
                x, z = (rng() * 2 - 1) * (self.ax - 1.0), (rng() * 2 - 1) * (self.az - 1.0)
                hd = rng() * math.pi * 2 - math.pi
                if min(math.hypot(px - x, pz - z) for px, pz in zip(lay["px"], lay["pz"])) > 2.0:
                    break
            pose[k] = (x, z, hd)
        A_t = torch.from_numpy(A).to(self.device)
        self.anchor_x[idx], self.anchor_z[idx], self.offset[idx], self.pace[idx] = A_t[:, 0], A_t[:, 1], A_t[:, 2], A_t[:, 3]
        self.alive[idx] = torch.from_numpy(alive).to(self.device)
        self.stamina[idx] = 1.0
        self.resting[idx] = False
        self.rest_left[idx] = 0.0
        self.pause_in[idx] = self._next_pause((n, MAX_PEOPLE))
        self.rest_steps[idx] = 0
        pose_t = torch.from_numpy(pose).to(self.device)
        self.x[idx], self.z[idx], self.heading[idx] = pose_t[:, 0], pose_t[:, 1], pose_t[:, 2]
        self.env_seed[idx] = torch.tensor(seeds, device=self.device, dtype=torch.long)
        self.t[idx] = 0.0
        self.speed[idx] = 0.0
        self.prev_drive[idx] = 0.0
        self.steps[idx] = 0
        self.facing_steps[idx] = 0
        self.path_len[idx] = 0.0
        self.dist_sum[idx] = 0.0
        self.visited[idx] = False
        self.blind_steps[idx] = 0
        self.idle_steps[idx] = 0
        self.alert[idx] = False
        self.pir_hold[idx] = 0
        self.wx[idx], self.wz[idx] = self._waypoints((n, MAX_PEOPLE))
        self.switches[idx] = 0
        self.locked[idx] = False
        self.first_frontal[idx] = False
        self.lock_step[idx] = 0; self.last_touch_step[idx] = -10 ** 6; self.touches[idx] = 0; self.track_steps[idx] = 0
        self.contact_steps[idx] = 0; self.contact_streak[idx] = 0; self.contact_best[idx] = 0
        self.track_streak[idx] = 0; self.track_best[idx] = 0
        self.last_contact_step[idx] = -10 ** 6; self.spun[idx] = False; self.spins[idx] = 0
        self.t_first[idx] = 0.0; self.path_first[idx] = 0.0; self.first_b[idx] = 0.0
        self._place_people(idx)
        d, _ = self._dist_bearing(idx)
        d_alive = torch.where(self.alive[idx], d, torch.full_like(d, 1e9))
        nearest, tgt = d_alive.min(dim=1)
        self.min_dist[idx], self.start_dist[idx], self.target[idx], self.target_d[idx] = nearest, nearest, tgt, nearest
        self.started_locked[idx] = False
        if self.track and self.start_locked > 0:
            pick = torch.rand(n, device=self.device) < self.start_locked
            if bool(pick.any()):
                self._start_at_contact(idx[pick], tgt[pick])

    def _start_at_contact(self, idx: torch.Tensor, tgt: torch.Tensor) -> None:
        """Begin these episodes at the moment of the first touch: the fly stands still, nose to the skin of its
        target, facing it from a random side, already locked on. Every tick from here is tracking practice
        (the touch itself is not paid again: it is counted as having happened at t = 0)."""
        px, pz = self.px[idx].gather(1, tgt[:, None])[:, 0], self.pz[idx].gather(1, tgt[:, None])[:, 0]
        hd = torch.rand(idx.numel(), device=self.device) * 2 * math.pi - math.pi
        back = self.person_r + self.nose + 0.5 * self.touch_m
        self.x[idx] = (px - torch.sin(hd) * back).clamp(-self.ax, self.ax)
        self.z[idx] = (pz - torch.cos(hd) * back).clamp(-self.az, self.az)
        self.heading[idx] = hd
        self.locked[idx] = True
        self.started_locked[idx] = True
        self.target[idx] = tgt
        self.first_frontal[idx] = True
        self.touches[idx] = 1
        self.lock_step[idx] = 0
        self.last_touch_step[idx] = 0
        d, _ = self._dist_bearing(idx)
        td = d.gather(1, tgt[:, None])[:, 0]
        self.target_d[idx], self.min_dist[idx], self.start_dist[idx] = td, td, td

    def _place_people(self, idx=None) -> None:
        sl = slice(None) if idx is None else idx
        th = self.t[sl, None] * self.pace[sl] + self.offset[sl]
        nx = (self.anchor_x[sl] + ANCHOR_RX * torch.cos(th)).clamp(-PEOPLE_X, PEOPLE_X)
        nz = (self.anchor_z[sl] + ANCHOR_RZ * torch.sin(th)).clamp(-PEOPLE_Z, PEOPLE_Z)
        # a person's speed is the distance moved since the last tick; on a reset the CPU arena measures it
        # from the origin (a fresh Person starts at 0, 0), so the first observation sees everyone as moving
        ox = self.px[sl] if idx is None else torch.zeros_like(nx)
        oz = self.pz[sl] if idx is None else torch.zeros_like(nz)
        self.pspeed[sl] = torch.hypot(nx - ox, nz - oz)
        self.px[sl], self.pz[sl] = nx, nz
        self.pyaw[sl] = torch.atan2(-ANCHOR_RX * torch.sin(th), ANCHOR_RZ * torch.cos(th))

    def _next_pause(self, shape) -> torch.Tensor:
        if self.pause_every <= 0:
            return torch.full(shape, 1e9, device=self.device)
        return self.pause_every * (0.5 + torch.rand(shape, device=self.device))      # uniform in [0.5, 1.5] x every_s

    def _waypoints(self, shape) -> tuple[torch.Tensor, torch.Tensor]:
        return ((torch.rand(shape, device=self.device) * 2 - 1) * (PEOPLE_X - 1.0), (torch.rand(shape, device=self.device) * 2 - 1) * (PEOPLE_Z - 1.0))

    def wander_drive(self) -> torch.Tensor:
        """A runner not minding the fly strolls to a random waypoint, then picks the next one."""
        dx, dz = self.wx - self.px, self.wz - self.pz
        arrived = torch.hypot(dx, dz) < 0.5
        nx, nz = self._waypoints(self.wx.shape)
        self.wx, self.wz = torch.where(arrived, nx, self.wx), torch.where(arrived, nz, self.wz)
        turn = (2.0 * wrap(torch.atan2(self.wx - self.px, self.wz - self.pz) - self.pyaw)).clamp(-1, 1)
        return torch.stack([torch.full_like(turn, self.walk_speed / self.hmax), turn], dim=-1)

    def _update_alert(self) -> None:
        if self.human_mode == "wander":                         # people going about their business: the fly is nothing to them
            self.alert = torch.zeros_like(self.alert)
            return
        d = torch.hypot(self.px - self.x[:, None], self.pz - self.z[:, None])
        self.alert = self.alive & ((self.alert & (d < self.calm_m)) | (d < self.alarm_m))

    def _move_people(self, hdrive: torch.Tensor) -> None:
        """Runners: `hdrive` [B, P, 2] = (run in [-1, 1], turn in [-1, 1], +ve left as the yaw grows). They stay
        inside the people box (the walls stop them) and dead slots stay put."""
        hdrive = hdrive.clamp(-1.0, 1.0)
        run, turn = hdrive[..., 0].clamp(min=0), hdrive[..., 1]
        yaw = wrap(self.pyaw + self.hturn * turn * self.dt)
        # forced stops: the scheduled pause (every ~pause_every s, for pause_for s) and exhaustion (stamina at zero:
        # stand for rest_s). Sprinting above half speed drains the stamina in sprint_s, standing restores it in rest_s.
        self.pause_in = self.pause_in - self.dt
        free = self.rest_left <= 0
        pause = free & (self.pause_in <= 0)
        self.rest_left = torch.where(pause, torch.full_like(self.rest_left, self.pause_for), self.rest_left)
        self.pause_in = torch.where(pause, self._next_pause(self.pause_in.shape), self.pause_in)
        exhausted = free & ~pause & (self.stamina <= 0.0)
        self.rest_left = torch.where(exhausted, torch.full_like(self.rest_left, self.rest_s), self.rest_left)
        self.resting = (self.rest_left > 0) & self.alive
        self.rest_left = (self.rest_left - self.dt).clamp(min=0)
        run = torch.where(self.resting, torch.zeros_like(run), run)                     # standing still
        v = self.hmax * run * self.alive.float()
        frac = v / self.hmax
        self.stamina = (self.stamina - self.dt / self.sprint_s * ((frac - 0.5) / 0.5).clamp(0, 1) + self.dt / self.rest_s * (1 - frac)).clamp(0, 1)
        nx = (self.px + torch.sin(yaw) * v * self.dt).clamp(-PEOPLE_X, PEOPLE_X)
        nz = (self.pz + torch.cos(yaw) * v * self.dt).clamp(-PEOPLE_Z, PEOPLE_Z)
        self.pspeed = torch.hypot(nx - self.px, nz - self.pz)
        self.px, self.pz, self.pyaw = nx, nz, torch.where(self.alive, yaw, self.pyaw)

    # ---- the humans' side ------------------------------------------------------------------------
    def _human_geometry(self):
        """Per person: distance and bearing of the fly in the person's own frame, plus the nearest other person."""
        dx, dz = self.x[:, None] - self.px, self.z[:, None] - self.pz
        s, c = torch.sin(self.pyaw), torch.cos(self.pyaw)
        fwd, left = dx * s + dz * c, dx * c - dz * s
        d, b = torch.hypot(dx, dz), torch.atan2(left, fwd)
        ox, oz = self.px[:, None, :] - self.px[:, :, None], self.pz[:, None, :] - self.pz[:, :, None]      # [B, P, P] i -> j
        od = torch.hypot(ox, oz)
        od = torch.where(self.alive[:, None, :] & ~torch.eye(od.shape[1], device=od.device, dtype=torch.bool)[None], od, torch.full_like(od, 1e9))
        nd, nj = od.min(dim=2)
        nox, noz = ox.gather(2, nj[..., None])[..., 0], oz.gather(2, nj[..., None])[..., 0]
        nb = torch.atan2(nox * c - noz * s, nox * s + noz * c)
        return d, b, nd.clamp(max=20.0), nb

    def observe_humans(self) -> torch.Tensor:
        """[B, P, HumanObs.size]: what each runner knows, in its own frame (see HumanObs)."""
        d, b, nd, nb = self._human_geometry()
        rel = self.heading[:, None] - self.pyaw
        walls = torch.stack([(PEOPLE_X - self.px), (PEOPLE_X + self.px), (PEOPLE_Z - self.pz), (PEOPLE_Z + self.pz)], dim=-1).div(5.0).clamp(max=1.0)
        mine = torch.nn.functional.one_hot(self.target, self.px.shape[1]).bool() & self.locked[:, None]
        obs = torch.cat([torch.stack([d / 10.0, torch.sin(b), torch.cos(b), self.speed[:, None].expand_as(d) / self.vmax, torch.sin(rel), torch.cos(rel),
                                      self.pspeed / self.dt / self.hmax], dim=-1),
                         walls, torch.stack([torch.sin(self.pyaw), torch.cos(self.pyaw), nd / 10.0, torch.sin(nb), torch.cos(nb),
                                             mine.float(), self.t[:, None].expand_as(d) / self.episode_s, self.stamina, self.resting.float()], dim=-1)], dim=-1)
        return obs

    def flee_drive(self) -> torch.Tensor:
        """The scripted runner: turn to put the fly straight behind, sprint while it is within flee_m, and
        steer toward the middle when a wall is near (so the fly cannot pin it in a corner for free)."""
        d, _, _, _ = self._human_geometry()
        ax, az = self.px - self.x[:, None], self.pz - self.z[:, None]                     # away from the fly (world frame)
        ax, az = ax / d.clamp(min=1e-6), az / d.clamp(min=1e-6)
        mx, mz = -self.px, -self.pz                                                     # toward the middle
        mn = torch.hypot(mx, mz).clamp(min=1e-6)
        near = (1 - torch.minimum(PEOPLE_X - self.px.abs(), PEOPLE_Z - self.pz.abs()) / self.wall_m).clamp(0, 1)
        vx, vz = (1 - near) * ax + near * mx / mn, (1 - near) * az + near * mz / mn      # blended escape direction
        turn = (2.0 * wrap(torch.atan2(vx, vz) - self.pyaw)).clamp(-1, 1)
        run = torch.where(d < self.flee_m, torch.ones_like(d), torch.full_like(d, 0.3))
        return torch.stack([run, turn], dim=-1)

    # ---- geometry ----------------------------------------------------------------------------
    def _dist_bearing(self, idx=None) -> tuple[torch.Tensor, torch.Tensor]:
        """distance and bearing [B, P] of every person; bearing +ve on the fly's left (heading 0 faces +z)."""
        sl = slice(None) if idx is None else idx
        dx, dz = self.px[sl] - self.x[sl, None], self.pz[sl] - self.z[sl, None]
        s, c = torch.sin(self.heading[sl])[:, None], torch.cos(self.heading[sl])[:, None]
        fwd, left = dx * s + dz * c, dx * c - dz * s
        return torch.hypot(dx, dz), torch.atan2(left, fwd)

    # ---- senses ------------------------------------------------------------------------------
    def observe(self, noise: bool = True) -> torch.Tensor:
        d, b = self._dist_bearing()
        heat, vis = sense(d, b, self.alive, self.pspeed, self.speed, self)
        if self.pir:
            heat = (self.pir_hold > 0).float().mul(self.heat_gain).clamp(max=1.0)
        body = torch.stack([self.speed / self.vmax, self.prev_drive[:, 0], self.prev_drive[:, 1], self.t / self.episode_s], dim=1)
        obs = torch.cat([heat, vis.reshape(self.B, -1), body], dim=1)
        if noise and self.obs_noise > 0:
            obs = obs + self.obs_noise * torch.randn_like(obs)
        return obs

    # ---- movement ----------------------------------------------------------------------------
    def step(self, drive: torch.Tensor, hdrive: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """One tick for every environment. `drive` [B, 2] = (forward in [-1, 1], turn in [-1, 1], +ve right).
        Returns (reward [B], done [B] bool, info with per-episode results for the environments that just
        finished). Finished environments are NOT reset here: call `reset(done.nonzero())` afterwards."""
        drive = drive.clamp(-1.0, 1.0).to(self.device)
        dt = self.dt
        d0, b0 = self._dist_bearing()
        d0a = torch.where(self.alive, d0, torch.full_like(d0, 1e9))
        nearest0, nearest_i0 = d0a.min(dim=1)
        facing = b0.gather(1, nearest_i0[:, None])[:, 0].abs() < math.radians(30)
        self.facing_steps += facing.long()
        self.steps += 1
        self.t += dt
        if self.human_mode == "patrol":
            self._place_people()
        else:
            self._update_alert()
            evade = hdrive if (hdrive is not None and self.human_mode == "learn") else self.flee_drive()
            self._move_people(torch.where(self.alert[:, :, None], evade, self.wander_drive()))
        fwd, turn_in = drive[:, 0], drive[:, 1]
        speed = self.vmax * fwd.clamp(min=0) - self.vrev * (-fwd).clamp(min=0)
        turn = self.turn_sign * self.wmax * self.turn_gain * turn_in
        to_center = torch.atan2(-self.x, -self.z)
        edge = torch.maximum(self.x.abs() - (self.ax - 1.2), self.z.abs() - (self.az - 1.2))
        want = torch.where(speed > 0, to_center, to_center + math.pi)
        turn = turn + torch.where((edge > 0) & (speed.abs() > 0.01), wrap(want - self.heading) * (2 + 6 * edge), torch.zeros_like(turn))
        heading = wrap(self.heading + turn * dt)
        nx = (self.x + torch.sin(heading) * speed * dt).clamp(-self.ax, self.ax)
        nz = (self.z + torch.cos(heading) * speed * dt).clamp(-self.az, self.az)
        solid = self.person_r + (self.nose if self.body_solid else 0.10)
        for i in range(MAX_PEOPLE):                       # people are solid, resolved one after another as on the CPU
            px, pz = self.px[:, i], self.pz[:, i]
            dd = torch.hypot(px - nx, pz - nz)
            push = self.alive[:, i] & (dd < solid) & (dd > 1e-6)
            nx = torch.where(push, px + (nx - px) / dd * solid, nx)
            nz = torch.where(push, pz + (nz - pz) / dd * solid, nz)
        self.path_len += torch.hypot(nx - self.x, nz - self.z)
        self.x, self.z, self.heading, self.speed = nx, nz, heading, speed
        self.prev_drive = drive
        d, b = self._dist_bearing()
        da = torch.where(self.alive, d, torch.full_like(d, 1e9))
        nearest, nearest_i = da.min(dim=1)
        self.min_dist = torch.minimum(self.min_dist, nearest)
        self.dist_sum += nearest
        # touch: the nose within touch_m of a person's skin; the most head-on contact counts
        nose_x, nose_z = self.x + torch.sin(self.heading) * self.nose, self.z + torch.cos(self.heading) * self.nose
        dn = torch.hypot(self.px - nose_x[:, None], self.pz - nose_z[:, None])
        hit = self.alive & (dn <= self.person_r + self.touch_m)
        hit_b = torch.where(hit, b.abs(), torch.full_like(b, 1e9))
        best_b, best_i = hit_b.min(dim=1)
        touching = hit.any(dim=1)
        first = touching & ~self.locked                                   # the first touch of the episode
        contact_b = b.gather(1, best_i[:, None])[:, 0]
        frontal = first & (best_b < self.frontal)
        self.t_first = torch.where(first, self.t, self.t_first)
        self.path_first = torch.where(first, self.path_len, self.path_first)
        self.first_b = torch.where(first, contact_b, self.first_b)
        self.first_frontal |= frontal
        self.lock_step = torch.where(first, self.steps, self.lock_step)
        # target commitment: the nearest person at the start, someone a margin closer until the first touch,
        # then the person touched, for good
        # what the fly can sense right now: seen (in the field of view, within see_m) or felt (within heat_m, or by the PIR)
        if self.pir:
            moving = (self.pspeed / dt > self.pir_motion) | ((self.speed.abs() > 0.05) | (turn_in.abs() > 0.1))[:, None]
            centres = [0.0] if self.pir_n == 1 else [self.pir_lobe, -self.pir_lobe]
            felt = torch.zeros_like(self.alive)
            for k, c in enumerate(centres):
                in_cone = self.alive & moving & ((b - c).abs() < self.pir_fov / 2) & (d < self.pir_range)
                felt |= in_cone
                self.pir_hold[:, k] = torch.where(in_cone.any(dim=1), torch.full_like(self.pir_hold[:, k], self.pir_hold_steps), (self.pir_hold[:, k] - 1).clamp(min=0))
            if self.pir_n == 1:
                self.pir_hold[:, 1] = self.pir_hold[:, 0]
        else:
            felt = d < (self.heat_m if self.heat_m > 0 else 1e9)
        sensed = self.alive & (((b.abs() < self.fov / 2) & (d < (self.see_m if self.see_m > 0 else 1e9))) | felt)
        ds = torch.where(sensed, d, torch.full_like(d, 1e9))
        nearest_s, nearest_si = ds.min(dim=1)
        any_sensed = nearest_s < 1e8
        tgt_d = da.gather(1, self.target[:, None])[:, 0]
        tgt_sensed = sensed.gather(1, self.target[:, None])[:, 0]
        switch = (~self.locked) & (~first) & any_sensed & tgt_sensed & (nearest_s < tgt_d - self.switch_margin)
        retarget = (~self.locked) & (~first) & any_sensed & ~tgt_sensed            # lost the target, found someone else: free
        self.target = torch.where(first, best_i, torch.where(switch | retarget, nearest_si, self.target))
        self.switches += switch.long()
        tgt_d = torch.where(first | switch | retarget, da.gather(1, self.target[:, None])[:, 0], tgt_d)
        tgt_sensed = sensed.gather(1, self.target[:, None])[:, 0] | first
        progress = torch.where(tgt_sensed, self.target_d - tgt_d, torch.zeros_like(tgt_d))
        self.target_d = tgt_d
        tgt_b = b.gather(1, self.target[:, None])[:, 0]
        # searching: nobody in range -> paid for every new cell of the room, charged for standing still
        cell = ((self.x + self.ax) / self.cell_m).long().clamp(0, self.gx - 1) * self.gz + ((self.z + self.az) / self.cell_m).long().clamp(0, self.gz - 1)
        new_cell = ~self.visited.gather(1, cell[:, None])[:, 0]
        self.visited.scatter_(1, cell[:, None], True)
        blind = ~any_sensed
        idle = blind & (self.speed.abs() < 0.2)
        self.blind_steps += blind.long()
        self.idle_steps += idle.long()
        reward = (self.r_progress * progress - self.r_time - self.r_switch * switch.float()
                  + self.r_facing * torch.cos(tgt_b).clamp(min=0) * tgt_sensed.float()
                  + self.r_explore * (new_cell & blind).float() - self.r_idle * idle.float())
        amount = torch.where(frontal, torch.ones_like(reward), torch.full_like(reward, self.side_reward))
        reward = reward + first.float() * (self.r_touch * amount + self.r_fast * (1 - self.t / self.episode_s).clamp(min=0))
        if self.track:
            # after the first touch: paid for keeping the locked person close and ahead, and for touching them again
            tracking = self.locked & (dn.gather(1, self.target[:, None])[:, 0] <= self.person_r + self.follow_m) & (tgt_b.abs() < self.follow_deg)
            self.track_steps += tracking.long()
            self.track_streak = torch.where(tracking, self.track_streak + 1, torch.zeros_like(self.track_streak))
            self.track_best = torch.maximum(self.track_best, self.track_streak)
            tramp = 1.0 + (self.track_max - 1.0) * (self.track_streak.float() * self.dt / max(self.track_ramp_s, 1e-6)).clamp(max=1.0) if self.track_ramp_s > 0 else torch.ones_like(tgt_b)
            lost = self.locked & (tgt_d > self.lost_m)
            retouch = self.locked & hit.gather(1, self.target[:, None])[:, 0] & (self.steps - self.last_touch_step >= self.cooldown_steps)
            # sustained frontal contact: nose on the locked person's skin with them ahead within frontal_deg
            contact = self.locked & hit.gather(1, self.target[:, None])[:, 0] & (tgt_b.abs() < self.frontal)
            self.contact_streak = torch.where(contact, self.contact_streak + 1, torch.zeros_like(self.contact_streak))
            self.contact_best = torch.maximum(self.contact_best, self.contact_streak)
            self.contact_steps += contact.long()
            ramp = 1.0 + (self.contact_max - 1.0) * (self.contact_streak.float() * self.dt / max(self.contact_ramp_s, 1e-6)).clamp(max=1.0)
            # spinning away right after a touch: the person behind lose_deg or out of the senses within hold_s of the last touch
            touching_target = hit.gather(1, self.target[:, None])[:, 0]
            self.last_contact_step = torch.where(touching_target | first, self.steps, self.last_contact_step)
            self.spun &= ~(touching_target | first)                                          # a new touch re-arms the charge
            gone = (tgt_b.abs() > self.lose_rad) | ~tgt_sensed
            spin = self.locked & ~touching_target & gone & ~self.spun & (self.steps - self.last_contact_step <= self.hold_steps)
            self.spun |= spin
            self.spins += spin.long()
            reward = (reward + self.r_track * tracking.float() * tramp - self.r_lost * lost.float() + self.r_retouch * retouch.float()
                      + self.r_track_facing * torch.cos(tgt_b) * self.locked.float() + self.r_contact * contact.float() * ramp
                      - self.r_lose * spin.float())
            self.touches += (first | retouch).long()
            self.last_touch_step = torch.where(first | retouch, self.steps, self.last_touch_step)
            self.locked |= first
            done = self.steps >= self.max_steps
            timeout = done & ~self.locked
        else:
            self.touches += first.long()
            self.locked |= first
            timeout = (~first) & (self.steps >= self.max_steps)
            done = first | timeout
        reward = reward - timeout.float() * self.r_timeout
        # the humans' reward: keep away; the one caught pays, the one tracked keeps paying
        d_fly = torch.hypot(self.px - self.x[:, None], self.pz - self.z[:, None])
        mine = torch.nn.functional.one_hot(self.target, self.px.shape[1]).float()
        self.rest_steps += (self.resting.float() * mine).sum(dim=1).long()
        hreward = self.hr_distance * (d_fly / self.alarm_m).clamp(max=1.0) - self.hr_caught * (first.float()[:, None] * mine)
        if self.track:
            hreward = (hreward - self.hr_tracked * tracking.float()[:, None] * mine - self.hr_retouch * retouch.float()[:, None] * mine
                       - self.hr_contact * contact.float()[:, None] * mine)
        info = {"human_reward": hreward * self.alive.float(), "human_active": self.alert.clone(), "sensed": sensed}
        if bool(done.any()):
            info["episodes"] = self.results(done.nonzero()[:, 0])
        return reward, done, info

    def results(self, idx: torch.Tensor) -> list[dict]:
        """Per-episode records in the CPU arena's `result()` format (so `sim.hunt.score` applies: the first
        touch, its time, angle and path efficiency), plus `touches` (contacts with the locked person) and
        `track` (the fraction of the ticks after the first touch spent close behind them)."""
        d, _ = self._dist_bearing(idx)
        final = torch.where(self.alive[idx], d, torch.full_like(d, 1e9)).min(dim=1).values
        touched = self.locked[idx]
        after = (self.steps[idx] - self.lock_step[idx]).clamp(min=1).float()
        cols = torch.stack([self.env_seed[idx].float(), touched.float(), self.t_first[idx], self.first_frontal[idx].float(), torch.rad2deg(self.first_b[idx]),
                            self.facing_steps[idx].float() / self.steps[idx].clamp(min=1).float(), self.min_dist[idx],
                            (self.start_dist[idx] / self.path_first[idx].clamp(min=1e-6)).clamp(max=1.0) * touched.float(),
                            final, self.switches[idx].float(), self.alive[idx].sum(dim=1).float(),
                            self.touches[idx].float(), self.track_steps[idx].float() / after, self.dist_sum[idx] / self.steps[idx].clamp(min=1).float(),
                            self.rest_steps[idx].float() / self.steps[idx].clamp(min=1).float(),
                            self.blind_steps[idx].float() / self.steps[idx].clamp(min=1).float(),
                            self.idle_steps[idx].float() / self.blind_steps[idx].clamp(min=1).float(), self.started_locked[idx].float(),
                            self.contact_steps[idx].float() / after, self.contact_best[idx].float() * self.dt, self.spins[idx].float(),
                            self.track_best[idx].float() * self.dt], dim=1).cpu().numpy()
        out = []
        for row in cols:
            seed, hit, t, fr, cb, facing, md, eff, fd, sw, n, touches, track, mean_d, rest, blind, idle, sl, contact, contact_s, spins, track_s = row
            out.append({"seed": int(seed), "touched": bool(hit), "t_touch": round(float(t), 2) if hit else None, "frontal": bool(fr),
                        "contact_deg": round(float(cb), 1) if hit else None, "facing": round(float(facing), 3), "min_dist": round(float(md), 2),
                        "efficiency": round(float(eff), 3), "final_dist": round(float(fd), 2), "switches": int(sw), "people": int(n),
                        "touches": int(touches), "track": round(float(track), 3) if hit else None, "mean_dist": round(float(mean_d), 2), "target_rest": round(float(rest), 3),
                        "blind": round(float(blind), 3), "idle": round(float(idle), 3), "started_locked": bool(sl),
                        "contact": round(float(contact), 3) if hit else None, "contact_s": round(float(contact_s), 2) if hit else None,
                        "spins": int(spins), "track_s": round(float(track_s), 2) if hit else None})
        return out

    def snapshot(self) -> dict:
        """Positions for a viewer or a test (CPU floats)."""
        return {"x": self.x.cpu().numpy(), "z": self.z.cpu().numpy(), "heading": self.heading.cpu().numpy(), "pir": (self.pir_hold > 0).cpu().numpy(),
                "px": self.px.cpu().numpy(), "pz": self.pz.cpu().numpy(), "pyaw": self.pyaw.cpu().numpy(), "alive": self.alive.cpu().numpy(), "t": self.t.cpu().numpy(),
                "stamina": self.stamina.cpu().numpy(), "resting": self.resting.cpu().numpy(), "alert": self.alert.cpu().numpy()}


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def sense(d: torch.Tensor, b: torch.Tensor, alive: torch.Tensor, pspeed: torch.Tensor, speed: torch.Tensor, a) -> tuple[torch.Tensor, torch.Tensor]:
    """Heat [B, 2] and the retinotopic vision map [B, bins, 2] from per-person distance / bearing [B, P].
    `a` supplies the sensor geometry (an arena or anything with the same attributes)."""
    warmth = torch.where(alive, 1.0 / (1.0 + (d / a.heat_scale) ** 2), torch.zeros_like(d))
    if a.heat_m > 0:
        warmth = warmth * ((a.heat_m - d) / 1.0).clamp(0, 1)                         # felt only within heat_m (fades over the last metre)
    hl = (warmth * torch.cos(b - a.heat_lobe).clamp(min=0)).sum(dim=1)
    hr = (warmth * torch.cos(b + a.heat_lobe).clamp(min=0)).sum(dim=1)
    heat = torch.stack([hl, hr], dim=1).mul(a.heat_gain).clamp(max=1.0)
    half = a.fov / 2
    width = 2 * torch.atan2(torch.full_like(d, 0.25), d.clamp(min=0.3))            # shoulders ~0.5 m across
    strength = (width / math.radians(25)).clamp(max=1.0)
    moving = torch.where((pspeed > 1e-3) | (speed.abs() > 0.05)[:, None], torch.ones_like(d), torch.full_like(d, 0.4))
    seen = alive & (b.abs() < half) & ((d < a.see_m) if a.see_m > 0 else torch.ones_like(alive))
    cover = ((b[:, :, None] - a.bin_c[None, None, :]).abs() <= (width[:, :, None] / 2 + a.bin_w / 2)) & seen[:, :, None]   # [B, P, K]
    obj = torch.where(cover, strength[:, :, None], torch.zeros_like(cover, dtype=strength.dtype)).amax(dim=1)
    bar = torch.where(cover, 0.8 * (strength * moving)[:, :, None], torch.zeros_like(cover, dtype=strength.dtype)).amax(dim=1)
    return heat, torch.stack([obj, bar], dim=2)


def scripted_drive(obs: torch.Tensor, spec: ObsSpec) -> torch.Tensor:
    """The hand-written hunter of the CPU arena on the batched observation: turn toward the warmer side
    and the seen object, drive forward. Proves the batch arena is solvable; the reference for the trained fly."""
    heat, vis, _ = spec.split(obs)
    obj = vis[..., 0]
    k = obj.shape[-1]
    # column azimuth -1 (left edge) .. +1 (right edge): the CPU `object_x` is +ve to the right of the midline
    az = (torch.arange(k, device=obs.device, dtype=obs.dtype) + 0.5) / k * 2 - 1
    strength, col = obj.max(dim=-1)
    turn = 3.0 * (heat[..., 1] - heat[..., 0]) + 1.5 * az[col] * strength
    return torch.stack([torch.ones_like(turn), turn.clamp(-1, 1)], dim=-1)
