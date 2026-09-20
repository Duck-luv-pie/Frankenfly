"""
arena.py -- batched 2D kinematic world. No rendering. Everything (B, ...) torch.

    arena = Arena(B, device)
    state = arena.reset(seed=0, stage="A")
    state, events = arena.step(forward (B,), turn (B,))
    rew = arena.reward(events)                      # (B,)
    arena.retina_inputs()                           # -> (robot_xy, robot_yaw, human_xy, human_r, valid)
    arena.heat()                                    # (B, 2) left/right leaky heat detectors
    arena.perturb_retina(r)                         # retina noise (per-episode sigma, column dropout)
    arena.gain                                      # (B,) global gain multiplier, +-10%

Stages: A one human spawned IN VIEW at 1-3 m; B humans out of view; C everything random.
Humans: random walk. Pick heading, walk 2-6 s, pause 0-2 s with probability 0.3, bounce off walls.
Not scripted, not evasive. Rover: differential drive, v_max = ratio * that episode's human speed.
Collision: oriented rover box vs human cylinders and walls. Touch = overlap with the front strip.
The rover cannot push through a human (its move is rejected); a human walking into the rover bounces.

Events (all (B,) bool): acquire (first time any human is in the camera FOV), front_contact (first
front-strip touch of the episode), sustained (front touch after the first), back_or_side_contact,
timeout (episode clock ran out this step), contact_now.

Reward (events -> scalar per env per step):
  +1.0 acquire (stage A/B only), +5.0 first front contact, +0.5 per step sustained front contact,
  -0.02 per step with no contact, 0 for back/side contact (never positive), -3.0 at timeout with no contact.
Randomized per episode: arena size, human count/radius/speed, spawn poses, rover start, retina noise
  (sigma 0.05 pres, 0.1 size, 2% column dropout), heat latency (0-200 ms) / false positives (<=1%),
  global gain +-10%.
Conventions: metres, radians, seconds. Yaw CCW-positive in world frame. Bearing + = robot's RIGHT.
"""
from __future__ import annotations

import math

import torch


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Arena:
    def __init__(self, B: int, device: str | torch.device | None = None, dt: float = 0.02,
                 size_range: tuple[float, float] = (5, 12), n_humans: tuple[int, int] = (1, 8),
                 rover_lw: tuple[float, float] = (0.32, 0.17), front_depth: float = 0.06,
                 human_r: tuple[float, float] = (0.22, 0.30),
                 human_speed: tuple[float, float] = (0.8, 1.4), fly_speed_ratio: float = 1.15,
                 episode_s: float = 20.0, hfov_deg: float = 98.43, omega_max: float = 2.5,
                 contact_margin: float = 0.04, n_cols: int = 24,
                 heat_range: float = 7.0, heat_half_cone_deg: float = 60.0, heat_axis_deg: float = 30.0,
                 heat_tau: float = 0.3, heat_max_delay_s: float = 0.2,
                 retina_sigma: tuple[float, float] = (0.05, 0.10), col_dropout: float = 0.02,
                 gain_jitter: float = 0.10, motor_tau: float = 0.0) -> None:
        from brain.lif import pick_device
        self.B, self.device, self.dt = int(B), pick_device(device), dt
        self.size_range, self.n_humans_range = size_range, n_humans
        self.K = int(n_humans[1])
        self.hl, self.hw = rover_lw[0] / 2, rover_lw[1] / 2
        self.front_depth = front_depth
        self.human_r_range, self.human_speed_range = human_r, human_speed
        self.ratio, self.T = fly_speed_ratio, episode_s
        self.T_steps = int(round(episode_s / dt))
        self.half_fov = math.radians(hfov_deg) / 2
        self.omega_max, self.margin = omega_max, contact_margin
        self.n_cols = n_cols
        self.heat_range, self.heat_half_cone = heat_range, math.radians(heat_half_cone_deg)
        self.heat_axis = torch.tensor([-math.radians(heat_axis_deg), math.radians(heat_axis_deg)],
                                      device=self.device)              # left sensor, right sensor
        self.heat_tau, self.heat_max_delay = heat_tau, int(round(heat_max_delay_s / dt))
        self.retina_sigma, self.col_dropout, self.gain_jitter = retina_sigma, col_dropout, gain_jitter
        self.motor_tau = motor_tau            # s; 0 = instant velocity (default); >0 = first-order chassis lag
        self.gen = torch.Generator(device="cpu")
        self.stage = "A"
        self.reset(seed=0, stage="A")

    # ------------------------------------------------------------ helpers
    def _u(self, *shape, lo=0.0, hi=1.0):
        return (torch.rand(*shape, generator=self.gen) * (hi - lo) + lo).to(self.device)

    def _rel(self):
        """Human position relative to rover: dist (B,K), bearing (B,K, + = right), local (u, v)."""
        d = self.hxy - self.rxy.unsqueeze(1)
        dist = d.norm(dim=-1).clamp(min=1e-6)
        bearing_ccw = _wrap(torch.atan2(d[..., 1], d[..., 0]) - self.ryaw.unsqueeze(1))
        c, s = torch.cos(self.ryaw).unsqueeze(1), torch.sin(self.ryaw).unsqueeze(1)
        u = c * d[..., 0] + s * d[..., 1]
        v = -s * d[..., 0] + c * d[..., 1]
        return dist, -bearing_ccw, u, v

    def _box_dist(self, u, v, u_lo, u_hi):
        du = torch.maximum(torch.maximum(u_lo - u, u - u_hi), torch.zeros_like(u))
        dv = (v.abs() - self.hw).clamp(min=0.0)
        return torch.sqrt(du * du + dv * dv)

    def _overlap(self, rxy, ryaw, extra=0.0):
        """(B,K) bool: human cylinder overlaps rover box at pose (rxy, ryaw), with extra margin."""
        d = self.hxy - rxy.unsqueeze(1)
        c, s = torch.cos(ryaw).unsqueeze(1), torch.sin(ryaw).unsqueeze(1)
        u = c * d[..., 0] + s * d[..., 1]
        v = -s * d[..., 0] + c * d[..., 1]
        return self.hvalid & (self._box_dist(u, v, -self.hl, self.hl) <= self.hr + extra)

    def in_view(self) -> torch.Tensor:
        """(B,K) bool: any part of the human inside the horizontal FOV."""
        dist, bearing, _, _ = self._rel()
        half_w = torch.asin((self.hr / dist).clamp(max=0.999))
        return self.hvalid & ((bearing - half_w) < self.half_fov) & ((bearing + half_w) > -self.half_fov)

    def nearest(self) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (dist (B,), abs bearing (B,)) to the nearest valid human."""
        dist, bearing, _, _ = self._rel()
        dist = torch.where(self.hvalid, dist, torch.full_like(dist, 1e9))
        j = dist.argmin(dim=1, keepdim=True)
        return dist.gather(1, j).squeeze(1), bearing.gather(1, j).squeeze(1).abs()

    # ------------------------------------------------------------ reset
    @torch.no_grad()
    def reset(self, seed: int | None = None, stage: str | None = None) -> dict:
        B, K, dev = self.B, self.K, self.device
        if seed is not None:
            self.gen.manual_seed(int(seed))
        if stage is not None:
            self.stage = stage
        st = self.stage
        self.L = self._u(B, lo=self.size_range[0], hi=self.size_range[1])
        half = self.L / 2
        # humans
        if st == "A":
            n = torch.ones(B, device=dev, dtype=torch.long)
        else:
            n = torch.randint(self.n_humans_range[0], self.n_humans_range[1] + 1, (B,),
                              generator=self.gen).to(dev)
        self.hvalid = torch.arange(K, device=dev).unsqueeze(0) < n.unsqueeze(1)
        self.hr = self._u(B, K, lo=self.human_r_range[0], hi=self.human_r_range[1])
        hs = self._u(B, lo=self.human_speed_range[0], hi=self.human_speed_range[1])   # per-episode speed
        self.hspeed = hs.unsqueeze(1) * self._u(B, K, lo=0.85, hi=1.15)
        self.hhead = self._u(B, K, lo=-math.pi, hi=math.pi)
        self.htimer = self._u(B, K, lo=2.0, hi=6.0)
        self.hpaused = torch.zeros(B, K, dtype=torch.bool, device=dev)
        # rover
        self.rxy = self._u(B, 2, lo=-1.0, hi=1.0) * (half - 0.6).unsqueeze(1)
        self.ryaw = self._u(B, lo=-math.pi, hi=math.pi)
        self.vmax = self.ratio * hs
        # human placement
        if st == "A":
            d = self._u(B, K, lo=1.0, hi=3.0)
            bearing = self._u(B, K, lo=-0.8, hi=0.8) * self.half_fov          # + = right
            ang = self.ryaw.unsqueeze(1) - bearing                              # world CCW angle
            hxy = self.rxy.unsqueeze(1) + d.unsqueeze(-1) * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
            outside = (hxy.abs() > (half - 0.3).view(B, 1, 1)).any(-1)          # (B,K)
            flip = outside.any(1)                                               # face inward instead
            self.ryaw = torch.where(flip, _wrap(self.ryaw + math.pi), self.ryaw)
            ang = self.ryaw.unsqueeze(1) - bearing
            hxy = self.rxy.unsqueeze(1) + d.unsqueeze(-1) * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        elif st == "B":
            d = self._u(B, K, lo=1.5, hi=6.0)
            off = self._u(B, K, lo=self.half_fov + 0.1, hi=2 * math.pi - self.half_fov - 0.1)
            ang = self.ryaw.unsqueeze(1) + off
            hxy = self.rxy.unsqueeze(1) + d.unsqueeze(-1) * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        else:
            hxy = self._u(B, K, 2, lo=-1.0, hi=1.0) * (half - 0.5).view(B, 1, 1)
        lim = (half - 0.35).view(B, 1, 1)
        self.hxy = torch.maximum(torch.minimum(hxy, lim), -lim)
        # never spawn overlapping the rover: push such humans 1 m straight ahead of themselves
        ov = self._overlap(self.rxy, self.ryaw, extra=0.3)
        self.hxy = torch.where(ov.unsqueeze(-1), self.hxy + torch.stack([torch.cos(self.hhead), torch.sin(self.hhead)], -1),
                               self.hxy)
        self.hxy = torch.maximum(torch.minimum(self.hxy, lim), -lim)
        # episode bookkeeping
        self.t = torch.zeros(B, device=dev)
        self.acquired = torch.zeros(B, dtype=torch.bool, device=dev)
        self.contacted = torch.zeros(B, dtype=torch.bool, device=dev)
        self.first_contact_t = torch.full((B,), float("nan"), device=dev)
        self.first_sight_t = torch.full((B,), float("nan"), device=dev)
        self.view_steps = torch.zeros(B, device=dev)
        self.n_steps = 0
        # domain randomization
        self.sigma_pres = self._u(B, lo=0.0, hi=self.retina_sigma[0])
        self.sigma_size = self._u(B, lo=0.0, hi=self.retina_sigma[1])
        self.col_mask = (self._u(B, self.n_cols) >= self.col_dropout).float()
        self.heat_delay = torch.randint(0, self.heat_max_delay + 1, (B,), generator=self.gen).to(dev)
        self.heat_fp = self._u(B, lo=0.0, hi=0.01)
        self.heat_h = torch.zeros(B, 2, device=dev)
        self.heat_buf = torch.zeros(B, self.heat_max_delay + 1, 2, device=dev)
        self.gain = self._u(B, lo=1 - self.gain_jitter, hi=1 + self.gain_jitter)
        self.v_cmd = torch.zeros(B, device=dev); self.w_cmd = torch.zeros(B, device=dev)
        return self.state()

    def state(self) -> dict:
        dist, bearing = self.nearest()
        return dict(rxy=self.rxy, ryaw=self.ryaw, hxy=self.hxy, hr=self.hr, hvalid=self.hvalid,
                    t=self.t, nearest_dist=dist, nearest_abs_bearing=bearing)

    # ------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, forward: torch.Tensor, turn: torch.Tensor) -> tuple[dict, dict]:
        B, K, dt, dev = self.B, self.K, self.dt, self.device
        half = self.L / 2
        # ---- humans: random walk with pauses, wall bounce
        self.htimer -= dt
        expired = self.htimer <= 0
        will_pause = expired & ~self.hpaused & (torch.rand(B, K, device=dev) < 0.3)
        start_walk = expired & ~will_pause
        self.hhead = torch.where(start_walk, torch.rand(B, K, device=dev) * 2 * math.pi - math.pi, self.hhead)
        self.htimer = torch.where(will_pause, torch.rand(B, K, device=dev) * 2.0,
                                  torch.where(start_walk, torch.rand(B, K, device=dev) * 4.0 + 2.0, self.htimer))
        self.hpaused = torch.where(expired, will_pause, self.hpaused)
        moving = (~self.hpaused & self.hvalid).float()
        prev_hxy = self.hxy
        step_vec = torch.stack([torch.cos(self.hhead), torch.sin(self.hhead)], -1) * (self.hspeed * moving * dt).unsqueeze(-1)
        hxy = self.hxy + step_vec
        lim = (half.view(B, 1) - self.hr)
        hit_x = hxy[..., 0].abs() > lim
        hit_y = hxy[..., 1].abs() > lim
        self.hhead = torch.where(hit_x, _wrap(math.pi - self.hhead), self.hhead)
        self.hhead = torch.where(hit_y, _wrap(-self.hhead), self.hhead)
        hxy = torch.maximum(torch.minimum(hxy, lim.unsqueeze(-1)), -lim.unsqueeze(-1))
        self.hxy = hxy
        # human walked into the rover: undo its move and bounce
        bumped = self._overlap(self.rxy, self.ryaw)
        self.hxy = torch.where(bumped.unsqueeze(-1), prev_hxy, self.hxy)
        self.hhead = torch.where(bumped, _wrap(self.hhead + math.pi), self.hhead)
        # ---- rover: differential drive, walls, cannot push through humans
        forward = forward.clamp(-1, 1); turn = turn.clamp(-1, 1)
        # turn > 0 = turn RIGHT (toward + azimuth) = clockwise = yaw decreases (yaw is CCW-positive)
        if self.motor_tau > 0:                      # chassis inertia / command latency: velocities chase the commands
            a = dt / (self.motor_tau + dt)
            self.v_cmd = self.v_cmd + a * (forward - self.v_cmd)
            self.w_cmd = self.w_cmd + a * (turn - self.w_cmd)
            forward, turn = self.v_cmd, self.w_cmd
        self.ryaw = _wrap(self.ryaw - turn * self.omega_max * dt)
        v = forward * self.vmax
        cand = self.rxy + torch.stack([torch.cos(self.ryaw), torch.sin(self.ryaw)], -1) * (v * dt).unsqueeze(-1)
        rlim = (half - math.hypot(self.hl, self.hw)).unsqueeze(1)
        cand = torch.maximum(torch.minimum(cand, rlim), -rlim)
        blocked = self._overlap(cand, self.ryaw).any(1)
        self.rxy = torch.where(blocked.unsqueeze(1), self.rxy, cand)
        # ---- contacts
        dist, bearing, u, v_ = self._rel()
        thr = self.hr + self.margin
        contact = self.hvalid & (self._box_dist(u, v_, -self.hl, self.hl) <= thr)
        # front touch = touching, and the human's centre lies ahead of the front strip's back edge
        front = contact & (u >= self.hl - self.front_depth)
        contact_any, front_any = contact.any(1), front.any(1)
        back_or_side = contact_any & ~front_any
        # ---- events
        self.t += dt
        self.n_steps += 1
        view_any = self.in_view().any(1)
        self.view_steps += view_any.float()
        acquire = view_any & ~self.acquired
        self.first_sight_t = torch.where(acquire, self.t, self.first_sight_t)
        self.acquired |= view_any
        first_front = front_any & ~self.contacted
        sustained = front_any & self.contacted
        self.first_contact_t = torch.where(first_front, self.t, self.first_contact_t)
        self.contacted |= front_any
        timeout = torch.full((B,), self.n_steps == self.T_steps, dtype=torch.bool, device=dev)
        events = dict(acquire=acquire, front_contact=first_front, sustained=sustained,
                      back_or_side_contact=back_or_side, timeout=timeout, contact_now=contact_any,
                      no_contact_ever=~self.contacted)
        return self.state(), events

    def reward(self, events: dict) -> torch.Tensor:
        r = torch.zeros(self.B, device=self.device)
        if self.stage in ("A", "B"):
            r += events["acquire"].float() * 1.0
        r += events["front_contact"].float() * 5.0
        r += events["sustained"].float() * 0.5
        r -= (~events["contact_now"]).float() * 0.02
        r -= (events["timeout"] & events["no_contact_ever"]).float() * 3.0
        return r

    # ------------------------------------------------------------ senses
    def retina_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.rxy, self.ryaw, self.hxy, self.hr, self.hvalid

    @torch.no_grad()
    def perturb_retina(self, r: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """In-place retina noise: gaussian on pres/size (per-episode sigma) and column dropout."""
        B = self.B
        r["pres"] = ((r["pres"] + torch.randn_like(r["pres"]) * self.sigma_pres.unsqueeze(1)) * self.col_mask).clamp(0, 1)
        r["size"] = ((r["size"] + torch.randn_like(r["size"]) * self.sigma_size.unsqueeze(1)) * self.col_mask).clamp(0, 1)
        return r

    @torch.no_grad()
    def heat(self) -> torch.Tensor:
        """(B, 2) left/right leaky heat detectors with per-episode latency and false positives."""
        dist, bearing, _, _ = self._rel()
        rel = _wrap(bearing.unsqueeze(-1) - self.heat_axis.view(1, 1, 2))            # (B,K,2)
        hit = self.hvalid.unsqueeze(-1) & (rel.abs() <= self.heat_half_cone) & (dist.unsqueeze(-1) <= self.heat_range)
        strength = hit.float() * (1.0 - 0.5 * dist.unsqueeze(-1) / self.heat_range)
        raw = strength.max(dim=1).values                                              # (B,2)
        raw = torch.maximum(raw, (torch.rand_like(raw) < self.heat_fp.unsqueeze(1)).float())
        self.heat_buf = torch.roll(self.heat_buf, 1, dims=1)
        self.heat_buf[:, 0] = raw
        delayed = self.heat_buf[torch.arange(self.B, device=self.device), self.heat_delay]  # (B,2)
        self.heat_h += (self.dt / self.heat_tau) * (delayed - self.heat_h)
        return self.heat_h.clamp(0, 1)

    # ------------------------------------------------------------ metrics
    def episode_stats(self) -> dict:
        c = self.contacted.float()
        ttc = self.first_contact_t[self.contacted]
        return dict(contact_rate=c.mean().item(),
                    mean_time_to_contact=(ttc.mean().item() if ttc.numel() else float("nan")),
                    track_fraction=(self.view_steps / max(self.n_steps, 1)).mean().item(),
                    acquire_rate=self.acquired.float().mean().item(),
                    mean_time_to_first_sight=(self.first_sight_t[self.acquired].mean().item() if self.acquired.any() else float("nan")))


if __name__ == "__main__":
    a = Arena(4, device="cpu")
    s = a.reset(seed=1, stage="A")
    print("stage A nearest dist", s["nearest_dist"], "abs bearing", s["nearest_abs_bearing"])
    for _ in range(100):
        s, ev = a.step(torch.ones(4), torch.zeros(4))
    print("after 2 s straight:", s["nearest_dist"], "contacted", a.contacted, "heat", a.heat())
    print(a.episode_stats())
