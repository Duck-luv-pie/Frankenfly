"""
explore.py -- EXPLORATORY STATE, INTERNAL DRIVE, NOT SENSORY.

Nothing here comes from the camera. This is an internal state, the reverse of the old tonic fallback:
when no human has been in the retina for `onset_s` (0.5 s) the fly is "searching", and searching means
  1. a small tonic current into DNa01_L/R (amp_tonic ~0.4)  = baseline walking drive, and
  2. a saccade generator: a 300 ms burst into DNa02_L, then DNa02_R, alternating every ~1.5 s with jitter,
     so the camera sweeps the room.
Both are gated OFF the moment any retina column shows presence (within one env step, 20 ms < 100 ms).
If the heat sensors disagree, bursts are biased toward the warmer side (thermo_bias); with no heat signal the
generator alternates strictly. Real flies do this: locomotor and search states are set by internal
neuromodulatory drive, not by the eye. It is labelled as such everywhere; it is not a learned or sensory pathway.

    ex = Exploratory(groups, N, B, device)
    ex.reset()
    exploring = ex.inject(I, retina_out, heat_LR)     # adds current into I in place, returns (B,) bool
"""
from __future__ import annotations

import torch


class Exploratory:
    def __init__(self, groups: dict[str, torch.Tensor], N: int, B: int, device: torch.device | str | None = None,
                 dt: float = 0.02, amp_tonic: float = 0.4, amp_saccade: float = 1.0, burst_s: float = 0.3,
                 period_s: float = 1.5, jitter: float = 0.3, onset_s: float = 0.5, presence_thr: float = 0.05,
                 thermo_bias: bool = True) -> None:
        from brain.lif import pick_device
        self.device = pick_device(device)
        self.B, self.N, self.dt = int(B), int(N), dt
        self.g = {k: groups[k].to(self.device) for k in ("DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R")}
        self.amp_tonic, self.amp_saccade = amp_tonic, amp_saccade
        self.burst_s, self.period_s, self.jitter, self.onset_s = burst_s, period_s, jitter, onset_s
        self.presence_thr, self.thermo_bias = presence_thr, thermo_bias
        self.reset()

    def reset(self) -> None:
        B, dev = self.B, self.device
        self.t_since_seen = torch.zeros(B, device=dev)              # starts at 0: exploration begins after onset_s
        self.clock = torch.zeros(B, device=dev)
        self.next_burst = self.period_s * torch.rand(B, device=dev) * 0.5   # first burst soon after onset
        self.burst_left = torch.zeros(B, device=dev)
        self.side = torch.where(torch.rand(B, device=dev) < 0.5, -torch.ones(B, device=dev), torch.ones(B, device=dev))
        self.exploring = torch.zeros(B, dtype=torch.bool, device=dev)
        self.explore_steps = torch.zeros(B, device=dev)
        self.n_bursts = torch.zeros(B, device=dev)

    @torch.no_grad()
    def inject(self, I: torch.Tensor, r: dict, heat_LR: torch.Tensor | None = None) -> torch.Tensor:
        dt, dev = self.dt, self.device
        present = (r["pres"] > self.presence_thr).any(dim=1)
        self.t_since_seen = torch.where(present, torch.zeros_like(self.t_since_seen), self.t_since_seen + dt)
        exploring = self.t_since_seen >= self.onset_s
        self.exploring = exploring
        self.explore_steps += exploring.float()
        ex = exploring.float().unsqueeze(1)
        # 1. tonic walking drive
        I[:, self.g["DNa01_L"]] += self.amp_tonic * ex
        I[:, self.g["DNa01_R"]] += self.amp_tonic * ex
        # 2. saccade generator
        self.clock = torch.where(exploring, self.clock + dt, torch.zeros_like(self.clock))
        start = exploring & (self.clock >= self.next_burst) & (self.burst_left <= 0)
        alt = -self.side
        new_side = alt
        if self.thermo_bias and heat_LR is not None:
            hl, hr = heat_LR[:, 0], heat_LR[:, 1]
            tot = hl + hr
            have_heat = tot > 0.05
            p_right = 0.5 + 0.45 * (hr - hl) / (tot + 1e-3)
            biased = torch.where(torch.rand_like(p_right) < p_right, torch.ones_like(alt), -torch.ones_like(alt))
            new_side = torch.where(have_heat, biased, alt)
        u = torch.rand(self.B, device=dev) * 2 - 1
        self.side = torch.where(start, new_side, self.side)
        self.burst_left = torch.where(start, self.burst_s * (1 + self.jitter * u), self.burst_left)
        u2 = torch.rand(self.B, device=dev) * 2 - 1
        self.next_burst = torch.where(start, self.clock + self.period_s * (1 + self.jitter * u2), self.next_burst)
        self.n_bursts += start.float()
        active = exploring & (self.burst_left > 0)
        I[:, self.g["DNa02_L"]] += (self.amp_saccade * (active & (self.side < 0)).float()).unsqueeze(1)
        I[:, self.g["DNa02_R"]] += (self.amp_saccade * (active & (self.side > 0)).float()).unsqueeze(1)
        self.burst_left = (self.burst_left - dt).clamp(min=0.0)
        # leaving exploration resets the generator so the next search starts fresh
        self.next_burst = torch.where(exploring, self.next_burst, self.period_s * torch.rand(self.B, device=dev) * 0.5)
        self.burst_left = torch.where(exploring, self.burst_left, torch.zeros_like(self.burst_left))
        return exploring

    def stats(self, n_steps: int) -> dict:
        return dict(explore_fraction=(self.explore_steps / max(n_steps, 1)).mean().item(),
                    bursts_per_env=self.n_bursts.mean().item())
