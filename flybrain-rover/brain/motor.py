"""
motor.py -- descending neurons -> (forward, turn) in [-1, 1].

turn    = k_t * (rate(DNa02_R) - rate(DNa02_L))       see-saw steering
forward = k_f * mean rate over DN_ALL                  population descending activity (default, 2026-09-18)
          or k_f * rate(DNa01_L + DNa01_R + DNp09)     forward_source="dna01_dnp09" (the original read-out;
                                                       nothing in the eye drives those two, see STATUS.md)
escape  = if GF bursts and dodge mode is on, override with back-and-turn
k_t, k_f are part of the parameter sweep after the global gain. k_f 0.3 for the DN_ALL mean (M3b sweep
2026-09-18: 0.3 → advance 100%, turn-toward 75%; 0.6 → 0.46 m in 2 s but 72%); 0.01 for the old read-out.
"""
from __future__ import annotations

import torch


class Motor:
    def __init__(self, groups: dict[str, torch.Tensor], k_t: float = 0.02, k_f: float = 0.3,
                 gf_thresh: float = 30.0, dodge: bool = False, device: str | torch.device | None = None,
                 forward_source: str = "dn_all") -> None:
        from brain.lif import pick_device
        device = pick_device(device)
        self.g = {k: v.to(device) for k, v in groups.items()}
        self.k_t, self.k_f, self.gf_thresh, self.dodge = k_t, k_f, gf_thresh, dodge
        if forward_source == "dn_all" and "DN_ALL" not in self.g:
            forward_source = "dna01_dnp09"          # old brain.npz without the DN_ALL key
        self.forward_source = forward_source

    @torch.no_grad()
    def decode(self, lif: object) -> tuple[torch.Tensor, torch.Tensor]:
        rL, rR = lif.rates(self.g["DNa02_L"]), lif.rates(self.g["DNa02_R"])
        if self.forward_source == "dn_all":
            fwd = lif.rates(self.g["DN_ALL"])                        # mean Hz over every descending neuron
        else:
            fwd = lif.rates(self.g["DNa01_L"]) + lif.rates(self.g["DNa01_R"]) + lif.rates(self.g["DNp09"])
        turn = (self.k_t * (rR - rL)).clamp(-1, 1)
        forward = (self.k_f * fwd).clamp(-1, 1)
        if self.dodge:
            gf = lif.rates(self.g["GF"]) > self.gf_thresh
            forward = torch.where(gf, torch.full_like(forward, -1.0), forward)
            turn = torch.where(gf, torch.ones_like(turn), turn)
        return forward, turn
