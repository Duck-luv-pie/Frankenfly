"""
senses.py -- retina columns + heat -> external current on the fly's eye neurons.

This is where the 180-degree-turn fix lives: small-object detectors (LC11/12/15)
are driven by pres*(1-size) so they fade as the human fills the view, LC10a
takes over with pres*size, and LC4/LPLC2 carry looming. Left columns feed
left-hemisphere neurons, right feed right.

    senses = Senses(groups, N)
    I_ext = senses.inject(retina_out, heat_LR)   # (B, N)
    senses.dopamine(I_ext, reward)               # reward (B,) -> PAM (+) / PPL1 (-)

amp_tonic (default 0 = off) is the tonic-locomotion FALLBACK: a constant current into DNa01_L/R
standing in for the fly's baseline walking drive, gated off on contact. It is a motor-side parameter
like k_f, not part of the connectome; any result using it must say so.
Amplitudes (mV per brain step on the driven neurons). amp_track was raised 1.2 -> 20 on 2026-09-17:
size = angular width / hfov is only 0.1-0.3 at 1-3 m, so pres*size*1.2 never crossed threshold and
LC10a (the only eye population that reaches DNa02, see scripts/probe.py) stayed silent. With 20 the
M3 check passes (72-77% of envs turn toward the human, scripts/milestones.py m3).
"""
from __future__ import annotations

import torch


class Senses:
    def __init__(self, groups: dict[str, torch.Tensor], N: int, n_cols: int = 24,
                 device: str | torch.device | None = None,
                 amp_small: float = 1.2, amp_track: float = 20.0, amp_loom: float = 2.0,
                 amp_heat: float = 0.8, amp_tonic: float = 0.0,
                 col_of: dict[str, torch.Tensor] | None = None) -> None:
        from brain.lif import pick_device
        device = pick_device(device)
        self.N, self.n, self.device = N, n_cols, device
        self.amp = dict(small=amp_small, track=amp_track, loom=amp_loom, heat=amp_heat, tonic=amp_tonic)
        half = n_cols // 2
        self.g = {k: v.to(device) for k, v in groups.items()}
        # each neuron in a retinotopic population gets one column within its hemisphere
        # each neuron in a retinotopic population gets one column within its hemisphere.
        # col_of (brain/retinotopy.py, real lobula footprints) overrides the bodyId-order spread below;
        # column -1 = receptive field outside the camera, never driven.
        self.col_of, self._drive_idx, self._drive_col = {}, {}, {}
        for pop in ["LC10a", "LC11", "LC12", "LC15"]:
            for side, offset in (("L", 0), ("R", half)):
                key = f"{pop}_{side}"
                idx = self.g[key]
                if col_of is not None and key in col_of:
                    col = col_of[key].to(device).long()
                else:
                    k = torch.arange(len(idx), device=device)
                    col = offset + (k * half) // max(len(idx), 1)
                self.col_of[key] = col
                ok = col >= 0
                self._drive_idx[key], self._drive_col[key] = idx[ok], col[ok]

    @torch.no_grad()
    def inject(self, r: dict[str, torch.Tensor], heat_LR: torch.Tensor,
               contact: torch.Tensor | None = None) -> torch.Tensor:
        B = r["pres"].shape[0]
        I = torch.zeros(B, self.N, device=self.device)
        if self.amp["tonic"] > 0:
            # FALLBACK, NOT BIOLOGY (OVERNIGHT.md Task 1): a constant "baseline walking" current into
            # DNa01_L/R, gated off while the rover is touching a human. Off by default (amp_tonic=0).
            gate = torch.ones(B, 1, device=self.device) if contact is None else (~contact).float().unsqueeze(1)
            for key in ("DNa01_L", "DNa01_R"):
                I[:, self.g[key]] = self.amp["tonic"] * gate
        small = (r["pres"] * (1.0 - r["size"])) * self.amp["small"]     # (B, n) prefers small
        track = (r["pres"] * r["size"]) * self.amp["track"]             # (B, n) prefers big
        for pop, drive in (("LC11", small), ("LC12", small), ("LC15", small), ("LC10a", track)):
            for side in ("L", "R"):
                key = f"{pop}_{side}"
                I[:, self._drive_idx[key]] = drive[:, self._drive_col[key]]
        for pop in ("LC4", "LPLC2"):
            I[:, self.g[f"{pop}_L"]] = (r["loom_L"] * self.amp["loom"]).unsqueeze(1)
            I[:, self.g[f"{pop}_R"]] = (r["loom_R"] * self.amp["loom"]).unsqueeze(1)
        I[:, self.g["THERMO_L"]] = (heat_LR[:, 0] * self.amp["heat"]).unsqueeze(1)
        I[:, self.g["THERMO_R"]] = (heat_LR[:, 1] * self.amp["heat"]).unsqueeze(1)
        return I

    @torch.no_grad()
    def dopamine(self, I: torch.Tensor, reward: torch.Tensor, amp: float = 2.0, pun_gain: float = 5.0) -> torch.Tensor:
        """reward (B,): positive -> PAM, negative -> PPL1. Modifies I in place."""
        I[:, self.g["PAM"]] += (reward.clamp(min=0) * amp).unsqueeze(1)
        # punishment channel weighted pun_gain x (2026-09-18): with 1x the -0.02/step penalty never reached PPL1
        # threshold (~0.35 mV/step), so learning was potentiation-only and saturated DNa02 at 300 Hz.
        I[:, self.g["PPL1"]] += ((-reward).clamp(min=0) * amp * pun_gain).unsqueeze(1)
        return I
