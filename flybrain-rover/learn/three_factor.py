"""
three_factor.py -- reward-modulated STDP on the plastic subset of the real connectome.

    learner = ThreeFactor(lif, groups, plastic="kc_mbon")
    per episode:  learner.reset_episode()
    per 1 ms:     learner.on_substep(lif)        # spike traces -> eligibility
    per env step: learner.update(lif, reward)    # dopamine-gated weight change
    learner.stats() -> {"dw_abs": mean |dw| per update, "dopamine": mean D}

Rule (connectome-pilot style, see README credits):
  eligibility e (nnz_plastic,) shared across the batch (mean over B), decays with tau_e = 1.5 s:
      e += A_plus * pre_trace(20 ms) * post_spk  -  A_minus * post_trace(20 ms) * pre_spk
  dopamine D = rate(PAM) - rate(PPL1), normalised by `dopamine_norm` Hz, averaged over B; every env step:
      dw  = eta * e * D
      |w| = clip(|w| + dw, 0, bound_mult * |w0|);  w = sign(w0) * |w|      (Dale's law by construction)
  A_plus = A_minus = 1.0, scaled by the initial weight magnitude. eta 3e-4 (was 1e-3 until 2026-09-18).
  Homeostatic synaptic scaling (added 2026-09-18, Taka's call): every env step, a plastic post neuron whose
  batch-mean rate exceeds r_target (150 Hz) multiplies the magnitude of all its incoming plastic synapses by
  (1 - eta_h * (rate/r_target - 1)), eta_h 0.02. Textbook Turrigiano scaling, not a new mechanism; it is what
  stops DNa02 pinning at 300 Hz under reward-only dopamine.
The dopamine neurons themselves are driven by the reward current in brain/senses.py, so the gate is
the fly's own PAM/PPL1 activity, not the scalar reward.
"""
from __future__ import annotations

import math

import torch

from brain.lif import expand_ranges
from learn.plastic import plastic_edges


class ThreeFactor:
    def __init__(self, lif: object, groups: dict[str, torch.Tensor], plastic: str = "kc_mbon",
                 eta: float = 3e-4, A_plus: float = 1.0, A_minus: float = 1.0,
                 tau_e: float = 1.5, tau_trace: float = 0.020, bound_mult: float = 3.0,
                 dopamine_norm: float = 100.0, r_target: float = 150.0, eta_h: float = 0.02) -> None:
        self.lif, self.groups = lif, groups
        self.eid = plastic_edges(lif, groups, plastic)
        self.pre_p, self.post_p = lif.pre_idx[self.eid], lif.post_idx[self.eid]
        w0 = lif.w0[self.eid]
        self.sign = torch.where(w0 < 0, -torch.ones_like(w0), torch.ones_like(w0))
        self.mag0 = w0.abs()
        self.hi = bound_mult * self.mag0
        self.eta, self.A_plus, self.A_minus, self.norm = eta, A_plus, A_minus, dopamine_norm
        self.r_target, self.eta_h = r_target, eta_h
        self.trace_decay = math.exp(-lif.dt / tau_trace)
        self.e_decay = math.exp(-lif.dt / tau_e)
        self.pam, self.ppl1 = groups["PAM"], groups["PPL1"]
        # plastic edges grouped by post neuron (LTP: a post spike touches its incoming plastic edges) and by pre
        # neuron (LTD: a pre spike touches its outgoing plastic edges), so each 1 ms update only visits edges
        # of neurons that spiked: ~spikes x plastic degree instead of B x nnz_plastic (the dense version).
        N, dev = lif.N, lif.device
        self.by_post = torch.argsort(self.post_p)
        cnt = torch.bincount(self.post_p, minlength=N)
        self.post_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)])
        self.by_pre = torch.argsort(self.pre_p)
        cnt = torch.bincount(self.pre_p, minlength=N)
        self.pre_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)])
        self.sparse_update = True
        self.reset_episode()

    @property
    def n_plastic(self) -> int:
        return int(self.eid.numel())

    def reset_episode(self) -> None:
        B, N, dev = self.lif.B, self.lif.N, self.lif.device
        self.e = torch.zeros(self.n_plastic, device=dev)
        self.pre_tr = torch.zeros(B, N, device=dev)
        self.post_tr = torch.zeros(B, N, device=dev)
        self._dw_sum, self._n_upd, self._D_sum, self._n_steps, self._n_scaled = 0.0, 0, 0.0, 0, 0

    @torch.no_grad()
    def on_substep(self, lif: object) -> None:
        if not self.sparse_update:
            return self._on_substep_dense(lif)
        spk = lif.spk
        B = spk.shape[0]
        self.e.mul_(self.e_decay)
        act = spk.nonzero()                                      # (m, 2) = [env, neuron], every spike this ms
        if act.numel():
            b, j = act[:, 0], act[:, 1]
            # LTP: post spiked -> e += A_plus * pre_trace[env, pre] * |w0|  on that post's incoming plastic edges
            pos, k = expand_ranges(self.post_ptr[j], self.post_ptr[j + 1] - self.post_ptr[j])
            if pos.numel():
                eid = self.by_post[pos]
                self.e.index_add_(0, eid, self.A_plus * self.pre_tr[b[k], self.pre_p[eid]] * self.mag0[eid] / B)
            # LTD: pre spiked -> e -= A_minus * post_trace[env, post] * |w0|  on that pre's outgoing plastic edges
            pos, k = expand_ranges(self.pre_ptr[j], self.pre_ptr[j + 1] - self.pre_ptr[j])
            if pos.numel():
                eid = self.by_pre[pos]
                self.e.index_add_(0, eid, -self.A_minus * self.post_tr[b[k], self.post_p[eid]] * self.mag0[eid] / B)
        spkf = spk.float()
        self.pre_tr.mul_(self.trace_decay).add_(spkf)
        self.post_tr.mul_(self.trace_decay).add_(spkf)

    @torch.no_grad()
    def _on_substep_dense(self, lif: object) -> None:
        """Reference implementation (B x nnz_plastic gathers); kept for the equivalence test."""
        spk = lif.spk.float()
        pre_s, post_s = spk[:, self.pre_p], spk[:, self.post_p]                 # (B, nnz_p)
        ltp = self.pre_tr[:, self.pre_p] * post_s
        ltd = self.post_tr[:, self.post_p] * pre_s
        de = (self.A_plus * ltp - self.A_minus * ltd).mean(0) * self.mag0
        self.e.mul_(self.e_decay).add_(de)
        self.pre_tr.mul_(self.trace_decay).add_(spk)
        self.post_tr.mul_(self.trace_decay).add_(spk)

    @torch.no_grad()
    def update(self, lif: object, reward: torch.Tensor | None = None) -> None:
        D = ((lif.rates(self.pam) - lif.rates(self.ppl1)).mean() / self.norm).item()
        self._D_sum += D; self._n_steps += 1
        mag = lif.w[self.eid].abs()
        new_mag = mag
        if abs(D) >= 1e-6:
            new_mag = (mag + self.eta * self.e * D).clamp(min=0.0)
        # homeostatic synaptic scaling (Turrigiano): a post neuron firing above r_target shrinks the magnitude
        # of ALL its incoming plastic synapses multiplicatively. Sign-blind, so Dale's law is untouched.
        if self.eta_h > 0:
            over = (lif.rate.mean(0)[self.post_p] / self.r_target - 1.0).clamp(min=0.0)
            if bool((over > 0).any()):
                new_mag = new_mag * (1.0 - self.eta_h * over).clamp(min=0.0)
                self._n_scaled += 1
        new_mag = torch.minimum(new_mag, self.hi)
        dw = new_mag - mag
        if bool((dw != 0).any()):
            lif.set_w_subset(self.eid, self.sign * new_mag)
            self._dw_sum += dw.abs().mean().item(); self._n_upd += 1

    def end_episode(self, lif: object, ret: torch.Tensor) -> None:
        pass

    def full_w(self, lif: object) -> torch.Tensor:
        return lif.w

    def stats(self) -> dict:
        return dict(dw_abs=self._dw_sum / max(self._n_upd, 1), dopamine=self._D_sum / max(self._n_steps, 1),
                    n_updates=self._n_upd, n_scaled=self._n_scaled)
