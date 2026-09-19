"""
es.py -- fallback: evolution strategies over per-synapse scale factors on the plastic subset.

    learner = ES(lif, groups, plastic="dn_in", sigma=0.05, top_frac=0.2)

Each env b runs its own copy of the plastic synapses, |w|_b = |w| * (1 + sigma * eps_b), for one episode.
The plastic edges are removed from the shared LIF operator and re-added per env through
`lif.plastic_hook` (a gather/scatter over the plastic edges only, so the rest of the brain stays one
shared matrix). At the end of the episode envs are ranked by return and |w| moves toward the mean of
the top-k perturbations. Same mask, same Dale's law (sign fixed), same bounds [0, bound_mult * |w0|].
Embarrassingly parallel. Switch to this if three_factor shows no contact-rate improvement.
"""
from __future__ import annotations

import torch

from learn.plastic import plastic_edges


class ES:
    def __init__(self, lif: object, groups: dict[str, torch.Tensor], plastic: str = "kc_mbon",
                 sigma: float = 0.05, top_frac: float = 0.2, bound_mult: float = 3.0,
                 lr: float = 1.0) -> None:
        self.lif = lif
        self.eid = plastic_edges(lif, groups, plastic)
        self.pre_p, self.post_p = lif.pre_idx[self.eid], lif.post_idx[self.eid]
        w0 = lif.w0[self.eid]
        self.sign = torch.where(w0 < 0, -torch.ones_like(w0), torch.ones_like(w0))
        self.hi = bound_mult * w0.abs()
        self.mag = lif.w[self.eid].abs().clone()                 # current base magnitudes
        self.sigma, self.top_frac, self.lr = sigma, top_frac, lr
        B = lif.B
        self.post_index = self.post_p.unsqueeze(0).expand(B, -1)
        lif.set_w_subset(self.eid, torch.zeros_like(self.mag))   # out of the shared operator...
        lif.plastic_hook = self.hook                             # ...and back in, per env
        self._last_dw, self._last_best, self._last_mean = 0.0, 0.0, 0.0
        self.reset_episode()

    @property
    def n_plastic(self) -> int:
        return int(self.eid.numel())

    def reset_episode(self) -> None:
        B, dev = self.lif.B, self.lif.device
        self.eps = torch.randn(B, self.n_plastic, device=dev)
        self.mag_b = torch.minimum((self.mag * (1 + self.sigma * self.eps)).clamp(min=0.0), self.hi)
        self.w_b = self.mag_b * self.sign * self.lif.g           # effective per-env weights (B, nnz_p)
        self.ret = torch.zeros(B, device=dev)

    @torch.no_grad()
    def hook(self, s: torch.Tensor) -> torch.Tensor:
        """s: spikes (B, N) as float -> per-env synaptic input from the plastic edges (B, N)."""
        contrib = s.float()[:, self.pre_p] * self.w_b
        out = torch.zeros(s.shape[0], self.lif.N, device=s.device)
        return out.scatter_add_(1, self.post_index, contrib)

    def on_substep(self, lif: object) -> None:
        pass

    def update(self, lif: object, reward: torch.Tensor) -> None:
        self.ret += reward

    @torch.no_grad()
    def end_episode(self, lif: object, ret: torch.Tensor) -> None:
        k = max(1, int(round(self.top_frac * lif.B)))
        top = ret.topk(k).indices
        target = self.mag_b[top].mean(0)
        new = torch.minimum((self.mag + self.lr * (target - self.mag)).clamp(min=0.0), self.hi)
        self._last_dw = (new - self.mag).abs().mean().item()
        self._last_best, self._last_mean = ret.max().item(), ret.mean().item()
        self.mag = new

    def full_w(self, lif: object) -> torch.Tensor:
        """Shared weights with the plastic edges filled back in (for checkpoints)."""
        w = lif.w.clone()
        w[self.eid] = self.mag * self.sign
        return w

    def stats(self) -> dict:
        return dict(dw_abs=self._last_dw, dopamine=0.0, es_best=self._last_best, es_mean=self._last_mean)
