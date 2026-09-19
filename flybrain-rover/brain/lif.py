"""
lif.py -- batched leaky integrate-and-fire over the real connectome.

    lif = LIF(W_indices, W_values, N, B)   # W[post, pre] signed synapse counts
    spikes = lif.step(I_ext)               # I_ext (B, N) -> spikes (B, N) bool
    lif.rates(groups["DNa02_L"])           # EMA firing rate per batch element, Hz
    lif.w / lif.set_w(w)                   # (nnz,) signed synapse weights, learners write here

Model (Shiu et al. 2024, Nature, "A Drosophila computational brain model reveals
sensorimotor processing"; constants as used in their Brian2 model):
    tau_m 20 ms, v_rest -52 mV, v_th -45 mV, v_reset -52 mV, t_ref 2.2 ms,
    exponential synapse tau_syn 5 ms, w_syn = 0.275 mV per synapse count.
    dV/dt = (v_rest - V + g_syn) / tau_m ;  g_syn += g * W . spikes ; g_syn decays with tau_syn
I_ext is added directly to V each step (mV per step), so a constant I_ext of x drives V
towards v_rest + x * tau_m / dt.

Global gain `g` (mV per synapse count) is THE first thing to sweep: too low = silent brain,
too high = every group at 200+ Hz (seizure). Shiu's whole-brain value is 0.275; on this 15k-neuron
cut the M2 sweep (scripts/milestones.py m2, 2026-09-17) passes for 0.025-0.15 and seizes at 0.275,
so the default is 0.05 (DNa02 ipsilateral 126 Hz, contralateral 0 Hz, nothing else above 110 Hz).

Engines:
  "sparse"  CSR matmul (CUDA/CPU): cost ~ nnz x B every step, whatever the activity.
  "dense"   N x N matrix (needed on Apple MPS, which has no sparse matmul).
  "event"   event-driven: gathers only the outgoing synapses of neurons that spiked this step
            (cost ~ spikes x out-degree, batched over B with one index_add_, no Python loop over envs).
            Fastest when activity is sparse (it is: a few hundred of 15k neurons per ms). Any device.
  "auto"    sparse on CUDA, dense on MPS, event on CPU.
Edges are stored sorted by (post, pre) so the (nnz,) weight vector is CSR-ordered; the event engine
keeps a permutation into CSC (by pre) order.
"""
from __future__ import annotations

import math

import torch


def pick_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def expand_ranges(start: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate integer ranges [start_k, start_k + length_k). Returns (positions (K,), owner k (K,))."""
    total = int(length.sum())
    if total == 0:
        e = torch.zeros(0, dtype=torch.long, device=start.device)
        return e, e
    first = torch.cumsum(length, 0) - length
    k = torch.repeat_interleave(torch.arange(len(length), device=start.device), length)
    pos = start[k] + torch.arange(total, device=start.device) - first[k]
    return pos, k


def load_brain(path: str = "data/brain.npz") -> tuple[object, int, dict[str, torch.Tensor]]:
    """-> (d, N, groups) where groups = {key: LongTensor of neuron indices}. No device placement."""
    import numpy as np
    d = np.load(path, allow_pickle=False)
    N = int(d["N"])
    skip = {"N", "W_indices", "W_values", "ids", "types", "instances", "nt", "sides", "superclass", "meta"}
    groups = {k: torch.as_tensor(d[k], dtype=torch.long) for k in d.files
              if k not in skip and d[k].dtype.kind in "iu" and not k.startswith("col_of_")}
    return d, N, groups


class LIF:
    def __init__(self, W_indices: torch.Tensor, W_values: torch.Tensor, N: int, B: int,
                 device: str | torch.device | None = None, engine: str = "auto",
                 tau_m: float = 0.020, v_rest: float = -52.0, v_th: float = -45.0, v_reset: float = -52.0,
                 t_ref: float = 0.0022, tau_syn: float = 0.005, dt: float = 0.001, g: float = 0.05,
                 rate_tau: float = 0.050, dtype: torch.dtype = torch.float32) -> None:
        self.device = pick_device(device)
        self.N, self.B, self.dt, self.dtype = int(N), int(B), dt, dtype
        self.tau_m, self.v_rest, self.v_th, self.v_reset = tau_m, v_rest, v_th, v_reset
        self.ref_steps = int(round(t_ref / dt))
        self.rate_a = dt / rate_tau
        self.syn_decay = math.exp(-dt / tau_syn)
        self.g = float(g)
        self.plastic_hook = None

        # ---- edges, sorted by (post, pre) so w is CSR ordered
        Wi = torch.as_tensor(W_indices, dtype=torch.long).cpu()
        Wv = torch.as_tensor(W_values, dtype=torch.float32).cpu()
        key = Wi[0] * self.N + Wi[1]
        uniq, inv = torch.unique(key, return_inverse=True)   # sorted by (post, pre); sums duplicates
        w = torch.zeros(uniq.numel(), dtype=torch.float32).scatter_add_(0, inv, Wv)
        self.post_idx = (uniq // self.N).to(self.device)
        self.pre_idx = (uniq % self.N).to(self.device)
        self._w = w.to(self.device)                      # signed synapse counts (unscaled)
        self.w0 = self._w.clone()
        self.nnz = int(self._w.numel())

        if engine == "auto":
            engine = {"mps": "dense", "cuda": "sparse"}.get(self.device.type, "event")
        self.engine = engine
        if engine == "sparse":
            counts = torch.bincount(self.post_idx, minlength=self.N)
            self.crow = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device),
                                   counts.cumsum(0)])
            self._build_sparse()
        elif engine == "dense":
            # Wt[pre, post] so that syn_in = spk (B, pre) @ Wt -> (B, post)
            self.Wt = torch.zeros(self.N, self.N, device=self.device, dtype=dtype)
            self._build_dense()
        elif engine == "event":
            # CSC view: edges grouped by pre neuron. csc_perm maps CSC position -> CSR edge id.
            self.csc_perm = torch.argsort(self.pre_idx * self.N + self.post_idx)
            self.csc_post = self.post_idx[self.csc_perm]
            counts = torch.bincount(self.pre_idx, minlength=self.N)
            self.colptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), counts.cumsum(0)])
            self._build_event()
        else:
            raise ValueError(engine)
        self.reset()

    # ------------------------------------------------------------ weights
    def _build_sparse(self):
        self.W_csr = torch.sparse_csr_tensor(self.crow, self.pre_idx, (self._w * self.g).to(self.dtype),
                                             size=(self.N, self.N), device=self.device)

    def _build_dense(self):
        self.Wt.zero_()
        self.Wt[self.pre_idx, self.post_idx] = (self._w * self.g).to(self.dtype)

    def _build_event(self):
        self.csc_val = (self._w * self.g)[self.csc_perm]

    def _build(self):
        {"sparse": self._build_sparse, "dense": self._build_dense, "event": self._build_event}[self.engine]()

    @property
    def w(self) -> torch.Tensor:
        return self._w

    @torch.no_grad()
    def set_w(self, w: torch.Tensor) -> None:
        """Replace all edge weights (nnz,), same order as lif.w. Rebuilds the operator."""
        self._w.copy_(w)
        self._build()

    @torch.no_grad()
    def set_w_subset(self, edge_ids: torch.Tensor, w_sub: torch.Tensor) -> None:
        """Cheaper update for a subset of edges (learners)."""
        self._w[edge_ids] = w_sub
        if self.engine == "dense":
            self.Wt[self.pre_idx[edge_ids], self.post_idx[edge_ids]] = (w_sub * self.g).to(self.dtype)
        else:
            self._build()

    def set_gain(self, g: float) -> None:
        """Global gain (rebuilds the operator). For per-env gain use lif.gain_b (B,) instead."""
        self.g = float(g)
        self._build()

    def set_gain_b(self, gain_b: list[float] | torch.Tensor) -> None:
        """Per-env multiplier on synaptic input, e.g. a gain sweep or +-10% domain randomization."""
        self.gain_b = torch.as_tensor(gain_b, dtype=torch.float32, device=self.device).reshape(self.B)

    # ------------------------------------------------------------ dynamics
    def reset(self) -> None:
        self.V = torch.full((self.B, self.N), self.v_rest, device=self.device)
        self.gsyn = torch.zeros((self.B, self.N), device=self.device)
        self.ref = torch.zeros((self.B, self.N), dtype=torch.int32, device=self.device)
        self.spk = torch.zeros((self.B, self.N), dtype=torch.bool, device=self.device)
        self.rate = torch.zeros((self.B, self.N), device=self.device)   # Hz, EMA
        if not hasattr(self, "gain_b"):
            self.gain_b = torch.ones(self.B, device=self.device)        # per-env synaptic gain multiplier

    def _syn_input(self):
        s = self.spk.to(self.dtype)
        if self.engine == "sparse":
            out = torch.sparse.mm(self.W_csr, s.t()).t().float()       # (post,pre)@(pre,B) -> (B,post)
        elif self.engine == "dense":
            out = (s @ self.Wt).float()
        else:
            out = self._event_input()
        if self.plastic_hook is not None:                              # per-env plastic edges (learn/es.py)
            out = out + self.plastic_hook(s)
        return out

    def _event_input(self):
        """Sum outgoing synapses of every (env, pre) that spiked, in one batched gather + index_add_."""
        out = torch.zeros(self.B * self.N, device=self.device)
        act = self.spk.nonzero()                                        # (m, 2) = [env, pre]
        if act.numel() == 0:
            return out.view(self.B, self.N)
        b, pre = act[:, 0], act[:, 1]
        start, length = self.colptr[pre], self.colptr[pre + 1] - self.colptr[pre]
        edge, k = expand_ranges(start, length)
        if edge.numel() == 0:
            return out.view(self.B, self.N)
        out.index_add_(0, b[k] * self.N + self.csc_post[edge], self.csc_val[edge])
        return out.view(self.B, self.N)

    @torch.no_grad()
    def step(self, I_ext: torch.Tensor) -> torch.Tensor:
        self.gsyn = self.gsyn * self.syn_decay + self._syn_input() * self.gain_b.unsqueeze(1)
        dV = (self.dt / self.tau_m) * (self.v_rest - self.V + self.gsyn) + I_ext
        active = self.ref <= 0
        self.V = torch.where(active, self.V + dV, self.V)
        self.spk = active & (self.V >= self.v_th)
        self.V = torch.where(self.spk, torch.full_like(self.V, self.v_reset), self.V)
        self.ref = torch.where(self.spk, torch.full_like(self.ref, self.ref_steps), self.ref - 1)
        self.rate += self.rate_a * (self.spk.float() / self.dt - self.rate)
        return self.spk

    # ------------------------------------------------------------ lesions
    @torch.no_grad()
    def lesion(self, group_idx: torch.Tensor, mode: str = "both") -> dict:
        """Silence a neuron group by zeroing its synapses. mode: "out" = outgoing (columns of Wt),
        "in" = incoming, "both" = the neuron is effectively removed. Returns a snapshot for restore()."""
        idx = torch.as_tensor(group_idx, dtype=torch.long, device=self.device)
        pre_cpu, post_cpu, idx_cpu = self.pre_idx.cpu(), self.post_idx.cpu(), idx.cpu()
        mask = torch.zeros(self.nnz, dtype=torch.bool)
        if mode in ("out", "both"):
            mask |= torch.isin(pre_cpu, idx_cpu)
        if mode in ("in", "both"):
            mask |= torch.isin(post_cpu, idx_cpu)
        eid = mask.nonzero().flatten().to(self.device)
        snap = dict(eid=eid, w=self._w[eid].clone(), mode=mode, n_neurons=int(idx.numel()))
        self.set_w_subset(eid, torch.zeros_like(snap["w"]))
        return snap

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        """Undo lesion(): puts the saved weights back on exactly the edges that were zeroed."""
        self.set_w_subset(snap["eid"], snap["w"])

    def rates(self, idx: torch.Tensor) -> torch.Tensor:
        """Mean EMA firing rate (Hz) over a neuron index group -> (B,)"""
        return self.rate[:, idx].mean(dim=1)

    def rate_table(self, groups: dict[str, torch.Tensor], keys: list[str] | None = None) -> dict[str, float]:
        keys = keys or list(groups)
        return {k: float(self.rates(groups[k]).mean()) for k in keys if len(groups[k])}


if __name__ == "__main__":
    import sys
    d, N, groups = load_brain()
    B = 8
    g = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, B, g=g)
    print(f"device {lif.device} engine {lif.engine} N {N} nnz {lif.nnz:,} g {g}")
    groups = {k: v.to(lif.device) for k, v in groups.items()}
    I = torch.zeros(B, N, device=lif.device)
    I[:, groups["LC10a_L"]] = 1.5          # poke the left eye only
    for t in range(300):                    # 300 ms
        lif.step(I)
    for k in ["LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF", "KC", "MBON", "PAM", "PPL1"]:
        print(f"{k:10s} {lif.rates(groups[k]).mean():7.1f} Hz")
    # M2 passes if DNa02_L > DNa02_R, nothing > 200 Hz, not all zero. If wrong: sweep g.
