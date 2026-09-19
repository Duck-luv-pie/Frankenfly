"""Leaky integrate-and-fire network, parameters and dynamics from Shiu et al. 2024:

    dv/dt = (v_rest - v + g) / tau_mem        (held during refractory period)
    dg/dt = -g / tau_syn                      (alpha synapse, g in mV)
    spike when v > v_thresh: v = v_reset, g = 0, refractory 2.2 ms
    presynaptic spike -> g_post += w after a 1.8 ms delay, w = 0.275 mV * synapse count * (+1 / -1)

Stimulated ("Poisson input") neurons spike as a Poisson process at their drive rate, exactly as
in the reference model, where each Poisson event carries a super-threshold weight.

Integration is exponential Euler at dt = 0.1 ms. The step kernel is compiled with numba when
available (pip extra `fast`), otherwise a vectorized NumPy version is used."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

try:  # optional accelerator
    import numba

    HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    numba = None
    HAVE_NUMBA = False


@dataclass
class LIFParams:
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_thresh_mv: float = -45.0
    tau_mem_ms: float = 20.0
    tau_syn_ms: float = 5.0
    refractory_ms: float = 2.2
    delay_ms: float = 1.8
    dt_ms: float = 0.1
    adapt_mv: float = 0.0        # spike-frequency adaptation: each spike adds this (mV) to a hyperpolarizing variable
    tau_adapt_ms: float = 200.0  # ... that decays with this time constant. 0 = reference model (none)
    depress_u: float = 0.0       # short-term synaptic depression: each presynaptic spike uses this fraction of the
    tau_depress_ms: float = 200.0  # ... neuron's transmitter resources, which recover with this time constant. 0 = none

    @classmethod
    def from_config(cls, lif_cfg: dict) -> "LIFParams":
        return cls(**{k: v for k, v in lif_cfg.items() if k in cls.__dataclass_fields__})


class LIFNetwork:
    def __init__(self, n: int, pre: np.ndarray, post: np.ndarray, weight_mv: np.ndarray,
                 params: LIFParams | None = None, seed: int | None = None, use_numba: bool | None = None):
        self.p = params or LIFParams()
        self.n = n
        W = sp.csr_matrix((weight_mv.astype(np.float32), (pre.astype(np.int64), post.astype(np.int64))), shape=(n, n))
        W.sum_duplicates()
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int32)
        self.data = W.data.astype(np.float32)

        p = self.p
        self.a_mem = math.exp(-p.dt_ms / p.tau_mem_ms)
        self.a_syn = math.exp(-p.dt_ms / p.tau_syn_ms)
        self.a_adapt = math.exp(-p.dt_ms / p.tau_adapt_ms) if p.tau_adapt_ms > 0 else 0.0
        self.a_depress = math.exp(-p.dt_ms / p.tau_depress_ms) if p.tau_depress_ms > 0 else 0.0
        self.ref_steps = max(1, int(round(p.refractory_ms / p.dt_ms)))
        self.delay_steps = max(1, int(round(p.delay_ms / p.dt_ms)))

        self.v = np.full(n, p.v_rest_mv, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.adapt = np.zeros(n, dtype=np.float32)       # adaptation variable (mV, subtracted from drive)
        self.resource = np.ones(n, dtype=np.float32)     # synaptic resources per presynaptic neuron (1 = full)
        self.th_offset = np.zeros(n, dtype=np.float32)   # per-neuron spike threshold offset (mV), e.g. Kenyon cells
        self.ref_left = np.zeros(n, dtype=np.int32)
        # delay ring: spike flags per step for the last delay_steps steps
        self.ring = np.zeros((self.delay_steps, n), dtype=np.bool_)
        self.ring_pos = 0
        self.step_count = 0
        self.rate = np.zeros(n, dtype=np.float32)        # stimulation rate in Hz per neuron
        self.background_hz = 0.0                          # spontaneous activity floor
        self.background_idx = np.arange(n, dtype=np.int32)   # neurons that get the floor
        self.rng = np.random.default_rng(seed)
        self.use_numba = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
        self._seed_numba = int(self.rng.integers(0, 2**31 - 1))

    # ---- stimulation ----------------------------------------------------------------------
    def set_rates(self, rates_hz: np.ndarray) -> None:
        self.rate[:] = rates_hz

    def set_rate(self, idx: np.ndarray, hz: float) -> None:
        self.rate[idx] = hz

    def set_background(self, hz: float, idx: np.ndarray | None = None) -> None:
        """Spontaneous activity: the neurons in `idx` (default: all) fire as a Poisson process at
        `hz` on top of their synaptic input. The reference model has none, but a real brain is
        never silent. Sensory neurons should be excluded so they only fire when stimulated."""
        self.background_hz = float(hz)
        if idx is not None:
            self.background_idx = np.asarray(idx, dtype=np.int32)

    def clear_rates(self) -> None:
        self.rate[:] = 0.0

    @property
    def t_ms(self) -> float:
        return self.step_count * self.p.dt_ms

    # ---- simulation -----------------------------------------------------------------------
    def run(self, steps: int, record: bool = True) -> np.ndarray:
        """Advance `steps` steps. Returns spike counts per neuron (int32[n]) for the interval."""
        counts = np.zeros(self.n, dtype=np.int32)
        if self.use_numba:
            self._seed_numba = (self._seed_numba * 6364136223846793005 + 1442695040888963407) % (2**31 - 1)
            driven_idx = np.nonzero(self.rate > 0)[0].astype(np.int32)
            bg_lambda = float(self.background_hz * self.p.dt_ms * 1e-3 * len(self.background_idx))
            self.ring_pos = _run_numba(
                steps, self.v, self.g, self.adapt, self.resource, self.th_offset, self.ref_left, self.ring, self.ring_pos, self.rate, driven_idx,
                self.background_idx, bg_lambda,
                self.indptr, self.indices, self.data, counts,
                np.float32(self.a_mem), np.float32(self.a_syn), np.float32(self.p.v_rest_mv),
                np.float32(self.p.v_reset_mv), np.float32(self.p.v_thresh_mv), self.ref_steps,
                np.float32(self.p.dt_ms * 1e-3), self._seed_numba, np.float32(self.p.adapt_mv), np.float32(self.a_adapt),
                np.float32(self.p.depress_u), np.float32(self.a_depress))
        else:
            for _ in range(steps):
                self._step_numpy(counts)
        self.step_count += steps
        return counts

    def run_ms(self, ms: float, record: bool = True) -> np.ndarray:
        return self.run(int(round(ms / self.p.dt_ms)), record)

    def _step_numpy(self, counts: np.ndarray) -> None:
        p = self.p
        # 1) deliver delayed spikes
        delivered = self.ring[self.ring_pos]
        src = np.nonzero(delivered)[0]
        if len(src):
            starts = self.indptr[src]
            lens = self.indptr[src + 1] - starts
            total = int(lens.sum())
            if total:
                offs = np.repeat(starts - (np.cumsum(lens) - lens), lens)
                sel = np.arange(total) + offs
                wsel = self.data[sel] * np.repeat(self.resource[src], lens)
                self.g += np.bincount(self.indices[sel], weights=wsel, minlength=self.n).astype(np.float32)
            if p.depress_u > 0:
                self.resource[src] *= (1.0 - p.depress_u)
        if p.depress_u > 0:
            self.resource += (1.0 - self.resource) * (1.0 - self.a_depress)
        # 2) integrate (skip refractory)
        active = self.ref_left <= 0
        self.g[active] *= self.a_syn
        self.adapt *= self.a_adapt
        self.v[active] = p.v_rest_mv + (self.v[active] - p.v_rest_mv) * self.a_mem + (self.g[active] - self.adapt[active]) * (1.0 - self.a_mem)
        self.ref_left[~active] -= 1
        # 3) spikes: threshold crossings + Poisson-driven
        spk = (self.v > p.v_thresh_mv + self.th_offset) & active
        driven = self.rate > 0
        if driven.any():  # driven neurons may fire even while refractory (reference model)
            u = self.rng.random(int(driven.sum()))
            spk[np.nonzero(driven)[0][u < self.rate[driven] * p.dt_ms * 1e-3]] = True
        if self.background_hz > 0 and len(self.background_idx):
            k = self.rng.poisson(self.background_hz * p.dt_ms * 1e-3 * len(self.background_idx))
            if k:
                spk[self.background_idx[self.rng.integers(0, len(self.background_idx), k)]] = True
        if spk.any():
            self.v[spk] = p.v_reset_mv
            self.g[spk] = 0.0
            self.adapt[spk] += p.adapt_mv
            self.ref_left[spk] = self.ref_steps
            counts[spk] += 1
        self.ring[self.ring_pos] = spk
        self.ring_pos = (self.ring_pos + 1) % self.delay_steps


if HAVE_NUMBA:

    @numba.njit(cache=True, fastmath=True)
    def _run_numba(steps, v, g, adapt, resource, th_offset, ref_left, ring, ring_pos, rate, driven_idx, bg_idx, bg_lambda,
                   indptr, indices, data, counts,
                   a_mem, a_syn, v_rest, v_reset, v_th, ref_steps, dt_s, seed, adapt_mv, a_adapt, depress_u, a_depress):
        np.random.seed(seed)
        n = v.shape[0]
        delay_steps = ring.shape[0]
        n_bg = bg_idx.shape[0]
        for _ in range(steps):
            # 1) deliver delayed spikes
            row = ring[ring_pos]
            for i in range(n):
                if row[i]:
                    x = resource[i]
                    for k in range(indptr[i], indptr[i + 1]):
                        g[indices[k]] += data[k] * x
                    if depress_u > 0.0:
                        resource[i] = x * (1.0 - depress_u)
            if depress_u > 0.0:
                for i in range(n):
                    resource[i] += (1.0 - resource[i]) * (1.0 - a_depress)
            # 2) integrate + threshold
            for i in range(n):
                adapt[i] *= a_adapt
                if ref_left[i] > 0:
                    ref_left[i] -= 1
                    row[i] = False
                    continue
                g[i] *= a_syn
                v[i] = v_rest + (v[i] - v_rest) * a_mem + (g[i] - adapt[i]) * (1.0 - a_mem)
                if v[i] > v_th + th_offset[i]:
                    v[i] = v_reset
                    g[i] = 0.0
                    adapt[i] += adapt_mv
                    ref_left[i] = ref_steps
                    counts[i] += 1
                    row[i] = True
                else:
                    row[i] = False
            # 3) forced spikes: sensory drive (no refractory period, as in the reference model)
            for j in range(driven_idx.shape[0]):
                i = driven_idx[j]
                if np.random.random() < rate[i] * dt_s and not row[i]:
                    v[i] = v_reset
                    g[i] = 0.0
                    adapt[i] += adapt_mv
                    ref_left[i] = ref_steps
                    counts[i] += 1
                    row[i] = True
            # 4) spontaneous background: a Poisson number of random eligible neurons
            if bg_lambda > 0.0 and n_bg > 0:
                k = np.random.poisson(bg_lambda)
                for _j in range(k):
                    i = bg_idx[np.random.randint(0, n_bg)]
                    if not row[i]:
                        v[i] = v_reset
                        g[i] = 0.0
                        adapt[i] += adapt_mv
                        ref_left[i] = ref_steps
                        counts[i] += 1
                        row[i] = True
            ring_pos = (ring_pos + 1) % delay_steps
        return ring_pos
