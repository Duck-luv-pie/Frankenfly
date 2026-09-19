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
        self.ref_steps = max(1, int(round(p.refractory_ms / p.dt_ms)))
        self.delay_steps = max(1, int(round(p.delay_ms / p.dt_ms)))

        self.v = np.full(n, p.v_rest_mv, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.ref_left = np.zeros(n, dtype=np.int32)
        # delay ring: spike flags per step for the last delay_steps steps
        self.ring = np.zeros((self.delay_steps, n), dtype=np.bool_)
        self.ring_pos = 0
        self.step_count = 0
        self.rate = np.zeros(n, dtype=np.float32)        # stimulation rate in Hz per neuron
        self.rng = np.random.default_rng(seed)
        self.use_numba = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
        self._seed_numba = int(self.rng.integers(0, 2**31 - 1))

    # ---- stimulation ----------------------------------------------------------------------
    def set_rates(self, rates_hz: np.ndarray) -> None:
        self.rate[:] = rates_hz

    def set_rate(self, idx: np.ndarray, hz: float) -> None:
        self.rate[idx] = hz

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
            self.ring_pos = _run_numba(
                steps, self.v, self.g, self.ref_left, self.ring, self.ring_pos, self.rate,
                self.indptr, self.indices, self.data, counts,
                np.float32(self.a_mem), np.float32(self.a_syn), np.float32(self.p.v_rest_mv),
                np.float32(self.p.v_reset_mv), np.float32(self.p.v_thresh_mv), self.ref_steps,
                np.float32(self.p.dt_ms * 1e-3), self._seed_numba)
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
                self.g += np.bincount(self.indices[sel], weights=self.data[sel], minlength=self.n).astype(np.float32)
        # 2) integrate (skip refractory)
        active = self.ref_left <= 0
        self.g[active] *= self.a_syn
        self.v[active] = p.v_rest_mv + (self.v[active] - p.v_rest_mv) * self.a_mem + self.g[active] * (1.0 - self.a_mem)
        self.ref_left[~active] -= 1
        # 3) spikes: threshold crossings + Poisson-driven
        spk = (self.v > p.v_thresh_mv) & active
        driven = self.rate > 0
        if driven.any():  # driven neurons may fire even while refractory (reference model)
            u = self.rng.random(int(driven.sum()))
            spk[np.nonzero(driven)[0][u < self.rate[driven] * p.dt_ms * 1e-3]] = True
        if spk.any():
            self.v[spk] = p.v_reset_mv
            self.g[spk] = 0.0
            self.ref_left[spk] = self.ref_steps
            counts[spk] += 1
        self.ring[self.ring_pos] = spk
        self.ring_pos = (self.ring_pos + 1) % self.delay_steps


if HAVE_NUMBA:

    @numba.njit(cache=True, fastmath=True)
    def _run_numba(steps, v, g, ref_left, ring, ring_pos, rate, indptr, indices, data, counts,
                   a_mem, a_syn, v_rest, v_reset, v_th, ref_steps, dt_s, seed):
        np.random.seed(seed)
        n = v.shape[0]
        delay_steps = ring.shape[0]
        for _ in range(steps):
            # 1) deliver delayed spikes
            row = ring[ring_pos]
            for i in range(n):
                if row[i]:
                    for k in range(indptr[i], indptr[i + 1]):
                        g[indices[k]] += data[k]
            # 2) integrate + detect
            for i in range(n):
                if ref_left[i] > 0:
                    # driven neurons have no refractory period in the reference model
                    if rate[i] > 0.0 and np.random.random() < rate[i] * dt_s:
                        v[i] = v_reset
                        g[i] = 0.0
                        ref_left[i] = ref_steps
                        counts[i] += 1
                        row[i] = True
                    else:
                        ref_left[i] -= 1
                        row[i] = False
                    continue
                g[i] *= a_syn
                v[i] = v_rest + (v[i] - v_rest) * a_mem + g[i] * (1.0 - a_mem)
                fired = v[i] > v_th
                if rate[i] > 0.0 and np.random.random() < rate[i] * dt_s:
                    fired = True
                if fired:
                    v[i] = v_reset
                    g[i] = 0.0
                    ref_left[i] = ref_steps
                    counts[i] += 1
                    row[i] = True
                else:
                    row[i] = False
            ring_pos = (ring_pos + 1) % delay_steps
        return ring_pos
