"""Real-time brain runner: paces the LIF network against the wall clock and keeps a sliding
window of spike counts per readout group."""
from __future__ import annotations

import time
from collections import deque

import numpy as np

from ..config import Config
from ..data.prune import Circuit
from .lif import LIFNetwork, LIFParams


class BrainRunner:
    def __init__(self, circuit: Circuit, cfg: Config, chunk_ms: float = 10.0, seed: int | None = None,
                 use_numba: bool | None = None):
        self.c = circuit
        self.cfg = cfg
        self.net = LIFNetwork(circuit.n, circuit.pre, circuit.post, circuit.weight_mv,
                              LIFParams.from_config(dict(cfg.lif)), seed=seed, use_numba=use_numba)
        self.chunk_ms = chunk_ms
        self.window_ms = float(cfg.decode.window_ms)
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(round(self.window_ms / chunk_ms))))
        self.readouts = list(cfg.readouts)
        self.wall_start: float | None = None
        self.brain_ms = 0.0
        self.behind_ms = 0.0
        self.last_chunk_wall_ms = 0.0

    # ---- drive ---------------------------------------------------------------------------
    def clear_drive(self) -> None:
        self.net.clear_rates()

    def drive(self, group: str, hz: float, side: str = "all") -> None:
        idx = self.c.idx(group, side)
        if len(idx):
            self.net.rate[idx] = np.maximum(self.net.rate[idx], hz)

    # ---- stepping ------------------------------------------------------------------------
    def step_chunk(self) -> np.ndarray:
        t0 = time.perf_counter()
        counts = self.net.run_ms(self.chunk_ms)
        self.last_chunk_wall_ms = (time.perf_counter() - t0) * 1e3
        self.history.append(counts)
        self.brain_ms += self.chunk_ms
        return counts

    def advance_to_wall(self, max_chunks: int = 50) -> int:
        """Advance brain time to catch up with wall-clock time. Returns chunks simulated."""
        now = time.perf_counter()
        if self.wall_start is None:
            self.wall_start = now
        target_ms = (now - self.wall_start) * 1e3
        n = 0
        while self.brain_ms < target_ms and n < max_chunks:
            self.step_chunk()
            n += 1
        self.behind_ms = max(0.0, target_ms - self.brain_ms)
        if n >= max_chunks and self.behind_ms > 1000:  # hopelessly behind: skip ahead
            self.brain_ms = target_ms
        return n

    # ---- readout -------------------------------------------------------------------------
    def rates(self) -> dict[str, dict[str, float]]:
        """Mean firing rate (Hz per neuron) for each readout group over the window, by side."""
        if not self.history:
            return {g: {"left": 0.0, "right": 0.0, "all": 0.0} for g in self.readouts}
        window_s = len(self.history) * self.chunk_ms * 1e-3
        total = np.sum(self.history, axis=0)
        out = {}
        for g in self.readouts:
            d = {}
            for side in ("left", "right", "all"):
                idx = self.c.idx(g, side)
                d[side] = float(total[idx].sum() / (len(idx) * window_s)) if len(idx) else 0.0
            out[g] = d
        return out

    def window_counts(self) -> np.ndarray:
        return np.sum(self.history, axis=0) if self.history else np.zeros(self.c.n, dtype=np.int32)
