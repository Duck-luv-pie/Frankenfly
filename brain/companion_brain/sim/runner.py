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
        bg_classes = set(cfg.lif.get("background_classes", ["central", "descending", "ascending", "visual_centrifugal", "endocrine"]))
        if circuit.super_class is not None:
            bg_mask = np.isin(circuit.super_class, list(bg_classes))
            import re
            for pat in cfg.lif.get("background_exclude_types", []):
                rx = re.compile(pat)
                bg_mask &= ~np.array([bool(rx.search(t)) for t in circuit.cell_type])
            bg_idx = np.nonzero(bg_mask)[0]
        else:
            bg_idx = np.arange(circuit.n)
        self.background_idx = bg_idx
        self.net.set_background(float(cfg.lif.get("background_hz", 0.0)), bg_idx)
        for spec in cfg.lif.get("threshold_offsets", []):        # e.g. Kenyon cells: high threshold -> sparse odor code
            import re
            rx = re.compile(spec["regex"])
            mask = np.array([bool(rx.search(t)) for t in circuit.cell_type]) if circuit.cell_type is not None else np.zeros(circuit.n, bool)
            self.net.th_offset[mask] += float(spec["mv"])
        # Synapse-class gains (config lif.synapse_gains): scale the synapses from cell types matching `pre`
        # onto cell types matching `post`. Used where the LIF calibration under-drives a known pathway, e.g.
        # the thermosensory (VP) projection neurons onto Kenyon cells (see docs/architecture.md, Hunting).
        for spec in cfg.lif.get("synapse_gains", []):
            if circuit.cell_type is None:
                break
            import re
            rpre, rpost = re.compile(spec["pre"]), re.compile(spec["post"])
            pre_ok = np.array([bool(rpre.search(t)) for t in circuit.cell_type])
            post_ok = np.array([bool(rpost.search(t)) for t in circuit.cell_type])
            rows = np.nonzero(pre_ok)[0]
            for i in rows:
                a, b = self.net.indptr[i], self.net.indptr[i + 1]
                m = post_ok[self.net.indices[a:b]]
                self.net.data[a:b][m] *= np.float32(spec["gain"])
        self.chunk_ms = chunk_ms
        self.window_ms = float(cfg.decode.window_ms)
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(round(1000.0 / chunk_ms))))   # last 1 s
        self.readouts = list(cfg.readouts)
        self.wall_start: float | None = None
        self.brain_ms = 0.0
        self.behind_ms = 0.0
        self.last_chunk_wall_ms = 0.0
        self.since_take = np.zeros(circuit.n, dtype=np.int32)   # spike counts since take_recent()
        self.plasticity = None
        if cfg.get("learning") and cfg.learning.get("enabled", True) and circuit.cell_type is not None:
            from .plasticity import MushroomBodyPlasticity
            self.plasticity = MushroomBodyPlasticity(circuit, self.net, cfg, chunk_ms)

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
        self.since_take += counts
        if self.plasticity is not None:
            self.plasticity.update(counts)
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
    def rates(self, window_ms: float | None = None) -> dict[str, dict[str, float]]:
        """Mean firing rate (Hz per neuron) for each readout group over the last `window_ms`, by side."""
        if not self.history:
            return {g: {"left": 0.0, "right": 0.0, "all": 0.0} for g in self.readouts}
        k_req = max(1, int(round((window_ms or self.window_ms) / self.chunk_ms)))
        k = min(len(self.history), k_req)
        chunks = list(self.history)[-k:]
        window_s = k_req * self.chunk_ms * 1e-3      # a short history counts as silence, never as a burst
        total = np.sum(chunks, axis=0)
        out = {}
        for g in self.readouts:
            d = {}
            for side in ("left", "right", "all"):
                idx = self.c.idx(g, side)
                d[side] = float(total[idx].sum() / (len(idx) * window_s)) if len(idx) else 0.0
            out[g] = d
        return out

    def take_recent(self) -> np.ndarray:
        """Spike counts per neuron since the previous call (for live display)."""
        out = self.since_take.copy()
        self.since_take[:] = 0
        return out

    def probe(self, drives: dict[str, float], seconds: float = 1.0, settle_s: float = 1.0, capture=None):
        """Measure readout rates in response to a stimulus on a scratch copy of the network state
        (same synaptic weights, so learned changes show), without disturbing the live brain.
        Returns (resting rates just before the stimulus, rates during the stimulus). `capture(runner)`
        is called at the end of the stimulus, before the state is restored."""
        net = self.net
        saved = (net.v.copy(), net.g.copy(), net.adapt.copy(), net.resource.copy(), net.ref_left.copy(), net.ring.copy(), net.ring_pos, net.rate.copy(), net.step_count)
        hist = list(self.history); since = self.since_take.copy()
        plast = self.plasticity
        enabled = plast.enabled if plast else False
        if plast:
            plast.enabled = False
        try:
            net.clear_rates()
            for _ in range(int(settle_s * 1000 / self.chunk_ms)):
                self.step_chunk()
            rest = self.rates(min(settle_s, 1.0) * 1000)
            for _ in range(int(seconds * 1000 / self.chunk_ms)):
                net.clear_rates()
                for g, hz in drives.items():
                    self.drive(g, hz)
                self.step_chunk()
            if capture is not None:
                capture(self)
            return rest, self.rates(seconds * 1000)
        finally:
            net.v[:], net.g[:], net.adapt[:], net.resource[:], net.ref_left[:] = saved[0], saved[1], saved[2], saved[3], saved[4]
            net.ring[:] = saved[5]; net.ring_pos = saved[6]; net.rate[:] = saved[7]; net.step_count = saved[8]
            self.history.clear(); self.history.extend(hist); self.since_take[:] = since
            if plast:
                plast.enabled = enabled
                plast.elig[:] = 0

    def calibrate(self, seconds: float, windows_ms: list[float] | None = None, warmup_s: float = 0.0,
                  verbose: bool = True) -> dict:
        """Run with no sensory input and record each readout group's resting mean/std rate for each
        rate window (per side). This is the fly's own baseline; behaviors are deviations from it.
        Returns {str(window_ms): {group: {side: {mean, std}}}}."""
        self.clear_drive()
        windows = sorted(set(float(w) for w in (windows_ms or [self.window_ms])))
        for _ in range(int(warmup_s * 1000 / self.chunk_ms)):   # let adaptation and activity settle
            self.step_chunk()
        samples = {w: {g: {"left": [], "right": [], "all": []} for g in self.readouts} for w in windows}
        n_chunks = int(seconds * 1000 / self.chunk_ms)
        for k in range(n_chunks):
            self.step_chunk()
            for w in windows:
                per = max(1, int(round(w / self.chunk_ms)))
                if k >= per and k % per == 0:
                    for g, d in self.rates(w).items():
                        for side, v in d.items():
                            samples[w][g][side].append(v)
        base = {str(int(w)): {g: {side: {"mean": float(np.mean(v)) if v else 0.0, "std": float(np.std(v)) if v else 0.0}
                                  for side, v in d.items()} for g, d in samples[w].items()} for w in windows}
        if verbose:
            w0 = str(int(windows[-1]))
            print(f"[calibrate] resting rates (Hz/neuron, {w0} ms windows): " + ", ".join(
                f"{g}={base[w0][g]['all']['mean']:.1f}±{base[w0][g]['all']['std']:.1f}" for g in self.readouts))
        self.history.clear()
        self.brain_ms = 0.0          # calibration time does not count against the wall clock
        self.wall_start = None
        return base

    def window_counts(self) -> np.ndarray:
        return np.sum(self.history, axis=0) if self.history else np.zeros(self.c.n, dtype=np.int32)
