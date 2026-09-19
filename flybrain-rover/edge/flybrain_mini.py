"""
flybrain_mini.py -- the fly brain with nothing but numpy. One file, no framework, no GPU.

    python edge/flybrain_mini.py --selftest          # must match the PyTorch implementation exactly
    python edge/flybrain_mini.py --bench             # milliseconds per camera frame on this machine
    python edge/flybrain_mini.py --demo              # a person crossing the view, printed as wheel commands

Why this exists: once the circuit is cut to the part that carries the behaviour (2,211 neurons and
10,387 synapses, RESULTS.md section 4), the simulation is small enough that PyTorch is the heaviest
thing in the room. This is the same spiking model, the same constants and the same read-outs in about
200 lines of numpy, so the brain can run on a Raspberry Pi, inside QNX, or anywhere `pip install numpy`
works. `--selftest` checks it spike for spike against `brain/lif.py` rather than asking you to trust it.

    from flybrain_mini import FlyBrain
    fly = FlyBrain("data/brain_tiny.npz")
    forward, turn = fly.step([(120, 40, 190, 230)], dt=1/30)   # person boxes in a 320x240 frame
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# leaky integrate-and-fire constants (Shiu et al. 2024), identical to brain/lif.py
TAU_M, V_REST, V_TH, V_RESET = 0.020, -52.0, -45.0, -52.0
T_REF, TAU_SYN, DT, RATE_TAU = 0.0022, 0.005, 0.001, 0.050
# camera and read-out defaults, identical to brain/retina.py, senses.py and motor.py
W_PX, H_PX, HFOV_DEG, N_COLS = 320, 240, 98.43, 24
AMP_SMALL, AMP_TRACK, AMP_LOOM, AMP_HEAT = 1.2, 20.0, 2.0, 0.8
K_T, K_F, GAIN = 0.02, 0.3, 0.05
SMALL_POPS, TRACK_POPS, LOOM_POPS = ("LC11", "LC12", "LC15"), ("LC10a",), ("LC4", "LPLC2")


class FlyBrain:
    def __init__(self, path="data/brain_tiny.npz", gain=GAIN, k_t=K_T, k_f=K_F, substeps=20,
                 amp_track=AMP_TRACK, weights=None, dtype=np.float32):
        self.dtype = dtype
        d = np.load(path, allow_pickle=False)
        self.N = int(d["N"])
        post, pre = d["W_indices"].astype(np.int64)
        w = (d["W_values"] if weights is None else weights).astype(np.float64)   # coalesce in double ...
        # coalesce duplicate (post, pre) pairs, then index the edges by their source neuron so a spike
        # only ever touches its own outgoing column: cost is spikes x out-degree, not neurons x neurons
        key = post * self.N + pre
        uniq, inv = np.unique(key, return_inverse=True)
        wc = np.zeros(uniq.size); np.add.at(wc, inv, w)
        post, pre = uniq // self.N, uniq % self.N
        order = np.argsort(pre * self.N + post, kind="stable")
        self.out_post = post[order]
        self.out_w = (wc[order] * gain).astype(dtype)   # ... then run in the same precision as the reference
        self.colptr = np.concatenate([[0], np.cumsum(np.bincount(pre, minlength=self.N))])
        self.nnz = wc.size

        self.g = {k: d[k].astype(np.int64) for k in d.files
                  if k not in ("N", "ids", "types", "instances", "nt", "sides", "superclass", "meta",
                               "W_indices", "W_values") and not k.startswith("col_of_")}
        self.col_of = {k[len("col_of_"):]: d[k].astype(np.int64) for k in d.files if k.startswith("col_of_")}
        self.k_t, self.k_f, self.substeps = k_t, k_f, substeps
        self.amp = dict(small=AMP_SMALL, track=amp_track, loom=AMP_LOOM, heat=AMP_HEAT)
        self.syn_decay = math.exp(-DT / TAU_SYN)
        self.ref_steps = int(round(T_REF / DT))
        self.rate_a = DT / RATE_TAU
        half = math.radians(HFOV_DEG) / 2
        self.edges = np.linspace(-half, half, N_COLS + 1)
        self.col_w = self.edges[1] - self.edges[0]
        self.fx = (W_PX / 2) / math.tan(half)
        self.reset()

    # ---------------------------------------------------------------- spiking simulation
    def reset(self):
        self.V = np.full(self.N, V_REST, self.dtype)
        self.gsyn = np.zeros(self.N, self.dtype)
        self.ref = np.zeros(self.N, np.int32)
        self.spk = np.zeros(self.N, bool)
        self.rate = np.zeros(self.N, self.dtype)
        self._prev_pres = None
        self._prev_width = None

    def lif_step(self, I_ext):
        fired = np.flatnonzero(self.spk)
        syn = np.zeros(self.N, self.dtype)
        if fired.size:
            starts, ends = self.colptr[fired], self.colptr[fired + 1]
            n = ends - starts
            if n.sum():
                # expand every spiking neuron's edge range into one flat gather
                pos = np.repeat(starts, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
                np.add.at(syn, self.out_post[pos], self.out_w[pos])
        self.gsyn = self.gsyn * self.syn_decay + syn
        dV = (np.float32(DT / TAU_M) * (V_REST - self.V + self.gsyn) + I_ext).astype(self.dtype)
        active = self.ref <= 0
        self.V = np.where(active, self.V + dV, self.V)
        self.spk = active & (self.V >= V_TH)
        self.V = np.where(self.spk, V_RESET, self.V)
        self.ref = np.where(self.spk, self.ref_steps, self.ref - 1)
        self.rate += (np.float32(self.rate_a) * (self.spk / np.float32(DT) - self.rate)).astype(self.dtype)
        return self.spk

    def rates(self, key):
        idx = self.g[key]
        return float(self.rate[idx].mean()) if idx.size else 0.0

    # ---------------------------------------------------------------- eye
    def retina(self, boxes, dt):
        """Person boxes in pixels -> presence, angular size and motion per angular column."""
        pres = np.zeros(N_COLS); size = np.zeros(N_COLS)
        w_lr = np.zeros(2)
        for x0, _, x1, _ in boxes:
            az0 = math.atan((x0 - W_PX / 2) / self.fx)
            az1 = math.atan((x1 - W_PX / 2) / self.fx)
            az0, az1 = max(az0, self.edges[0]), min(az1, self.edges[-1])
            if az1 <= az0:
                continue
            ov = np.clip(np.minimum(az1, self.edges[1:]) - np.maximum(az0, self.edges[:-1]), 0, None) / self.col_w
            hit = ov > pres
            size[hit] = (az1 - az0) / math.radians(HFOV_DEG)
            pres = np.maximum(pres, ov)
            w_lr[0 if 0.5 * (az0 + az1) < 0 else 1] += az1 - az0
        mot = np.zeros(N_COLS) if self._prev_pres is None else np.clip((pres - self._prev_pres) / max(dt, 1e-3), -1, 1)
        loom = np.zeros(2) if self._prev_width is None else np.clip((w_lr - self._prev_width) / max(dt, 1e-3), 0, None)
        a = min(1.0, dt / 0.05)
        if self._prev_pres is not None:
            pres = a * pres + (1 - a) * self._prev_pres
        self._prev_pres, self._prev_width = pres.copy(), w_lr.copy()
        return pres, size, mot, loom

    def senses(self, pres, size, mot, loom, heat=(0.0, 0.0)):
        """Columns onto the fly's own visual projection neurons, the way brain/senses.py does it."""
        I = np.zeros(self.N, self.dtype)
        small = pres * (1.0 - size) * self.amp["small"]
        track = pres * size * self.amp["track"]
        for pops, drive in ((SMALL_POPS, small), (TRACK_POPS, track)):
            for pop in pops:
                for s, offset in (("L", 0), ("R", N_COLS // 2)):
                    key = f"{pop}_{s}"
                    idx = self.g.get(key)
                    if idx is None or not idx.size:
                        continue
                    col = self.col_of.get(key)
                    if col is None:
                        col = offset + (np.arange(idx.size) * (N_COLS // 2)) // max(idx.size, 1)
                    ok = col >= 0
                    I[idx[ok]] = drive[col[ok]]
        for pop in LOOM_POPS:
            for s, v in (("L", loom[0]), ("R", loom[1])):
                idx = self.g.get(f"{pop}_{s}")
                if idx is not None and idx.size:
                    I[idx] = v * self.amp["loom"]
        for s, v in (("L", heat[0]), ("R", heat[1])):
            idx = self.g.get(f"THERMO_{s}")
            if idx is not None and idx.size:
                I[idx] = v * self.amp["heat"]
        return I

    # ---------------------------------------------------------------- one camera frame
    def step(self, boxes, dt=1 / 30, heat=(0.0, 0.0)):
        pres, size, mot, loom = self.retina(boxes, dt)
        I = self.senses(pres, size, mot, loom, heat)
        for _ in range(self.substeps):
            self.lif_step(I)
        turn = np.clip(self.k_t * (self.rates("DNa02_R") - self.rates("DNa02_L")), -1, 1)
        fwd = np.clip(self.k_f * self.rates("DN_ALL"), -1, 1)
        return float(fwd), float(turn)


def sweeping_boxes(n, w=70):
    """A person walking across the view, for the bench and the demo."""
    for i in range(n):
        cx = W_PX * (0.15 + 0.7 * (0.5 - 0.5 * math.cos(2 * math.pi * i / 90)))
        yield [(cx - w / 2, 30.0, cx + w / 2, float(H_PX))]


def selftest(path):
    """Spike for spike against brain/lif.py, which is the only reason to trust a second implementation."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from brain.lif import LIF, load_brain
    d, N, groups = load_brain(path)
    ref = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1, device="cpu",
              engine="event", g=GAIN)
    mini = FlyBrain(path)
    rng = np.random.default_rng(0)
    I = (rng.random(N) * 1.4).astype(np.float32)
    It = torch.as_tensor(I, dtype=torch.float32).unsqueeze(0)
    bad = 0
    for t in range(300):
        a = ref.step(It)[0].numpy()
        b = mini.lif_step(I)
        if not np.array_equal(a, b):
            bad += 1
            if bad == 1:
                print(f"  first divergence at step {t}: {int((a != b).sum())} of {N} neurons differ")
    r_ref = float(ref.rates(groups["DNa02_L"].to(ref.device)).item())
    r_min = mini.rates("DNa02_L")
    print(f"selftest on {path}: {N:,} neurons, 300 steps, {bad} steps with any difference")
    print(f"  DNa02_L rate  pytorch {r_ref:.3f} Hz   numpy {r_min:.3f} Hz   difference {abs(r_ref - r_min):.4f}")
    ok = bad == 0 and abs(r_ref - r_min) < 1e-3
    print("  identical" if ok else "  MISMATCH")
    return ok


def bench(path, frames=120):
    fly = FlyBrain(path)
    for boxes in sweeping_boxes(10):
        fly.step(boxes)
    t0 = time.time()
    for boxes in sweeping_boxes(frames):
        fly.step(boxes)
    ms = (time.time() - t0) / frames * 1000
    print(f"{path}: {fly.N:,} neurons, {fly.nnz:,} synapses, numpy only")
    print(f"  {ms:.1f} ms per camera frame ({fly.substeps} brain steps) -> {1000 / ms:.0f} Hz control loop")
    print(f"  {ms / fly.substeps:.3f} ms per 1 ms of brain time")


def demo(path, frames=90):
    fly = FlyBrain(path)
    print(f"{path}: a person walking left to right, wheel commands only\n")
    print(f"{'frame':>5s} {'box centre':>11s} {'LC10a L/R':>13s} {'DNa02 L/R':>13s} {'forward':>8s} {'turn':>7s}")
    for i, boxes in enumerate(sweeping_boxes(frames)):
        fwd, turn = fly.step(boxes)
        if i % 10 == 0:
            cx = 0.5 * (boxes[0][0] + boxes[0][2])
            print(f"{i:5d} {cx:10.0f}px {fly.rates('LC10a_L'):6.1f}/{fly.rates('LC10a_R'):6.1f} "
                  f"{fly.rates('DNa02_L'):6.1f}/{fly.rates('DNa02_R'):6.1f} {fwd:8.2f} {turn:+7.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain_tiny.npz")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if not (a.selftest or a.bench or a.demo):
        a.bench = True
    ok = True
    if a.selftest:
        ok = selftest(a.brain)
    if a.bench:
        bench(a.brain)
    if a.demo:
        demo(a.brain)
    sys.exit(0 if ok else 1)
