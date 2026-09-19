"""
bench_brain.py -- how fast does this brain run on THIS computer? One command, no robot, no camera.

    python scripts/bench_brain.py                                  # data/brain.npz, batch 1, CPU
    python scripts/bench_brain.py --brain data/brain_pruned.npz    # the pruned runtime circuit
    python scripts/bench_brain.py --engine sparse --device cuda --batch 512

Drives the eye the way the retina does when a person fills a third of the view, runs the spiking
network, and reports milliseconds per 1 ms brain step plus the implied camera-to-command time at
--substeps steps per frame. Anything under 50 ms per frame leaves room for a detector inside the
100 ms budget. Run it on the Raspberry Pi to decide whether the brain can live on the robot.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event", choices=["event", "sparse", "dense", "auto"])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave alone)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    d, N, groups = load_brain(a.brain)
    t0 = time.time()
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, a.batch,
              device=a.device, engine=a.engine, g=a.gain)
    build_s = time.time() - t0
    groups = {k: v.to(lif.device) for k, v in groups.items()}

    I = torch.zeros(a.batch, N, device=lif.device)          # a person filling a third of the left view
    I[:, groups["LC10a_L"][: max(1, len(groups["LC10a_L"]) // 3)]] = 20.0
    for k in ("LC11_L", "LC12_L", "LC15_L"):
        if len(groups.get(k, [])):
            I[:, groups[k]] = 1.2
    for k in ("LC4_L", "LPLC2_L"):
        if len(groups.get(k, [])):
            I[:, groups[k]] = 0.6

    for _ in range(50):                                      # warm up, let rates settle
        lif.step(I)
    spikes = 0
    t0 = time.time()
    for _ in range(a.steps):
        spikes += int(lif.step(I).sum())
    dt = (time.time() - t0) / a.steps * 1000

    print(f"brain   {a.brain}: {N:,} neurons, {lif.nnz:,} synapses, built in {build_s:.1f} s")
    print(f"machine {a.device} / {lif.engine} engine / batch {a.batch}"
          + (f" / {torch.get_num_threads()} threads" if a.device == "cpu" else ""))
    print(f"step    {dt:.2f} ms per 1 ms of brain time  ({spikes / a.steps / a.batch:.0f} spikes per step per env)")
    frame = dt * a.substeps
    print(f"frame   {frame:.0f} ms for {a.substeps} steps -> {1000 / frame:.0f} Hz control loop"
          f"  ({'real time with room for a detector' if frame < 50 else 'tight' if frame < 100 else 'too slow: prune, fewer substeps, or a faster machine'})")


if __name__ == "__main__":
    main()
