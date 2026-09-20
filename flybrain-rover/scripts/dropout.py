"""
dropout.py -- how much of the connectome can you destroy before the behaviour goes?

    python scripts/dropout.py                       # both orders, 64 arenas each
    python scripts/dropout.py --envs 16 --quick

Deletes a fraction of the synapses and re-runs the turn-toward and advance test, in two orders:

  random        a uniformly random fraction of all 2,334,959 synapses;
  weakest       the same fraction, but the smallest synapses first.

The gap between the two lines is the claim: if weakest-first tolerates far more deletion than random,
the behaviour is carried by a small set of strong connections rather than spread thinly over the whole
graph. Neurons are never removed here, only synapses, so this is a different question from the lesion
table, which removes named cell types.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build, run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--fractions", default="0,0.25,0.5,0.75,0.9,0.95,0.99")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="logs/dropout.json")
    a = ap.parse_args()
    fracs = [float(x) for x in a.fractions.split(",")]
    if a.quick:
        fracs = [0, 0.5, 0.9, 0.99]

    parts = build(a.envs, device=a.device, engine=a.engine, stage="A", brain_path=a.brain,
                  seed=a.seed, episode_s=a.seconds, checkpoint=a.checkpoint)
    lif = parts["lif"]
    w0 = lif.w.detach().clone()
    order = {"random": torch.randperm(w0.numel(), generator=torch.Generator().manual_seed(0)).to(w0.device),
             "weakest": torch.argsort(w0.abs())}                      # smallest synapses deleted first

    rows = []
    print(f"{a.brain}, {a.envs} arenas, {a.seconds} s, people frozen, {w0.numel():,} synapses\n")
    print(f"{'order':9s} {'deleted':>8s} {'turn-toward':>12s} {'advance':>9s} {'DNa02 Hz':>9s} {'final dist':>11s}")
    for name, idx in order.items():
        for f in fracs:
            w = w0.clone()
            if f > 0:
                w[idx[: int(f * w0.numel())]] = 0.0
            lif.set_w(w)
            t0 = time.time()
            st = run_episode(parts, seconds=a.seconds, seed=a.seed, stage="A", freeze_humans=True)
            b0, b1, d0, d1 = st["bearing_abs_start"], st["bearing_abs_end"], st["dist_start"], st["dist_end"]
            row = dict(order=name, fraction=f,
                       turn_toward=round((b1 < b0 - 0.02).float().mean().item(), 3),
                       advance=round((d1 < d0 - 0.05).float().mean().item(), 3),
                       dna02=round(0.5 * (st["dn_rates"]["DNa02_L"] + st["dn_rates"]["DNa02_R"]), 1),
                       final_dist=round(d1.mean().item(), 2), seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(f"{name:9s} {100 * f:7.0f}% {100 * row['turn_toward']:11.0f}% {100 * row['advance']:8.0f}% "
                  f"{row['dna02']:9.1f} {row['final_dist']:10.2f} m", flush=True)
        print()
    lif.set_w(w0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
