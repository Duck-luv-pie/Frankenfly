"""
lesion_relay.py -- a prediction from the pathway analysis, tested.

    python scripts/lesion_relay.py [--envs 256] [--seconds 2]

RESULTS.md section 4b says the anterior optic tubercle carries about half of the LC10a to DNa02 route and
AOTU plus LAL carry 89% of it. That analysis only looked at anatomy: weights on a graph, no simulation.
It makes a falsifiable prediction. If those relays really are the route, then removing them should abolish
tracking as thoroughly as removing the eye does, even though they are not sensory neurons and the eye is
left completely intact.

Removing neurons always hurts, so the comparison that matters is against a size-matched random lesion:
the same number of neurons, drawn from the rest of the brain, removed the same way. If AOTU is special,
the AOTU row collapses and the random row does not.

Protocol is M3: untrained brain, stage A, humans frozen, turn-toward = fraction of arenas whose |bearing|
to the person fell by more than 0.02 rad over the episode.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def by_prefix(types, prefix):
    return torch.tensor([i for i, t in enumerate(types) if str(t).startswith(prefix)], dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="sparse", help="sparse (CPU/CUDA) or dense (MPS)")
    a = ap.parse_args()

    from train import build, run_episode
    parts = build(a.envs, device=a.device, engine=a.engine, stage="A", brain_path=a.brain,
                  seed=a.seed, episode_s=a.seconds)
    lif = parts["lif"] if "lif" in parts else parts["brain"].lif
    d = dict(__import__("numpy").load(a.brain, allow_pickle=False))
    types = [str(t) for t in d["types"]]
    g = {k: torch.as_tensor(d[k]) for k in ("LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R")}

    aotu = by_prefix(types, "AOTU")
    lal = by_prefix(types, "LAL")
    lc10a = torch.cat([g["LC10a_L"], g["LC10a_R"]])
    gen = torch.Generator().manual_seed(a.seed)

    def random_like(n, exclude):
        """n neurons drawn from everything that is not a named interface cell or the group under test."""
        mask = torch.ones(lif.N, dtype=torch.bool)
        mask[exclude] = False
        pool = mask.nonzero().flatten()
        return pool[torch.randperm(pool.numel(), generator=gen)[:n]]

    rows = [("intact", None)]
    rows.append((f"LC10a lesioned ({lc10a.numel()} cells)", lc10a))
    rows.append((f"AOTU lesioned ({aotu.numel()} cells)", aotu))
    rows.append((f"LAL lesioned ({lal.numel()} cells)", lal))
    rows.append((f"random {aotu.numel()} cells", random_like(aotu.numel(), torch.cat([aotu, lc10a]))))
    rows.append((f"random {lal.numel()} cells", random_like(lal.numel(), torch.cat([lal, lc10a]))))

    print(f"{a.brain}  {a.envs} arenas  {a.seconds}s  stage A, humans frozen, untrained\n")
    print(f"  {'condition':32s} {'turn-toward':>12s} {'advance':>9s} {'DNa02 L/R (Hz)':>18s}", flush=True)
    base = None
    for label, idx in rows:
        snap = lif.lesion(idx, mode="both") if idx is not None else None
        st = run_episode(parts, seconds=a.seconds, seed=a.seed, stage="A", freeze_humans=True)
        if snap is not None:
            lif.restore(snap)
        b0, b1 = st["bearing_abs_start"], st["bearing_abs_end"]
        d0, d1 = st["dist_start"], st["dist_end"]
        tt = 100 * (b1 < b0 - 0.02).float().mean().item()
        adv = 100 * (d1 < d0 - 0.05).float().mean().item()
        dl, dr = st["dn_rates"]["DNa02_L"], st["dn_rates"]["DNa02_R"]
        if base is None:
            base = tt
        print(f"  {label:32s} {tt:11.0f}% {adv:8.0f}% {dl:8.0f} /{dr:7.0f}", flush=True)
    print(f"\nintact turn-toward is {base:.0f}%; chance is 50%. A relay lesion that lands at chance while a "
          f"same-size\nrandom lesion does not is the pathway analysis being right about the anatomy.")


if __name__ == "__main__":
    main()
