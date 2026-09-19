"""
minimal_circuit.py -- the smallest piece of this connectome that still finds a person.

    python scripts/minimal_circuit.py --keep 0.05          # keep the strongest 5% of synapses
    python scripts/minimal_circuit.py --keep 0.02 --out data/brain_min2.npz --test

The dropout experiment showed that deleting the weakest 95% of synapses leaves the behaviour intact
(RESULTS.md section 4). This takes that seriously: keep only the strongest fraction, drop every neuron
left with nothing attached, and write a real brain file that every other script here accepts. It then
reports which cell types own the surviving connections, which is the honest way to ask whether the
"load-bearing" wiring is the pathway we named or something else.

Named neurons (the eye populations we inject into, the descending neurons we read out of, the dopamine
cells) are always kept so the sensory and motor interfaces still exist, even if a particular cell ends
up with no strong synapse of its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--keep", type=float, default=0.05, help="fraction of synapses to keep, strongest first")
    ap.add_argument("--out", default="data/brain_minimal.npz")
    ap.add_argument("--test", action="store_true", help="also run the turn-toward test on the result")
    ap.add_argument("--envs", type=int, default=32)
    a = ap.parse_args()

    d = dict(np.load(a.brain, allow_pickle=False))
    N = int(d["N"])
    post, pre = d["W_indices"]
    w = d["W_values"]
    types = [str(t) for t in d["types"]]
    group_keys = [k for k in d if k not in ("N", "ids", "types", "instances", "nt", "sides", "superclass",
                                            "meta", "W_indices", "W_values") and not k.startswith("col_of_")]

    k = int(round(a.keep * w.size))
    thresh = np.partition(np.abs(w), -k)[-k]
    edge = np.abs(w) >= thresh
    print(f"{a.brain}: {N:,} neurons, {w.size:,} synapses")
    print(f"keeping |weight| >= {thresh:.0f}: {int(edge.sum()):,} synapses ({100 * edge.mean():.1f}%)")

    named = np.zeros(N, bool)
    for gk in group_keys:
        named[d[gk]] = True
    touched = np.zeros(N, bool)
    touched[post[edge]] = True
    touched[pre[edge]] = True
    keep = touched | named
    idx = np.flatnonzero(keep)
    remap = np.full(N, -1, np.int64); remap[idx] = np.arange(idx.size)
    print(f"neurons with a strong synapse: {int(touched.sum()):,}; plus named interface cells -> {idx.size:,} kept "
          f"({100 * idx.size / N:.0f}%)")

    # which cell types own the surviving wiring?
    pairs = Counter()
    for e in np.flatnonzero(edge):
        pairs[(types[pre[e]], types[post[e]])] += 1
    print("\nstrongest surviving connections, by cell type:")
    print(f"  {'from':22s} {'to':22s} {'edges':>7s}")
    for (s, t), c in pairs.most_common(12):
        print(f"  {s:22s} {t:22s} {c:7d}")

    out = {"N": np.int64(idx.size),
           "W_indices": np.stack([remap[post[edge]], remap[pre[edge]]]).astype(np.int64),
           "W_values": w[edge].astype(np.float32)}
    for key in ("ids", "types", "instances", "nt", "sides", "superclass"):
        if key in d:
            out[key] = d[key][idx]
    for gk in group_keys:
        g = remap[d[gk]]
        out[gk] = g[g >= 0]
    for ck in [c for c in d if c.startswith("col_of_")]:
        grp = ck[len("col_of_"):]
        out[ck] = d[ck][remap[d[grp]] >= 0]
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    meta.update(minimal_from=a.brain, minimal_keep=a.keep, minimal_threshold=float(thresh))
    out["meta"] = json.dumps(meta)
    np.savez(a.out, **out)
    empty = [gk for gk in group_keys if out[gk].size == 0]
    print(f"\nwrote {a.out}: {idx.size:,} neurons, {int(edge.sum()):,} synapses"
          + (f"  (empty groups: {empty})" if empty else "  (every named group survives)"))

    if a.test:
        from train import build, run_episode
        parts = build(a.envs, device="cpu", engine="event", stage="A", brain_path=a.out, seed=0, episode_s=2.0)
        st = run_episode(parts, seconds=2.0, seed=0, stage="A", freeze_humans=True)
        b0, b1, d0, d1 = st["bearing_abs_start"], st["bearing_abs_end"], st["dist_start"], st["dist_end"]
        print(f"\nturn-toward {100 * (b1 < b0 - 0.02).float().mean():.0f}%   "
              f"advance {100 * (d1 < d0 - 0.05).float().mean():.0f}%   "
              f"{d0.mean():.2f} m -> {d1.mean():.2f} m   DNa02 {st['dn_rates']['DNa02_L']:.0f}/"
              f"{st['dn_rates']['DNa02_R']:.0f} Hz")


if __name__ == "__main__":
    main()
