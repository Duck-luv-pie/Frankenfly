"""
prune_brain.py -- cut the runtime circuit down to the neurons that carry the behaviour.

    python scripts/prune_brain.py [--brain data/brain.npz] [--out data/brain_pruned.npz] [--hops 1]

Keeps, by name and by connectivity, nothing else:
  * every sensory population we inject into: LC10a LC11 LC12 LC15 LC4 LPLC2 THERMO (both sides);
  * every descending neuron (DN_ALL, including GF) we read out of;
  * the dopamine neurons PAM and PPL1 (the reward/punish keys in the demo);
  * the **bridge**: neurons that receive from the sensory set AND project to the descending set.
    With --hops 2 the bridge is widened to neurons two synapses from the eye that still reach a DN.

Everything else (Kenyon cells, mushroom-body output neurons, and one-hop neighbours that lead
nowhere) is dropped. Those are silent in this task, so the point of pruning is speed on a small
computer, not a change of claim: rerun `scripts/milestones.py m3 --brain data/brain_pruned.npz`
and `scripts/lock_demo.py` on the result and quote whatever those say.

Index groups, retinotopy columns and the signed weights are remapped, so the pruned file is a
drop-in for every script here (`--brain data/brain_pruned.npz`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENSORY_PREFIX = ("LC10a", "LC11", "LC12", "LC15", "LC4", "LPLC2", "THERMO")
KEEP_GROUPS = ("PAM", "PPL1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--out", default="data/brain_pruned.npz")
    ap.add_argument("--hops", type=int, default=1, choices=[1, 2], help="bridge width between eye and descending")
    a = ap.parse_args()

    d = dict(np.load(a.brain, allow_pickle=False))
    N = int(d["N"])
    post, pre = d["W_indices"]
    w = d["W_values"]
    group_keys = [k for k in d if k not in ("N", "ids", "types", "instances", "nt", "sides", "superclass",
                                            "meta", "W_indices", "W_values") and not k.startswith("col_of_")]

    eye = np.unique(np.concatenate([d[k] for k in group_keys if k.startswith(SENSORY_PREFIX)]))
    dn = d["DN_ALL"]
    in_eye = np.zeros(N, bool); in_eye[eye] = True
    in_dn = np.zeros(N, bool); in_dn[dn] = True

    driven = np.zeros(N, bool); driven[post[in_eye[pre]]] = True          # receives from the eye
    reaches = np.zeros(N, bool); reaches[pre[in_dn[post]]] = True         # projects onto a descending neuron
    if a.hops == 2:
        driven[post[driven[pre]]] = True                                  # two synapses from the eye
        reaches[pre[reaches[post]]] = True                                # two synapses from a DN
    bridge = driven & reaches & ~in_eye & ~in_dn

    keep = in_eye | in_dn | bridge
    for k in KEEP_GROUPS:
        keep[d[k]] = True
    idx = np.flatnonzero(keep)
    remap = np.full(N, -1, np.int64); remap[idx] = np.arange(idx.size)

    edge = keep[post] & keep[pre]
    out = {"N": np.int64(idx.size),
           "W_indices": np.stack([remap[post[edge]], remap[pre[edge]]]).astype(np.int64),
           "W_values": w[edge].astype(np.float32)}
    for k in ("ids", "types", "instances", "nt", "sides", "superclass"):
        if k in d:
            out[k] = d[k][idx]
    for k in group_keys:
        g = remap[d[k]]
        out[k] = g[g >= 0]
    for k in [k for k in d if k.startswith("col_of_")]:                   # columns follow their neurons
        grp = k[len("col_of_"):]
        kept = remap[d[grp]] >= 0
        out[k] = d[k][kept]
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    meta.update(pruned_from=a.brain, pruned_hops=a.hops, pruned_kept=int(idx.size), pruned_from_N=N,
                pruned_rule="sensory + descending + dopamine + neurons that both receive from the eye and reach a DN")
    out["meta"] = json.dumps(meta)
    np.savez(a.out, **out)

    print(f"{a.brain}: N {N:,}  nnz {w.size:,}")
    print(f"  eye {eye.size}  descending {dn.size}  bridge ({a.hops} hop) {int(bridge.sum()):,}  dopamine {len(d['PAM']) + len(d['PPL1'])}")
    print(f"{a.out}: N {idx.size:,} ({100 * idx.size / N:.0f}%)  nnz {int(edge.sum()):,} ({100 * edge.sum() / w.size:.0f}%)")
    empty = [k for k in group_keys if out[k].size == 0]
    print("  every named group survives" if not empty else f"  EMPTY GROUPS: {empty}")


if __name__ == "__main__":
    main()
