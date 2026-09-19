"""
probe_dns.py -- which descending-neuron TYPES in brain.npz does LC10a (or any eye group) actually drive?

    python scripts/probe_dns.py [--source LC10a_L] [--drive 1.5] [--gain 0.05] [--ms 300] [--top 25]

brain.npz contains DNa02/DNa01/DNp09/DNp01 as seed groups, but its 1-hop neighbours include many other
descending neurons (types starting with "DN"). Drive one eye group, report every DN type's mean rate.
Also asks neuPrint (anonymous) for every DN type receiving direct LC10a input (>=5 synapses) in the whole
male CNS, so candidates missing from brain.npz show up too.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--source", default="LC10a_L")
    ap.add_argument("--drive", type=float, default=1.5)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    d, N, groups = load_brain(a.brain)
    types, sides = d["types"], d["sides"]
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1, device=a.device, engine="event", g=a.gain)
    I = torch.zeros(1, N, device=lif.device)
    I[0, groups[a.source].to(lif.device)] = a.drive
    for _ in range(a.ms):
        lif.step(I)
    rate = lif.rate[0].cpu().numpy()
    by_type = defaultdict(list)
    for i, t in enumerate(types):
        if t.startswith("DN"):
            by_type[(t, sides[i])].append(rate[i])
    rows = sorted(((t, s, float(np.mean(v)), len(v)) for (t, s), v in by_type.items()), key=lambda r: -r[2])
    print(f"{a.source} driven at {a.drive} mV/step, gain {a.gain}, {a.ms} ms; DN types in {a.brain}: {len(by_type)}")
    print(f"{'DN type':14s} side  n   rate Hz")
    for t, s, r, n in rows[: a.top]:
        print(f"{t:14s} {s:4s} {n:3d} {r:8.1f}")
    print(f"... {sum(1 for r in rows if r[2] > 5)} DN (type,side) entries above 5 Hz")
    # whole-CNS check: DN types with direct LC10a input
    try:
        from data.pull_connectome import cypher
        _, data = cypher("MATCH (a:Neuron {type:'LC10a'})-[c:ConnectsTo]->(b:Neuron) WHERE b.type STARTS WITH 'DN' AND c.weight >= 5 "
                         "RETURN b.type, count(DISTINCT a) AS n_lc10a, sum(c.weight) AS syn ORDER BY syn DESC LIMIT 20")
        print("\nneuPrint: DN types receiving direct LC10a input (>=5 syn per edge), whole male CNS:")
        for t, n, syn in data:
            print(f"  {t:14s} from {n:3d} LC10a neurons, {syn:5d} synapses")
    except Exception as e:  # noqa: BLE001
        print("neuPrint query failed:", e)


if __name__ == "__main__":
    main()
