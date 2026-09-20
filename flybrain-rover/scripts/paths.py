"""
paths.py -- who feeds the descending neurons, and is any of it reachable from the eye?

    python scripts/paths.py [--top 12]

For each DN group prints its strongest presynaptic partners inside brain.npz (type, side, signed weight)
and, for each partner, the total synapse count it receives from LC / THERMO seed groups (1 hop) and from
the direct targets of those groups (2 hops). Use it to understand why a DN is silent.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import load_brain  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=12); a = ap.parse_args()
    d, N, g = load_brain()
    post, pre = torch.as_tensor(d["W_indices"]); w = torch.as_tensor(d["W_values"])
    types, sides = d["types"], d["sides"]
    W = torch.sparse_coo_tensor(torch.stack([post, pre]), w, (N, N)).coalesce()
    Wd = W.to_dense()                                     # 15k x 15k floats, fine on CPU (900 MB)
    eye = torch.cat([g[k] for k in g if k.startswith(("LC", "LPLC2", "THERMO"))])
    from_eye = Wd[:, eye].clamp(min=0).sum(1)              # excitatory synapses each neuron gets from the eye
    hop1 = (from_eye > 0).nonzero().flatten()
    from_hop1 = Wd[:, hop1].clamp(min=0).sum(1)            # from neurons that the eye drives
    for dn in ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF"]:
        idx = g[dn]
        inp = Wd[idx].sum(0)                                # (N,) summed signed input to this DN group
        exc_total, inh_total = inp.clamp(min=0).sum().item(), inp.clamp(max=0).sum().item()
        order = inp.abs().argsort(descending=True)[: a.top]
        print(f"\n{dn}: {len(idx)} neuron(s), exc in {exc_total:.0f}, inh in {inh_total:.0f}; "
              f"exc from eye-driven neurons {(inp.clamp(min=0) * (from_eye[:] > 0)).sum():.0f} (1 hop), "
              f"{(inp.clamp(min=0) * (from_hop1 > 0)).sum():.0f} (2 hops)")
        print(f"  {'type':28s} {'side':4s} {'w':>7s} {'fromEye':>8s} {'fromHop1':>9s}")
        for j in order.tolist():
            print(f"  {types[j]:28s} {sides[j]:4s} {inp[j].item():7.0f} {from_eye[j].item():8.0f} {from_hop1[j].item():9.0f}")
