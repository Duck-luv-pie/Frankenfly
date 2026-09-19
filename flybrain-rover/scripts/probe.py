"""
probe.py -- which eye population drives which descending neuron, and on which side?

    python scripts/probe.py [--drive 1.5] [--ms 300] [--gain 0.05]

One batch element per source group; each gets a constant drive on all its neurons.
Prints the DN / MB rate table. Use it to choose senses amplitudes and to sanity check laterality.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402

SOURCES = ["LC10a_L", "LC10a_R", "LC11_L", "LC11_R", "LC12_L", "LC12_R", "LC15_L", "LC15_R",
           "LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "THERMO_L", "THERMO_R", "PAM", "PPL1",
           "LC10a_L+LC10a_R", "LC11_L+LC11_R", "LC10a_L+LC11_L+LC12_L+LC15_L", "THERMO_L+THERMO_R",
           "LC10a_L+LC10a_R+THERMO_L+THERMO_R"]
TARGETS = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF", "KC", "MBON", "PAM", "PPL1"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", type=float, default=1.5)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of the source group driven (retinotopic realism)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    d, N, groups = load_brain()
    B = len(SOURCES)
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, B, device=a.device, g=a.gain)
    groups = {k: v.to(lif.device) for k, v in groups.items()}
    I = torch.zeros(B, N, device=lif.device)
    for b, k in enumerate(SOURCES):
        for part in k.split("+"):
            idx = groups[part]
            idx = idx[: max(1, int(len(idx) * a.frac))]
            I[b, idx] = a.drive
    for _ in range(a.ms):
        lif.step(I)
    print(f"gain {a.gain} drive {a.drive} mV/step on {a.frac:.0%} of each source, {a.ms} ms; rates in Hz")
    print(f"{'source':34s} {'self':>6s} " + " ".join(f"{k:>8s}" for k in TARGETS))
    for b, k in enumerate(SOURCES):
        row = [lif.rates(groups[t])[b].item() for t in TARGETS]
        selfr = lif.rates(groups[k.split("+")[0]])[b].item()
        print(f"{k:34s} {selfr:6.0f} " + " ".join(f"{v:8.1f}" for v in row))
