"""
why_silent.py -- why does the eye not drive this descending neuron, when the anatomy says it should?

    python scripts/why_silent.py --target DNa10
    python scripts/why_silent.py --target DNp09 --source LC4_L,LC4_R

Drives a sensory population, measures the target descending neuron, then asks whether the silence is
imposed by feedforward inhibition: it ranks the inhibitory inputs onto the target by how hard they are
themselves being driven, removes them one group at a time (largest first), and reports the target's
rate after each removal. If the rate comes up, the pathway exists and is being actively suppressed; if
it stays at zero, the excitation genuinely never arrives.

This is the tool behind the claim that anatomy is not drive: LC10a puts 801 synapses straight onto
DNa10 and never fires it, while DNa02, which receives no direct LC10a synapse at all, fires at 126 Hz
through a three-hop path.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--target", default="DNa10", help="descending neuron type to explain")
    ap.add_argument("--source", default="LC10a_L,LC10a_R", help="group keys to drive")
    ap.add_argument("--drive", type=float, default=1.5)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--top", type=int, default=6, help="how many inhibitory sources to peel off")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    a = ap.parse_args()

    d, N, groups = load_brain(a.brain)
    types = [str(t) for t in d["types"]]
    post, pre = torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_indices"])[1]
    post, pre = torch.as_tensor(d["W_indices"])[0], torch.as_tensor(d["W_indices"])[1]
    w = torch.as_tensor(d["W_values"])
    tgt = torch.tensor([i for i, t in enumerate(types) if t == a.target])
    if not len(tgt):
        sys.exit(f"{a.target} is not in {a.brain}")
    src = torch.cat([groups[k] for k in a.source.split(",")])

    onto = torch.isin(post, tgt)
    direct = onto & torch.isin(pre, src)
    print(f"{a.target}: {len(tgt)} neurons, excitatory input {float(w[onto & (w > 0)].sum()):.0f}, "
          f"inhibitory {float(w[onto & (w < 0)].sum()):.0f}, direct from {a.source} "
          f"{float(w[direct].sum()):.0f} synapses over {int(direct.sum())} edges")

    def run(lif, gg):
        lif.reset()
        I = torch.zeros(1, N, device=lif.device)
        I[:, src.to(lif.device)] = a.drive
        for _ in range(a.ms):
            lif.step(I)
        return lif.rate[0]

    lif = LIF(torch.as_tensor(d["W_indices"]), w, N, 1, device=a.device, engine=a.engine, g=a.gain)
    rate = run(lif, groups)
    base = float(rate[tgt.to(lif.device)].mean())
    print(f"baseline with the eye driven at {a.drive} mV/step: {a.target} {base:.1f} Hz\n")

    # inhibitory inputs onto the target, ranked by (synapses x how fast that neuron is actually firing)
    inh = onto & (w < 0)
    rates_cpu = rate.cpu()
    by_type = defaultdict(lambda: [0.0, 0.0, []])          # type -> [synapses, drive-weighted, neuron ids]
    for e in inh.nonzero().flatten().tolist():
        p = int(pre[e]); t = types[p]
        by_type[t][0] += float(-w[e])
        by_type[t][1] += float(-w[e]) * float(rates_cpu[p])
        by_type[t][2].append(p)
    ranked = sorted(by_type.items(), key=lambda kv: -kv[1][1])[: a.top]
    print(f"{'inhibitory source':22s} {'synapses':>9s} {'its rate':>9s} {'suppression':>12s}")
    for t, (syn, weighted, ids) in ranked:
        r = float(rates_cpu[torch.tensor(ids)].mean())
        print(f"{t:22s} {syn:9.0f} {r:9.1f} {weighted:12.0f}")

    print(f"\npeeling those off, largest first (each row also keeps the ones above it):")
    print(f"{'removed':34s} {a.target + ' rate':>12s}")
    cut = []
    for t, (syn, weighted, ids) in ranked:
        cut += ids
        lif2 = LIF(torch.as_tensor(d["W_indices"]), w, N, 1, device=a.device, engine=a.engine, g=a.gain)
        lif2.lesion(torch.tensor(cut), mode="both")
        r2 = float(run(lif2, groups)[tgt.to(lif2.device)].mean())
        print(f"{('+ ' + t):34s} {r2:11.1f} Hz")
    print("\nA rate that climbs means the excitation is there and something is sitting on it. A rate that "
          "stays at zero means the drive never arrives in the first place.")


if __name__ == "__main__":
    main()
