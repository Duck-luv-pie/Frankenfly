"""
trace_pathway.py -- which neurons actually carry the signal, and are they the ones the literature names?

    python scripts/trace_pathway.py --from LC10a_L --to DNa02_L
    python scripts/trace_pathway.py --from LC4_L --to GF --hops 1
    python scripts/trace_pathway.py --from LC10a_L --to DNa02_L --brain data/brain_tiny.npz

The dropout experiment (RESULTS.md section 4) showed the behaviour survives deleting the weakest 95% of
synapses, which is a claim about redundancy but not about anatomy. This asks the sharper question: of the
paths from an eye population to a descending neuron, which intermediate cell types carry the most weight,
and do the same ones survive pruning? If the surviving relay is a named cell type that the fly literature
already identifies with this pathway, that is evidence the simulation is riding the real circuit rather
than an accident of thresholding.

Ranks each intermediate by the bottleneck weight of its path, min(in, out), summed over the population,
because a relay is only as good as its weaker half. Reports sign, since an inhibitory relay means the
opposite of what the raw weight suggests.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import load_brain  # noqa: E402


def direct(pre_idx, post_idx, pre, post, w):
    m = torch.isin(pre, pre_idx) & torch.isin(post, post_idx)
    return float(w[m].sum()), int(m.sum())


def relays(src, dst, pre, post, w, N):
    """For every neuron X with src -> X -> dst, the bottleneck min(|in|, |out|) and both signs."""
    m_in = torch.isin(pre, src)
    m_out = torch.isin(post, dst)
    w_in = torch.zeros(N).index_add_(0, post[m_in], w[m_in])
    w_out = torch.zeros(N).index_add_(0, pre[m_out], w[m_out])
    seen_in = torch.zeros(N, dtype=torch.bool); seen_in[post[m_in]] = True
    seen_out = torch.zeros(N, dtype=torch.bool); seen_out[pre[m_out]] = True
    mid = (seen_in & seen_out).clone()
    mid[src] = False
    mid[dst] = False
    idx = mid.nonzero().flatten()
    bottleneck = torch.minimum(w_in[idx].abs(), w_out[idx].abs())
    return idx, bottleneck, w_in[idx], w_out[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--from", dest="src", default="LC10a_L")
    ap.add_argument("--to", dest="dst", default="DNa02_L")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--hops", type=int, default=2, help="2 = through one relay; 1 = direct only")
    a = ap.parse_args()

    d, N, g = load_brain(a.brain)
    types = [str(t) for t in d["types"]]
    post, pre = torch.as_tensor(d["W_indices"])
    w = torch.as_tensor(d["W_values"])
    if a.src not in g or a.dst not in g:
        raise SystemExit(f"unknown group; have {sorted(k for k in g)}")
    src, dst = g[a.src], g[a.dst]
    print(f"{a.brain}: {N:,} neurons, {w.numel():,} synapses")
    print(f"{a.src} ({src.numel()} cells) -> {a.dst} ({dst.numel()} cells)\n")

    tot, n_e = direct(src, dst, pre, post, w)
    print(f"direct {a.src} -> {a.dst}: {n_e} synapses, net weight {tot:+.0f}")
    if a.hops < 2:
        return

    idx, bottleneck, w_in, w_out = relays(src, dst, pre, post, w, N)
    if idx.numel() == 0:
        print(f"no two-hop relay between them in this brain file")
        return

    # group by cell type: individual cells of the same type are the same anatomical claim
    by_type = defaultdict(lambda: [0.0, 0, 0.0, 0.0])
    for k in range(idx.numel()):
        t = types[int(idx[k])]
        e = by_type[t]
        e[0] += float(bottleneck[k]); e[1] += 1
        e[2] += float(w_in[k]); e[3] += float(w_out[k])
    order = sorted(by_type.items(), key=lambda kv: -kv[1][0])

    total = sum(v[0] for v in by_type.values())
    print(f"\n{idx.numel()} relay neurons of {len(by_type)} cell types carry {a.src} -> X -> {a.dst}")
    print(f"ranked by bottleneck weight min(in, out), which is what a relay can actually pass\n")
    print(f"  {'relay type':18s} {'cells':>5s} {'bottleneck':>11s} {'share':>6s} {'in':>9s} {'out':>9s}  sign")
    cum = 0.0
    for t, (b, n, wi, wo) in order[: a.top]:
        cum += b
        sign = "excitatory" if wo > 0 else "INHIBITORY"
        print(f"  {t:18s} {n:5d} {b:11.0f} {100 * b / max(total, 1):5.1f}% {wi:+9.0f} {wo:+9.0f}  {sign}")
    print(f"\ntop {min(a.top, len(order))} of {len(order)} types carry {100 * cum / max(total, 1):.0f}% "
          f"of the bottleneck weight")
    top_t, top_v = order[0]
    print(f"strongest relay: {top_t} ({top_v[1]} cells, {100 * top_v[0] / max(total, 1):.0f}% of the path)")

    # Roll up to anatomical families. Individual cell names split a population that the literature
    # treats as one relay, so AOTU015 and AOTU012 separately look minor while the tubercle as a whole
    # carries the pathway. The family is the claim worth making.
    fam = defaultdict(lambda: [0.0, 0, 0])
    for t, (b, n, wi, wo) in by_type.items():
        key = "".join(ch for ch in t if not ch.isdigit()).split("_")[0].rstrip("abc") or t
        f = fam[key]
        f[0] += b; f[1] += n; f[2] += 1
    print(f"\nby anatomical family:")
    for k, (b, n, nt) in sorted(fam.items(), key=lambda kv: -kv[1][0])[:6]:
        print(f"  {k:12s} {n:3d} cells, {nt:2d} types, {100 * b / max(total, 1):5.1f}% of the pathway")


if __name__ == "__main__":
    main()
