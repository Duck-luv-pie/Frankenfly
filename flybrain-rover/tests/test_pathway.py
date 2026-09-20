"""
The headline anatomy claim in RESULTS.md section 4b, locked down.

LC10a reaches DNa02 only through relays, and those relays are the anterior optic tubercle and the lateral
accessory lobe: the steering pathway the courtship literature describes. It is a claim about the data, so
it should fail loudly if the connectome file is ever rebuilt differently.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BRAIN = os.path.join(ROOT, "data", "brain.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(BRAIN), reason="needs data/brain.npz")


def families(brain, src_key, dst_key):
    from brain.lif import load_brain
    from trace_pathway import relays
    d, N, g = load_brain(brain)
    types = [str(t) for t in d["types"]]
    post, pre = torch.as_tensor(d["W_indices"])
    w = torch.as_tensor(d["W_values"])
    idx, bottleneck, _, _ = relays(g[src_key], g[dst_key], pre, post, w, N)
    out = {}
    for k in range(idx.numel()):
        t = types[int(idx[k])]
        fam = "".join(ch for ch in t if not ch.isdigit()).split("_")[0].rstrip("abc")
        out[fam] = out.get(fam, 0.0) + float(bottleneck[k])
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def test_lc10a_reaches_dna02_only_through_relays():
    from brain.lif import load_brain
    from trace_pathway import direct
    d, N, g = load_brain(BRAIN)
    post, pre = torch.as_tensor(d["W_indices"])
    w = torch.as_tensor(d["W_values"])
    _, n = direct(g["LC10a_L"], g["DNa02_L"], pre, post, w)
    assert n == 0, f"expected no direct LC10a->DNa02 synapses, found {n}"


def test_steering_pathway_is_aotu_then_lal():
    f = families(BRAIN, "LC10a_L", "DNa02_L")
    assert f.get("AOTU", 0) > 0.40, f"AOTU carries only {f.get('AOTU', 0):.1%} of LC10a->DNa02"
    assert f.get("AOTU", 0) + f.get("LAL", 0) > 0.75, f"AOTU+LAL below 75%: {f}"


def test_escape_pathway_is_monosynaptic():
    """LC4 synapses straight onto the giant fibre. That is why the badge app needs only 150 neurons."""
    from brain.lif import load_brain
    from trace_pathway import direct
    d, N, g = load_brain(BRAIN)
    post, pre = torch.as_tensor(d["W_indices"])
    w = torch.as_tensor(d["W_values"])
    tot, n = direct(g["LC4_L"], g["GF"], pre, post, w)
    assert n >= 50 and tot > 0, f"expected a strong direct LC4->GF projection, got {n} synapses at {tot:+.0f}"
