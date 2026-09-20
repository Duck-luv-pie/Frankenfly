"""
export_badge_circuit.py -- the fly's escape circuit, small enough to compile on a Hacker Badge.

    python badge/export_badge_circuit.py                      # defaults: 150 neurons, <=900 synapses
    python badge/export_badge_circuit.py --direct-only        # the fallback: LC4/LPLC2 -> GF edges only
    python badge/export_badge_circuit.py --neurons 120 --max-edges 600

Takes LC4 and LPLC2 (both sides), the giant fibre, and the intermediate neurons that sit on a two-hop
path between them, then shrinks that to something an ESP32-C3 can hold: the badge gives an app 48 KiB
of Lua heap (96 with `heap_kb=96`) and 64 KiB of source, and a big table literal can exhaust memory
while it is still compiling. So:

  * neurons are ordered [LC4 | LPLC2 | LC10a | intermediates | GF] and membership is a range check,
    which costs no memory at all;
  * the eye populations are subsampled (311 LC4+LPLC2 cells do not fit) by total synaptic weight;
  * only the strongest synapses among the kept neurons survive, and weights are quantized to int8;
  * the whole thing is emitted as two Lua strings, decoded with string.byte at runtime, because a
    string of 3N bytes costs a fraction of what a table of 3N numbers costs.

Everything that survives is a real connectome synapse with its real sign. Nothing is invented, but
this is a *subsample*, and the generated file says so on its first line.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import load_brain  # noqa: E402

EYE = ("LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R")


def pick(strength, pool, n):
    """The n neurons of `pool` with the most synaptic weight on the path we care about."""
    order = torch.argsort(strength[pool], descending=True)
    return pool[order[:n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--out", default="badge/flybadge_circuit.lua")
    ap.add_argument("--neurons", type=int, default=150)
    ap.add_argument("--max-edges", type=int, default=900)
    ap.add_argument("--lc10a", type=int, default=12, help="LC10a cells for the bump display (0 disables)")
    ap.add_argument("--direct-only", action="store_true", help="fallback: only LC4/LPLC2 -> GF edges")
    a = ap.parse_args()

    d, N, g = load_brain(a.brain)
    types = [str(t) for t in d["types"]]
    post, pre = torch.as_tensor(d["W_indices"])
    w = torch.as_tensor(d["W_values"])
    eye_all = torch.cat([g[k] for k in EYE])
    gf = g["GF"]

    # how much each eye cell projects toward the giant fibre, directly or through one intermediate
    to_gf = torch.zeros(N, dtype=torch.bool); to_gf[pre[torch.isin(post, gf)]] = True
    from_eye = torch.zeros(N, dtype=torch.bool); from_eye[post[torch.isin(pre, eye_all)]] = True
    mid_pool = (from_eye & to_gf).clone(); mid_pool[eye_all] = False; mid_pool[gf] = False
    mid_pool = mid_pool.nonzero().flatten()

    useful = torch.zeros(N)                                   # eye cell -> weight onto GF or an intermediate
    tgt = torch.zeros(N, dtype=torch.bool); tgt[gf] = True; tgt[mid_pool] = True
    m = torch.isin(pre, eye_all) & tgt[post]
    useful.index_add_(0, pre[m], w[m].abs())
    mid_strength = torch.zeros(N)                             # intermediate -> weight onto GF
    m2 = torch.isin(post, gf) & torch.isin(pre, mid_pool)
    mid_strength.index_add_(0, pre[m2], w[m2].abs())

    n_lc10a = 0 if a.direct_only else a.lc10a
    budget = a.neurons - gf.numel() - n_lc10a
    n_mid = 0 if a.direct_only else max(0, min(mid_pool.numel(), budget // 5))
    n_eye = max(2, budget - n_mid)
    # balanced across the two populations: ranking the pooled eye cells by strength picks only LC4,
    # and the escape response is a real LC4 + LPLC2 convergence onto the giant fibre
    lc4_pool = torch.cat([g["LC4_L"], g["LC4_R"]])
    lplc2_pool = torch.cat([g["LPLC2_L"], g["LPLC2_R"]])
    eye = torch.cat([pick(useful, lc4_pool, n_eye // 2),
                     pick(useful, lplc2_pool, n_eye - n_eye // 2)])
    mid = pick(mid_strength, mid_pool, n_mid)
    lc10a = pick(torch.ones(N), torch.cat([g["LC10a_L"], g["LC10a_R"]]), n_lc10a) if n_lc10a else torch.zeros(0, dtype=torch.long)

    lc4_mask = torch.isin(eye, lc4_pool)
    order = torch.cat([eye[lc4_mask], eye[~lc4_mask], lc10a, mid, gf])     # LC4 | LPLC2 | LC10a | mid | GF
    n_lc4 = int(lc4_mask.sum()); n_lplc2 = int((~lc4_mask).sum())
    keep = torch.zeros(N, dtype=torch.long) - 1
    keep[order] = torch.arange(order.numel())

    sel = (keep[pre] >= 0) & (keep[post] >= 0)
    if a.direct_only:
        sel &= torch.isin(pre, eye) & torch.isin(post, gf)
    ei, wi = sel.nonzero().flatten(), w[sel]
    if ei.numel() > a.max_edges:                                          # keep the strongest
        ei = ei[torch.argsort(wi.abs(), descending=True)[: a.max_edges]]
    p_i, q_i, wv = keep[pre[ei]].numpy(), keep[post[ei]].numpy(), w[ei].numpy()

    scale = max(1.0, float(np.abs(wv).max()) / 127.0)                     # int8 quantization
    q = np.clip(np.round(wv / scale), -127, 127).astype(np.int8)
    nz = q != 0
    p_i, q_i, q = p_i[nz], q_i[nz], q[nz]
    srt = np.lexsort((q_i, p_i))                                          # group by source neuron
    p_i, q_i, q = p_i[srt], q_i[srt], q[srt]
    colptr = np.concatenate([[0], np.cumsum(np.bincount(p_i, minlength=order.numel()))])

    def lua_b64(arr):
        import base64
        return base64.b64encode(bytes(int(v) & 0xFF for v in arr)).decode()

    n_neurons, n_edges = int(order.numel()), int(q.size)
    kept_types = {}
    for i in order.tolist():
        kept_types[types[i]] = kept_types.get(types[i], 0) + 1
    top = sorted(kept_types.items(), key=lambda kv: -kv[1])[:6]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"""-- flybadge_circuit.lua -- GENERATED by badge/export_badge_circuit.py, do not edit.
-- A subsample of the male Drosophila CNS connectome (neuPrint male-cns:v1.0): {n_neurons} neurons and
-- {n_edges} synapses of the looming-to-escape pathway. Every synapse below is real, with its real sign;
-- the sampling is ours ({n_lc4} of the LC4 cells, {n_lplc2} of LPLC2, ranked by how much they project
-- toward the giant fibre) because the full population is 311 neurons and 20,607 synapses, which does
-- not compile in 48 KiB of Lua heap.
-- Neuron order: [1..{n_lc4}] LC4, [{n_lc4 + 1}..{n_lc4 + n_lplc2}] LPLC2, """
                f"""[{n_lc4 + n_lplc2 + 1}..{n_lc4 + n_lplc2 + n_lc10a}] LC10a, """
                f"""[{n_lc4 + n_lplc2 + n_lc10a + 1}..{n_neurons - gf.numel()}] intermediates, """
                f"""[{n_neurons - gf.numel() + 1}..{n_neurons}] giant fibre.
-- Synapse weight = byte value (signed, -127..127) times {scale:.4f} raw synapse counts.
-- post_b64 / w_b64 are base64: shorter than decimal escapes and free of backslashes, which a serial
-- push cannot mangle. main.lua decodes them once at load.
local M = {{}}
M.n = {n_neurons}
M.n_lc4 = {n_lc4}
M.n_lplc2 = {n_lplc2}
M.n_lc10a = {n_lc10a}
M.eye_last = {n_lc4 + n_lplc2}          -- neurons 1..eye_last are LC4/LPLC2: the lesion target
M.lc10a_first = {n_lc4 + n_lplc2 + 1}
M.lc10a_last = {n_lc4 + n_lplc2 + n_lc10a}
M.gf_first = {n_neurons - gf.numel() + 1}
M.gf_last = {n_neurons}
M.n_edges = {n_edges}
M.w_scale = {scale:.6f}
-- outgoing edges grouped by source neuron; colptr[i]..colptr[i+1]-1 are neuron i's edges
M.colptr = {{{",".join(str(int(v)) for v in colptr)}}}
M.post_b64 = "{lua_b64(q_i + 1)}"
M.w_b64 = "{lua_b64(np.where(q < 0, q.astype(np.int16) + 256, q))}"   -- int8 as unsigned bytes
return M
""")
    size = os.path.getsize(a.out)
    print(f"{a.out}: {n_neurons} neurons, {n_edges} synapses, {size / 1024:.1f} KB of Lua source")
    print(f"  LC4 {n_lc4}, LPLC2 {n_lplc2}, LC10a {n_lc10a}, intermediates {int(mid.numel())}, GF {int(gf.numel())}")
    print(f"  weight scale {scale:.4f} raw synapses per unit; strongest kept edge {int(np.abs(q).max())}")
    print(f"  cell types: " + ", ".join(f"{t} x{c}" for t, c in top))
    print(f"  direct eye->GF synapses kept: {int(((q_i + 1) >= n_neurons - gf.numel() + 1).sum())}")
    if size > 40000:
        print("  WARNING: over 40 KB of source; the badge compiles main.lua in RAM, so shrink it")


if __name__ == "__main__":
    main()
