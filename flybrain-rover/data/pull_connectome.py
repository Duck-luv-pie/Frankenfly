"""
pull_connectome.py -- neuPrint (male-cns:v1.0) -> data/brain.npz

    python data/pull_connectome.py [--max-n 15000] [--min-weight 1] [--hop-min-weight 5]

Talks to neuPrint's /api/custom/custom cypher endpoint directly (verified 2026-09-17:
male-cns:v1.0 answers without a token; a NEUPRINT_TOKEN in .env is sent if present).

Steps
1. Seed populations (field names verified against the live dataset):
     visual projection : LC10a LC11 LC12 LC15 LC4 LPLC2          (type exact, side from somaSide / instance suffix)
     descending        : DNa02 DNa01 DNp09 DNp01 (giant fiber -> key GF)
     dopamine          : PAM.* PPL1.*
     mushroom body     : KC.* MBON.*
     thermosensory     : TRN_VP3a / TRN_VP3b (hot receptor neurons) + VP3-glomerulus projection neurons
                         (no type matches "hot"/"thermo"/"TPN"; VP3 is the hot glomerulus, Marin et al. 2020)
2. Expand one synaptic hop (weight >= hop_min_weight). Neighbours are ranked per seed family
   (visual / descending / mushroom-body) so the LC->DN path is not crowded out by KC partners.
   Total N capped at max_n.
3. Fetch all ConnectsTo edges among the final set with weight >= min_weight.
   Sign from the PRE neuron's consensusNt (fallback predictedNt, celltypePredictedNt):
   acetylcholine +1, gaba -1, glutamate -1, histamine -1, anything else (unclear/dopamine/...) +1.
4. Save data/brain.npz:
     N, ids (bodyId int64), types (U), instances (U), nt (U), sides (U),
     W_indices (2, nnz) int64 as [post, pre], W_values (nnz,) float32 signed synapse counts,
     one int64 index array per group key in CLAUDE.md, meta (json string).
5. Print group -> count. M1 passes if every group is non-empty and 3k <= N <= 15k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass

SERVER = os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org")
DATASET = os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0")
TOKEN = os.environ.get("NEUPRINT_TOKEN", "").strip()

VISUAL = ["LC10a", "LC11", "LC12", "LC15", "LC4", "LPLC2"]
DESCENDING = ["DNa02", "DNa01", "DNp09", "DNp01"]
SIDED = VISUAL + ["DNa02", "DNa01"]           # groups that get _L / _R keys
GROUP_KEYS = [f"{t}_{s}" for t in SIDED for s in ("L", "R")] + \
             ["THERMO_L", "THERMO_R", "DNp09", "GF", "PAM", "PPL1", "KC", "MBON"]

# cypher predicates on a node variable, one per seed family
def _pred_visual(v):
    return f"{v}.type IN {json.dumps(VISUAL)}"

def _pred_dn(v):
    return f"{v}.type IN {json.dumps(DESCENDING)}"

def _pred_mb(v):
    return f"({v}.type =~ 'PAM.*' OR {v}.type =~ 'PPL1.*' OR {v}.type =~ 'KC.*' OR {v}.type =~ 'MBON.*')"

def _pred_thermo(v):
    return f"({v}.type STARTS WITH 'TRN_VP3' OR ({v}.type CONTAINS 'VP3' AND {v}.type ENDS WITH 'PN'))"

def _pred_seed(v):
    return f"({_pred_visual(v)} OR {_pred_dn(v)} OR {_pred_mb(v)} OR {_pred_thermo(v)})"


NEURON_FIELDS = "bodyId, type, instance, somaSide, consensusNt, predictedNt, celltypePredictedNt, superclass"


def cypher(q: str, timeout: int = 600, retries: int = 3) -> tuple[list, list]:
    hdr = {"Content-Type": "application/json"}
    if TOKEN:
        hdr["Authorization"] = f"Bearer {TOKEN}"
    for attempt in range(retries):
        try:
            r = requests.post(f"{SERVER}/api/custom/custom", headers=hdr,
                              json={"cypher": q, "dataset": DATASET}, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            return j["columns"], j["data"]
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"  neuPrint error ({e}); retry {attempt + 1}/{retries}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def side_of(soma_side: str | None, instance: str | None) -> str:
    if soma_side in ("L", "R"):
        return soma_side
    if isinstance(instance, str):
        if instance.endswith("_L"):
            return "L"
        if instance.endswith("_R"):
            return "R"
    return "?"


def nt_sign(row: dict) -> tuple[float, str]:
    nt = row.get("consensusNt") or row.get("predictedNt") or row.get("celltypePredictedNt") or ""
    nt = str(nt).lower()
    if nt in ("gaba", "glutamate", "histamine"):
        return -1.0, nt
    return 1.0, (nt or "unknown")


def rows_to_dicts(cols: list, data: list) -> list[dict]:
    return [dict(zip(cols, r)) for r in data]


def fetch_seed() -> list[dict]:
    q = f"MATCH (n:Neuron) WHERE {_pred_seed('n')} RETURN n.bodyId AS bodyId, n.type AS type, " \
        f"n.instance AS instance, n.somaSide AS somaSide, n.consensusNt AS consensusNt, " \
        f"n.predictedNt AS predictedNt, n.celltypePredictedNt AS celltypePredictedNt, n.superclass AS superclass"
    cols, data = cypher(q)
    return rows_to_dicts(cols, data)


def fetch_neighbours(hop_min_weight: int) -> list[dict]:
    """Neighbours of the seed (either direction), with summed weight per seed family."""
    q = f"""
    MATCH (s:Neuron)-[c:ConnectsTo]-(b:Neuron)
    WHERE {_pred_seed('s')} AND NOT {_pred_seed('b')} AND c.weight >= {int(hop_min_weight)}
    WITH b,
         sum(CASE WHEN {_pred_visual('s')} THEN c.weight ELSE 0 END) AS w_vis,
         sum(CASE WHEN {_pred_dn('s')} THEN c.weight ELSE 0 END) AS w_dn,
         sum(CASE WHEN {_pred_mb('s')} THEN c.weight ELSE 0 END) AS w_mb,
         sum(CASE WHEN {_pred_thermo('s')} THEN c.weight ELSE 0 END) AS w_th,
         sum(c.weight) AS w_tot
    RETURN b.bodyId AS bodyId, b.type AS type, b.instance AS instance, b.somaSide AS somaSide,
           b.consensusNt AS consensusNt, b.predictedNt AS predictedNt,
           b.celltypePredictedNt AS celltypePredictedNt, b.superclass AS superclass,
           w_vis, w_dn, w_mb, w_th, w_tot
    """
    cols, data = cypher(q, timeout=1200)
    return rows_to_dicts(cols, data)


def select_neighbours(nb: list[dict], k_total: int,
                       quotas: tuple = (("w_vis", 0.35), ("w_dn", 0.30), ("w_th", 0.05), ("w_mb", 0.15))) -> list[dict]:
    """Per-family top-k (by summed weight into that family), remainder by global total."""
    chosen, chosen_ids = [], set()
    for key, frac in quotas:
        k = int(k_total * frac)
        ranked = sorted((r for r in nb if r[key] > 0 and r["bodyId"] not in chosen_ids),
                        key=lambda r: -r[key])[:k]
        for r in ranked:
            chosen.append(r); chosen_ids.add(r["bodyId"])
    rest = sorted((r for r in nb if r["bodyId"] not in chosen_ids), key=lambda r: -r["w_tot"])
    for r in rest[: max(0, k_total - len(chosen))]:
        chosen.append(r); chosen_ids.add(r["bodyId"])
    return chosen


def fetch_edges(ids: list, min_weight: int, chunk: int = 1000) -> tuple:
    ids = [int(i) for i in ids]
    all_lit = json.dumps(ids)
    pre, post, w = [], [], []
    n_chunks = (len(ids) + chunk - 1) // chunk
    for ci in range(n_chunks):
        sub = ids[ci * chunk:(ci + 1) * chunk]
        q = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN {json.dumps(sub)} AND b.bodyId IN {all_lit} AND c.weight >= {int(min_weight)}
        RETURN a.bodyId, b.bodyId, c.weight
        """
        t0 = time.time()
        _, data = cypher(q, timeout=1200)
        for a, b, cw in data:
            pre.append(a); post.append(b); w.append(cw)
        print(f"  edges chunk {ci + 1}/{n_chunks}: +{len(data):,} ({time.time() - t0:.0f}s, total {len(w):,})",
              flush=True)
    return np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64), np.asarray(w, dtype=np.float32)


def build_groups(neurons: list[dict]) -> dict:
    """neurons: list of dicts in final index order. Returns {key: int64 array}."""
    types = [str(n.get("type") or "") for n in neurons]
    sides = [n["_side"] for n in neurons]
    idx = {k: [] for k in GROUP_KEYS}
    for i, (t, s) in enumerate(zip(types, sides)):
        if t in SIDED and s in ("L", "R"):
            idx[f"{t}_{s}"].append(i)
        if t == "DNp09":
            idx["DNp09"].append(i)
        if t == "DNp01":
            idx["GF"].append(i)
        if t.startswith("PAM"):
            idx["PAM"].append(i)
        if t.startswith("PPL1"):
            idx["PPL1"].append(i)
        if t.startswith("KC"):
            idx["KC"].append(i)
        if t.startswith("MBON"):
            idx["MBON"].append(i)
        if t.startswith("TRN_VP3") or ("VP3" in t and t.endswith("PN")):
            if s in ("L", "R"):
                idx[f"THERMO_{s}"].append(i)
    # fallback for any sided group that came out empty: split the un-sided members by parity
    for t in SIDED:
        if not idx[f"{t}_L"] or not idx[f"{t}_R"]:
            members = [i for i, tt in enumerate(types) if tt == t]
            if members:
                print(f"  WARNING: {t} had no L/R labels; splitting {len(members)} by parity")
                idx[f"{t}_L"], idx[f"{t}_R"] = members[0::2], members[1::2]
    return {k: np.asarray(v, dtype=np.int64) for k, v in idx.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=15000)
    ap.add_argument("--min-weight", type=int, default=1, help="min synapse count for an edge to be kept")
    ap.add_argument("--hop-min-weight", type=int, default=5, help="min weight for a 1-hop neighbour")
    ap.add_argument("--out", default="data/brain.npz")
    ap.add_argument("--include-ids", default=None,
                    help="JSON list of bodyIds to force into the circuit (e.g. from data/find_paths.py)")
    args = ap.parse_args()

    t0 = time.time()
    print(f"neuPrint {SERVER} dataset {DATASET} token={'yes' if TOKEN else 'no'}")
    seed = fetch_seed()
    print(f"seed neurons: {len(seed)} ({time.time() - t0:.0f}s)")
    seed_ids = {r["bodyId"] for r in seed}

    forced = []
    if args.include_ids:
        want = {int(x) for x in json.load(open(args.include_ids))} - seed_ids
        cols, data = cypher(f"MATCH (n:Neuron) WHERE n.bodyId IN {json.dumps(sorted(want))} RETURN "
                            f"n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.somaSide AS somaSide, "
                            f"n.consensusNt AS consensusNt, n.predictedNt AS predictedNt, "
                            f"n.celltypePredictedNt AS celltypePredictedNt, n.superclass AS superclass")
        forced = rows_to_dicts(cols, data)
        print(f"forced path neurons: {len(forced)} (of {len(want)} requested outside the seed)")
    forced_ids = {r["bodyId"] for r in forced}
    k_nb = max(0, args.max_n - len(seed) - len(forced))
    nb = fetch_neighbours(args.hop_min_weight)
    nb = [r for r in nb if r["bodyId"] not in forced_ids]
    print(f"1-hop neighbours (w>={args.hop_min_weight}): {len(nb):,}; keeping {k_nb:,} ({time.time() - t0:.0f}s)")
    chosen = select_neighbours(nb, k_nb)

    neurons = seed + forced + chosen
    neurons.sort(key=lambda r: r["bodyId"])
    for r in neurons:
        r["_side"] = side_of(r.get("somaSide"), r.get("instance"))
        r["_sign"], r["_nt"] = nt_sign(r)
    ids = np.asarray([r["bodyId"] for r in neurons], dtype=np.int64)
    N = len(ids)
    local = {int(b): i for i, b in enumerate(ids)}

    pre_b, post_b, w = fetch_edges(ids, args.min_weight)
    pre_i = np.asarray([local[int(b)] for b in pre_b], dtype=np.int64)
    post_i = np.asarray([local[int(b)] for b in post_b], dtype=np.int64)
    sign = np.asarray([r["_sign"] for r in neurons], dtype=np.float32)
    vals = (w * sign[pre_i]).astype(np.float32)

    # dedupe (post, pre) pairs by summing, just in case
    key = post_i * N + pre_i
    uniq, inv = np.unique(key, return_inverse=True)
    vals_u = np.zeros(len(uniq), dtype=np.float32)
    np.add.at(vals_u, inv, vals)
    post_u, pre_u = uniq // N, uniq % N
    W_indices = np.stack([post_u, pre_u]).astype(np.int64)

    groups = build_groups(neurons)
    meta = dict(server=SERVER, dataset=DATASET, max_n=args.max_n, min_weight=args.min_weight,
                hop_min_weight=args.hop_min_weight, n_seed=len(seed), n_neighbours=len(chosen),
                n_forced=len(forced), include_ids=args.include_ids,
                pulled=time.strftime("%Y-%m-%d %H:%M:%S"),
                sign_rule="pre consensusNt: gaba/glutamate/histamine -1, else +1")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, N=np.int64(N), ids=ids,
             types=np.asarray([str(r.get("type") or "") for r in neurons], dtype="U"),
             instances=np.asarray([str(r.get("instance") or "") for r in neurons], dtype="U"),
             nt=np.asarray([r["_nt"] for r in neurons], dtype="U"),
             sides=np.asarray([r["_side"] for r in neurons], dtype="U"),
             superclass=np.asarray([str(r.get("superclass") or "") for r in neurons], dtype="U"),
             W_indices=W_indices, W_values=vals_u, meta=json.dumps(meta), **groups)

    print(f"\nsaved {args.out}: N={N:,} nnz={len(vals_u):,} "
          f"(exc {int((vals_u > 0).sum()):,} / inh {int((vals_u < 0).sum()):,}) in {time.time() - t0:.0f}s")
    ok = 3000 <= N <= 15000
    print(f"{'group':10s} count")
    for k in GROUP_KEYS:
        c = len(groups[k]); ok &= c > 0
        print(f"{k:10s} {c:6d}{'' if c else '   <-- EMPTY'}")
    print("M1", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
