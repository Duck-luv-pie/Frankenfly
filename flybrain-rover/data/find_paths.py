"""
find_paths.py -- shortest synaptic paths from the eye to the forward descending neurons.

    python data/find_paths.py [--max-hops 4] [--min-weight 5] [--out data/paths_v2.json]

For every (source type, target type) in LC10a/LC4 x DNp09/DNa01, asks neuPrint for the shortest
ConnectsTo path (<= max_hops, every edge weight >= min_weight) from each source neuron to each target
neuron, and collects every neuron on those paths. Output: nodes (bodyId, type, instance) and per-pair
stats, for data/pull_connectome.py --include-ids.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pull_connectome import cypher  # noqa: E402

SOURCES = ["LC10a", "LC4"]
TARGETS = ["DNp09", "DNa01"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--out", default="data/paths_v2.json")
    a = ap.parse_args()
    nodes, pairs = {}, {}
    for s in SOURCES:
        for t in TARGETS:
            q = f"""
            MATCH (a:Neuron {{type:'{s}'}}), (b:Neuron {{type:'{t}'}})
            MATCH p = shortestPath((a)-[:ConnectsTo*..{a.max_hops}]->(b))
            WHERE ALL(r IN relationships(p) WHERE r.weight >= {a.min_weight})
            RETURN a.bodyId, b.bodyId, b.instance, length(p),
                   [n IN nodes(p) | n.bodyId], [n IN nodes(p) | n.type], [n IN nodes(p) | n.instance],
                   [r IN relationships(p) | r.weight]
            """
            t0 = time.time()
            _, rows = cypher(q, timeout=900)
            hops = [r[3] for r in rows]
            inter = {}
            for r in rows:
                for bid, ty, inst in zip(r[4], r[5], r[6]):
                    nodes[int(bid)] = dict(type=ty, instance=inst)
                for ty in r[5][1:-1]:
                    inter[ty] = inter.get(ty, 0) + 1
            top = sorted(inter.items(), key=lambda kv: -kv[1])[:8]
            pairs[f"{s}->{t}"] = dict(n_paths=len(rows), by_hops={h: hops.count(h) for h in sorted(set(hops))},
                                     top_intermediates=top, seconds=round(time.time() - t0, 1))
            print(f"{s}->{t}: {len(rows)} paths, hops {pairs[f'{s}->{t}']['by_hops']}, "
                  f"intermediates {top} ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(dict(max_hops=a.max_hops, min_weight=a.min_weight, pairs=pairs,
                       nodes={str(k): v for k, v in nodes.items()}), f, indent=1)
    print(f"{len(nodes)} distinct neurons on shortest paths -> {a.out}")


if __name__ == "__main__":
    main()
