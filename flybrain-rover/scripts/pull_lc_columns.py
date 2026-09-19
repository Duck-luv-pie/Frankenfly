"""
pull_lc_columns.py -- lobula column footprint of every LC neuron -> data/lc_columns.json

    python scripts/pull_lc_columns.py [--types LC10a,LC11,LC12,LC15,LC4,LPLC2] [--out data/lc_columns.json]

For each neuron, neuPrint's roiInfo lists `LO_{L,R}_col_{hex1}_{hex2}` ROIs with pre/post synapse counts.
The postsynapse-weighted centroid in hex space gives the neuron's receptive-field position:
    H = hex1 - hex2 + 1   (horizontal, along the horizon; Reiser eyemap: hex1 -> q, hex2 -> p, H = q - p)
    V = hex1 + hex2 - 37  (vertical; origin [18, 19])
Output per bodyId: type, side (instance suffix), n_cols, post, H, V, hex1, hex2. Used by brain/retinotopy.py.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pull_connectome import cypher  # noqa: E402

PAT = re.compile(r"^LO_([LR])_col_(\d+)_(\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="LC10a,LC11,LC12,LC15,LC4,LPLC2")
    ap.add_argument("--out", default="data/lc_columns.json")
    a = ap.parse_args()
    types = a.types.split(",")
    _, data = cypher(f"MATCH (n:Neuron) WHERE n.type IN {json.dumps(types)} "
                     f"RETURN n.bodyId, n.type, n.instance, n.roiInfo")
    out = {}
    for bid, t, inst, ri in data:
        ri = json.loads(ri) if isinstance(ri, str) else ri
        cells = [(m.group(1), int(m.group(2)), int(m.group(3)), int(v["post"]))
                 for k, v in ri.items() if (m := PAT.match(k)) and v.get("post", 0) > 0]
        if not cells:
            continue
        side = inst[-1] if inst and inst[-1] in "LR" else cells[0][0]
        h1 = np.array([c[1] for c in cells], float); h2 = np.array([c[2] for c in cells], float)
        w = np.array([c[3] for c in cells], float)
        out[str(bid)] = dict(type=t, side=side, n_cols=len(cells), post=int(w.sum()),
                             H=float(((h1 - h2 + 1) * w).sum() / w.sum()), V=float(((h1 + h2 - 37) * w).sum() / w.sum()),
                             hex1=float((h1 * w).sum() / w.sum()), hex2=float((h2 * w).sum() / w.sum()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"{len(out)} neurons with lobula footprints (of {len(data)} queried) -> {a.out}")
    for t in types:
        for s in "LR":
            H = np.array([o["H"] for o in out.values() if o["type"] == t and o["side"] == s])
            if H.size:
                print(f"  {t:6s}{s}: n {H.size:3d}  H mean {H.mean():6.2f}  range [{H.min():5.1f}, {H.max():5.1f}]")


if __name__ == "__main__":
    main()
