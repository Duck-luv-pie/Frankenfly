"""
retinotopy.py -- give each LC neuron the retina column its real dendrites look at.

Before 2026-09-18 senses.py spread each LC population over the 12 columns of its hemisphere in bodyId
order, which is not retinotopic: a centred human drove DNa02_R (68 Hz) but not DNa02_L (0 Hz). This module
uses each neuron's lobula dendritic footprint from neuPrint (roiInfo keys `LO_{L,R}_col_{hex1}_{hex2}`,
postsynapse counts; pulled by scripts/pull_lc_columns.py into data/lc_columns.json).

Hex -> visual axes (Reiser lab eyemap docs, docs/coordinate-systems.md + column_coord.png):
    hex1 -> q, hex2 -> p, origin [hex1, hex2] = [18, 19]; v (vertical) = p + q; the axis "parallel to the
    perceived horizon" is q - p. So per neuron: H = hex1 - hex2 + 1 (horizontal, ~5.3 deg per unit),
    V = hex1 + hex2 - 37 (vertical). Both hemispheres share the convention (L/R statistics match).
Anterior sign: which end of H is frontal is not stated in the docs. LC10a, the courtship-tracking
population, is biased to +H (mean +3.5 vs -2.4 for LC11, which tiles the whole eye), so +H is taken as
anterior. `anterior_sign=-1` flips it; steering does not depend on it (hemisphere assignment is by side),
only the centre-vs-edge ordering within a hemisphere does.

Modes
  rank   (default) within a hemisphere, neurons ranked by anteriorness fill that hemisphere's n/2 columns by
         quantile: the most anterior neurons get the column next to the midline (az ~ 0), the most posterior
         the outermost column (az ~ -49 deg on the left). Every neuron gets a column; L/R are mirror images.
  angle  az_deg = side * deg_per_col * (h_edge - anterior_sign * H) with h_edge the anterior edge (~16) at
         az 0; the neuron gets the retina column containing that azimuth, or -1 (never driven) if its field
         lies outside the camera's +-49 deg. Faithful scale, but ~2/3 of LC11/12/15 fall outside the FOV.
  index  the old bodyId-order spread (for A/B comparisons).

    col_of = column_assignment(groups, ids, n_cols=24, mode="rank")   # {"LC10a_L": LongTensor(n), ...}
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

POPS = ("LC10a", "LC11", "LC12", "LC15")


def load_columns(path: str = "data/lc_columns.json") -> dict[int, dict]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def column_assignment(groups: dict[str, torch.Tensor], ids: np.ndarray, n_cols: int = 24, mode: str = "rank",
                      anterior_sign: int = 1, path: str = "data/lc_columns.json", pops: tuple = POPS,
                      hfov_deg: float = 98.43, deg_per_col: float = 5.3, h_edge: float = 16.0,
                      device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    """-> {f"{pop}_{side}": LongTensor of retina column per neuron (aligned with groups[key]); -1 = none}."""
    half = n_cols // 2
    ids = np.asarray(ids)
    cols_db = load_columns(path) if (mode != "index" and os.path.exists(path)) else None
    out = {}
    for pop in pops:
        for side, offset in (("L", 0), ("R", half)):
            key = f"{pop}_{side}"
            idx = groups[key].cpu()
            n = int(idx.numel())
            if mode == "index" or cols_db is None:
                k = torch.arange(n)
                col = offset + (k * half) // max(n, 1)
                out[key] = col.to(device) if device is not None else col
                continue
            H = np.array([cols_db.get(int(ids[i]), {}).get("H", np.nan) for i in idx.tolist()], dtype=float)
            if np.isnan(H).any():                      # neurons without a footprint: put them at the median
                H[np.isnan(H)] = np.nanmedian(H) if not np.isnan(H).all() else 0.0
            anterior = anterior_sign * H              # larger = more frontal
            if mode == "rank":
                order = np.argsort(anterior, kind="stable")      # posterior ... anterior
                rank = np.empty(n, dtype=float); rank[order] = np.arange(n)
                q = (rank + 0.5) / max(n, 1)                      # 0 = most posterior, 1 = most anterior
                within = np.minimum((q * half).astype(int), half - 1)   # 0 = most posterior ... half-1 = most anterior
                # left eye: column 0 is az -49 (lateral), column half-1 is az ~0 (next to the midline)
                # right eye: column half is az ~0, column n-1 is az +49
                col = within if side == "L" else half + (half - 1 - within)
            elif mode == "angle":
                sgn = -1.0 if side == "L" else 1.0
                az = sgn * deg_per_col * (h_edge - anterior)      # degrees, + = right
                col_w = hfov_deg / n_cols
                col = np.floor((az + hfov_deg / 2) / col_w).astype(int)
                col = np.where((col >= 0) & (col < n_cols), col, -1)
            else:
                raise ValueError(mode)
            t = torch.as_tensor(np.asarray(col, dtype=np.int64))
            out[key] = t.to(device) if device is not None else t
    return out


def describe(col_of: dict[str, torch.Tensor], n_cols: int = 24) -> str:
    lines = []
    for key, col in col_of.items():
        c = col.cpu().numpy()
        used = np.bincount(c[c >= 0], minlength=n_cols)
        lines.append(f"{key:8s} n={len(c):3d} silent={int((c < 0).sum()):3d} per-column counts {used.tolist()}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from brain.lif import load_brain
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--mode", default="rank", choices=["rank", "angle", "index"])
    ap.add_argument("--anterior-sign", type=int, default=1)
    a = ap.parse_args()
    d, N, groups = load_brain(a.brain)
    print(describe(column_assignment(groups, d["ids"], mode=a.mode, anterior_sign=a.anterior_sign)))
