"""Map simulated FlyWire (female FAFB) neurons onto the MaleCNS neuron atlas (Janelia male CNS,
the dataset behind the Fly / Neural Atlas viewer) so spikes can light up real neuron
morphologies and cell-body positions.

The two datasets are different animals with different neuron ids, so the mapping is by
identity, not id: cell type and soma side. Every simulated neuron is assigned a MaleCNS
neuron of the same type on the same side (round-robin, preferring neurons whose full
skeleton is bundled in the atlas, then those with a mapped cell body). Types that only differ
by subtype suffix fall back to a prefix match (pC1a -> pC1_*, KCab -> KCab-*). Optic-lobe
types that the two nomenclatures name differently stay unmapped and are simply not shown."""
from __future__ import annotations

import collections
import gzip
import json
import re
from pathlib import Path

import numpy as np

SIDE = {"left": "L", "right": "R"}


def load_catalog(atlas_dir: Path):
    rows = json.loads(gzip.open(atlas_dir / "catalog.json.gz").read())
    manifest = json.loads((atlas_dir / "manifest.json").read_text())
    bundled = {n["id"] for n in manifest["neurons"]}
    return rows, manifest, bundled


def build_mapping(circuit, sides: np.ndarray, atlas_dir: Path) -> tuple[np.ndarray, dict]:
    rows, manifest, bundled = load_catalog(atlas_dir)
    by_type_side: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    by_type: dict[str, list[int]] = collections.defaultdict(list)
    for slot, r in enumerate(rows):
        t, side, soma = r[1], r[4], r[6]
        if not t:
            continue
        # rank: bundled skeleton first, then has soma, then the rest
        by_type_side[(t, side)].append(slot)
        by_type[t].append(slot)
    rank = {slot: (0 if r[0] in bundled else 1 if r[6] else 2) for slot, r in enumerate(rows)}
    for d in (by_type_side, by_type):
        for k, v in d.items():
            v.sort(key=lambda s: rank[s])
    prefixes = sorted(by_type.keys())
    prefix_cache: dict[str, list[int]] = {}

    def prefix_candidates(t: str) -> list[int]:
        if t in prefix_cache:
            return prefix_cache[t]
        out: list[int] = []
        for stem in (t, re.sub(r"[a-z0-9_\-]+$", "", t) if re.search(r"[A-Z]", t) else t):
            if len(stem) < 3:
                continue
            for k in prefixes:
                if k.startswith(stem) and k != t:
                    out += by_type[k]
            if out:
                break
        out.sort(key=lambda s: rank[s])
        prefix_cache[t] = out
        return out

    counters: dict = collections.Counter()
    slots = np.full(circuit.n, -1, dtype=np.int32)
    how = collections.Counter()
    for i in range(circuit.n):
        t = str(circuit.cell_type[i]) if circuit.cell_type is not None else ""
        side = SIDE.get(str(sides[i]), "")
        cands = by_type_side.get((t, side)) or by_type.get(t) or prefix_candidates(t)
        if not cands:
            how["unmapped"] += 1
            continue
        key = (t, side)
        slots[i] = cands[counters[key] % len(cands)]
        counters[key] += 1
        how["exact" if (t, side) in by_type_side else "type" if t in by_type else "prefix"] += 1
    stats = {"mapped": int((slots >= 0).sum()), "n": int(circuit.n), "how": dict(how),
             "atlas_neurons": len(rows), "skeletons": len(bundled), "dataset": manifest.get("dataset")}
    return slots, stats
