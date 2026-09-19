"""Build the simulated circuit from the connectome.

The full brain (139k neurons, 15M synaptic pairs) is too slow for real time on a laptop, so
we keep only neurons that lie on short paths from the stimulated input groups to the readout
groups: forward BFS from inputs, backward BFS from readouts, intersection (plus the groups
themselves), and every connection among the kept neurons. `--full` keeps everything.

The result is cached as an .npz next to a .json describing the groups."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import Config
from .annotations import Group, load_annotations, resolve_groups
from .download import raw_paths


@dataclass
class Circuit:
    root_ids: np.ndarray          # int64 [N] FlyWire root id of each simulated neuron
    pre: np.ndarray               # int32 [M] presynaptic index
    post: np.ndarray              # int32 [M] postsynaptic index
    weight_mv: np.ndarray         # float32 [M] signed synaptic weight in mV (count * sign * w_syn)
    groups: dict[str, dict[str, np.ndarray]]   # name -> {left,right,all} -> indices into root_ids
    meta: dict
    pos: np.ndarray | None = None            # float32 [N,3] anchor position (FAFB voxels), for display
    cell_type: np.ndarray | None = None      # str [N] annotation cell type, for display

    @property
    def n(self) -> int:
        return len(self.root_ids)

    @property
    def m(self) -> int:
        return len(self.pre)

    def idx(self, group: str, side: str = "all") -> np.ndarray:
        return self.groups[group][side]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"g__{g}__{s}": v for g, d in self.groups.items() for s, v in d.items()}
        extra = {}
        if self.pos is not None:
            extra["pos"] = self.pos
        if self.cell_type is not None:
            extra["cell_type"] = np.asarray(self.cell_type, dtype=str)
        np.savez_compressed(path, root_ids=self.root_ids, pre=self.pre, post=self.post,
                            weight_mv=self.weight_mv, **flat, **extra)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Circuit":
        z = np.load(path)
        groups: dict[str, dict[str, np.ndarray]] = {}
        for k in z.files:
            if k.startswith("g__"):
                _, g, s = k.split("__")
                groups.setdefault(g, {})[s] = z[k]
        meta = json.loads(path.with_suffix(".json").read_text()) if path.with_suffix(".json").exists() else {}
        return cls(z["root_ids"], z["pre"], z["post"], z["weight_mv"], groups, meta,
                   pos=z["pos"] if "pos" in z.files else None,
                   cell_type=z["cell_type"] if "cell_type" in z.files else None)


def circuit_key(cfg: Config, full: bool) -> str:
    material = json.dumps({"groups": cfg.groups, "inputs": cfg.inputs, "readouts": cfg.readouts,
                           "prune": cfg.prune, "min_synapses": cfg.lif.min_synapses,
                           "w_syn": cfg.lif.w_syn_mv, "full": full, "schema": 2}, sort_keys=True, default=str)
    return hashlib.sha1(material.encode()).hexdigest()[:10]


def cache_path(cfg: Config, full: bool) -> Path:
    return cfg.path("data.cache_dir") / f"circuit_{'full' if full else 'pruned'}_{circuit_key(cfg, full)}.npz"


def load_or_build(cfg: Config, full: bool = False, rebuild: bool = False, verbose: bool = True) -> Circuit:
    path = cache_path(cfg, full)
    if path.exists() and not rebuild:
        c = Circuit.load(path)
        if verbose:
            print(f"[prune] loaded cached circuit {path.name}: {c.n:,} neurons, {c.m:,} connections")
        return c
    c = build(cfg, full=full, verbose=verbose)
    c.save(path)
    if verbose:
        print(f"[prune] cached to {path}")
    return c


def build(cfg: Config, full: bool = False, verbose: bool = True) -> Circuit:
    paths = raw_paths(cfg)
    log = print if verbose else (lambda *a, **k: None)

    log("[prune] loading connectivity ...")
    con = pd.read_parquet(paths["connectivity"],
                          columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"])
    con = con[con["Connectivity"] >= cfg.lif.min_synapses]
    comp = pd.read_csv(paths["completeness"], index_col=0)
    all_ids = comp.index.to_numpy(dtype=np.int64)
    ann = load_annotations(paths["annotations"])
    groups = resolve_groups(ann, cfg, restrict_to=all_ids)

    # Index all model neurons
    id2i = pd.Series(np.arange(len(all_ids), dtype=np.int32), index=all_ids)
    pre = id2i.reindex(con["Presynaptic_ID"].to_numpy()).to_numpy()
    post = id2i.reindex(con["Postsynaptic_ID"].to_numpy()).to_numpy()
    ok = ~(np.isnan(pre) | np.isnan(post))
    pre = pre[ok].astype(np.int32)
    post = post[ok].astype(np.int32)
    count = con["Connectivity"].to_numpy()[ok].astype(np.float32)
    sign = con["Excitatory"].to_numpy()[ok].astype(np.float32)
    N = len(all_ids)
    log(f"[prune] full model: {N:,} neurons, {len(pre):,} connections with >= {cfg.lif.min_synapses} synapses")

    if full:
        keep = np.ones(N, dtype=bool)
    else:
        adj = sp.csr_matrix((np.ones(len(pre), dtype=np.int8), (pre, post)), shape=(N, N))
        src = np.zeros(N, dtype=bool)
        for g in cfg.inputs:
            src[id2i.reindex(groups[g].all).dropna().to_numpy().astype(int)] = True
        dst = np.zeros(N, dtype=bool)
        for g in cfg.readouts:
            dst[id2i.reindex(groups[g].all).dropna().to_numpy().astype(int)] = True
        fwd = _bfs(adj, src, cfg.prune.hops_forward)
        bwd = _bfs(adj.T.tocsr(), dst, cfg.prune.hops_backward)
        keep = (fwd & bwd) | src | dst
        log(f"[prune] forward reach {fwd.sum():,}, backward reach {bwd.sum():,}, kept {keep.sum():,}")

    new_index = np.full(N, -1, dtype=np.int32)
    new_index[keep] = np.arange(keep.sum(), dtype=np.int32)
    emask = keep[pre] & keep[post]
    kept_ids = all_ids[keep]
    ann_idx = ann.set_index("root_id")
    ann_kept = ann_idx.reindex(kept_ids)
    pos = ann_kept[["pos_x", "pos_y", "pos_z"]].fillna(0).to_numpy(dtype=np.float32)
    cell_type = ann_kept["cell_type"].fillna("").replace("", "unnamed").to_numpy(dtype=str)
    circuit_groups = {}
    for name, g in groups.items():
        circuit_groups[name] = {s: _to_new(id2i, new_index, g.side(s)) for s in ("left", "right", "all")}

    c = Circuit(
        root_ids=kept_ids,
        pre=new_index[pre[emask]],
        post=new_index[post[emask]],
        weight_mv=(count[emask] * sign[emask] * cfg.lif.w_syn_mv).astype(np.float32),
        groups=circuit_groups,
        meta={
            "full": full, "n": int(keep.sum()), "m": int(emask.sum()),
            "group_sizes": {k: {s: int(len(v[s])) for s in v} for k, v in circuit_groups.items()},
            "group_types": {k: g.meta.get("types", []) for k, g in groups.items()},
        },
        pos=pos,
        cell_type=cell_type,
    )
    log(f"[prune] circuit: {c.n:,} neurons, {c.m:,} connections")
    for name in list(cfg.inputs) + list(cfg.readouts):
        d = c.groups[name]
        log(f"         {name:15s} L={len(d['left']):4d} R={len(d['right']):4d} all={len(d['all']):4d}")
    return c


def _bfs(adj: sp.csr_matrix, start: np.ndarray, hops: int) -> np.ndarray:
    reached = start.copy()
    frontier = start.copy()
    for _ in range(hops):
        nxt = (adj.T @ frontier.astype(np.int32)) > 0  # rows=pre, so successors of frontier
        nxt = np.asarray(nxt).ravel() & ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt
    return reached


def _to_new(id2i: pd.Series, new_index: np.ndarray, ids: np.ndarray) -> np.ndarray:
    old = id2i.reindex(ids).dropna().to_numpy().astype(int)
    new = new_index[old]
    return np.sort(new[new >= 0]).astype(np.int32)
