"""Resolve named neuron groups from the FlyWire annotation table.

A group selector is a dict with any of: cell_type (list), hemibrain_type (list), regex
(against cell_type), top_nt, super_class. Matches are OR-ed. Every group is resolved to
root ids for 'left', 'right' and 'all' (side comes from the annotation `side` column;
neurons with no or 'center' side end up only in 'all')."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config

COLUMNS = ["root_id", "super_class", "cell_class", "cell_type", "hemibrain_type", "top_nt", "side"]
POS_COLUMNS = ["pos_x", "pos_y", "pos_z"]


def load_annotations(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(path, sep="\t", usecols=COLUMNS + POS_COLUMNS, low_memory=False)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    for c in POS_COLUMNS:
        ann[c] = ann[c].fillna(0).astype(np.float32)
    for c in COLUMNS[1:]:
        ann[c] = ann[c].fillna("")
    return ann


@dataclass
class Group:
    name: str
    left: np.ndarray
    right: np.ndarray
    all: np.ndarray
    meta: dict = field(default_factory=dict)

    def side(self, which: str) -> np.ndarray:
        return {"left": self.left, "right": self.right, "all": self.all}[which]

    def __len__(self) -> int:
        return len(self.all)


def select(ann: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mask = np.zeros(len(ann), dtype=bool)
    if "cell_type" in spec:
        mask |= ann["cell_type"].isin(spec["cell_type"]).to_numpy()
    if "hemibrain_type" in spec:
        mask |= ann["hemibrain_type"].isin(spec["hemibrain_type"]).to_numpy()
    if "regex" in spec:
        pat = re.compile(spec["regex"])
        mask |= ann["cell_type"].map(lambda s: bool(pat.search(s))).to_numpy()
    if "top_nt" in spec:
        mask |= (ann["top_nt"] == spec["top_nt"]).to_numpy()
    if "super_class" in spec:
        mask |= (ann["super_class"] == spec["super_class"]).to_numpy()
    return ann[mask]


def resolve_groups(ann: pd.DataFrame, cfg: Config, restrict_to: np.ndarray | None = None) -> dict[str, Group]:
    """Resolve every group in cfg.groups. `restrict_to` limits ids to those present in the model."""
    keep = None if restrict_to is None else set(int(x) for x in restrict_to)
    groups: dict[str, Group] = {}
    for name, spec in cfg.groups.items():
        sub = select(ann, dict(spec))
        if keep is not None:
            sub = sub[sub["root_id"].isin(keep)]
        ids = sub["root_id"].to_numpy(dtype=np.int64)
        side = sub["side"].to_numpy()
        groups[name] = Group(
            name=name,
            left=ids[side == "left"],
            right=ids[side == "right"],
            all=ids,
            meta={"types": sorted(sub["cell_type"].unique().tolist())[:12]},
        )
        if len(ids) == 0:
            print(f"[annotations] warning: group {name!r} matched no neurons ({spec})")
    return groups
