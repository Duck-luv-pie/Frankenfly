"""Tests that need the downloaded connectome."""
import numpy as np

from companion_brain.data.prune import load_or_build
from companion_brain.sim.runner import BrainRunner
from conftest import CFG, needs_data


@needs_data
def test_pruned_circuit_has_groups():
    c = load_or_build(CFG, verbose=False)
    assert 1000 < c.n < 139000
    for g in ("LC4", "LPLC2", "GF", "DNp09", "MDN"):
        assert len(c.idx(g)) > 0, g
    assert len(c.idx("GF")) == 2 and len(c.idx("GF", "left")) == 1


@needs_data
def test_looming_triggers_giant_fiber():
    c = load_or_build(CFG, verbose=False)
    r = BrainRunner(c, CFG, seed=3)
    assert sum(int(r.step_chunk().sum()) for _ in range(10)) == 0
    r.drive("LC4", 100.0); r.drive("LPLC2", 100.0)
    gf = c.idx("GF")
    spikes = sum(int(r.step_chunk()[gf].sum()) for _ in range(30))
    assert spikes > 0
