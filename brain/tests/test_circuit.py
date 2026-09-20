"""Tests that need the downloaded connectome."""
import numpy as np

import copy

from companion_brain.data.prune import load_or_build
from companion_brain.sim.runner import BrainRunner
from companion_brain.body.decode import Decoder
from conftest import CFG, needs_data


def reference_cfg():
    """Paper-faithful settings: no spontaneous activity, no adaptation."""
    cfg = copy.deepcopy(CFG)
    cfg.lif["background_hz"] = 0.0
    cfg.lif["adapt_mv"] = 0.0
    return cfg


@needs_data
def test_pruned_circuit_has_groups():
    c = load_or_build(CFG, verbose=False)
    assert 1000 < c.n < 139000
    for g in ("LC4", "LPLC2", "GF", "DNp09", "MDN"):
        assert len(c.idx(g)) > 0, g
    assert len(c.idx("GF")) == 2 and len(c.idx("GF", "left")) == 1


@needs_data
def test_looming_triggers_giant_fiber():
    cfg = reference_cfg()
    c = load_or_build(cfg, verbose=False)
    r = BrainRunner(c, cfg, seed=3)
    assert sum(int(r.step_chunk().sum()) for _ in range(10)) == 0
    r.drive("LC4", 100.0); r.drive("LPLC2", 100.0)
    gf = c.idx("GF")
    spikes = sum(int(r.step_chunk()[gf].sum()) for _ in range(30))
    assert spikes > 0


@needs_data
def test_live_brain_is_spontaneously_active_but_quiet_until_looming():
    c = load_or_build(CFG, verbose=False)
    r = BrainRunner(c, CFG, seed=4)
    from companion_brain.body.decode import decode_windows
    base = r.calibrate(3.0, decode_windows(CFG), warmup_s=1.0, verbose=False)
    assert base["500"]["DN_all"]["all"]["mean"] > 0.1     # descending neurons fire spontaneously
    dec = Decoder(CFG, base)
    rbw = lambda: {w: r.rates(w) for w in dec.windows}
    # resting: escape should not trigger
    esc_rest = max(dec.decode(rbw(), now=t * 0.05).scores["escape"] for t in range(20) if r.step_chunk() is not None)
    assert esc_rest < 0.6
    # looming: escape score saturates
    r.drive("LC4", 100.0); r.drive("LPLC2", 100.0)
    best = 0.0
    for t in range(30):
        r.step_chunk(); r.drive("LC4", 100.0); r.drive("LPLC2", 100.0)
        best = max(best, dec.decode(rbw(), now=10 + t * 0.05).scores["escape"])
    assert best >= 0.9
