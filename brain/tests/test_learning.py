"""Mushroom-body plasticity: reward paired with odor A depresses A's Kenyon-cell synapses onto
reward-compartment MBONs and leaves odor B's alone (needs the connectome data)."""
import numpy as np

from companion_brain.data.prune import load_or_build
from companion_brain.sim.runner import BrainRunner
from conftest import CFG, needs_data


@needs_data
def test_reward_depresses_only_the_paired_odor():
    c = load_or_build(CFG, verbose=False)
    r = BrainRunner(c, CFG, seed=2)
    pl = r.plasticity
    assert pl is not None and pl.n_syn > 1000 and pl.n_kc > 500
    for _ in range(600):                      # settle
        r.step_chunk()
    # which Kenyon cells does each odor excite?
    def excited(group):
        found = {}
        r.probe({group: 120.0}, seconds=1.0, capture=lambda rr: found.update(e=pl.elig[pl.kc_idx].copy()))
        return set(pl.kc_idx[np.nonzero(found["e"] > 0.05)[0]].tolist())
    a, b = excited("ORN_fruit"), excited("ORN_lemon")
    assert len(a) > 50 and len(b) > 20
    # train: odor A + sugar + reward dopamine for 15 s
    for _ in range(1500):
        r.clear_drive(); r.drive("ORN_fruit", 120.0); r.drive("GRN_sugar", 150.0); r.drive("DAN_reward", 20.0); r.step_chunk()
    w = pl.net.data[pl.syn] / np.where(pl.w0 == 0, 1, pl.w0)
    rew = np.array([pl.kind_of[int(q)] == "reward" for q in pl.post])
    only_a = np.array([k in a and k not in b for k in pl.pre]) & rew
    only_b = np.array([k in b and k not in a for k in pl.pre]) & rew
    assert only_a.sum() > 20 and only_b.sum() > 20
    assert w[only_a].mean() < 0.6                      # paired odor's synapses depressed
    assert w[only_b].mean() > w[only_a].mean() + 0.3   # unpaired odor's synapses spared


@needs_data
def test_learning_can_be_switched_off():
    c = load_or_build(CFG, verbose=False)
    r = BrainRunner(c, CFG, seed=2)
    pl = r.plasticity
    pl.enabled = False
    for _ in range(600):
        r.step_chunk()
    for _ in range(800):
        r.clear_drive(); r.drive("ORN_fruit", 120.0); r.drive("DAN_reward", 20.0); r.step_chunk()
    assert np.allclose(pl.net.data[pl.syn], pl.w0)
    pl.reset()
    assert np.allclose(pl.net.data[pl.syn], pl.w0)
