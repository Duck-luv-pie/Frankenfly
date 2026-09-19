"""The real fly's brain in the GPU arena (skipped without torch and the cached circuit): the hunter
subcircuit is cut from the connectome, the rate model is lateralized like the spiking brain, its learned
gains round-trip through hunter_brain.npz onto a spiking BrainRunner, and the trainer runs."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from companion_brain.config import load_config, apply_dotted  # noqa: E402
from companion_brain.data.prune import cache_path  # noqa: E402

CFG = load_config(overrides={"learning": {"enabled": False}, "hunt_gpu": {"humans": {"mode": "flee"}}})
apply_dotted(CFG, dict(CFG.hunt.get("overrides", {})))
pytestmark = pytest.mark.skipif(not cache_path(CFG, False).exists(), reason="needs the cached pruned circuit (companion prune)")


@pytest.fixture(scope="module")
def brain_and_circuit():
    from companion_brain.hunt_gpu.brain_train import build_brain
    hc, brain = build_brain(CFG, hops=(1, 1), device="cpu", verbose=False)
    return hc, brain


def test_the_hunter_subcircuit_is_cut_from_the_connectome(brain_and_circuit):
    hc, _ = brain_and_circuit
    c = hc.circuit
    assert 500 < c.n < 5000 and c.m > 5000 and hc.hops == (1, 1)
    for g in ("LC10a", "LC11", "TRN_hot", "DNa02", "DN_all"):
        assert len(c.groups[g]["all"]) > 0
    assert len(c.groups["DNa02"]["left"]) == 1 and len(c.groups["DNa02"]["right"]) == 1
    assert c.pre.max() < c.n and c.post.max() < c.n and len(c.root_ids) == c.n
    assert (c.weight_mv != 0).all()


def test_the_rate_model_steers_like_the_spiking_brain(brain_and_circuit):
    hc, brain = brain_and_circuit
    rest = brain.probe({})
    left = brain.probe({("LC10a", "left"): 150, ("LC11", "left"): 150})
    right = brain.probe({("LC10a", "right"): 150, ("LC11", "right"): 150})
    assert left[("DNa02", "left")] > 5 * max(rest[("DNa02", "left")], 1.0)              # a left object fires the left DNa02 ...
    assert left[("DNa02", "left")] > 5 * left[("DNa02", "right")]                        # ... and not the right one
    assert right[("DNa02", "right")] > 5 * right[("DNa02", "left")]
    assert 0.5 < rest[("DN_all", "all")] < 3.0                                            # the descending population rests near its 1 Hz floor
    # the decoder turns that into a turn drive toward the object
    arena_obs = torch.zeros(1, brain.spec.size)
    heat, vis, body = brain.spec.split(arena_obs)
    vis[0, 2, 0] = 1.0                                                                    # a strong object in a left column
    st = brain.initial_state(1)
    for _ in range(20):
        mean, st, _ = brain.step(arena_obs, st)
    assert float(mean[0, 1]) < -0.2                                                       # turn +ve = right: it turns left
    assert torch.isfinite(mean).all()


def test_learned_gains_round_trip_into_the_spiking_brain(brain_and_circuit, tmp_path):
    from companion_brain.hunt_gpu.connectome import load_brain, apply_to_runner
    from companion_brain.sim.runner import BrainRunner
    hc, brain = brain_and_circuit
    with torch.no_grad():
        brain.u_gain.normal_(0, 0.3)
        brain.u_th.normal_(0, 0.3)
    path = brain.save(tmp_path / "hunter_brain.npz", meta={"note": "test"})
    b = load_brain(path)
    assert b["meta"]["note"] == "test" and len(b["gain"]) == hc.m and len(b["th_mv"]) == hc.n
    assert (b["gain"] > 0.1).all() and (b["gain"] < 10).all() and (np.abs(b["th_mv"]) <= 4).all()
    r = BrainRunner(hc.circuit, CFG, seed=0)
    before = r.net.data.copy()
    stats = apply_to_runner(r, b, verbose=False)
    assert stats["synapses"] == hc.m and stats["neurons"] == hc.n                         # every synapse and neuron found by root id
    ratio = r.net.data / before
    assert np.allclose(np.sort(ratio), np.sort(b["gain"]), rtol=1e-4)                     # the gains landed on the right synapses
    assert np.allclose(np.sort(r.net.th_offset), np.sort(b["th_mv"]), atol=1e-5)
    # and load_params restores the rate model
    from companion_brain.hunt_gpu.connectome import RateBrain
    fresh = RateBrain(hc, CFG, {}, dict(CFG.hunt_gpu.brain))
    fresh.load_params(path)
    assert torch.allclose(fresh.gain(), brain.gain()) and torch.allclose(fresh.th_offset(), brain.th_offset())


def test_the_brain_trainer_runs_a_tiny_update(tmp_path):
    from companion_brain.hunt_gpu.brain_train import BrainTrainer
    cfg = load_config(overrides={"learning": {"enabled": False}, "hunt_gpu": {"humans": {"mode": "flee"}, "brain": {"hops": [1, 1], "ppo": {"steps": 6, "epochs": 1, "minibatches": 2}}}})
    apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))
    tr = BrainTrainer(cfg, n_envs=4, device="cpu", seed=1, out_dir=tmp_path, hops=(1, 1))
    g0 = tr.brain.gain().detach().clone()
    res = tr.train(updates=2, verbose=False, save_every=1)
    assert res["updates"] == 2 and (tmp_path / "hunter_brain_last.npz").exists()
    assert not torch.equal(g0, tr.brain.gain().detach())                                  # the synapses moved
    assert torch.isfinite(tr.brain.gain()).all() and torch.isfinite(tr.brain.th_offset()).all()


def test_the_spiking_hunter_drives_a_room(brain_and_circuit):
    from companion_brain.hunt_gpu.brain_train import SpikingHunter, evaluate_spiking
    hc, _ = brain_and_circuit
    cfg = load_config(overrides={"learning": {"enabled": False}, "hunt": {"episode_s": 2.0}, "hunt_gpu": {"humans": {"mode": "flee"}}, "decode": {"calibrate_s": 2.0, "warmup_s": 1.0}})
    apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))
    h = SpikingHunter(cfg, hc, None, verbose=False)
    res = evaluate_spiking(cfg, h, 1, seed=3, verbose=False)
    assert len(res) == 1 and "touched" in res[0] and h.spikes.sum() >= 0


def test_a_brain_file_loads_into_the_full_spiking_circuit(brain_and_circuit, tmp_path):
    from companion_brain.sim.hunt import run_hunt, load_hunter, is_brain_file
    from companion_brain.hunt_gpu.connectome import load_brain
    hc, brain = brain_and_circuit
    with torch.no_grad():
        brain.u_gain.normal_(0, 0.2)
    path = brain.save(tmp_path / "hunter_brain.npz")
    assert is_brain_file(path)
    _, gains, meta = load_hunter(path)
    assert meta["brain"] and "decode.motor.channels.turn.z_ref" in gains
    cfg = load_config(overrides={"learning": {"enabled": False}, "hunt": {"episode_s": 3.0, "punish_s": 0.5, "reward_s": 0.5}})
    apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))
    apply_dotted(cfg, gains)
    res = run_hunt(cfg, seed=3, episodes=1, learn=False, brain=load_brain(path))           # the whole pruned circuit, gains applied by root id
    assert len(res["episodes"]) == 1 and "touched" in res["episodes"][0]
