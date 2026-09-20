"""Unit tests of the LIF engine on tiny hand-built networks (no connectome needed)."""
import numpy as np
import pytest

from companion_brain.sim.lif import HAVE_NUMBA, LIFNetwork, LIFParams

ENGINES = [False] + ([True] if HAVE_NUMBA else [])


@pytest.mark.parametrize("use_numba", ENGINES)
def test_silent_without_input(use_numba):
    net = LIFNetwork(3, np.array([0, 1]), np.array([1, 2]), np.array([5.0, 5.0]), use_numba=use_numba)
    assert net.run_ms(100).sum() == 0
    assert np.allclose(net.v, LIFParams().v_rest_mv)


@pytest.mark.parametrize("use_numba", ENGINES)
def test_driven_neuron_fires_at_rate(use_numba):
    net = LIFNetwork(1, np.array([], dtype=int), np.array([], dtype=int), np.array([]), seed=0, use_numba=use_numba)
    net.set_rate(np.array([0]), 100.0)
    counts = net.run_ms(5000)
    assert 400 < counts[0] < 600          # ~100 Hz for 5 s


@pytest.mark.parametrize("use_numba", ENGINES)
def test_strong_synapse_transmits_with_delay(use_numba):
    # 0 -> 1 with a huge excitatory weight; 1 -> 2 inhibitory so 2 never fires.
    net = LIFNetwork(3, np.array([0, 1]), np.array([1, 2]), np.array([40.0, -40.0]), seed=0, use_numba=use_numba)
    net.set_rate(np.array([0]), 200.0)
    total = np.zeros(3, dtype=int)
    first_post = None
    for step in range(5000):
        c = net.run(1)
        total += c
        if c[1] and first_post is None:
            first_post = step
    assert total[0] > 70 and total[1] > 30 and total[2] == 0
    assert first_post is not None and first_post >= net.delay_steps


@pytest.mark.parametrize("use_numba", ENGINES)
def test_weak_synapses_summate(use_numba):
    # 20 presynaptic neurons each with 1 mV onto neuron 20: alone too weak, together enough.
    pre = np.arange(20); post = np.full(20, 20)
    net = LIFNetwork(21, pre, post, np.full(20, 1.0), seed=2, use_numba=use_numba)
    net.set_rate(pre, 150.0)
    assert net.run_ms(500)[20] > 0
    net2 = LIFNetwork(21, pre, post, np.full(20, 1.0), seed=2, use_numba=use_numba)
    net2.set_rate(pre[:2], 150.0)
    assert net2.run_ms(500)[20] == 0


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_engines_agree_statistically():
    rng = np.random.default_rng(0)
    n, m = 200, 2000
    pre, post = rng.integers(0, n, m), rng.integers(0, n, m)
    w = rng.choice([-1.0, 1.0], m) * rng.integers(5, 30, m) * 0.275
    res = []
    for use_numba in (False, True):
        net = LIFNetwork(n, pre, post, w, seed=0, use_numba=use_numba)
        net.set_rate(np.arange(20), 150.0)
        res.append(net.run_ms(1000).sum())
    assert abs(res[0] - res[1]) < 0.25 * max(res)
