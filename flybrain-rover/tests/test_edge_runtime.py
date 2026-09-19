"""The numpy-only runtime must stay bit-identical to brain/lif.py, or it is not the same brain."""
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))
from brain.lif import LIF
from flybrain_mini import FlyBrain


def tiny_brain(tmp_path, N=180, nnz=1400, seed=0):
    """A small connectome-shaped file with the group keys the runtime expects."""
    g = np.random.default_rng(seed)
    post = g.integers(0, N, nnz); pre = g.integers(0, N, nnz)
    w = (g.integers(1, 40, nnz) * np.where(g.random(nnz) < 0.3, -1.0, 1.0)).astype(np.float32)
    d = dict(N=np.int64(N), W_indices=np.stack([post, pre]).astype(np.int64), W_values=w,
             types=np.array(["x"] * N, dtype="U"), ids=np.arange(N, dtype=np.int64))
    k = 0
    for name in ("LC10a_L", "LC10a_R", "LC11_L", "LC11_R", "LC12_L", "LC12_R", "LC15_L", "LC15_R",
                 "LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "THERMO_L", "THERMO_R",
                 "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF", "DN_ALL"):
        d[name] = np.arange(k, k + 6, dtype=np.int64) % N
        k += 6
    p = str(tmp_path / "tiny.npz")
    np.savez(p, **d)
    return p, N


def test_numpy_runtime_matches_pytorch_spike_for_spike(tmp_path):
    path, N = tiny_brain(tmp_path)
    ref = LIF(*[torch.as_tensor(np.load(path)[k]) for k in ("W_indices", "W_values")], N, 1,
              device="cpu", engine="event", g=0.05)
    mini = FlyBrain(path)
    rng = np.random.default_rng(1)
    I = (rng.random(N) * 1.5).astype(np.float32)
    It = torch.as_tensor(I).unsqueeze(0)
    fired = 0
    for _ in range(250):
        a = ref.step(It)[0].numpy()
        b = mini.lif_step(I)
        assert np.array_equal(a, b)
        fired += int(b.sum())
    assert fired > 0, "the test brain never spiked, so it proves nothing"
    assert abs(float(ref.rates(torch.as_tensor(np.load(path)["DNa02_L"]))) - mini.rates("DNa02_L")) < 1e-4


def test_numpy_runtime_produces_lateralized_commands(tmp_path):
    path, _ = tiny_brain(tmp_path)
    fly = FlyBrain(path)
    left = [(40.0, 30.0, 110.0, 239.0)]
    for _ in range(20):
        fwd, turn = fly.step(left)
    assert -1.0 <= turn <= 1.0 and 0.0 <= fwd <= 1.0
    assert fly.rates("LC10a_L") >= 0.0
