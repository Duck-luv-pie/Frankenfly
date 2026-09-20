"""LIF engine tests on a tiny hand-made graph (no brain.npz needed)."""
import torch
import pytest
from brain.lif import LIF, pick_device


def chain_graph():
    # 0 -> 1 (exc, 40 synapses), 1 -> 2 (exc, 40), 3 -> 2 (inh, -40)
    W_indices = torch.tensor([[1, 2, 2], [0, 1, 3]])  # [post, pre]
    W_values = torch.tensor([40.0, 40.0, -40.0])
    return W_indices, W_values, 4


@pytest.mark.parametrize("engine", ["sparse", "dense", "event"])
def test_chain_propagates(engine):
    Wi, Wv, N = chain_graph()
    lif = LIF(Wi, Wv, N, B=2, device="cpu", engine=engine, g=1.0)
    I = torch.zeros(2, N)
    I[:, 0] = 1.0
    for _ in range(300):
        lif.step(I)
    r = lif.rate[0]
    assert r[0] > 50, "driven neuron should fire"
    assert r[1] > 0, "downstream neuron should be recruited"
    assert r[2] > 0
    assert r[3] == 0, "unconnected, undriven neuron stays silent"


def test_inhibition_reduces_rate():
    Wi, Wv, N = chain_graph()
    lif = LIF(Wi, Wv, N, B=1, device="cpu", engine="sparse", g=1.0)
    I = torch.zeros(1, N); I[:, 0] = 1.0
    for _ in range(300):
        lif.step(I)
    r_no_inh = lif.rate[0, 2].item()
    lif.reset()
    I[:, 3] = 1.0  # now also drive the inhibitory neuron
    for _ in range(300):
        lif.step(I)
    assert lif.rate[0, 2].item() < r_no_inh


def test_engines_agree():
    torch.manual_seed(0)
    N, nnz = 60, 400
    post = torch.randint(0, N, (nnz,)); pre = torch.randint(0, N, (nnz,))
    Wi = torch.stack([post, pre]); Wv = torch.randint(1, 5, (nnz,)).float() * torch.sign(torch.randn(nnz))
    a = LIF(Wi, Wv, N, B=3, device="cpu", engine="sparse", g=0.5)
    b = LIF(Wi, Wv, N, B=3, device="cpu", engine="dense", g=0.5)
    c = LIF(Wi, Wv, N, B=3, device="cpu", engine="event", g=0.5)
    I = torch.rand(3, N) * 0.6
    for _ in range(100):
        sa, sb, sc = a.step(I), b.step(I), c.step(I)
        assert torch.equal(sa, sb)
        assert torch.equal(sa, sc)


def test_set_w_subset_updates_operator():
    Wi, Wv, N = chain_graph()
    for engine in ("sparse", "dense", "event"):
        lif = LIF(Wi, Wv, N, B=1, device="cpu", engine=engine, g=1.0)
        e = torch.tensor([0])  # edge 0 (sorted by post,pre): post 1 pre 0
        lif.set_w_subset(e, torch.tensor([0.0]))
        I = torch.zeros(1, N); I[:, 0] = 1.0
        for _ in range(200):
            lif.step(I)
        assert lif.rate[0, 1] == 0, f"{engine}: cut synapse should silence neuron 1"


def test_refractory_caps_rate():
    Wi, Wv, N = chain_graph()
    lif = LIF(Wi, Wv, N, B=1, device="cpu", g=1.0)
    I = torch.zeros(1, N); I[:, 0] = 50.0  # absurd drive
    for _ in range(500):
        lif.step(I)
    assert lif.rate[0, 0] <= 1000 / (lif.ref_steps + 1) + 1


def test_pick_device_returns_device():
    assert isinstance(pick_device(None), torch.device)
    assert pick_device("cpu").type == "cpu"


def test_lesion_and_restore_are_exact():
    Wi, Wv, N = chain_graph()
    for engine in ("sparse", "dense", "event"):
        lif = LIF(Wi, Wv, N, B=1, device="cpu", engine=engine, g=1.0)
        w0 = lif.w.clone()
        snap = lif.lesion(torch.tensor([0]), mode="out")      # cut 0 -> 1
        I = torch.zeros(1, N); I[:, 0] = 1.0
        for _ in range(200):
            lif.step(I)
        assert lif.rate[0, 1] == 0, f"{engine}: lesioned neuron 0 must not drive neuron 1"
        lif.restore(snap)
        assert torch.equal(lif.w, w0)
        lif.reset()
        for _ in range(200):
            lif.step(I)
        assert lif.rate[0, 1] > 0, f"{engine}: restore must bring the pathway back"
