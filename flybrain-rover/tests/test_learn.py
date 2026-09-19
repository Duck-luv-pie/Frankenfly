"""Learning rules: mask, Dale's law, bounds, and ES hook equivalence. Tiny random graph, CPU."""
import torch
import pytest
from brain.lif import LIF
from learn.plastic import plastic_edges, check_dale
from learn.three_factor import ThreeFactor
from learn.es import ES


def toy(B=4, N=40, nnz=300, seed=0):
    g = torch.Generator().manual_seed(seed)
    post = torch.randint(0, N, (nnz,), generator=g); pre = torch.randint(0, N, (nnz,), generator=g)
    vals = torch.randint(1, 6, (nnz,), generator=g).float() * torch.where(torch.rand(nnz, generator=g) < 0.3, -1.0, 1.0)
    lif = LIF(torch.stack([post, pre]), vals, N, B, device="cpu", engine="dense", g=1.0)
    groups = dict(KC=torch.arange(0, 10), MBON=torch.arange(10, 15), PAM=torch.arange(15, 18), PPL1=torch.arange(18, 20),
                  LC10a_L=torch.arange(20, 25), LC10a_R=torch.arange(25, 30),
                  DNa02_L=torch.tensor([30]), DNa02_R=torch.tensor([31]), DNa01_L=torch.tensor([32]),
                  DNa01_R=torch.tensor([33]), DNp09=torch.tensor([34]), GF=torch.tensor([35]))
    return lif, groups


def test_masks_are_subsets_of_edges():
    lif, groups = toy()
    for which in ("kc_mbon", "dn_in", "lc_dn", "both"):
        eid = plastic_edges(lif, groups, which)
        assert eid.numel() > 0 and eid.max() < lif.nnz
    kc = plastic_edges(lif, groups, "kc_mbon")
    assert torch.isin(lif.pre_idx[kc], groups["KC"]).all() and torch.isin(lif.post_idx[kc], groups["MBON"]).all()


def test_three_factor_keeps_dale_and_bounds():
    lif, groups = toy()
    tf = ThreeFactor(lif, groups, plastic="both", eta=0.5, bound_mult=3.0)
    I = torch.rand(lif.B, lif.N) * 2.0
    for t in range(200):
        lif.step(I)
        tf.on_substep(lif)
        if t % 20 == 19:
            tf.update(lif, None)
    assert tf.stats()["n_updates"] > 0 and tf.stats()["dw_abs"] > 0
    assert check_dale(lif, tf.eid)
    assert (lif.w[tf.eid].abs() <= 3.0 * lif.w0[tf.eid].abs() + 1e-6).all()
    untouched = torch.ones(lif.nnz, dtype=torch.bool); untouched[tf.eid] = False
    assert torch.equal(lif.w[untouched], lif.w0[untouched]), "non-plastic edges must never change"


def test_three_factor_sign_of_dopamine():
    lif, groups = toy()
    tf = ThreeFactor(lif, groups, plastic="kc_mbon", eta=1.0)
    I = torch.zeros(lif.B, lif.N); I[:, groups["KC"]] = 1.0; I[:, groups["MBON"]] = 0.6
    for _ in range(100):
        lif.step(I); tf.on_substep(lif)
    w_before = lif.w[tf.eid].abs().clone()
    I[:, groups["PAM"]] = 3.0                      # reward -> PAM fires -> D > 0
    for _ in range(60):
        lif.step(I); tf.on_substep(lif)
    e = tf.e.clone()
    tf.update(lif, None)
    dw = lif.w[tf.eid].abs() - w_before
    assert tf.stats()["dopamine"] > 0
    moved = dw.abs() > 1e-9
    assert moved.any()
    assert (torch.sign(dw[moved]) == torch.sign(e[moved])).float().mean() > 0.9


def test_es_hook_reproduces_original_brain_at_sigma_zero():
    lif_ref, groups = toy(seed=1)
    lif_es, _ = toy(seed=1)
    es = ES(lif_es, groups, plastic="lc_dn", sigma=0.0)
    I = torch.rand(lif_ref.B, lif_ref.N) * 1.5
    for _ in range(150):
        a, b = lif_ref.step(I), lif_es.step(I)
        assert torch.equal(a, b)


def test_es_update_moves_toward_top_k_and_respects_bounds():
    lif, groups = toy(seed=2)
    es = ES(lif, groups, plastic="dn_in", sigma=0.5, top_frac=0.5, bound_mult=2.0)
    before = es.mag.clone()
    ret = torch.tensor([0.0, 1.0, 5.0, -1.0])          # env 2 and 1 are the top half
    es.end_episode(lif, ret)
    target = es.mag_b[[2, 1]].mean(0).clamp(max=2.0 * lif.w0[es.eid].abs())
    assert torch.allclose(es.mag, target)
    assert (es.mag >= 0).all() and (es.mag <= 2.0 * lif.w0[es.eid].abs() + 1e-6).all()
    assert es.stats()["dw_abs"] > 0 and not torch.equal(before, es.mag)
    w = es.full_w(lif)
    assert (w[es.eid] * lif.w0[es.eid] >= 0).all()      # Dale


def test_synaptic_scaling_shrinks_hot_neuron_inputs_and_keeps_dale():
    lif, groups = toy()
    tf = ThreeFactor(lif, groups, plastic="both", eta=0.0, r_target=50.0, eta_h=0.1)   # no dopamine term, scaling only
    I = torch.zeros(lif.B, lif.N); I[:, :] = 3.0                                          # drive everything hard
    for _ in range(100):
        lif.step(I)
    before = lif.w[tf.eid].abs().clone()
    tf.update(lif, None)
    after = lif.w[tf.eid].abs()
    assert (after <= before + 1e-6).all() and (after < before).any(), "hot post neurons must shrink incoming plastic synapses"
    assert check_dale(lif, tf.eid)
    assert tf.stats()["n_scaled"] == 1


def test_sparse_eligibility_matches_dense_reference():
    import copy
    lif_a, groups = toy(seed=3)
    lif_b, _ = toy(seed=3)
    a = ThreeFactor(lif_a, groups, plastic="both", eta=0.0)
    b = ThreeFactor(lif_b, groups, plastic="both", eta=0.0)
    b.sparse_update = False
    I = torch.rand(lif_a.B, lif_a.N) * 2.0
    for _ in range(120):
        sa, sb = lif_a.step(I), lif_b.step(I)
        assert torch.equal(sa, sb)
        a.on_substep(lif_a); b.on_substep(lif_b)
    assert a.e.abs().sum() > 0
    assert torch.allclose(a.e, b.e, atol=1e-5, rtol=1e-4), (a.e - b.e).abs().max()
    assert torch.allclose(a.pre_tr, b.pre_tr) and torch.allclose(a.post_tr, b.post_tr)
