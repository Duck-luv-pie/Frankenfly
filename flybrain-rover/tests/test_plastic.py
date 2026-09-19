"""Lobotomize/restore: reset plastic synapses to the connectome, optionally lesion LC10a, undo exactly.
Tiny random graph, CPU. Also exercises the CLI end to end against a synthetic brain.npz."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

from brain.lif import LIF
from learn.plastic import plastic_edges
from brain.plastic import snapshot_plastic, lobotomize, restore

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


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


def _perturb(lif, eid):
    """Move the given edges away from w0 so a lobotomize actually has something to undo."""
    lif.set_w_subset(eid, lif.w0[eid] + torch.arange(1, eid.numel() + 1, dtype=torch.float32))


# ------------------------------------------------------------------ snapshot_plastic
def test_snapshot_plastic_default_is_every_edge():
    lif, groups = toy()
    snap = snapshot_plastic(lif)
    assert snap["eid"].numel() == lif.nnz
    assert torch.equal(snap["w"], lif.w.cpu())


def test_snapshot_plastic_subset_matches_current_weights():
    lif, groups = toy()
    eid = plastic_edges(lif, groups, "both")
    _perturb(lif, eid)
    snap = snapshot_plastic(lif, eid)
    assert torch.equal(snap["eid"].cpu(), eid.cpu())
    assert torch.equal(snap["w"], lif.w[eid].cpu())


# ------------------------------------------------------------------ lobotomize / restore
def test_lobotomize_resets_plastic_edges_to_w0():
    lif, groups = toy()
    eid = plastic_edges(lif, groups, "both")
    _perturb(lif, eid)
    assert not torch.equal(lif.w[eid], lif.w0[eid])   # sanity: perturbation took

    weights_before = lif.w.clone()
    snap = lobotomize(lif, eid=eid)
    assert torch.equal(lif.w[eid], lif.w0[eid])
    untouched = torch.ones(lif.nnz, dtype=torch.bool); untouched[eid] = False
    assert torch.equal(lif.w[untouched], weights_before[untouched]), "edges outside eid must not move"

    restore(lif, snap)
    assert torch.equal(lif.w, weights_before), "restore must be bit-identical"


def test_lobotomize_default_eid_resets_every_edge():
    lif, groups = toy()
    lif.set_w(lif.w0 + 1.0)
    weights_before = lif.w.clone()

    snap = lobotomize(lif)
    assert torch.equal(lif.w, lif.w0)

    restore(lif, snap)
    assert torch.equal(lif.w, weights_before)


def test_lobotomize_with_zero_lc10a_lesions_and_restores():
    lif, groups = toy()
    eid = plastic_edges(lif, groups, "both")
    _perturb(lif, eid)
    weights_before = lif.w.clone()

    lc10a = torch.cat([groups["LC10a_L"], groups["LC10a_R"]])
    snap = lobotomize(lif, eid=eid, zero_lc10a=lc10a)

    touch = torch.isin(lif.pre_idx, lc10a) | torch.isin(lif.post_idx, lc10a)
    assert touch.any(), "toy graph should have at least one LC10a edge"
    # plastic edges NOT touching LC10a reset to their connectome value; those that do are zeroed by the lesion
    eid_touch = touch[eid]
    assert torch.equal(lif.w[eid][~eid_touch], lif.w0[eid][~eid_touch])
    assert torch.equal(lif.w[eid][eid_touch], torch.zeros_like(lif.w[eid][eid_touch]))
    assert torch.equal(lif.w[touch], torch.zeros_like(lif.w[touch]))
    assert snap["lesion"] is not None and snap["lesion"]["n_neurons"] == lc10a.numel()

    restore(lif, snap)
    assert torch.equal(lif.w, weights_before), "restore must undo both the lesion and the weight reset"


def test_lobotomize_without_zero_lc10a_has_no_lesion_snapshot():
    lif, groups = toy()
    eid = plastic_edges(lif, groups, "kc_mbon")
    _perturb(lif, eid)
    snap = lobotomize(lif, eid=eid)
    assert snap["lesion"] is None


# ------------------------------------------------------------------ CLI
def _make_tiny_brain_npz(path):
    """Minimal synthetic brain.npz: N, W_indices/W_values, plus LC10a_L/LC10a_R groups."""
    N = 10
    post = np.array([4, 5, 6, 7, 8, 9, 4, 0, 2], dtype=np.int64)
    pre = np.array([0, 1, 2, 3, 4, 5, 1, 3, 1], dtype=np.int64)
    W_indices = np.stack([post, pre])
    sign = np.where(np.arange(len(post)) % 3 == 0, -1.0, 1.0).astype(np.float32)
    W_values = (np.arange(len(post), dtype=np.float32) + 1.0) * sign
    LC10a_L = np.array([0, 1], dtype=np.int64)
    LC10a_R = np.array([2, 3], dtype=np.int64)
    np.savez(path, N=np.int64(N), W_indices=W_indices, W_values=W_values,
              LC10a_L=LC10a_L, LC10a_R=LC10a_R)


@pytest.mark.skipif(not VENV_PY.exists(), reason=".venv not found at repo root")
def test_cli_lobotomize_and_restore_round_trip(tmp_path):
    brain_path = tmp_path / "brain.npz"
    _make_tiny_brain_npz(brain_path)
    d = np.load(brain_path)

    # Build the same edge ordering the CLI will build, to construct a checkpoint and to predict output.
    Wi = torch.as_tensor(d["W_indices"], dtype=torch.long)
    Wv = torch.as_tensor(d["W_values"], dtype=torch.float32)
    N = int(d["N"])
    lif = LIF(Wi, Wv, N, 1, device="cpu", engine="event")
    w0 = lif.w0.clone()
    learned_w = w0 + 1.0   # pretend training moved every synapse away from the raw connectome

    ck_path = tmp_path / "ck.pt"
    torch.save(dict(w=learned_w, gen=3, stage="B", gain=0.05, run="test",
                     k_t=0.02, k_f=0.01, learn="three_factor", amps=None), ck_path)

    lobo_path = tmp_path / "ck_lobotomized.pt"
    r = subprocess.run(
        [str(VENV_PY), "-m", "brain.plastic", "--lobotomize",
         "--checkpoint", str(ck_path), "--out", str(lobo_path),
         "--zero-lc10a", "--brain", str(brain_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "lobotomized" in r.stdout

    ck2 = torch.load(lobo_path, map_location="cpu", weights_only=True)
    assert ck2["lobotomized"] is True
    assert ck2["lobotomized_from"] == str(ck_path)
    assert Path(ck2["snapshot_path"]).exists()
    # untracked keys carried through unchanged
    assert ck2["gen"] == 3 and ck2["stage"] == "B" and ck2["run"] == "test"

    lc10a = torch.cat([torch.as_tensor(d["LC10a_L"], dtype=torch.long),
                        torch.as_tensor(d["LC10a_R"], dtype=torch.long)])
    touch = torch.isin(lif.pre_idx, lc10a) | torch.isin(lif.post_idx, lc10a)
    expected = w0.clone()
    expected[touch] = 0.0
    assert torch.equal(ck2["w"], expected)
    # input checkpoint itself must never be touched
    ck_orig = torch.load(ck_path, map_location="cpu", weights_only=True)
    assert torch.equal(ck_orig["w"], learned_w)

    r2 = subprocess.run(
        [str(VENV_PY), "-m", "brain.plastic", "--restore",
         "--checkpoint", str(lobo_path), "--brain", str(brain_path)],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert "restored" in r2.stdout

    restored_path = tmp_path / "ck_lobotomized_restored.pt"
    assert restored_path.exists()
    ck3 = torch.load(restored_path, map_location="cpu", weights_only=True)
    assert torch.equal(ck3["w"], learned_w), "restore must reproduce the original checkpoint's weights exactly"
    assert "lobotomized" not in ck3 and "snapshot_path" not in ck3


@pytest.mark.skipif(not VENV_PY.exists(), reason=".venv not found at repo root")
def test_cli_requires_exactly_one_mode(tmp_path):
    r = subprocess.run(
        [str(VENV_PY), "-m", "brain.plastic", "--checkpoint", "does_not_matter.pt"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert r.returncode != 0
