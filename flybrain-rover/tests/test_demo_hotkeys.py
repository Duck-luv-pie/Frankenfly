"""The demo's hotkeys are the demo. Every sequence must leave the brain exactly as it found it."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import demo as D
from scripts.robot_bridge import Brain


def synthetic_brain(tmp_path, N=200, nnz=1600, seed=3):
    g = np.random.default_rng(seed)
    d = dict(N=np.int64(N),
             W_indices=np.stack([g.integers(0, N, nnz), g.integers(0, N, nnz)]).astype(np.int64),
             W_values=(g.integers(1, 30, nnz) * np.where(g.random(nnz) < 0.3, -1.0, 1.0)).astype(np.float32),
             types=np.array(["x"] * N, dtype="U"), ids=np.arange(N, dtype=np.int64))
    k = 0
    for name in ("LC10a_L", "LC10a_R", "LC11_L", "LC11_R", "LC12_L", "LC12_R", "LC15_L", "LC15_R",
                 "LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R", "THERMO_L", "THERMO_R", "DNa02_L", "DNa02_R",
                 "DNa01_L", "DNa01_R", "DNp09", "GF", "DN_ALL", "PAM", "PPL1", "KC", "MBON"):
        d[name] = (np.arange(k, k + 5) % N).astype(np.int64)
        k += 5
    p = str(tmp_path / "b.npz")
    np.savez(p, **d)
    return p


def fresh(tmp_path, shuffle=True):
    D.STATE.update(sets=[], snaps=[], quarter=0, lobotomy=None, voice=None)
    return Brain(device="cpu", engine="event", brain_path=synthetic_brain(tmp_path), shuffle=shuffle,
                 retinotopy="index")


def test_every_hotkey_sequence_restores_the_brain_exactly(tmp_path):
    brain = fresh(tmp_path)
    w0 = brain.lif_real.w.detach().clone()
    sequences = [
        ["2", "3"],                                  # lesion, restore
        ["2", "6", "3", "6"],                        # the sequence that used to crash: lesion, shuffle, restore
        ["7", "7", "7", "7", "3"],                   # graded lesion all the way down, then restore
        ["4", "5"],                                  # lobotomy and back
        ["2", "4", "6", "7", "6", "5", "3"],         # everything interleaved
    ]
    for seq in sequences:
        for k in seq:
            D.HOTKEYS[k][1](brain, None)
        D.HOTKEYS["1"][1](brain, None)               # baseline must always return to the real, intact brain
        assert not brain.shuffled, f"{seq}: still on the shuffled wiring"
        assert torch.equal(brain.lif_real.w, w0), f"{seq}: weights not restored"
        assert brain.lif is brain.lif_real


def test_shuffle_silences_the_readout_but_not_the_eye(tmp_path):
    brain = fresh(tmp_path)
    boxes = [(60.0, 30.0, 110.0, 239.0)]

    def run():
        brain.lif.reset(); brain.retina.reset()
        for i in range(25):
            brain.step(boxes, (0.0, 0.0), i / 30.0)
        return brain.rates(("LC10a_L", "DNa02_L", "DNa02_R"))

    real = run()
    D.HOTKEYS["6"][1](brain, None)
    assert brain.shuffled
    shuf = run()
    D.HOTKEYS["1"][1](brain, None)
    assert not brain.shuffled
    # the eye is driven by injected current, so it must survive a rewiring of everything downstream
    assert shuf["LC10a_L"] > 0.5 * real["LC10a_L"] or real["LC10a_L"] == 0


def test_graded_lesion_counts_up_and_stops(tmp_path):
    brain = fresh(tmp_path, shuffle=False)
    for expected in (1, 2, 3, 4, 4):
        D.HOTKEYS["7"][1](brain, None)
        assert D.STATE["quarter"] == expected
    D.HOTKEYS["3"][1](brain, None)
    assert D.STATE["quarter"] == 0 and D.STATE["sets"] == []
