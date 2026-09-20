"""The two demo controls: the lobotomy and the stop.

The lobotomy is the thing an audience watches, so it has to be a real lesion and it has to *look* like
something. Both halves are asserted here, because the first version of this failed the second half: the
visual pathway was cut but the search state gated on the retina, so a blind fly still "saw" someone,
never started searching, and just stood there. From the front row that is indistinguishable from a bug.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "demo_brain.pt"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="checkpoints/demo_brain.pt not built")

PERSON = [(120.0, 80.0, 200.0, 220.0)]     # someone standing a little right of centre
HEAT = (0.0, 0.0)


@pytest.fixture(scope="module")
def brain():
    from scripts.robot_bridge import Brain
    return Brain(str(CKPT), gain=0.05, device="cpu", engine="event", substeps=20, explore=True)


def drive(b, n=90, boxes=PERSON):
    """Run n frames at 30 Hz and report the steady-state behaviour, dropping the first second."""
    t, fwd, turn = time.time(), [], []
    for _ in range(n):
        t += 1 / 30
        f, w = b.step(boxes, HEAT, t)
        fwd.append(f); turn.append(w)
    fwd, turn = fwd[30:], turn[30:]
    flips = sum(1 for i in range(1, len(turn)) if (turn[i] > 0) != (turn[i - 1] > 0))
    return dict(forward=sum(fwd) / len(fwd), swing=max(turn) - min(turn), flips=flips)


def test_lobotomy_is_slow_and_searching(brain):
    seeing = drive(brain)
    assert brain.set_blind(True) is True
    blind = drive(brain)

    # it crawls
    assert abs(blind["forward"]) < 0.5 * abs(seeing["forward"]), (seeing, blind)
    # and it sweeps, which the intact fly locked onto a person does not do
    assert blind["flips"] > seeing["flips"]
    assert blind["swing"] > seeing["swing"]

    brain.set_blind(False)
    woken = drive(brain)
    assert woken["forward"] == pytest.approx(seeing["forward"], rel=0.35, abs=0.05)


def test_lobotomy_really_cuts_the_visual_pathway(brain):
    """The cells keep spiking; it is their output that goes. Assert on the synapses, not the rates,
    because asserting on the rates is what a demo script would do and it would be wrong."""
    lif = brain.lif
    eye = torch.cat([brain.groups[k] for k in brain.BLIND_GROUPS if k in brain.groups])
    out_edges = torch.isin(lif.pre_idx.cpu(), eye.cpu()).nonzero().flatten()
    assert out_edges.numel() > 0, "the visual groups have no outgoing synapses to cut"

    before = lif._w[out_edges.to(lif.device)].abs().sum().item()
    assert before > 0.0

    brain.set_blind(True)
    try:
        assert lif._w[out_edges.to(lif.device)].abs().sum().item() == 0.0, "the projection is still live"
        drive(brain, n=45)
        assert max(brain.rates()[k] for k in ("LC10a_L", "LC10a_R")) > 0.0, \
            "the cells should still be spiking: this is a severed projection, not a dead population"
    finally:
        brain.set_blind(False)
    assert lif._w[out_edges.to(lif.device)].abs().sum().item() == pytest.approx(before, rel=1e-6)


def test_set_blind_is_idempotent(brain):
    assert brain.set_blind(True) is True
    assert brain.set_blind(True) is True          # a second cut must not stack another lesion
    brain.set_blind(False)
    brain.set_blind(False)
    drive(brain, n=45)
    assert max(brain.rates()[k] for k in ("LC10a_L", "LC10a_R")) > 0.0


def test_stop_zeroes_the_wheels_but_not_the_brain(brain):
    brain.halted = True
    try:
        out = drive(brain, n=45)
        assert out["forward"] == 0.0 and out["swing"] == 0.0
        # the point of stop rather than silence: it is still thinking, so the feed stays alive
        assert sum(brain.rates().values()) > 0.0
    finally:
        brain.halted = False
    assert drive(brain)["forward"] != 0.0


def test_explore_gates_on_the_pathway_not_the_retina():
    """The bug this file exists for: presence must come from what the brain is told, not from the camera."""
    from brain.explore import Exploratory
    groups = {k: torch.tensor([i]) for i, k in enumerate(("DNa01_L", "DNa01_R", "DNa02_L", "DNa02_R"))}
    ex = Exploratory(groups, N=4, B=1, device="cpu", dt=0.1, onset_s=0.5)
    r = {"pres": torch.ones(1, 8)}                # a person filling the retina

    for _ in range(20):
        seeing = ex.inject(torch.zeros(1, 4), r)
    assert not bool(seeing[0]), "a fly that can see someone should not be searching"

    ex.reset()
    for _ in range(20):
        blind = ex.inject(torch.zeros(1, 4), r, blind=True)
    assert bool(blind[0]), "a fly with the pathway cut should be searching, whatever the retina says"


def test_the_viewer_gets_the_buttons():
    from scripts.viz_adapter import brighten
    ui = ROOT.parent / "hunting-fly" / "brain" / "companion_brain" / "ui" / "hunt_gpu.html"
    if not ui.exists():
        pytest.skip("Ducks's viewer is not checked out beside us")
    html = brighten(ui).decode()
    assert html.count("</body>") >= 1
    for needle in ("op-lobo", "op-stop", '{lobotomy:lobo}', '{halted:halt}', "read_only"):
        assert needle in html, f"the operator panel lost {needle!r}"
