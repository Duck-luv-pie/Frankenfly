"""Arena geometry, events and reward rules (CPU, tiny batches)."""
import math
import torch
import pytest
from env.arena import Arena


def place(a, rxy, ryaw, hxy, hr=0.25):
    """Put one human per env at explicit poses (B envs)."""
    B = a.B
    a.rxy = torch.tensor(rxy, dtype=torch.float32).view(B, 2)
    a.ryaw = torch.tensor(ryaw, dtype=torch.float32).view(B)
    a.hxy = torch.zeros(B, a.K, 2); a.hxy[:, 0] = torch.tensor(hxy, dtype=torch.float32).view(B, 2)
    a.hr = torch.full((B, a.K), hr); a.hvalid = torch.zeros(B, a.K, dtype=torch.bool); a.hvalid[:, 0] = True
    a.hspeed = torch.zeros(B, a.K); a.htimer = torch.full((B, a.K), 1e9)   # humans stand still


def test_stage_a_spawns_in_view_at_1_to_3m():
    a = Arena(64, device="cpu"); s = a.reset(seed=3, stage="A")
    assert a.in_view().any(1).all()
    assert (s["nearest_dist"] >= 0.9).all() and (s["nearest_dist"] <= 3.1).all()
    assert (a.hvalid.sum(1) == 1).all()


def test_stage_b_spawns_out_of_view():
    a = Arena(64, device="cpu"); a.reset(seed=4, stage="B")
    assert a.in_view().any(1).float().mean() < 0.15   # wall clamping can drag a few into view


def test_front_contact_reward_and_events():
    a = Arena(1, device="cpu"); a.reset(seed=0, stage="A")
    place(a, [0.0, 0.0], [0.0], [1.0, 0.0])
    a.vmax[:] = 1.0
    got_first, total = False, 0.0
    for _ in range(100):
        _, ev = a.step(torch.ones(1), torch.zeros(1))
        r = a.reward(ev)
        total += r.item()
        if ev["front_contact"].item():
            got_first = True
            assert r.item() >= 5.0
            assert not ev["back_or_side_contact"].item()
    assert got_first and a.contacted.all()
    assert ev["sustained"].item() and ev["contact_now"].item()
    # rover never pushes through the human
    assert (a.hxy[0, 0, 0] - a.rxy[0, 0]) > a.hl + a.hr[0, 0] - 0.05


def test_back_contact_is_not_rewarded():
    a = Arena(1, device="cpu"); a.reset(seed=0, stage="A")
    place(a, [0.0, 0.0], [0.0], [-(0.16 + 0.25 + 0.02), 0.0])
    _, ev = a.step(torch.zeros(1), torch.zeros(1))
    assert ev["back_or_side_contact"].item() and not ev["front_contact"].item()
    assert a.reward(ev).item() == pytest.approx(0.0 + (1.0 if ev["acquire"].item() else 0.0))


def test_side_contact_is_not_front():
    a = Arena(1, device="cpu"); a.reset(seed=0, stage="A")
    place(a, [0.0, 0.0], [0.0], [0.0, 0.085 + 0.25 + 0.02])
    _, ev = a.step(torch.zeros(1), torch.zeros(1))
    assert ev["back_or_side_contact"].item() and not ev["front_contact"].item()


def test_no_contact_penalty_and_timeout():
    a = Arena(2, device="cpu", episode_s=0.1); a.reset(seed=0, stage="A")
    place(a, [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], [[-3.0, 0.0], [-3.0, 0.0]])   # behind, out of view
    rs = []
    for _ in range(5):
        _, ev = a.step(torch.zeros(2), torch.zeros(2)); rs.append(a.reward(ev))
    assert all(abs(r[0].item() + 0.02) < 1e-6 for r in rs[:-1])
    assert ev["timeout"].all() and rs[-1][0].item() == pytest.approx(-3.02)


def test_walls_contain_rover_and_humans():
    a = Arena(16, device="cpu"); a.reset(seed=5, stage="C")
    for _ in range(600):
        a.step(torch.ones(16), torch.zeros(16))
    half = (a.L / 2).view(16, 1)
    assert (a.rxy.abs() <= half).all()
    assert (a.hxy.abs()[a.hvalid] <= half.expand(16, a.K).unsqueeze(-1).expand(16, a.K, 2)[a.hvalid] + 1e-4).all()


def test_heat_lateralized():
    a = Arena(2, device="cpu"); a.reset(seed=0, stage="A")
    a.heat_fp[:] = 0.0
    # env0: human dead ahead 3 m; env1: human 70 deg to the LEFT (bearing negative -> world angle +70)
    place(a, [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0],
          [[3.0, 0.0], [3 * math.cos(math.radians(70)), 3 * math.sin(math.radians(70))]])
    for _ in range(60):
        h = a.heat()
    assert h[0, 0] > 0.3 and h[0, 1] > 0.3
    assert h[1, 0] > 0.3 and h[1, 1] < 1e-3


def test_retina_noise_keeps_range():
    a = Arena(8, device="cpu"); a.reset(seed=0, stage="A")
    r = {"pres": torch.rand(8, 24), "size": torch.rand(8, 24)}
    a.perturb_retina(r)
    assert (r["pres"] >= 0).all() and (r["pres"] <= 1).all() and (r["size"] <= 1).all()
    assert 0.8 <= a.gain.min() <= a.gain.max() <= 1.2


def test_positive_turn_steers_toward_human_on_the_right():
    a = Arena(1, device="cpu"); a.reset(seed=0, stage="A")
    place(a, [0.0, 0.0], [0.0], [2.0, -1.0])        # human ahead-right (world -y is to the right of +x heading)
    _, b0 = a.nearest()
    for _ in range(10):
        a.step(torch.zeros(1), torch.ones(1))       # turn = +1 -> right
    _, b1 = a.nearest()
    assert b1 < b0, "positive turn must reduce the bearing of a human on the right"
