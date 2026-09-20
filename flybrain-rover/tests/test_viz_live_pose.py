"""Live mode has no world pose, so viz_adapter dead-reckons one from the motor command. Synthetic frames only:
Source never opens its websocket outside run(), no GPU, no brain file."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts import viz_adapter as V

DT = 0.02


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def live_adapter(**kw):
    return V.Adapter(V.Source(None, "ws://never-opened"), viz_neurons=[], **kw)


def drive(ad, forward, turn, seconds, t0=0.0, boxes=()):
    """Frames at t0, t0+DT, ... covering `seconds` of motion (the frame at t0 only sets the clock)."""
    st = None
    for i in range(int(round(seconds / DT)) + 1):
        st = ad.state({"wheels_live": True, "t": t0 + i * DT, "forward": forward, "turn": turn, "boxes": list(boxes)})
    return st


def test_starts_at_the_origin_facing_up():
    st = live_adapter().state({"wheels_live": True, "t": 0.0, "forward": 1.0, "turn": 1.0})
    assert st["fly"][:3] == [0.0, 0.0, 0.0]          # his heading 0 = +z = our +y; first frame integrates nothing
    assert st["pose_estimated"] is True


def test_forward_for_one_second_moves_0_7_m_up():
    ad = live_adapter()
    st = drive(ad, 1.0, 0.0, 1.0)
    x, z, heading, _ = st["fly"]                      # his [x, z, heading, speed]: his x = our x, his z = our y
    assert x == pytest.approx(0.0, abs=1e-3)
    assert z == pytest.approx(0.7, abs=1e-2)
    assert heading == pytest.approx(0.0, abs=1e-6)
    assert ad.pose[0] == pytest.approx(0.0, abs=1e-9) and ad.pose[1] == pytest.approx(0.7, abs=1e-9)


def test_turn_right_for_one_second_is_minus_half_pi_yaw():
    ad = live_adapter()
    st0 = ad.state({"wheels_live": True, "t": 0.0, "forward": 0.0, "turn": 0.0})
    st = drive(ad, 0.0, 1.0, 1.0)
    assert ad.pose[2] == pytest.approx(0.0, abs=1e-6)                             # pi/2 - pi/2: clockwise
    assert wrap(st["fly"][2] - st0["fly"][2]) == pytest.approx(-math.pi / 2, abs=1e-4)  # his heading = yaw - pi/2: clockwise decreases it
    assert math.sin(st["fly"][2]) == pytest.approx(-1.0, abs=1e-4)                # his x = sin(heading): faces our +x
    assert st["fly"][:2] == [0.0, 0.0]


def test_turn_left_is_counter_clockwise():
    ad = live_adapter()
    drive(ad, 0.0, -1.0, 0.5)
    assert ad.pose[2] == pytest.approx(math.pi / 2 + math.pi / 4, abs=1e-6)


def test_backwards_clock_resets_the_pose():
    ad = live_adapter()
    drive(ad, 1.0, 0.3, 2.0)
    assert ad.pose[:2] != [0.0, 0.0]
    st = ad.state({"wheels_live": True, "t": 0.1, "forward": 1.0, "turn": 0.0})     # demo restarted: t went backwards
    assert st["fly"][:3] == [0.0, 0.0, 0.0]
    assert st["episode"] == 1
    st = ad.state({"wheels_live": True, "t": 0.12, "forward": 1.0, "turn": 0.0})    # and it walks again from there
    assert st["fly"][1] == pytest.approx(0.014, abs=1e-3)


def test_dt_is_clamped_to_100_ms():
    ad = live_adapter()
    ad.state({"wheels_live": True, "t": 0.0, "forward": 1.0, "turn": 0.0})
    st = ad.state({"wheels_live": True, "t": 5.0, "forward": 1.0, "turn": 0.0})     # stalled feed: at most 0.1 s of motion
    assert st["fly"][1] == pytest.approx(0.07, abs=1e-3)


@pytest.mark.parametrize("forward,turn", [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.05), (1.0, -0.02), (-1.0, 0.03)])
def test_pose_never_leaves_the_arena(forward, turn):
    ad = live_adapter()
    half = ad.arena_half
    touched = False
    for i in range(int(60.0 / DT)):                            # 42 m of travel on curves wider than the 12 m arena
        st = ad.state({"wheels_live": True, "t": i * DT, "forward": forward, "turn": turn})
        assert abs(st["fly"][0]) <= half + 1e-9 and abs(st["fly"][1]) <= half + 1e-9
        touched |= max(abs(ad.pose[0]), abs(ad.pose[1])) >= half - 1e-9
    assert touched


def test_people_are_placed_relative_to_the_estimated_pose():
    ad = live_adapter()
    drive(ad, 1.0, 0.0, 1.0)                                    # at (0, 0.7) facing +y
    box = [140.0, 20.0, 180.0, 240.0]                           # centred in the image: bearing 0
    az0, az1 = math.atan((140 - 160) / 138.1), math.atan((180 - 160) / 138.1)
    dist = min(6.0, 0.5 / (az1 - az0))
    st = ad.state({"wheels_live": True, "t": 1.02, "forward": 0.0, "turn": 0.0, "boxes": [box]})
    assert st["people"][0][:2] == pytest.approx([0.0, 0.7 + dist], abs=2e-3)
    drive(ad, 0.0, 1.0, 1.0, t0=1.02)                           # quarter turn right: same box now lies along our +x
    st = ad.state({"wheels_live": True, "t": 2.04, "forward": 0.0, "turn": 0.0, "boxes": [box]})
    assert st["people"][0][:2] == pytest.approx([-dist, 0.7], abs=2e-3)   # his x = -our x: the fly's right is his -x


def test_v_max_and_w_max_override():
    ad = live_adapter(v_max=1.0, w_max_deg=45.0)
    st = drive(ad, 1.0, 0.0, 1.0)
    assert st["fly"][1] == pytest.approx(1.0, abs=1e-2)
    drive(ad, 0.0, 1.0, 1.0, t0=1.0)
    assert ad.pose[2] == pytest.approx(math.pi / 2 - math.pi / 4, abs=1e-6)


def test_replay_mode_keeps_the_true_pose_and_no_estimate_flag(tmp_path):
    frames = [{"t": 0.0, "rxy": [1.0, -2.0], "ryaw": 0.3, "hxy": [[2.0, 2.0]], "forward": 1.0, "turn": 0.5},
              {"t": 0.02, "rxy": [1.5, -2.5], "ryaw": 0.4, "hxy": [[2.0, 2.0]], "forward": 1.0, "turn": 0.5}]
    p = tmp_path / "ep.json"
    p.write_text(json.dumps({"meta": {"arena_L": 6.0, "dt": 0.02, "seed": 7}, "frames": frames}))
    ad = V.Adapter(V.Source(str(p), None), viz_neurons=[])
    assert not ad.live and ad.pose is None
    for f in frames:
        st = ad.state(f)
        assert "pose_estimated" not in st
        assert st["fly"][:3] == [-f["rxy"][0], f["rxy"][1], round(wrap(f["ryaw"] - math.pi / 2), 4)]
        assert st["people"][0][:2] == [-2.0, 2.0]
    assert ad.pose is None


def test_live_dead_reckons_whether_or_not_the_wheels_are_live():
    """The viewer shows what the brain commands. A dry run still moves the fly, because the command is
    real even when the wheels are not; the state says pose_estimated so nobody mistakes it for odometry."""
    ad = live_adapter()
    for i in range(60):
        st = ad.state({"t": i * DT, "forward": 1.0, "turn": 0.0})
    assert st["fly"][1] > 0.5, "one second of full forward should have moved it"
    assert st["pose_estimated"] is True
