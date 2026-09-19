"""talk.py's Brain tools, offline: no ElevenLabs API, no live demo.

`last_frame()` reads talk.FRAMES (monkeypatched here to a temp file written in replay/FORMAT.md /
scripts/robot_bridge.py VizFeed shape). `Brain.press()` sends one hotkey byte over UDP to
127.0.0.1:<port>; the test binds its own socket on that port instead of talking to a running demo.
"""
from __future__ import annotations

import json
import os
import socket
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import talk  # noqa: E402


DEFAULT_RATES = {"LC10a_L": 10.0, "LC10a_R": 60.0, "DNa02_L": 5.0, "DNa02_R": 90.0,
                  "DNa01_L": 0.0, "DNa01_R": 0.0, "DNp09": 0.0, "GF": 0.0, "PAM": 0.0, "PPL1": 0.0}


def write_frame(path, **overrides):
    frame = {
        "t": 1.23, "forward": 0.4, "turn": -0.2,
        "pres": [0.0] * talk.N_COLS, "size": [0.0] * talk.N_COLS, "mot": [0.0] * talk.N_COLS,
        "rates": dict(DEFAULT_RATES), "exploring": False, "lobotomy": False,
    }
    frame.update(overrides)
    with open(path, "a") as f:
        f.write(json.dumps(frame) + "\n")
    return frame


def side_box(side, width=4, level=0.8):
    """A person filling only the leftmost or rightmost `width` retina columns."""
    pres = [0.0] * talk.N_COLS
    idx = range(width) if side == "left" else range(talk.N_COLS - width, talk.N_COLS)
    for i in idx:
        pres[i] = level
    return pres


@pytest.fixture
def frames_path(tmp_path, monkeypatch):
    p = tmp_path / "demo_live.jsonl"
    monkeypatch.setattr(talk, "FRAMES", str(p))
    return str(p)


# --------------------------------------------------------------------------------------------------
# what_do_you_see: bearing sign matches CLAUDE.md convention (+ = the fly's right)

def test_bearing_positive_for_right_side_box(frames_path):
    write_frame(frames_path, pres=side_box("right"))
    out = talk.Brain(port=0).what_do_you_see()
    assert out["person_in_view"] is True
    assert out["bearing_deg"] > 0
    assert out["side"] == "right"


def test_bearing_negative_for_left_side_box(frames_path):
    write_frame(frames_path, pres=side_box("left"))
    out = talk.Brain(port=0).what_do_you_see()
    assert out["bearing_deg"] < 0
    assert out["side"] == "left"


def test_bearing_symmetric_for_mirrored_boxes(frames_path):
    write_frame(frames_path, pres=side_box("left"))
    left = talk.Brain(port=0).what_do_you_see()
    os.remove(frames_path)
    write_frame(frames_path, pres=side_box("right"))
    right = talk.Brain(port=0).what_do_you_see()
    assert left["bearing_deg"] == pytest.approx(-right["bearing_deg"], abs=1e-6)


def test_no_person_in_view_when_presence_is_near_zero(frames_path):
    write_frame(frames_path, pres=[0.02] * talk.N_COLS)
    out = talk.Brain(port=0).what_do_you_see()
    assert out["person_in_view"] is False


def test_what_do_you_see_with_no_frame_file(monkeypatch, tmp_path):
    monkeypatch.setattr(talk, "FRAMES", str(tmp_path / "missing.jsonl"))
    out = talk.Brain(port=0).what_do_you_see()
    assert "no frames" in out["status"]


# --------------------------------------------------------------------------------------------------
# neural_state: the steering string

def test_neural_state_steering_right(frames_path):
    write_frame(frames_path)  # DNa02_R 90 > DNa02_L 5 + 10
    out = talk.Brain(port=0).neural_state()
    assert out["steering"] == "turning right"
    assert out["firing_hz"]["DNa02_R"] == 90.0
    assert out["firing_hz"]["DNa02_L"] == 5.0


def test_neural_state_steering_left(frames_path):
    write_frame(frames_path, rates={**DEFAULT_RATES, "DNa02_L": 90.0, "DNa02_R": 5.0})
    out = talk.Brain(port=0).neural_state()
    assert out["steering"] == "turning left"


def test_neural_state_not_turning_within_10hz(frames_path):
    write_frame(frames_path, rates={**DEFAULT_RATES, "DNa02_L": 40.0, "DNa02_R": 42.0})
    out = talk.Brain(port=0).neural_state()
    assert out["steering"] == "not turning"


def test_neural_state_reports_lobotomy_and_motor(frames_path):
    write_frame(frames_path, lobotomy=True, forward=0.7, turn=-0.5)
    out = talk.Brain(port=0).neural_state()
    assert out["learning_wiped"] is True
    assert out["motor_forward"] == 0.7
    assert out["motor_turn"] == -0.5


def test_neural_state_with_no_frame_file(monkeypatch, tmp_path):
    monkeypatch.setattr(talk, "FRAMES", str(tmp_path / "missing.jsonl"))
    out = talk.Brain(port=0).neural_state()
    assert "no frames" in out["status"]


# --------------------------------------------------------------------------------------------------
# operator tools: press the right hotkey over a UDP socket bound in the test

@pytest.fixture
def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    port = sock.getsockname()[1]
    yield sock, port
    sock.close()


def test_lesion_eye_sends_key_2(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    result = b.lesion({"target": "eye"})
    data, _ = sock.recvfrom(16)
    assert data == b"2"
    assert result["done"] == "lesion eye"
    assert b.lesioned["eye"] is True


def test_restore_eye_sends_key_3(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.lesion({"target": "eye"})
    sock.recvfrom(16)  # drain the lesion press
    b.restore({"target": "eye"})
    data, _ = sock.recvfrom(16)
    assert data == b"3"
    assert b.lesioned["eye"] is False


def test_lesion_learning_sends_key_4(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.lesion({"target": "learning"})
    data, _ = sock.recvfrom(16)
    assert data == b"4"


def test_restore_learning_sends_key_5(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.restore({"target": "learning"})
    data, _ = sock.recvfrom(16)
    assert data == b"5"


def test_lesion_defaults_to_eye_with_no_params(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.lesion(None)
    data, _ = sock.recvfrom(16)
    assert data == b"2"


def test_lesion_unknown_target_errors_without_sending(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    out = b.lesion({"target": "wings"})
    assert "error" in out
    sock.settimeout(0.2)
    with pytest.raises(socket.timeout):
        sock.recvfrom(16)  # nothing should have been sent for an invalid target


def test_shuffle_wiring_sends_key_6_and_toggles(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    out = b.shuffle_wiring(None)
    data, _ = sock.recvfrom(16)
    assert data == b"6"
    assert out["wiring_shuffled"] is True
    assert b.shuffled is True


def test_baseline_sends_key_1_and_clears_state(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.lesion({"target": "eye"})
    sock.recvfrom(16)
    b.shuffled = True
    b.baseline()
    data, _ = sock.recvfrom(16)
    assert data == b"1"
    assert b.lesioned == {"eye": False, "learning": False}
    assert b.shuffled is False


def test_dopamine_reward_and_punish_send_different_keys(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.dopamine({"kind": "reward"})
    data, _ = sock.recvfrom(16)
    assert data == b"r"
    b.dopamine({"kind": "punish"})
    data, _ = sock.recvfrom(16)
    assert data == b"p"


def test_recent_events_records_presses_most_recent_last(udp_listener):
    sock, port = udp_listener
    b = talk.Brain(port=port)
    b.lesion({"target": "eye"})
    sock.recvfrom(16)
    b.restore({"target": "eye"})
    sock.recvfrom(16)
    out = b.recent_events()
    whats = [e["what"] for e in out["events"]]
    assert whats == ["lesion eye", "restore eye"]
