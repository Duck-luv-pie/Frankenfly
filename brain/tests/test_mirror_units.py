"""Unit tests for the sim-to-robot Mirror: representative and boundary drives, yaw-sign inversion,
and the send-failure / no-port paths (Task 2.3). A fake serial stands in for the S1 so no hardware
port is opened. Covers Requirements 4.5, 4.6, 4.7, 5.1, 5.4."""
import pytest

from companion_brain.body.s1 import S1Body, decode, stick, CENTER, LOW, HIGH
from companion_brain.hunt_gpu.mirror import Mirror


class FakeSerial:
    """A serial stub: it records the frames written and never touches a real port."""

    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, b):
        self.frames.append(bytes(b))

    def close(self):
        self.closed = True


def _body(**cfg):
    """An S1Body on a fake serial, thread stopped, so tests read frames deterministically."""
    base = {"timeout_s": 5.0}          # a long failsafe window so frames don't go stale mid-test
    base.update(cfg)
    return S1Body(base, ser=FakeSerial(), start=False)


# --- Mirror.sticks: the pure drive -> stick mapping (R5.1, calibrated constants) ---

def test_neutral_drive_maps_to_centre():
    m = Mirror(_body())
    assert m.sticks(0.0, 0.0) == {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}
    ch = m.channels(0.0, 0.0)
    assert ch == {"forward": CENTER, "strafe": CENTER, "yaw": CENTER}


def test_full_forward_uses_the_calibrated_stick_forward():
    m = Mirror(_body())                                  # stick_forward = 0.5 (DEFAULTS): full drive = half throw
    assert m.sticks(1.0, 0.0)["forward"] == pytest.approx(0.5)
    assert m.channels(1.0, 0.0)["forward"] == stick(0.5)


def test_full_reverse_mirrors_forward():
    m = Mirror(_body())
    assert m.sticks(-1.0, 0.0)["forward"] == pytest.approx(-0.5)
    assert m.channels(-1.0, 0.0)["forward"] == stick(-0.5)


def test_full_left_and_right_turn_saturate_the_yaw_stick():
    m = Mirror(_body())                                  # stick_yaw = 1.0 (DEFAULTS): full turn = full throw
    assert m.sticks(0.0, 1.0)["yaw"] == pytest.approx(1.0)
    assert m.sticks(0.0, -1.0)["yaw"] == pytest.approx(-1.0)
    assert m.channels(0.0, 1.0)["yaw"] == HIGH
    assert m.channels(0.0, -1.0)["yaw"] == LOW


def test_combined_forward_and_turn():
    m = Mirror(_body())
    s = m.sticks(1.0, 0.5)
    assert s["forward"] == pytest.approx(0.5) and s["yaw"] == pytest.approx(0.5)


def test_out_of_range_drive_clamps_to_the_boundary():
    m = Mirror(_body())
    assert m.sticks(5.0, -9.0) == m.sticks(1.0, -1.0)    # R4.4: nearest boundary before mapping


def test_sign_yaw_inverted_reverses_the_yaw_stick():
    normal = Mirror(_body(), sign_yaw="normal")
    inverted = Mirror(_body(), sign_yaw="inverted")
    assert inverted.sticks(0.0, 1.0)["yaw"] == pytest.approx(-normal.sticks(0.0, 1.0)["yaw"])
    assert inverted.sticks(0.0, -1.0)["yaw"] == pytest.approx(-normal.sticks(0.0, -1.0)["yaw"])
    assert inverted.channels(0.0, 1.0)["yaw"] == LOW     # right turn inverted -> left stick


# --- Mirror.send through S1Body on a fake serial: the frame the robot would receive ---

def test_send_drives_the_fake_s1():
    body = _body()
    m = Mirror(body)
    assert m.send(1.0, 1.0) is True
    ch = decode(body.frame())
    assert ch[1] == stick(0.5)      # forward channel: full drive -> calibrated 0.5 throw
    assert ch[3] == HIGH            # yaw channel: full right turn


def test_centered_send_ignores_the_drive():
    body = _body()
    m = Mirror(body)
    assert m.send(1.0, 1.0, centered=True) is True
    ch = decode(body.frame())
    assert ch[1] == CENTER and ch[3] == CENTER          # E-stop / paused / failsafe: neutral centre


# --- R4.7: no S1 port configured -> the Mirror sends nothing but the caller keeps running ---

def test_no_body_sends_nothing():
    m = Mirror(None)
    assert m.send(1.0, 1.0) is False
    assert m.send(0.0, 0.0, centered=True) is False


# --- R4.6: a send failure is surfaced (returns False) without propagating, so the sim keeps running ---

def test_send_failure_is_swallowed():
    class RaisingBody:
        cfg = None
        def send(self, packet):
            raise IOError("S-Bus write failed")

    m = Mirror(RaisingBody())
    result = m.send(1.0, 0.5)                            # must not raise
    assert result is False
