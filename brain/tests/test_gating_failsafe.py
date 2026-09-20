"""Gating and failsafe for the sim-to-robot mirror (sim-mirror-webapp).

Property 3 (E-stop / paused / failsafe dominance): whenever the operator's E-stop is active, the sim
is paused, or the per-tick feed has timed out, the channel frame the S1 would receive is fully centred
(neutral on every axis) regardless of the fly's drive. Plus unit tests for the failsafe gap, the
out-of-range rejection that holds the last neutral-safe state, and E-stop release resuming the drive.

Validates: Requirements 6.2, 7.1, 7.3, 7.4.
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from companion_brain.body.s1 import CENTER, S1Body, decode
from companion_brain.hunt_gpu.mirror import Mirror


class FakeSerial:
    """An in-process stand-in for the S-Bus port so no real hardware is opened."""

    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, b):
        self.frames.append(bytes(b))

    def close(self):
        self.closed = True


def _body(**cfg):
    base = {"stick_forward": 1.0, "stick_yaw": 1.0, "timeout_s": 0.2, "rotation_only": False}
    return S1Body({**base, **cfg}, ser=FakeSerial(), start=False)


def _is_centered(frame: bytes) -> bool:
    """The strafe, forward and yaw axes all sit at the neutral centre."""
    ch = decode(frame)
    return ch[0] == CENTER and ch[1] == CENTER and ch[3] == CENTER


# a drive component from the whole real line, including out-of-range magnitudes, NaN and +/-inf
drive_component = st.floats(allow_nan=True, allow_infinity=True, width=32)


# Feature: sim-mirror-webapp, Property 3
@settings(max_examples=200)
@given(forward=drive_component, turn=drive_component, condition=st.sampled_from(["estop", "paused", "failsafe"]))
def test_property_estop_paused_failsafe_dominance(forward, turn, condition):
    """For any drive, an E-stop, a pause, or a feed timeout centres every channel sent to S1Body.

    Validates: Requirements 6.2, 7.1, 7.3.
    """
    body = _body()
    mirror = Mirror(body)
    if condition == "failsafe":
        # a real per-tick command lands, then the feed stops: past the timeout the sender centres (R7.1)
        mirror.send(forward, turn, centered=False)
        stale = body.last_packet + float(body.cfg["timeout_s"]) + 0.05
        assert _is_centered(body.frame(now=stale))
    else:
        # E-stop (R6.2) and paused (R7.3) both drive the mirror's centred path regardless of the drive
        mirror.send(forward, turn, centered=True)
        assert _is_centered(body.frame(now=body.last_packet))


def test_failsafe_centres_after_a_send_gap_longer_than_timeout():
    """A gap longer than timeout_s makes the background sender emit centred channels (R7.1)."""
    body = _body(timeout_s=0.2)
    body.send({"motor": {"forward": 1.0, "turn": 1.0}})            # a live, fully-deflected command
    driven = decode(body.frame(now=body.last_packet + 0.05))       # within the window: the sticks are live
    assert driven[1] != CENTER or driven[3] != CENTER
    gapped = decode(body.frame(now=body.last_packet + 0.2 + 0.05)) # gap > timeout_s: everything centres
    assert gapped[0] == CENTER and gapped[1] == CENTER and gapped[3] == CENTER


def test_out_of_range_command_is_rejected_and_last_safe_state_held():
    """An out-of-range stick command is rejected, the last neutral-safe state is kept, and the
    caller sees the validation failure (R7.4)."""
    body = _body(timeout_s=5.0)
    assert body.sticks == {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}   # the neutral-safe state at rest
    ok = body.send({"motor": {"turn": 5.0}})                            # yaw stick would be 5.0: out of [-1, 1]
    assert ok is False and body.rejects == 1 and body.last_valid is False
    assert body.sticks == {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}   # unchanged: the bad command moved nothing
    assert decode(body.frame(now=body.last_packet))[3] == CENTER
    # a subsequent valid command is accepted again, and a later out-of-range one holds that state
    assert body.send({"motor": {"turn": 0.5}}) is True and body.last_valid is True
    assert body.send({"motor": {"turn": 3.0}}) is False and body.rejects == 2
    assert abs(body.sticks["yaw"] - 0.5) < 1e-9                         # still the last valid sticks


def test_estop_release_resumes_drive_within_one_tick():
    """While the E-stop is active the mirror centres; clearing it resumes the fly's drive on the
    very next tick (R6.2, R6.3)."""
    torch = pytest.importorskip("torch")
    from companion_brain.config import load_config
    from companion_brain.hunt_gpu.viewer import Watch

    cfg = load_config(overrides={"hunt": {"episode_s": 3.0}})
    body = _body(timeout_s=5.0)
    w = Watch(cfg, None, seed=5, body=body)                             # scripted hunter, one room, mirroring the S1

    gates = []
    orig_send = w.mirror.send

    def spy(f, t, centered=False):
        gates.append(bool(centered))
        return orig_send(f, t, centered=centered)

    w.mirror.send = spy

    w.set_estop(True)
    w.tick()
    assert gates[-1] is True                                           # E-stop active: the centred path is taken

    w.set_estop(False)
    w.tick()
    assert gates[-1] is False                                          # released: the drive resumes on the next tick
