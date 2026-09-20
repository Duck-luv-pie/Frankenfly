"""Property-based tests for the sim-to-robot Mirror drive->stick mapping (fake serial, no hardware).

The Mirror (hunt_gpu/mirror.py) turns the fly's per-tick drive [forward, turn] into S-Bus stick
deflections and hands them to an S1Body. These tests drive arbitrary Drive_Vectors -- including
out-of-range, extreme and non-finite (NaN / inf) inputs -- through a Mirror wired to an S1Body that
holds a fake serial, and assert the two universally-quantified properties from the design's
Correctness Properties section. See .kiro/specs/sim-mirror-webapp/design.md, "Property 1" / "Property 2".
"""
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from companion_brain.body.s1 import DEFAULTS, S1Body
from companion_brain.hunt_gpu.mirror import Mirror

# minimum 100 iterations per property (the design requires >= 100)
PBT = settings(max_examples=200, deadline=None)

SPEED_FULL = DEFAULTS["speed_mps_full"]   # 0.85 m/s at full forward stick (R5.1)
YAW_FULL = DEFAULTS["yaw_dps_full"]       # 90 deg/s at full yaw stick (R5.1)


class FakeSerial:
    """A stand-in for the S1's serial port: records frames, never touches hardware."""

    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, b):
        self.frames.append(bytes(b))

    def close(self):
        self.closed = True


def make_body(**cfg):
    """An S1Body wired to a fake serial with its background sender stopped (no real port opens)."""
    return S1Body(cfg, ser=FakeSerial(), start=False)


# a Drive_Vector component: in-range floats plus out-of-range, extreme, and NaN / inf values
drive_component = st.one_of(
    st.floats(min_value=-1.0, max_value=1.0),
    st.floats(min_value=-1000.0, max_value=1000.0),
    st.floats(allow_nan=True, allow_infinity=True),
)


# Feature: sim-mirror-webapp, Property 1: Drive-to-stick mapping is total and clamped
# Validates: Requirements 4.3, 4.4, 5.7
@PBT
@given(forward=drive_component, turn=drive_component)
def test_property1_mapping_is_total_and_clamped(forward, turn):
    """For any Drive_Vector (incl. out-of-range / extreme / non-finite), the Mirror produces a
    defined stick command with every axis within the S1 stick range [-1, 1]."""
    mirror = Mirror(body=make_body(), cfg=None)
    sticks = mirror.sticks(forward, turn)
    for axis, value in sticks.items():
        assert math.isfinite(value), f"axis {axis} not finite for drive ({forward}, {turn})"
        assert -1.0 <= value <= 1.0, f"axis {axis}={value} out of [-1, 1] for drive ({forward}, {turn})"


# Feature: sim-mirror-webapp, Property 1: [0, 0] maps to the neutral centre on every axis
# Validates: Requirements 4.3
@PBT
@given(
    v_max=st.floats(min_value=0.0, max_value=SPEED_FULL),
    w_max=st.floats(min_value=0.0, max_value=YAW_FULL),
    sign_yaw=st.sampled_from(["normal", "inverted"]),
)
def test_property1_zero_drive_is_neutral_center(v_max, w_max, sign_yaw):
    """A Drive_Vector of exactly [0.0, 0.0] maps to the neutral centre (0.0) on every axis, for any
    valid speed / turn cap and yaw-sign configuration."""
    mirror = Mirror(body=make_body(), cfg=None, v_max=v_max, w_max=w_max, sign_yaw=sign_yaw)
    sticks = mirror.sticks(0.0, 0.0)
    for axis, value in sticks.items():
        assert value == 0.0, f"axis {axis}={value} not neutral for zero drive"


# Feature: sim-mirror-webapp, Property 2: Speed and turn caps hold
# Validates: Requirements 5.2, 5.3, 5.5, 5.6
@PBT
@given(
    forward=drive_component,
    turn=drive_component,
    v_max=st.floats(min_value=0.0, max_value=SPEED_FULL),
    w_max=st.floats(min_value=0.0, max_value=YAW_FULL),
    sign_yaw=st.sampled_from(["normal", "inverted"]),
)
def test_property2_speed_and_turn_caps_hold(forward, turn, v_max, w_max, sign_yaw):
    """For any Drive_Vector and any valid caps (v_max in 0..0.85 m/s, w_max in 0..90 deg/s), the
    scaled forward stick never asks for more than v_max and the yaw stick never asks for more than
    w_max, in physical units."""
    mirror = Mirror(body=make_body(), cfg=None, v_max=v_max, w_max=w_max, sign_yaw=sign_yaw)
    sticks = mirror.sticks(forward, turn)
    eps = 1e-9
    # the stick fraction is a fraction of the calibrated full-scale, so the physical command is fraction * full
    forward_mps = abs(sticks["forward"]) * SPEED_FULL
    yaw_dps = abs(sticks["yaw"]) * YAW_FULL
    assert forward_mps <= v_max + eps, f"forward {forward_mps} m/s exceeds cap {v_max} for drive ({forward}, {turn})"
    assert yaw_dps <= w_max + eps, f"yaw {yaw_dps} deg/s exceeds cap {w_max} for drive ({forward}, {turn})"
