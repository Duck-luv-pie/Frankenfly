import time
from companion_brain.body import s1
from companion_brain.body.s1 import encode, decode, stick, motor_to_sticks, channels, S1Body, MultiBody, CENTER, LOW, HIGH


class FakeSerial:
    def __init__(self):
        self.frames = []
        self.closed = False

    def write(self, b):
        self.frames.append(bytes(b))

    def close(self):
        self.closed = True


def test_frame_round_trips():
    f = encode([LOW, HIGH, 352, 1696, CENTER])
    assert len(f) == 25 and f[0] == 0x0F and f[24] == 0
    assert decode(f)[:5] == [352, 1696, 352, 1696, 1024] and decode(f)[5:] == [1024] * 11
    assert decode(encode([]))[0] == CENTER
    assert encode([], lost=True)[23] & 0x04


def test_stick_range():
    assert stick(0) == CENTER and stick(1) == HIGH and stick(-1) == LOW and stick(5) == HIGH


def test_physical_units_and_rotation_only():
    cfg = {"yaw_dps_full": 90.0, "speed_mps_full": 0.85, "rotation_only": True, "stick_forward": 1.0}
    st = motor_to_sticks({"forward": 1.0, "turn": 1.0, "yaw_dps": 45.0, "speed_mps": 0.85}, cfg)
    assert st["yaw"] == 0.5 and st["forward"] == 0.0            # rotation only: forward stays centred
    cfg["rotation_only"] = False
    st = motor_to_sticks({"yaw_dps": -180.0, "speed_mps": 0.425}, cfg)
    assert st["yaw"] == -1.0 and st["forward"] == 0.5           # saturates at the robot's full stick


def test_motor_mapping_follows_the_hunt_vehicle():
    cfg = {"stick_forward": 0.5, "stick_strafe": 0.5, "stick_yaw": 0.5, "rotation_only": False}
    st = motor_to_sticks({"forward": 1.0, "turn": -0.5}, cfg)
    assert st["forward"] == 0.5 and st["yaw"] == -0.25 and st["strafe"] == 0
    st = motor_to_sticks({"forward": 0.2, "backward": 0.6}, cfg)
    assert abs(st["forward"] - (-0.2)) < 1e-9
    st = motor_to_sticks({"turn": 1.0}, {"stick_yaw": 1.0, "sign_yaw": -1, "rotation_only": False})
    assert st["yaw"] == -1.0


def test_channels_modes():
    ch = channels({}, {"free_mode": True, "speed": "slow"})
    assert ch[:4] == [CENTER] * 4 and ch[4] == LOW and ch[5] == LOW and ch[6] == HIGH
    ch = channels({}, {"free_mode": False, "speed": "fast"}, released=True)
    assert ch[4] == HIGH and ch[5] == HIGH and ch[6] == LOW


def test_body_streams_and_fails_safe():
    ser = FakeSerial()
    b = S1Body({"stick_forward": 1.0, "stick_yaw": 1.0, "timeout_s": 0.2, "rotation_only": False}, ser=ser, start=False)
    assert decode(b.frame())[1] == CENTER                  # nothing received yet: centred, lost flag
    b.send({"state": "track", "motor": {"forward": 1.0, "turn": 0.5}})
    ch = decode(b.frame())
    assert ch[1] == HIGH and ch[3] == stick(0.5) and ch[6] == HIGH
    ch = decode(b.frame(now=time.monotonic() + 1.0))       # packets stopped: sticks centre
    assert ch[1] == CENTER and ch[3] == CENTER
    b.send({"state": "sleep", "motor": {"forward": 1.0}})
    assert decode(b.frame())[6] == LOW                     # asleep: chassis released
    b.close()
    assert ser.closed


def test_body_thread_writes_frames():
    ser = FakeSerial()
    b = S1Body({}, ser=ser)
    time.sleep(0.1)
    b.close()
    assert len(ser.frames) >= 4 and all(len(f) == 25 for f in ser.frames)


def test_multibody_fans_out():
    class B:
        def __init__(self): self.got = []; self.pir = 0; self.busy = 0; self.last_rx = 0.0; self.addr = ("x", 1); self.closed = False
        def send(self, p): self.got.append(p)
        def poll(self): return {"pir": 1}
        def close(self): self.closed = True
    a, c = B(), B()
    a.pir = 1
    m = MultiBody(a, None, c)
    m.send({"motor": {}})
    assert a.got and c.got and m.pir == 1 and m.poll() == {"pir": 1}
    m.close()
    assert a.closed and c.closed


def test_heading_tracking_follows_the_fly_and_ignores_wobble():
    b = S1Body({"yaw_dps_full": 90.0, "heading_gain": 3.0, "timeout_s": 5.0}, ser=FakeSerial(), start=False)
    b.send({"episode": 1, "motor": {"heading_deg": 10.0}})
    assert b.est_heading == 10.0 and decode(b.frame())[3] == CENTER          # aligned on the first packet
    b.send({"episode": 1, "motor": {"heading_deg": 100.0}})                  # the fly turned 90 deg right
    ch = decode(b.frame())
    assert ch[3] == HIGH                                                     # full clockwise stick
    for _ in range(int(2.0 / s1.FRAME_S)):                                   # ~2 s later the estimate has caught up
        b.frame()
    assert abs(b.status()["error_deg"]) < 3 and decode(b.frame())[3] == CENTER
    for i in range(40):                                                      # a wobble of +/-30 deg nets to nothing
        b.send({"episode": 1, "motor": {"heading_deg": 100.0 + (30 if i % 2 else -30)}}); b.frame()
    b.send({"episode": 1, "motor": {"heading_deg": 100.0}})
    for _ in range(30):
        b.frame()
    assert abs(b.status()["error_deg"]) < 3
    b.send({"episode": 2, "motor": {"heading_deg": 250.0}})                  # new episode: the fly teleported, no spin to match
    assert b.est_heading == 250.0
