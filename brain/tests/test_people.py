import numpy as np
from companion_brain.senses.people import PeopleSense


def sense(boxes):
    return PeopleSense(bins=24, fly_fov_deg=150, cam_fov_deg=60, detector=lambda g: boxes, hold_s=0.1, tick_s=0.05)


def test_person_ahead_lights_the_middle_columns():
    s = sense([(150, 60, 20, 120)])                    # 20 px wide of 320 at 60 deg -> 3.75 deg wide, centred
    vis, boxes = s.observe(np.zeros((240, 320), np.uint8))
    lit = np.nonzero(vis[:, 0])[0]
    assert set(lit) == {11, 12} and abs(vis[11, 0] - 3.75 / 25) < 1e-3
    assert abs(vis[11, 1] - 0.8 * 0.4 * vis[11, 0]) < 1e-6   # nobody moves: 0.4 of the object


def test_left_of_image_is_left_columns_and_width_caps():
    s = sense([(0, 40, 160, 160)])                      # the left half of the frame: 30 deg wide, bearing +15 (left)
    vis, _ = s.observe(np.zeros((240, 320), np.uint8))
    assert vis[:, 0].max() == 1.0                       # 30 deg > 25 deg: capped
    lit = np.nonzero(vis[:, 0])[0]
    assert lit.min() < 12 and lit.max() <= 12           # left side of the eye only
    assert vis[23, 0] == 0                              # the eye's right edge (beyond the camera) stays dark


def test_motion_from_frame_difference_and_fly_motion():
    s = sense([(150, 60, 20, 120)])
    a = np.zeros((240, 320), np.uint8)
    b = a.copy(); b[60:180, 150:170] = 200              # the person's patch changed
    s.observe(a); vis, _ = s.observe(b)
    assert abs(vis[11, 1] - 0.8 * vis[11, 0]) < 1e-6    # moving: full 0.8
    vis, _ = s.observe(b, fly_moving=True)
    assert abs(vis[11, 1] - 0.8 * vis[11, 0]) < 1e-6    # the fly moving counts too


def test_lost_detection_is_held_briefly_then_cleared():
    boxes = [[(150, 60, 20, 120)]]
    s = PeopleSense(bins=24, fly_fov_deg=150, cam_fov_deg=60, detector=lambda g: boxes[0], hold_s=0.1, tick_s=0.05)
    f = np.zeros((240, 320), np.uint8)
    assert s.observe(f)[0][:, 0].max() > 0
    boxes[0] = []
    assert s.observe(f)[0][:, 0].max() > 0             # held for 2 ticks
    s.observe(f)
    assert s.observe(f)[0][:, 0].max() == 0
    assert s.observe(None)[0].shape == (24, 2)          # no frame at all is fine


def test_scripted_chaser_steers_at_the_nearest_person_and_searches_when_lost():
    from companion_brain.hunt_gpu.real import ScriptedChaser
    c = ScriptedChaser(cam_fov_deg=60.0)
    f, t = c.act([(275, 60, 40, 120), (100, 90, 20, 40)], 320)     # the big box on the far right wins
    assert t > 0 and f == 0.5 and c.bearing_deg > 20                 # far right: turn first, half throttle
    f, t = c.act([(150, 60, 20, 120)], 320)                          # dead ahead
    assert abs(t) < 0.1 and f == 1.0
    f, t = c.act([], 320)                                            # lost after seeing them on the right: spin right
    assert f == 0.0 and t > 0
    c.act([(10, 60, 40, 120)], 320); f, t = c.act([], 320)           # last seen on the left: spin left
    assert t < 0
