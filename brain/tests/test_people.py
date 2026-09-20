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


def test_every_level_including_the_coarsest_keeps_its_anchors():
    import math
    from companion_brain.senses.people import NanoDetPeople
    d = object.__new__(NanoDetPeople)                  # no model load: anchors() is pure numpy
    d._anchor_cache = {}
    d.strides, d.size = (8, 16, 32, 64), 416
    for st, rows in zip(d.strides, (2704, 676, 169, 49)):   # NanoDet-Plus-m 416's four levels
        g = int(round(math.sqrt(rows)))
        assert d.anchors(st, g).shape == (rows, 2), f"stride {st} lost its head"
    assert d.anchors(64, 7)[0].tolist() == [31.5, 31.5]     # centre of the first 64 px cell
