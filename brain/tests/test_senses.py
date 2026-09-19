import numpy as np

from companion_brain.config import load_config
from companion_brain.senses.camera import SyntheticLooming
from companion_brain.senses.optic_lobe import OpticLobe, features_to_rates


def test_synthetic_looming_drives_loom_and_object():
    cfg = load_config()
    src = SyntheticLooming(80, 60, 15.0)
    lobe = OpticLobe(80, 60, 15.0)
    loom_max = {"left": 0.0, "right": 0.0}
    obj_max = 0.0
    while (frame := src.read()) is not None:
        f = lobe.process(frame)
        for s in loom_max:
            loom_max[s] = max(loom_max[s], f.side(s).loom_fast)
        obj_max = max(obj_max, f.object_strength)
        rates = features_to_rates(f, cfg)
        assert all(0 <= v <= cfg.senses.max_rate_hz for v in rates.values())
    assert loom_max["right"] > 0.3          # disc expands in the right hemifield
    assert obj_max > 0.3                     # small object was seen


def test_static_scene_is_quiet():
    lobe = OpticLobe(80, 60, 15.0)
    frame = np.full((60, 80), 128, dtype=np.uint8)
    for _ in range(5):
        f = lobe.process(frame)
    assert f.motion_energy == 0 and f.left.loom_fast == 0 and f.right.small_object == 0
