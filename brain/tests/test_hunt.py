"""The hunt arena (no connectome needed): seeded layouts, senses, touch and timeout, and a scripted
hunter that proves the arena is solvable. Plus the CMA-ES optimizer on a toy problem."""
import math

import numpy as np

from companion_brain.config import load_config
from companion_brain.sim.hunt_arena import HuntArena, mulberry32, scripted_policy

CFG = load_config()


def test_mulberry32_matches_the_javascript_generator():
    # reference values from the fly project's seededRandom (human.js), node v24
    r = mulberry32(2048)
    assert np.allclose([r() for _ in range(5)], [0.8132568097207695, 0.9656568083446473, 0.124044309835881, 0.1343851420097053, 0.17017259332351387])
    r = mulberry32(4294967295)
    assert np.allclose([r() for _ in range(3)], [0.8964226141106337, 0.189478256739676, 0.7156526781618595])


def test_layout_is_seeded_and_people_stay_in_the_room():
    a, b = HuntArena(CFG.hunt, seed=7), HuntArena(CFG.hunt, seed=7)
    assert [(p.x, p.z, p.pace) for p in a.people] == [(p.x, p.z, p.pace) for p in b.people]
    assert (a.x, a.z, a.heading) == (b.x, b.z, b.heading)
    c = HuntArena(CFG.hunt, seed=8)
    assert (a.x, a.z) != (c.x, c.z)
    for _ in range(600):
        a.sense(); a.step({"forward": 0.0, "turn": 0.0}, 0.0, 0.05)
    for p in a.people:
        assert abs(p.x) <= 7.1 and abs(p.z) <= 5.8 and p.speed > 0
    assert a.nearest()[0] > 0.36


def test_heat_and_vision_are_lateralized():
    a = HuntArena({**CFG.hunt, "people": 1}, seed=1)
    p = a.people[0]
    a.x, a.z = p.x, p.z - 3.0; a.heading = 0.0            # person 3 m straight ahead (heading 0 faces +z)
    s = a.sense()
    assert abs(s.heat[0] - s.heat[1]) < 1e-6 and s.heat[0] > 0.2
    assert s.feats.object_strength > 0 and abs(s.feats.object_x) < 0.05
    a.heading = -math.radians(40)                          # turn right: the person is now on the fly's left
    s = a.sense()
    assert s.nearest_bearing > 0 and s.heat[0] > s.heat[1]
    assert s.feats.left.small_object > 0 and s.feats.right.small_object == 0 and s.feats.object_x < 0
    a.heading = math.pi                                    # facing away: unseen, heat weak on both
    s = a.sense()
    assert s.feats.object_strength == 0 and max(s.heat) < 0.2


def test_touch_rewards_and_timeout_punishes():
    a = HuntArena({**CFG.hunt, "people": 1, "episode_s": 5.0, "reward_s": 1.0, "punish_s": 1.0}, seed=3)
    p = a.people[0]
    p.pace = 0.0; a._place_people()                          # a person standing still
    a.x, a.z, a.heading = p.x, p.z + 1.2, math.pi           # facing -z, toward the person
    t = 0.0
    while not a.done and t < 10:
        a.sense(); a.step({"forward": 1.0}, 0.0, 0.05); t += 0.05
    res = a.result()
    assert res["touched"] and res["frontal"] and res["t_touch"] < 2.0 and a.done and a.reward_amount == 1.0
    # brushing past a person sideways is not a touch: only the nose counts
    c = HuntArena({**CFG.hunt, "people": 1, "episode_s": 3.0, "punish_s": 0.5}, seed=3)
    p = c.people[0]
    p.pace = 0.0; c._place_people()
    c.x, c.z, c.heading = p.x + 0.34, p.z - 1.0, 0.0                   # driving +z, 0.34 m to the person's side
    while not c.done:
        c.sense(); c.step({"forward": 1.0}, 0.0, 0.05)
    assert not c.result()["touched"] and c.result()["min_dist"] < 0.5
    b = HuntArena({**CFG.hunt, "people": 1, "episode_s": 5.0, "punish_s": 1.0}, seed=3)
    seen_bitter = False
    while not b.done:
        s = b.sense(); seen_bitter |= s.bitter > 0
        b.step({"forward": 0.0}, 0.0, 0.05)
    assert not b.result()["touched"] and seen_bitter and abs(b.t - 6.0) < 0.1


def test_scripted_hunter_catches_someone():
    times = []
    for seed in range(6):
        a = HuntArena(CFG.hunt, seed=seed)
        a.h["_valence_steering"] = 0.0
        while not a.done:
            s = a.sense(); a.step(scripted_policy(s), 0.0, 0.05)
        r = a.result()
        assert r["touched"], r
        times.append(r["t_touch"])
    assert np.mean(times) < 20.0


def test_azimuth_weighting_grades_the_object_drive():
    from companion_brain.senses.optic_lobe import Features, features_to_rates
    from companion_brain.config import load_config
    cfg = load_config(overrides={"senses": {"features": {"small_object": {"azimuth_weight": 1.0}}}})
    f = Features(); f.left.small_object = 1.0; f.left.object_x = 1.0          # at the midline of the left hemifield
    centre = features_to_rates(f, cfg).get(("LC10a", "left"), 0.0)
    f.left.object_x = -1.0                                                   # far out in the periphery
    edge = features_to_rates(f, cfg)[("LC10a", "left")]
    assert centre == 0.0 and edge == cfg.senses.max_rate_hz


def test_cmaes_minimizes_a_shifted_sphere():
    from companion_brain.sim.hunt import CMAES
    es = CMAES(np.zeros(4), 0.5, popsize=12, seed=0)
    target = np.array([0.3, -0.5, 0.1, 0.7])
    for _ in range(60):
        xs = es.ask()
        es.tell(xs, [-float(np.sum((x - target) ** 2)) for x in xs])
    assert np.linalg.norm(es.mean - target) < 0.05
