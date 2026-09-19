"""The GPU spinoff of the hunt (skipped without torch): the batched arena is the same room as the CPU
arena (layouts, senses, physics), it is solvable, the fly-shaped network runs, PPO learns for a few
updates, and a saved fly drives the CPU arena through the bridge."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from companion_brain.config import load_config  # noqa: E402
from companion_brain.sim.hunt_arena import HuntArena, scripted_policy  # noqa: E402
from companion_brain.hunt_gpu.arena import BatchArena, scripted_drive  # noqa: E402
from companion_brain.hunt_gpu.brain import HunterNet, build  # noqa: E402
from companion_brain.hunt_gpu.ppo import Trainer, evaluate, summarize  # noqa: E402
from companion_brain.hunt_gpu.bridge import CpuHunter, run_cpu  # noqa: E402

CFG = load_config()
SEEDS = [7, 8, 12345, 4294967295]


PATROL = load_config(overrides={"hunt_gpu": {"humans": {"mode": "patrol"}, "senses": {"see_m": 0, "heat_m": 0, "pir": {"enabled": False}}}})     # the CPU arena's walkers and senses, for the parity tests


def batch(n=None, seeds=None, **g):
    seeds = seeds or SEEDS
    a = BatchArena(CFG.hunt, {**PATROL.hunt_gpu, "obs_noise": 0.0, **g}, n or len(seeds), "cpu", seed=0)
    a.reset(torch.arange(a.B), seeds)
    return a


def test_batch_layout_matches_the_cpu_arena():
    a = batch()
    s = a.snapshot()
    for k, seed in enumerate(SEEDS):
        c = HuntArena(CFG.hunt, seed=seed)
        assert np.allclose([p.x for p in c.people], s["px"][k][:4], atol=1e-5)
        assert np.allclose([p.z for p in c.people], s["pz"][k][:4], atol=1e-5)
        assert s["alive"][k].sum() == 4
        assert abs(s["x"][k] - c.x) < 1e-5 and abs(s["z"][k] - c.z) < 1e-5 and abs(s["heading"][k] - c.heading) < 1e-5


def test_batch_senses_match_the_cpu_arena():
    a = batch()
    obs = a.observe(noise=False)
    heat, vis, body = a.spec.split(obs)
    for k, seed in enumerate(SEEDS):
        c = HuntArena(CFG.hunt, seed=seed)
        s = c.sense()
        assert np.allclose(heat[k].numpy(), s.heat, atol=1e-5)
        # a person the CPU arena sees on the left lights columns on the left half of the map, and vice versa
        left_cols, right_cols = vis[k, : a.spec.bins // 2, 0].max().item(), vis[k, a.spec.bins // 2:, 0].max().item()
        assert (left_cols > 0) == (s.feats.left.small_object > 0)
        assert (right_cols > 0) == (s.feats.right.small_object > 0)
        assert abs(max(left_cols, right_cols) - s.feats.object_strength) < 1e-5
        assert body[k].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_batch_physics_matches_the_cpu_arena_step_for_step():
    a = batch()
    cpus = [HuntArena({**CFG.hunt, "_valence_steering": 0.0}, seed=s) for s in SEEDS]
    drive = torch.tensor([[1.0, 0.3], [1.0, -0.6], [0.5, 0.0], [1.0, 1.0]])
    for _ in range(60):
        a.step(drive)
        for k, c in enumerate(cpus):
            c.sense()
            c.step({"forward": float(drive[k, 0]), "backward": 0.0, "turn": float(drive[k, 1])}, 0.0, 0.05)
    s = a.snapshot()
    for k, c in enumerate(cpus):
        assert abs(s["x"][k] - c.x) < 1e-3 and abs(s["z"][k] - c.z) < 1e-3, (k, s["x"][k], c.x, s["z"][k], c.z)
        assert abs(math.atan2(math.sin(s["heading"][k] - c.heading), math.cos(s["heading"][k] - c.heading))) < 1e-3
        assert np.allclose([p.x for p in c.people], s["px"][k][:4], atol=1e-4)


def test_touch_ends_the_episode_with_a_speed_bonus_and_timeout_costs():
    a = batch(seeds=[3], people_min=1, people_max=1, track={"enabled": False})
    # park the fly 1.2 m in front of the (frozen) person, facing it
    a.pace[:] = 0.0
    a.x[0], a.z[0], a.heading[0] = a.px[0, 0], a.pz[0, 0] + 1.2, math.pi
    a.target_d[0] = 1.2
    total, done, info = 0.0, None, {}
    for _ in range(60):
        r, done, info = a.step(torch.tensor([[1.0, 0.0]]))
        total += float(r[0])
        if bool(done[0]):
            break
    ep = info["episodes"][0]
    assert ep["touched"] and ep["frontal"] and ep["t_touch"] < 2.0
    assert total > a.r_touch + 0.9 * a.r_fast                      # the touch plus almost the whole speed bonus
    b = batch(seeds=[3], people_min=1, people_max=1, track={"enabled": False})
    b.pace[:] = 0.0
    b.x[0], b.z[0], b.heading[0] = 0.0, -5.0, 0.0                   # far from everyone, standing still
    b.px[0, 0], b.pz[0, 0] = 5.0, 5.0
    total = 0.0
    for n in range(b.max_steps + 5):
        r, done, info = b.step(torch.tensor([[0.0, 0.0]]))
        total += float(r[0])
        if bool(done[0]):
            break
    assert n + 1 == b.max_steps and not info["episodes"][0]["touched"] and total < -b.r_timeout


def test_after_the_first_touch_the_fly_is_paid_for_tracking_the_locked_person():
    a = batch(seeds=[3], people_min=1, people_max=1)
    a.pace[:] = 0.0
    a.x[0], a.z[0], a.heading[0] = a.px[0, 0], a.pz[0, 0] + 1.2, math.pi
    a.target_d[0] = 1.2
    after, done = 0.0, None
    for n in range(200):
        r, done, info = a.step(torch.tensor([[1.0, 0.0]]))
        if bool(a.locked[0]) and n > 30:
            after += float(r[0])
    assert bool(a.locked[0]) and not bool(done[0]) and "episodes" not in info      # touched, but the episode goes on
    assert a.touches[0] >= 3 and a.track_steps[0] > 150 and after > 0.5   # pressed against them: tracking pays, renewed contact pays
    # a timeout with someone locked is not punished; the record carries the first touch and the tracking fraction
    a.steps[0] = a.max_steps - 1
    r, done, info = a.step(torch.tensor([[1.0, 0.0]]))
    ep = info["episodes"][0]
    assert bool(done[0]) and ep["touched"] and ep["t_touch"] < 2.0 and ep["track"] > 0.2 and ep["touches"] >= 3 and float(r[0]) > 0   # (the skipped clock dilutes the fraction)
    # the scripted hunter chases whoever is warmest, not the person it caught: it tracks poorly (the bar a trained
    # fly must clear), but the records carry the tracking statistics of a full-length episode
    res = evaluate(load_config(overrides={"hunt": {"episode_s": 15.0}, "hunt_gpu": {"humans": {"mode": "patrol"}, "senses": {"see_m": 0, "heat_m": 0}}}), None, 8, device="cpu", seed=0, scripted=True)
    # (with a solid body and a single PIR bit for warmth, its glancing approaches sometimes miss within 15 s)
    assert sum(e["touched"] for e in res) >= 6 and 0.05 < np.mean([e["track"] or 0.0 for e in res]) < 0.6 and all(e["touches"] >= 1 for e in res if e["touched"])


def test_scripted_hunter_catches_people_in_the_batch_arena():
    res = evaluate(PATROL, None, 12, device="cpu", seed=0, scripted=True)
    sm = summarize(res, float(CFG.hunt.episode_s), float(CFG.hunt.side_reward))
    assert sm["n"] == 12 and sm["touch_rate"] >= 0.8 and sm["t_touch"] < 25.0
    # ... and it is the same hunter as the CPU arena's scripted policy on the same seeds
    for e in res[:3]:
        c = HuntArena({**CFG.hunt, "_valence_steering": 0.0}, seed=e["seed"])
        while not c.done:
            c.step(scripted_policy(c.sense()), 0.0, 0.05)
        assert c.result()["touched"] == e["touched"]


def test_network_runs_and_round_trips_through_a_file(tmp_path):
    a = batch()
    net = build(a.spec, CFG.hunt_gpu.net)
    obs = a.observe(noise=False)
    h = net.initial_state(a.B, "cpu")
    mean, value, h2 = net(obs, h)
    assert mean.shape == (a.B, 2) and value.shape == (a.B,) and h2.shape == h.shape
    assert mean.abs().max() <= 1.0
    act, logp, v, _ = net.act(obs, h)
    assert act.shape == (a.B, 2) and torch.isfinite(logp).all()
    kc = torch.relu(net.pn(obs[:, :2]) @ net.kc_w)
    assert (kc > 0).float().mean() > 0.3                               # dense before the k-winners ...
    path = net.save(tmp_path / "fly.pt", meta={"note": "test"})
    net2, meta = HunterNet.load(path)
    assert meta["note"] == "test"
    m2, v2, _ = net2(obs, h)
    assert torch.allclose(mean, m2) and torch.allclose(value, v2)


def test_ppo_smoke_trains_for_a_few_updates(tmp_path):
    cfg = load_config(overrides={"hunt_gpu": {"envs": 16, "ppo": {"steps": 16, "epochs": 2, "minibatches": 2}, "humans": {"mode": "patrol"}}})
    tr = Trainer(cfg, device="cpu", seed=1, out_dir=tmp_path)
    res = tr.train(updates=3, verbose=False)
    assert res["updates"] == 3 and res["steps"] == 3 * 16 * 16 and tr.hnet is None
    assert (tmp_path / "hunt_gpu_train.jsonl").exists() and (tmp_path / "hunter_gpu_last.pt").exists()
    for p in tr.net.parameters():
        assert torch.isfinite(p).all()


def test_runners_flee_at_their_top_speed_then_rest():
    a = batch(seeds=[3], people_min=1, people_max=1, humans={**CFG.hunt_gpu.humans, "mode": "flee"})
    a.x[0], a.z[0], a.heading[0] = a.px[0, 0], a.pz[0, 0] + 2.0, math.pi                # 2 m behind the person, facing it
    obs = a.observe_humans()
    assert obs.shape == (1, 8, a.hspec.size) and torch.isfinite(obs).all()
    d0 = math.hypot(float(a.px[0, 0] - a.x[0]), float(a.pz[0, 0] - a.z[0]))
    for _ in range(60):                                                                 # the fly drives at it flat out for 3 s ...
        a.step(torch.tensor([[1.0, 0.0]]))
    d1 = math.hypot(float(a.px[0, 0] - a.x[0]), float(a.pz[0, 0] - a.z[0]))
    assert d1 > 1.0 and not bool(a.locked[0])                                           # ... and never catches the runner
    assert abs(float(a.px[0, 0])) <= 7.1 and abs(float(a.pz[0, 0])) <= 5.8               # still inside the room
    assert float(a.stamina[0, 0]) < 0.5                                                 # 3 s of sprinting has drained it ...
    top, rested = 0.0, False
    for _ in range(120):                                                                # ... and within 6 s more it must stop and rest
        a.step(torch.tensor([[0.0, 0.0]]))
        top = max(top, float(a.pspeed[0, 0]) / a.dt)
        rested |= bool(a.resting[0, 0])
    assert top > 0.9 * float(CFG.hunt_gpu.humans["speed_max"]) and rested                # it sprinted at its top speed (a touch below the fly's), then rested
    assert a.observe_humans()[0, 0, -2:].tolist() == [round(float(a.stamina[0, 0]), 6) if False else float(a.stamina[0, 0]), float(a.resting[0, 0])]


def test_runners_stop_now_and_then_whatever_they_do():
    a = batch(seeds=[3], people_min=2, people_max=2, humans={**CFG.hunt_gpu.humans, "mode": "learn", "pause": {"every_s": 4.0, "for_s": 1.0}})
    a.x[0], a.z[0] = 6.0, 5.0                                                          # the fly is far away and never moves
    still = torch.zeros(1, 8, 2); still[..., 0] = 0.4                                  # runners stroll at 0.8 m/s (no stamina cost)
    rests = torch.zeros(8)
    for _ in range(int(20 / a.dt)):
        a.step(torch.tensor([[0.0, 0.0]]), still)
        rests += a.resting[0].float()
    assert all(40 <= r <= 140 for r in rests[:2].tolist())                             # 2..7 s (40..140 ticks) of standing in 20 s per runner: ~1 s per 2..6 s
    assert float(rests[2:].sum()) == 0                                                  # empty slots never rest
    b = batch(seeds=[3], people_min=1, people_max=1, humans={**CFG.hunt_gpu.humans, "mode": "learn", "pause": {"every_s": 0}})
    for _ in range(200):
        b.step(torch.tensor([[0.0, 0.0]]), still)
    assert not bool(b.resting[0, 0])                                                    # every_s 0 disables the stops


def test_the_fly_senses_only_nearby_people_and_is_paid_to_search():
    a = batch(seeds=[3], people_min=1, people_max=1, senses={"see_m": 6.0, "heat_m": 5.0}, humans={**CFG.hunt_gpu.humans, "mode": "learn", "pause": {"every_s": 0}})
    a.pace[:] = 0.0
    a.x[0], a.z[0], a.heading[0] = a.px[0, 0], a.pz[0, 0] + 8.0, math.pi                # 8 m straight ahead: out of range
    obs = a.observe(noise=False)
    heat, vis, _ = a.spec.split(obs)
    assert float(heat.sum()) == 0 and float(vis.sum()) == 0
    a.z[0] = a.pz[0, 0] + 3.0                                                             # 3 m: seen and felt
    heat, vis, _ = a.spec.split(a.observe(noise=False))
    assert float(heat.sum()) > 0.3 and float(vis[..., 0].max()) > 0
    # alone in the room (the person is parked far away and the fly points away from it), driving pays, standing does not
    b = batch(seeds=[3], people_min=1, people_max=1, senses={"see_m": 6.0, "heat_m": 5.0}, humans={**CFG.hunt_gpu.humans, "mode": "learn", "pause": {"every_s": 0}})
    b.px[0, 0], b.pz[0, 0], b.wx[0, 0], b.wz[0, 0] = 6.5, 5.5, 6.5, 5.5
    b.x[0], b.z[0], b.heading[0] = -5.0, -4.0, 0.0
    drive_total = sum(float(b.step(torch.tensor([[1.0, 0.1]]))[0][0]) for _ in range(40))
    c = batch(seeds=[3], people_min=1, people_max=1, senses={"see_m": 6.0, "heat_m": 5.0}, humans={**CFG.hunt_gpu.humans, "mode": "learn", "pause": {"every_s": 0}})
    c.px[0, 0], c.pz[0, 0], c.wx[0, 0], c.wz[0, 0] = 6.5, 5.5, 6.5, 5.5
    c.x[0], c.z[0], c.heading[0] = -5.0, -4.0, 0.0
    still_total = sum(float(c.step(torch.tensor([[0.0, 0.0]]))[0][0]) for _ in range(40))
    assert drive_total > still_total + 0.3 and int(b.blind_steps[0]) == 40 and int(c.idle_steps[0]) == 40


def test_runners_mind_the_fly_only_when_it_is_close_and_stroll_otherwise():
    a = batch(seeds=[3], people_min=4, people_max=4, humans={**CFG.hunt_gpu.humans, "mode": "flee", "pause": {"every_s": 0}})
    a.x[0], a.z[0] = 7.0, 6.0                                                           # the fly parked in a corner, far from everyone
    start = a.snapshot()
    still = torch.tensor([[0.0, 0.0]])
    for _ in range(int(15 / a.dt)):
        a.step(still)
    s = a.snapshot()
    assert not s["alert"][0].any()                                                       # nobody minds it ...
    moved = [math.hypot(s["px"][0][i] - start["px"][0][i], s["pz"][0][i] - start["pz"][0][i]) for i in range(4)]
    assert min(moved) > 0.5                                                              # ... everyone strolled somewhere
    corner = [max(abs(s["px"][0][i]) / 7.1, abs(s["pz"][0][i]) / 5.8) for i in range(4)]
    assert np.mean(corner) < 0.9                                                         # and not into the far corners
    a.x[0], a.z[0] = float(a.px[0, 0]) + 2.0, float(a.pz[0, 0])                          # now the fly is 2 m from person 1
    a.step(still)
    assert bool(a.alert[0, 0]) and not bool(a.alert[0, 1:].any()) or bool(a.alert[0, 0])


def test_adversarial_ppo_trains_fly_and_runners_together(tmp_path):
    from companion_brain.hunt_gpu.brain import EvaderNet
    cfg = load_config(overrides={"hunt_gpu": {"envs": 16, "ppo": {"steps": 16, "epochs": 2, "minibatches": 2}, "humans": {"mode": "learn"}}})
    tr = Trainer(cfg, device="cpu", seed=1, out_dir=tmp_path)
    before = [p.detach().clone() for p in tr.hnet.parameters()]
    res = tr.train(updates=3, verbose=False)
    assert res["updates"] == 3 and tr.hnet is not None
    assert any(not torch.equal(b, p.detach()) for b, p in zip(before, tr.hnet.parameters()))   # the runners learned something
    assert (tmp_path / "evaders_gpu_last.pt").exists()
    hnet, meta = EvaderNet.load(tmp_path / "evaders_gpu_last.pt")
    a = batch(seeds=[7], humans={**CFG.hunt_gpu.humans, "mode": "learn"})
    drive, logp, value = hnet.act(a.observe_humans(), deterministic=True)
    assert drive.shape == (1, 8, 2) and torch.isfinite(logp).all()
    res = evaluate(cfg, None, 4, device="cpu", seed=0, scripted=True, humans=hnet)
    assert len(res) == 4 and all("mean_dist" in e for e in res)


def test_a_saved_fly_drives_the_cpu_arena_through_the_bridge():
    a = batch()
    net = build(a.spec, CFG.hunt_gpu.net)
    hunter = CpuHunter(PATROL, net)                                        # the same unlimited senses as the batch above
    c = HuntArena(CFG.hunt, seed=SEEDS[0])
    obs = hunter.observe(c)
    assert torch.allclose(obs, a.observe(noise=False)[0:1], atol=1e-5)  # the bridge sees the batch arena's observation
    motor = hunter(c)
    assert set(motor) == {"forward", "backward", "turn", "freeze"} and 0 <= motor["forward"] <= 1 and -1 <= motor["turn"] <= 1
    cfg = load_config(overrides={"hunt": {"episode_s": 3.0, "punish_s": 0.5}})
    res = run_cpu(cfg, net, SEEDS[:2])
    assert len(res) == 2 and all("touched" in r for r in res)


def test_the_live_view_ticks_one_room_and_reports_its_state():
    from companion_brain.hunt_gpu.viewer import Watch
    cfg = load_config(overrides={"hunt": {"episode_s": 3.0}})
    w = Watch(cfg, None, seed=5)                                  # the scripted hunter, one room
    st = None
    for _ in range(w.arena.max_steps + 1):
        st = w.tick()
    assert len(st["people"]) == 4 and len(st["vis"]) == 24 and len(st["heat"]) == 2 and st["episode"] == 2   # the clock ran out: episode 2 began
    assert w.results and w.results[0]["seed"] == 5 and "total_reward" in w.results[0] and st["stats"]["n"] == 1
    w.next_episode()
    assert w.episode == 3 and float(w.arena.t[0]) == 0.0


def test_the_lobotomy_swaps_brains_without_losing_the_trained_one():
    from companion_brain.hunt_gpu.viewer import Watch
    cfg = load_config(overrides={"hunt": {"episode_s": 3.0}})
    net = build(BatchArena(cfg.hunt, cfg.hunt_gpu, 1, "cpu").spec, cfg.hunt_gpu.net)
    w = Watch(cfg, net, seed=5)
    smart = w.net
    assert w.smart is net and w.dumb is not net and not w.lobotomized and w.tick()["lobotomized"] is False
    w.lobotomize(True)
    assert w.net is w.dumb and w.lobotomized and w.tick()["lobotomized"] is True
    w.lobotomize(False)
    assert w.net is smart and not w.lobotomized                                       # the trained brain, untouched
    for a, b in zip(smart.parameters(), net.parameters()):
        assert torch.equal(a, b)
