"""Run many headless conditioning assays in parallel (one brain per worker process) and
aggregate: naive vs trained preference for the rewarded scent, with and without plasticity."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

TEST_S, TRAIN_S, TRAIN_FED_S = 60.0, 75.0, 15.0
TICK_S = 0.05


def run_assay(cfg, seed: int, learn: bool, verbose: bool = False) -> dict:
    """One fly, one assay: naive test, confined training with sugar, test. Returns a result dict."""
    from .data.prune import load_or_build, circuit_key
    from .sim.runner import BrainRunner
    from .sim.arena import Arena
    from .body.decode import Decoder, load_or_calibrate
    c = load_or_build(cfg, verbose=False)
    r = BrainRunner(c, cfg, seed=seed)
    base = load_or_calibrate(r, cfg, circuit_key(cfg, False))
    dec = Decoder(cfg, base)
    pl = r.plasticity
    pl.enabled = learn
    arena = Arena(float(cfg.learning.get("valence_steering", 1.0)), seed=seed)
    odor_a, odor_b = cfg.senses.world.odor, cfg.senses.world.odor_b
    sugar_spec, reward = cfg.senses.world.sugar, cfg.learning.reward
    chunks_per_tick = max(1, int(round(TICK_S * 1000 / r.chunk_ms)))
    for _ in range(300):
        r.step_chunk()

    def tick(training: bool):
        arena.sense(training)
        r.clear_drive()
        for side, v in zip(("left", "right"), arena.s.odor_a):
            for g in odor_a.groups: r.drive(g, v * float(odor_a.max_hz), side)
        for side, v in zip(("left", "right"), arena.s.odor_b):
            for g in odor_b.groups: r.drive(g, v * float(odor_b.max_hz), side)
        if arena.s.sugar > 0:
            for g in sugar_spec.groups: r.drive(g, arena.s.sugar * float(sugar_spec.max_hz))
            for g in reward.groups: r.drive(g, arena.s.sugar * float(reward.hz))
        for _ in range(chunks_per_tick):
            r.step_chunk()
        d = dec.decode({w: r.rates(w) for w in dec.windows}, now=r.brain_ms / 1000)
        arena.step(d.motor, d.valence, TICK_S, training)

    def test_phase() -> float:
        arena.s.x, arena.s.z = 0.0, 0.0
        arena.s.heading = float(arena.rng.uniform(-math.pi, math.pi))   # released facing a random way
        ta = tb = 0.0
        for _ in range(int(TEST_S / TICK_S)):
            tick(False)
            if arena.s.x < -1: ta += TICK_S
            elif arena.s.x > 1: tb += TICK_S
        return (ta - tb) / max(1e-6, ta + tb)

    t0 = time.time()
    naive = test_phase()
    arena.s.x, arena.s.z, arena.s.satiety = -AX_START, 0.0, 0.0
    fed, t = 0.0, 0.0
    while fed < TRAIN_FED_S and t < TRAIN_S:
        tick(True)
        if arena.s.feeding > 0.2: fed += TICK_S
        t += TICK_S
    arena.s.satiety = 0.0
    trained = test_phase()
    w = pl.net.data[pl.syn] / np.where(pl.w0 == 0, 1, pl.w0)
    rew = np.array([pl.kind_of[int(q)] == "reward" for q in pl.post])
    out = {"seed": seed, "learn": learn, "naive_pi": round(naive, 3), "trained_pi": round(trained, 3), "fed_s": round(fed, 1),
           "train_s": round(t, 1), "w_reward_comp": round(float(w[rew].mean()), 3), "wall_s": round(time.time() - t0, 1)}
    if verbose:
        print(json.dumps(out), flush=True)
    out["weights"] = pl.net.data[pl.syn].astype(np.float32) if learn else None
    return out


AX_START = 9.0


def _worker(args):
    cfg_path, overrides, seed, learn = args
    from .config import load_config
    cfg = load_config(cfg_path, overrides=overrides)
    return run_assay(cfg, seed, learn)


def run_batch(cfg_path, overrides: dict, runs: int, workers: int, control: bool, save: Path | None = None) -> list[dict]:
    import multiprocessing as mp
    jobs = []
    for i in range(runs):
        jobs.append((cfg_path, overrides, i, True))
        if control:
            jobs.append((cfg_path, overrides, 1000 + i, False))
    workers = max(1, min(workers, len(jobs), os.cpu_count() or 1))
    print(f"[batch] {len(jobs)} assays on {workers} workers ({os.cpu_count()} cores): each = 60 s naive test + training + 60 s test of brain time", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker, jobs)
    wall = time.time() - t0
    for res in results:
        print(f"  seed {res['seed']:4d} learning {'ON ' if res['learn'] else 'OFF'}: PI naive {res['naive_pi']:+.2f} -> trained {res['trained_pi']:+.2f} | fed {res['fed_s']:4.1f} s in {res['train_s']:3.0f} s | KC->MBON reward comps {res['w_reward_comp']:.2f} | {res['wall_s']:.0f} s wall")
    for learn in (True, False):
        sub = [x for x in results if x["learn"] == learn]
        if not sub:
            continue
        d = np.array([x["trained_pi"] - x["naive_pi"] for x in sub])
        print(f"[batch] learning {'ON ' if learn else 'OFF'} (n={len(sub)}): naive PI {np.mean([x['naive_pi'] for x in sub]):+.2f} -> trained PI {np.mean([x['trained_pi'] for x in sub]):+.2f}; "
              f"change {d.mean():+.2f} ± {d.std():.2f}", flush=True)
    print(f"[batch] {len(jobs)} assays ({len(jobs) * (2 * TEST_S + TRAIN_S) / 60:.0f} min of brain time) in {wall / 60:.1f} min wall", flush=True)
    if save is not None:
        learned = [x for x in results if x["learn"] and x["weights"] is not None]
        if learned:
            best = max(learned, key=lambda x: x["trained_pi"] - x["naive_pi"])
            save.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(save, weights=best["weights"], seed=best["seed"], naive_pi=best["naive_pi"], trained_pi=best["trained_pi"])
            print(f"[batch] saved the learned synapses of seed {best['seed']} to {save} (load with: companion run --load-weights {save})", flush=True)
    return results
