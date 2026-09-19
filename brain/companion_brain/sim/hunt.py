"""Training the fly to hunt: the connectome brain in the hunt arena, many flies in parallel.

Two things learn, both are the fly's own:

  1. Mushroom-body plasticity (sim/plasticity.py). A touch delivers sugar + reward dopamine while the
     Kenyon cells that encoded the last seconds (a warm, visible person close ahead) are still eligible,
     so those synapses onto avoidance MBONs are depressed and the situation acquires approach valence,
     which steers the vehicle (arena.step). A timeout delivers bitter + punishment dopamine.
  2. A handful of gains around the fixed wiring (PARAMS below: sensory gains, readout thresholds,
     steering ratio, plasticity time constants) tuned by CMA-ES on the hunting score of whole flies.
     Every candidate is a fresh brain that lives through `episodes` episodes with plasticity on; its
     fitness is the score of the later episodes, so what is optimized is "a fly that learns to hunt".

The trained fly = the learned KC->MBON synapses + the tuned gains, saved together by `save_hunter`
and loadable by `companion run --load-weights` and `companion hunt --load`."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from .hunt_arena import HuntArena, scripted_policy

# (dotted config key, default, low, high, log-scale)
PARAMS = [
    ("senses.features.small_object.gain", 1.0, 0.2, 4.0, True),
    ("senses.features.bar.gain", 0.4, 0.05, 2.0, True),
    ("senses.features.small_object.azimuth_weight", 0.5, 0.0, 1.0, False),
    ("hunt.heat_gain", 1.0, 0.2, 3.0, True),
    ("hunt.heat_scale_m", 3.0, 1.0, 8.0, True),
    ("hunt.heat_lobe_deg", 45.0, 15.0, 80.0, False),
    ("hunt.efference_copy_gain", 0.3, 0.0, 0.9, False),
    ("hunt.turn_gain", 1.0, 0.3, 3.0, True),
    ("decode.motor.channels.forward.z_ref", 1.2, 0.4, 4.0, True),
    ("decode.motor.channels.turn.z_ref", 4.0, 1.0, 10.0, True),
    ("decode.motor.channels.turn.z0", 1.0, 0.0, 2.0, False),
    ("decode.motor.smoothing_ms", 400.0, 100.0, 1200.0, True),
    ("learning.valence_steering", 1.0, 0.0, 4.0, False),
    ("learning.eta", 0.1, 0.02, 0.5, True),
    ("learning.tau_elig_s", 2.0, 0.5, 6.0, True),
    ("learning.dopamine_ref_hz", 5.0, 1.0, 15.0, True),
]


# ---- gains <-> normalized coordinates ------------------------------------------------------------
def gains_from_u(u: np.ndarray) -> dict:
    out = {}
    for (key, _d, lo, hi, log), x in zip(PARAMS, np.clip(u, -1, 1)):
        f = (x + 1) / 2
        out[key] = float(lo * (hi / lo) ** f) if log else float(lo + (hi - lo) * f)
    return out


def u_from_gains(g: dict) -> np.ndarray:
    u = []
    for key, d, lo, hi, log in PARAMS:
        v = float(g.get(key, d))
        f = math.log(v / lo) / math.log(hi / lo) if log else (v - lo) / (hi - lo)
        u.append(2 * min(1.0, max(0.0, f)) - 1)
    return np.array(u)


def default_gains() -> dict:
    return {k: d for k, d, *_ in PARAMS}


def nested(dotted: dict) -> dict:
    out: dict = {}
    for key, v in dotted.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


# ---- one fly, several episodes ----------------------------------------------------------------
def run_hunt(cfg, seed: int, episodes: int, learn: bool = True, weights: np.ndarray | None = None,
             scripted: bool = False, verbose: bool = False, on_tick=None, brain: dict | None = None) -> dict:
    """A brain lives through `episodes` hunts in the arena. Returns per-episode results, the score and
    (with learning) the learned synapses."""
    from ..data.prune import load_or_build, circuit_key
    from .runner import BrainRunner
    from ..body.decode import Decoder, load_or_calibrate
    from ..senses.optic_lobe import features_to_rates
    from ..config import apply_dotted
    t0 = time.time()
    apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))     # the hunter's fixed settings (e.g. no looming -> no escape)
    c = load_or_build(cfg, verbose=False)
    r = BrainRunner(c, cfg, seed=seed)
    if brain is not None:                                  # a GPU-trained connectome brain (hunter_brain.npz): synapse gains + thresholds,
        from ..hunt_gpu.connectome import apply_to_runner   # then a resting baseline measured on the changed brain
        apply_to_runner(r, brain, verbose=verbose)
        from ..body.decode import decode_windows
        base = r.calibrate(float(cfg.decode.calibrate_s), decode_windows(cfg), warmup_s=float(cfg.decode.get("warmup_s", 2.0)), verbose=verbose)
    else:
        base = load_or_calibrate(r, cfg, circuit_key(cfg, False))
    dec = Decoder(cfg, base)
    pl = r.plasticity
    if pl is not None:
        pl.enabled = learn
        if weights is not None and len(weights) == pl.n_syn:
            r.net.data[pl.syn] = np.asarray(weights, dtype=np.float32)
    h = dict(cfg.hunt)
    h["_valence_steering"] = float(cfg.learning.get("valence_steering", 1.0))
    arena = HuntArena(h, seed)
    tick_s = float(h.get("tick_s", 0.05))
    chunks = max(1, int(round(tick_s * 1000 / r.chunk_ms)))
    heat_spec, sugar_spec, bitter_spec = cfg.senses.world.heat, cfg.senses.world.sugar, cfg.senses.world.get("bitter")
    reward, punish = cfg.learning.get("reward"), cfg.learning.get("punish")
    eff = float(h.get("efference_copy_gain", 0.3))
    for _ in range(300):                                   # settle
        r.step_chunk()
    results = []
    for ep in range(episodes):
        arena.reset((seed * 7919 + ep * 104729) & 0xFFFFFFFF)
        while not arena.done:
            s = arena.sense()
            r.clear_drive()
            sup = max(0.0, 1.0 - eff * s.self_motion)
            for (g, side), hz in features_to_rates(s.feats, cfg).items():
                r.drive(g, hz * sup, side)
            for side, v in zip(("left", "right"), s.heat):
                for g in heat_spec.groups:
                    r.drive(g, min(1.0, v) * float(heat_spec.max_hz), side)
            if s.sugar > 0:
                for g in sugar_spec.groups:
                    r.drive(g, s.sugar * float(sugar_spec.max_hz))
                if reward:
                    for g in reward.groups:
                        r.drive(g, s.sugar * float(reward.hz))
            if s.bitter > 0:
                if bitter_spec:
                    for g in bitter_spec.groups:
                        r.drive(g, s.bitter * float(bitter_spec.max_hz))
                if punish:
                    for g in punish.groups:
                        r.drive(g, s.bitter * float(punish.hz))
            for _ in range(chunks):
                r.step_chunk()
            d = dec.decode({w: r.rates(w) for w in dec.windows}, now=r.brain_ms / 1000)
            motor = scripted_policy(s) if scripted else d.motor
            arena.step(motor, d.valence, tick_s)
            if on_tick is not None:
                on_tick(arena, s, d)
        res = arena.result()
        res["episode"] = ep
        if pl is not None:
            res["w_reward_comp"] = round(pl.summary()["weight_reward_comp"], 3)
        results.append(res)
        if verbose:
            print(f"  episode {ep + 1:2d}: {('TOUCH at %5.1f s %s' % (res['t_touch'], 'head-on' if res['frontal'] else 'glancing %+.0f°' % res['contact_deg'])) if res['touched'] else 'timeout                '} | facing {res['facing']:.0%} | path eff {res['efficiency']:.2f} | closest {res['min_dist']:.2f} m"
                  + (f" | KC->MBON reward comps {res['w_reward_comp']:.2f}" if pl is not None else ""), flush=True)
    out = {"seed": seed, "learn": learn, "episodes": results, "score": score(results), "score_late": score(results[len(results) // 2:]),
           "touch_rate": float(np.mean([e["touched"] for e in results])), "wall_s": round(time.time() - t0, 1)}
    if pl is not None:
        out["weights"] = r.net.data[pl.syn].astype(np.float32)
    return out


def score(results: list[dict], episode_s: float | None = None, side_reward: float = 0.5) -> float:
    """Hunting score: touches (frontal 1, glancing side_reward) + 0.5 x how early + 0.3 x time facing someone
    + 0.3 x path efficiency (straight-line distance / distance driven, touched episodes only)."""
    if not results:
        return 0.0
    touch = np.mean([(1.0 if e.get("frontal", True) else side_reward) if e["touched"] else 0.0 for e in results])
    early = np.mean([(1 - e["t_touch"] / max(1e-6, episode_s or 45.0)) if e["touched"] else 0.0 for e in results])
    facing = np.mean([e["facing"] for e in results])
    eff = np.mean([e.get("efficiency", 0.0) for e in results])
    return float(touch + 0.5 * early + 0.3 * facing + 0.3 * eff)


# ---- CMA-ES ------------------------------------------------------------------------------------
class CMAES:
    """(mu/mu_w, lambda)-CMA-ES (Hansen 2016), maximizing. Small and dependency-free."""

    def __init__(self, x0, sigma0: float, popsize: int | None = None, seed: int = 0):
        self.mean = np.array(x0, dtype=float)
        n = self.n = len(self.mean)
        self.sigma = float(sigma0)
        self.lam = popsize or 4 + int(3 * math.log(n))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.C, self.B, self.D = np.eye(n), np.eye(n), np.ones(n)
        self.rng = np.random.default_rng(seed)
        self.gen = 0

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        self._y = (self.B @ (self.D[:, None] * z.T)).T
        return self.mean + self.sigma * self._y

    def tell(self, xs, fitnesses) -> None:
        xs = np.asarray(xs, dtype=float)
        order = np.argsort(-np.asarray(fitnesses, dtype=float))
        ys = (xs[order[: self.mu]] - self.mean) / self.sigma
        ymean = self.w @ ys
        self.mean = self.mean + self.sigma * ymean
        cinv_y = self.B @ ((self.B.T @ ymean) / self.D)
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * cinv_y
        self.gen += 1
        hsig = float(np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean
        rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(self.w, ys))
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.C = (self.C + self.C.T) / 2
        evals, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(np.maximum(evals, 1e-20))


# ---- parallel training -----------------------------------------------------------------------
def _worker(args):
    cfg_path, overrides, gains, seed, episodes, learn, scripted = args
    from ..config import load_config
    cfg = load_config(cfg_path, overrides=_merge(overrides, nested(gains)))
    out = run_hunt(cfg, seed, episodes, learn=learn, scripted=scripted)
    out.pop("weights", None)
    return out


def _merge(a: dict, b: dict) -> dict:
    import copy
    from ..config import _deep_update
    return _deep_update(copy.deepcopy(a or {}), b)


def train(cfg_path, overrides: dict, generations: int, popsize: int | None, episodes: int, workers: int,
          out_dir: Path, seed: int = 0, final_episodes: int = 24, start_gains: dict | None = None) -> dict:
    """CMA-ES over PARAMS; each candidate is a fresh fly that learns through `episodes` hunts. The best
    gains are written every generation; at the end the best fly is trained for `final_episodes` and
    saved with its synapses as hunter.npz."""
    import multiprocessing as mp
    from ..config import load_config
    cfg = load_config(cfg_path, overrides=overrides)
    # warm the caches (circuit, baseline, numba) once here, not in every worker at once
    from ..data.prune import load_or_build, circuit_key
    from .runner import BrainRunner
    from ..body.decode import load_or_calibrate
    c = load_or_build(cfg, verbose=True)
    load_or_calibrate(BrainRunner(c, cfg, seed=0), cfg, circuit_key(cfg, False))
    es = CMAES(u_from_gains(start_gains or default_gains()), 0.35, popsize=popsize, seed=seed)
    workers = max(1, min(workers, es.lam, os.cpu_count() or 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "hunt_train.jsonl"
    print(f"[hunt] CMA-ES over {len(PARAMS)} gains: population {es.lam}, {episodes} episodes x {cfg.hunt.episode_s:.0f} s per fly, "
          f"{generations} generations on {workers} workers", flush=True)
    best = {"score": -1.0, "gains": start_gains or default_gains()}
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        for gen in range(generations):
            xs = es.ask()
            jobs = [(cfg_path, overrides, gains_from_u(x), seed * 100003 + gen * 1009, episodes, True, False) for x in xs]   # common layouts per generation: candidates are ranked on the same episodes
            tg = time.time()
            results = pool.map(_worker, jobs)
            fit = [res["score_late"] for res in results]
            es.tell(xs, fit)
            k = int(np.argmax(fit))
            if fit[k] > best["score"]:
                best = {"score": fit[k], "gains": gains_from_u(xs[k]), "touch_rate": results[k]["touch_rate"], "generation": gen}
                (out_dir / "hunter_gains.json").write_text(json.dumps(best, indent=1))
            mean_u = gains_from_u(es.mean)
            (out_dir / "hunter_gains_mean.json").write_text(json.dumps({"gains": mean_u, "generation": gen, "sigma": es.sigma}, indent=1))
            line = {"generation": gen, "best": round(max(fit), 3), "mean": round(float(np.mean(fit)), 3), "touch_rate_best": results[k]["touch_rate"],
                    "sigma": round(es.sigma, 3), "wall_s": round(time.time() - tg), "best_ever": round(best["score"], 3)}
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"[hunt] gen {gen + 1:3d}/{generations}: score best {line['best']:.2f} mean {line['mean']:.2f} (touch rate of best {line['touch_rate_best']:.0%}) "
                  f"| best ever {best['score']:.2f} | sigma {es.sigma:.2f} | {line['wall_s']} s", flush=True)
    print(f"[hunt] search done in {(time.time() - t0) / 60:.1f} min; best gains -> {out_dir / 'hunter_gains.json'}", flush=True)
    # the final fly: learn for longer with the best gains, then save synapses + gains together
    final_cfg = load_config(cfg_path, overrides=_merge(overrides, nested(best["gains"])))
    print(f"[hunt] training the final fly for {final_episodes} episodes with the best gains ...", flush=True)
    res = run_hunt(final_cfg, seed=seed + 1, episodes=final_episodes, learn=True, verbose=True)
    path = save_hunter(out_dir / "hunter.npz", res, best["gains"])
    print(f"[hunt] saved the trained fly to {path}: touch rate {res['touch_rate']:.0%}, score {res['score']:.2f} "
          f"(run it: companion run --load-weights {path})", flush=True)
    return {"best": best, "final": {k: v for k, v in res.items() if k != "weights"}}


def save_hunter(path: Path, res: dict, gains: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=res["weights"], seed=res["seed"], gains=json.dumps(gains),
                        touch_rate=res["touch_rate"], score=res["score"], episodes=json.dumps(res["episodes"]),
                        naive_pi=0.0, trained_pi=0.0)
    return path


def is_brain_file(path) -> bool:
    """A hunter_brain.npz (per-synapse gains keyed by root id, from `companion hunt-brain`) rather than a hunter.npz."""
    return "brain" in np.load(path).files


def load_hunter(path) -> tuple[np.ndarray, dict, dict]:
    z = np.load(path)
    if "brain" in z.files:
        gains = json.loads(str(z["gains"]))
        return np.zeros(0, np.float32), gains, {"brain": True, **json.loads(str(z["meta"]))}
    gains = json.loads(str(z["gains"])) if "gains" in z.files else {}
    meta = {k: (float(z[k]) if z[k].ndim == 0 and z[k].dtype.kind == "f" else z[k]) for k in z.files if k not in ("weights", "gains", "episodes")}
    return z["weights"], gains, meta
