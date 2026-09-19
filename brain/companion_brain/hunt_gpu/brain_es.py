"""Gradient-free training of the real fly's brain: CMA-ES over the decoder and sensory tunables and a gain per
synapse class of the hunter subcircuit, scored on whole hunts in the batch arena. Needs only forward passes
of the rate model, so it runs where backpropagation cannot (a memory-starved laptop), and it is the same
kind of search the connectome fly's gains came from (`sim/hunt.py`). The result is a `hunter_brain.npz`
like the PPO trainer's: the class gains are folded into the per-synapse gains on save."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..sim.hunt import CMAES
from .brain import EvaderNet
from .brain_train import build_brain, evaluate_brain
from .connectome import RateBrain
from .ppo import summarize

# (parameter, log-scale, low, high) of the scalars; class gains follow, as log gains in [-LOG_CLASS, LOG_CLASS]
SCALARS = [("p_obj_gain", True, 0.2, 4.0), ("p_bar_gain", True, 0.05, 3.0), ("p_azimuth", False, 0.0, 1.0), ("p_heat_gain", True, 0.2, 3.0),
           ("p_efference", False, 0.0, 0.9), ("p_turn_gain", True, 0.3, 3.0), ("p_fwd_zref", True, 0.2, 5.0), ("p_fwd_z0", False, -3.0, 1.0),
           ("p_turn_zref", True, 1.0, 12.0), ("p_turn_z0", False, 0.0, 2.0), ("p_smooth", True, 40.0, 800.0), ("p_freeze_zref", True, 0.5, 10.0),
           ("p_loom_gain", False, 0.0, 4.0), ("p_loom_ref", True, 0.5, 8.0), ("p_near_deg", False, 10.0, 75.0), ("p_near_gain", False, 0.0, 4.0), ("p_freeze_z0", False, 0.0, 6.0)]
LOG_CLASS = 1.5


KEY = ("LC10a", "LC11", "LC12", "LC15", "TRN_hot", "DNa01", "DNa02", "DNp09", "MDN")
EXTRA = {"central->central", "central->DN", "DN->central", "DN->DN", "visual_projection->central", "central->visual_projection",
         "visual_projection->DN", "DN->visual_projection"}


def searched_classes(brain: RateBrain) -> list[int]:
    """The synapse classes worth searching: anything touching a hunter group, plus the central / descending backbone."""
    out = []
    for i, name in enumerate(brain.class_names):
        a, b = name.split("->")
        if a in KEY or b in KEY or name in EXTRA:
            out.append(i)
    return out


def get_vector(brain: RateBrain, cls: list[int]) -> np.ndarray:
    """Normalized coordinates in [-1, 1] of the brain's current scalars and the searched class gains."""
    u = []
    for name, log, lo, hi in SCALARS:
        v = float(getattr(brain, name).detach())
        v = math.exp(v) if log else v
        f = (math.log(max(v, lo) / lo) / math.log(hi / lo)) if log else (v - lo) / (hi - lo)
        u.append(2 * min(1.0, max(0.0, f)) - 1)
    return np.concatenate([np.array(u), brain.class_log_gain.cpu().numpy()[cls] / LOG_CLASS])


@torch.no_grad()
def set_vector(brain: RateBrain, x: np.ndarray, cls: list[int]) -> None:
    x = np.clip(x, -1, 1)
    for (name, log, lo, hi), xi in zip(SCALARS, x):
        f = (xi + 1) / 2
        v = lo * (hi / lo) ** f if log else lo + (hi - lo) * f
        getattr(brain, name).fill_(math.log(v) if log else v)
    g = torch.zeros_like(brain.class_log_gain)
    g[torch.as_tensor(cls)] = torch.as_tensor(x[len(SCALARS):] * LOG_CLASS, dtype=torch.float32)
    brain.class_log_gain.copy_(g)


def fitness(sm: dict) -> float:
    """Tracking over ALL episodes (the tracking fraction of touched episodes times the touch rate, so a fly that
    rarely catches anyone but parks on the one it catches does not win), then touching everyone early, then
    contacts, with a penalty for catching fewer than 85% (the benchmark catches 99%)."""
    if not sm.get("n"):
        return 0.0
    touch = sm["touch_rate"]
    early = (1 - (sm["t_touch"] or 45.0) / 45.0) if sm.get("t_touch") else 0.0
    return 4.0 * (sm.get("track") or 0.0) * touch + 1.0 * touch + 0.3 * early + 0.02 * (sm.get("touches") or 0.0) - 3.0 * max(0.0, 0.85 - touch)


def evolve(cfg, generations: int, popsize: int | None, episodes: int, out_dir: Path, seed: int = 0, start: str | None = None,
           humans: EvaderNet | None = None, hops=(2, 1), sigma0: float = 0.3, device: str = "cpu", heldout: int = 64, spiking_episodes: int = 8) -> dict:
    """Every candidate of every generation is scored on the same `episodes` seeded hunts (common random numbers, so
    ranks compare like with like); a candidate that beats the best is re-scored on `heldout` other seeds before it
    is saved, so the saved brain is never a lucky draw."""
    from .connectome import load_brain
    gains = load_brain(start)["gains"] if start else {}
    hc, brain = build_brain(cfg, hops=hops, gains=gains, device=device, verbose=True)
    if start:
        brain.load_params(start)
    cls = searched_classes(brain)
    x0 = get_vector(brain, cls)
    es = CMAES(x0, sigma0, popsize=popsize, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "hunt_brain_es.jsonl"
    std = {g: {s: float(brain.base_std[i]) for i, (gg, s) in enumerate(brain.read_keys) if gg == g} for g in hc.readouts}
    ep_s, side = float(cfg.hunt.episode_s), float(cfg.hunt.side_reward)
    print(f"[brain-es] CMA-ES over {len(SCALARS)} tunables + {len(cls)} of {len(brain.class_names)} synapse classes: population {es.lam}, "
          f"{episodes} common episodes per candidate, {heldout} held-out for the record, {generations} generations, brake {'on' if brain.brake else 'off'}", flush=True)
    best = {"fitness": -1.0}
    spiking_best = {"fitness": -1.0}
    t0 = time.time()
    for gen in range(generations):
        xs = es.ask()
        fits, sms = [], []
        tg = time.time()
        for x in xs:
            set_vector(brain, x, cls)
            brain.calibrate(std=std)
            brain.readout_noise = True                                                                          # the spiking brain's jitter, so precision it does not have cannot be exploited
            res = evaluate_brain(cfg, brain, episodes, device=device, seed=seed * 100003, humans=humans)     # the same seeds every time
            sm = summarize(res, ep_s, side)
            fits.append(fitness(sm)); sms.append(sm)
        es.tell(xs, fits)
        k = int(np.argmax(fits))
        checked = None
        if fits[k] > best.get("train", -1.0):
            set_vector(brain, xs[k], cls)
            brain.calibrate(std=std)
            brain.readout_noise = True
            hsm = summarize(evaluate_brain(cfg, brain, heldout, device=device, seed=777, humans=humans), ep_s, side)
            checked = fitness(hsm)
            if checked > best["fitness"]:
                best = {"fitness": checked, "train": fits[k], "generation": gen, **{kk: hsm.get(kk) for kk in ("touch_rate", "t_touch", "track", "touches")}}
                path = brain.save(out_dir / "hunter_brain.npz", meta={"trainer": "cma-es", "heldout": heldout, **best})
                if spiking_episodes > 0:                      # what matters is the spiking brain: check the transfer, keep the spiking-best
                    from .brain_train import SpikingHunter, evaluate_spiking
                    import copy
                    scfg = copy.deepcopy(cfg)
                    ssm = summarize(evaluate_spiking(scfg, SpikingHunter(scfg, hc, path, verbose=False), spiking_episodes, seed=4242, humans=humans, verbose=False), ep_s, side)
                    sfit = fitness(ssm)
                    best["spiking"] = {kk: ssm.get(kk) for kk in ("touch_rate", "t_touch", "track", "touches")}
                    if sfit > spiking_best["fitness"]:
                        spiking_best = {"fitness": sfit, "generation": gen, **best["spiking"]}
                        brain.save(out_dir / "hunter_brain_spiking.npz", meta={"trainer": "cma-es", "spiking_episodes": spiking_episodes, **spiking_best})
                    print(f"[brain-es]   as spiking neurons ({spiking_episodes} episodes): touch {ssm['touch_rate']:.0%}, track {ssm.get('track')}, {ssm.get('touches')} touches"
                          f" | spiking best: track {spiking_best.get('track')} touch {spiking_best.get('touch_rate')} (gen {spiking_best.get('generation')})", flush=True)
        line = {"generation": gen, "best": round(max(fits), 3), "mean": round(float(np.mean(fits)), 3), "sigma": round(es.sigma, 3),
                "track_best": sms[k].get("track"), "touch_best": sms[k].get("touch_rate"), "heldout_fitness": checked, "wall_s": round(time.time() - tg), "best_ever": best}
        with open(log, "a") as f:
            f.write(json.dumps(line) + "\n")
        print(f"[brain-es] gen {gen + 1:3d}/{generations}: fitness best {line['best']:.3f} mean {line['mean']:.3f} | best candidate: touch {sms[k]['touch_rate']:.0%}, "
              f"track {sms[k].get('track')}{'' if checked is None else ' -> held-out %.3f (track %s)' % (checked, best.get('track') if best.get('train') == fits[k] else 'lower')} "
              f"| best ever (held-out) track {best.get('track')} touch {best.get('touch_rate')} | sigma {es.sigma:.2f} | {line['wall_s']} s", flush=True)
    print(f"[brain-es] done in {(time.time() - t0) / 60:.1f} min -> {out_dir / 'hunter_brain.npz'}", flush=True)
    return best


def evolve_spiking(cfg, generations: int, popsize: int | None, episodes: int, out_dir: Path, seed: int = 0, start: str | None = None,
                   humans: EvaderNet | None = None, hops=(2, 1), sigma0: float = 0.15, heldout: int = 8) -> dict:
    """The polish: the same search, scored on the spiking neurons themselves (a LIF subcircuit per candidate, one
    room at a time on the CPU), so what is optimized is exactly what runs. Slow (about a minute per candidate), so
    it starts from a rate-model result with a small step and a small population."""
    from .brain_train import SpikingHunter, evaluate_spiking
    from .connectome import load_brain
    import copy
    gains = load_brain(start)["gains"] if start else {}
    hc, brain = build_brain(cfg, hops=hops, gains=gains, device="cpu", verbose=True)
    if start:
        brain.load_params(start)
    cls = searched_classes(brain)
    es = CMAES(get_vector(brain, cls), sigma0, popsize=popsize or 8, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    std = {g: {s: float(brain.base_std[i]) for i, (gg, s) in enumerate(brain.read_keys) if gg == g} for g in hc.readouts}
    ep_s, side = float(cfg.hunt.episode_s), float(cfg.hunt.side_reward)
    tmp = out_dir / "candidate.npz"
    if humans is not None:
        humans = humans.to("cpu").eval()

    def score(x, n, s):
        set_vector(brain, x, cls)
        brain.calibrate(std=std)
        brain.save(tmp)
        scfg = copy.deepcopy(cfg)
        return summarize(evaluate_spiking(scfg, SpikingHunter(scfg, hc, tmp, verbose=False), n, seed=s, humans=humans, verbose=False), ep_s, side)

    print(f"[brain-es] spiking polish: CMA-ES over {len(SCALARS)} tunables + {len(cls)} synapse classes on the LIF subcircuit: population {es.lam}, "
          f"{episodes} common episodes per candidate, {heldout} held-out, {generations} generations", flush=True)
    sm0 = score(es.mean, heldout, 777)
    best = {"fitness": fitness(sm0), "generation": -1, **{kk: sm0.get(kk) for kk in ("touch_rate", "t_touch", "track", "touches")}}
    print(f"[brain-es] start (held-out, spiking): touch {sm0['touch_rate']:.0%}, track {sm0.get('track')}, {sm0.get('touches')} touches", flush=True)
    t0 = time.time()
    for gen in range(generations):
        xs = es.ask()
        tg = time.time()
        sms = [score(x, episodes, seed * 100003) for x in xs]
        fits = [fitness(sm) for sm in sms]
        es.tell(xs, fits)
        k = int(np.argmax(fits))
        hsm = score(xs[k], heldout, 777)
        checked = fitness(hsm)
        if checked > best["fitness"]:
            best = {"fitness": checked, "generation": gen, **{kk: hsm.get(kk) for kk in ("touch_rate", "t_touch", "track", "touches")}}
            set_vector(brain, xs[k], cls); brain.calibrate(std=std)
            brain.save(out_dir / "hunter_brain_spiking.npz", meta={"trainer": "cma-es-spiking", "heldout": heldout, **best})
        with open(out_dir / "hunt_brain_es_spiking.jsonl", "a") as f:
            f.write(json.dumps({"generation": gen, "best": round(max(fits), 3), "mean": round(float(np.mean(fits)), 3), "heldout": round(checked, 3),
                                "track_best": sms[k].get("track"), "touch_best": sms[k].get("touch_rate"), "best_ever": best, "wall_s": round(time.time() - tg)}) + "\n")
        print(f"[brain-es] gen {gen + 1:3d}/{generations} (spiking): fitness best {max(fits):.3f} mean {float(np.mean(fits)):.3f} | best candidate: touch {sms[k]['touch_rate']:.0%}, "
              f"track {sms[k].get('track')} -> held-out track {hsm.get('track')} touch {hsm['touch_rate']:.0%} | best ever: track {best.get('track')} touch {best.get('touch_rate')} | {round(time.time() - tg)} s", flush=True)
    print(f"[brain-es] spiking polish done in {(time.time() - t0) / 60:.1f} min -> {out_dir / 'hunter_brain_spiking.npz'}", flush=True)
    return best
