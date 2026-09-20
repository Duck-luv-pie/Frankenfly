"""
train.py -- curriculum loop.

    python train.py --stage A --envs 512 --gain 0.05 --learn none|three_factor|es --minutes 60
    python train.py --sweep gain 0.01:0.2:10 --envs 80          # batched: envs split across values
    python train.py --sweep k_t 0.005:0.05:5                     # one generation per value
    python train.py --record logs/replay.json --envs 8 --seconds 10   # dump env 0 for the viz

Per generation:
  arena.reset(stage); lif.reset(); retina.reset()
  for t in range(T):                                  # T = episode_s / 0.02
      r = retina.from_state(*arena.retina_inputs(), dt=0.02); arena.perturb_retina(r)
      I = senses.inject(r, arena.heat()); senses.dopamine(I, last_reward)
      for _ in range(20): spikes = lif.step(I)          # 20 x 1 ms
      forward, turn = motor.decode(lif)
      state, events = arena.step(forward, turn); last_reward = arena.reward(events)
      learner.update(lif, last_reward)
  log CSV: contact_rate, mean_time_to_contact, track_fraction, DN rates, |dw|
  checkpoint W to checkpoints/ every generation; promote stage when target met over 200 episodes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque

import torch

from brain.lif import LIF, load_brain, pick_device
from brain.motor import Motor
from brain.retina import Camera, Retina
from brain.retinotopy import column_assignment
from brain.explore import Exploratory
from brain.senses import Senses
from env.arena import Arena

DN_KEYS = ["DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF"]
VIZ_KEYS = ["LC10a_L", "LC10a_R", "LC11_L", "LC11_R", "LC12_L", "LC12_R", "LC15_L", "LC15_R", "LC4_L", "LC4_R",
            "LPLC2_L", "LPLC2_R", "THERMO_L", "THERMO_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09",
            "GF", "PAM", "PPL1", "KC", "MBON"]
STAGE_TARGET = {"A": 0.90, "B": 0.70, "C": 0.70}
NEXT_STAGE = {"A": "B", "B": "C", "C": "C"}


def build(B: int, gain: float = 0.05, device: str | torch.device | None = None, engine: str = "auto",
          stage: str = "A", brain_path: str = "data/brain.npz", seed: int = 0,
          episode_s: float = 20.0, k_t: float = 0.02, k_f: float = 0.3, forward_source: str = "dn_all",
          retinotopy: str = "rank", anterior_sign: int = 1, explore: bool = False, amps: dict | None = None,
          dodge: bool = False, checkpoint: str | None = None, dtype: torch.dtype = torch.float32) -> dict:
    device = pick_device(device)
    torch.manual_seed(seed)
    d, N, groups = load_brain(brain_path)
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, B,
              device=device, engine=engine, g=gain, dtype=dtype)
    if checkpoint:
        # a checkpoint carries the gain / motor gains / senses amps it was trained with; they win over CLI defaults
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        lif.set_w(ck["w"].to(device))
        if "gain" in ck:
            gain = float(ck["gain"]); lif.set_gain(gain)
        k_t, k_f = ck.get("k_t", k_t), ck.get("k_f", k_f)
        forward_source = ck.get("forward_source", "dna01_dnp09")   # checkpoints before 2026-09-18 used the old read-out
        retinotopy = ck.get("retinotopy", "index")                  # ... and the bodyId-order column spread
        explore = bool(ck.get("explore", explore))
        amps = {**(ck.get("amps") or {}), **(amps or {})} if ck.get("amps") else amps
        print(f"loaded {checkpoint} (run {ck.get('run')}, gen {ck.get('gen')}, stage {ck.get('stage')}, gain {gain}, "
              f"k_f {k_f}, forward {forward_source})")
    col_of = None
    if retinotopy != "index" and os.path.exists("data/lc_columns.json"):
        col_of = column_assignment(groups, d["ids"], mode=retinotopy, anterior_sign=anterior_sign, device=device)
    else:
        retinotopy = "index"
    groups = {k: v.to(device) for k, v in groups.items()}
    retina = Retina(Camera(), device=device)
    senses = Senses(groups, N, device=device, col_of=col_of, **(amps or {}))
    motor = Motor(groups, k_t=k_t, k_f=k_f, dodge=dodge, device=device, forward_source=forward_source)
    arena = Arena(B, device=device, episode_s=episode_s)
    arena.reset(seed=seed, stage=stage)
    explorer = Exploratory(groups, N, B, device=device, dt=arena.dt) if explore else None   # internal drive, not sensory
    return dict(lif=lif, groups=groups, retina=retina, senses=senses, motor=motor, arena=arena,
                device=device, N=N, B=B, stage=stage, gain=gain, retinotopy=retinotopy, anterior_sign=anterior_sign,
                explorer=explorer, explore=explore)


@torch.no_grad()
def run_episode(parts: dict, seconds: float | None = None, learner: object | None = None,
                log_bearing: bool = False, seed: int | None = None, stage: str | None = None,
                substeps: int = 20, gain_b: torch.Tensor | None = None, record: str | None = None,
                progress: bool = False, freeze_humans: bool = False) -> dict:
    lif, retina, senses, motor, arena = (parts[k] for k in ("lif", "retina", "senses", "motor", "arena"))
    groups, B = parts["groups"], parts["B"]
    arena.reset(seed=seed, stage=stage)
    lif.reset(); retina.reset()
    explorer = parts.get("explorer")
    if explorer is not None:
        explorer.reset()
    lif.set_gain_b(arena.gain if gain_b is None else gain_b)
    if freeze_humans:
        arena.hspeed.zero_()
    if learner is not None:
        learner.reset_episode()
    T = int(round((seconds or arena.T) / arena.dt))
    last_reward = torch.zeros(B, device=lif.device)
    contact_now = torch.zeros(B, dtype=torch.bool, device=lif.device)
    ret = torch.zeros(B, device=lif.device)
    dist0, bear0 = arena.nearest()
    dn_acc = {k: 0.0 for k in DN_KEYS}
    grp_keys = [k for k in groups if len(groups[k])]
    grp_acc = {k: 0.0 for k in grp_keys}; grp_peak = {k: 0.0 for k in grp_keys}
    fwd_acc = turn_acc = turn_signed = 0.0
    back_steps = torch.zeros(B, device=lif.device)      # steps with back/side contact
    front_steps = torch.zeros(B, device=lif.device)     # steps with front contact (first or sustained)
    t_2s = int(round(2.0 / arena.dt))
    bear_2s, dist_2s = bear0.clone(), dist0.clone()
    frames = []
    viz_idx = None
    if record is not None:
        gen = torch.Generator().manual_seed(0)
        viz_idx = torch.randperm(parts["N"], generator=gen)[:512].to(lif.device)   # fixed neuron sample
    t0 = time.time()
    for t in range(T):
        r = retina.from_state(*arena.retina_inputs(), dt=arena.dt)
        arena.perturb_retina(r)
        heat = arena.heat()
        I = senses.inject(r, heat, contact=contact_now)
        if explorer is not None:
            explorer.inject(I, r, heat)                       # exploratory state, internal drive, not sensory
        senses.dopamine(I, last_reward, pun_gain=parts.get("pun_gain", 5.0))
        spk_sum = None
        for _ in range(substeps):
            spikes = lif.step(I)
            if learner is not None:
                learner.on_substep(lif)
            if record is not None:
                s0 = spikes[0, viz_idx].float()
                spk_sum = s0 if spk_sum is None else spk_sum + s0
        forward, turn = motor.decode(lif)
        state, events = arena.step(forward, turn)
        last_reward = arena.reward(events)
        contact_now = events["contact_now"]
        ret += last_reward
        back_steps += events["back_or_side_contact"].float()
        front_steps += (events["front_contact"] | events["sustained"]).float()
        if t + 1 == t_2s:
            dist_2s, bear_2s = arena.nearest()
        if learner is not None:
            learner.update(lif, last_reward)
        for k in DN_KEYS:
            dn_acc[k] += lif.rates(groups[k]).mean().item() / T
        for k in grp_keys:
            v = lif.rates(groups[k]).mean().item(); grp_acc[k] += v / T; grp_peak[k] = max(grp_peak[k], v)
        fwd_acc += forward.mean().item() / T
        turn_acc += turn.abs().mean().item() / T
        turn_signed += turn.mean().item() / T
        if record is not None:
            frames.append(dict(
                t=round(t * arena.dt, 3), rxy=state["rxy"][0].tolist(), ryaw=round(state["ryaw"][0].item(), 4),
                hxy=state["hxy"][0][state["hvalid"][0]].tolist(), hr=state["hr"][0][state["hvalid"][0]].tolist(),
                forward=round(forward[0].item(), 3), turn=round(turn[0].item(), 3),
                reward=round(last_reward[0].item(), 3), heat=[round(x, 3) for x in arena.heat_h[0].tolist()],
                pres=[round(x, 3) for x in r["pres"][0].tolist()], size=[round(x, 3) for x in r["size"][0].tolist()],
                mot=[round(x, 3) for x in r["mot"][0].tolist()],
                loom=[round(r["loom_L"][0].item(), 3), round(r["loom_R"][0].item(), 3)],
                rates={k: round(lif.rates(groups[k])[0].item(), 1) for k in VIZ_KEYS},
                spikes=spk_sum.nonzero().flatten().tolist()))
        if progress and (t + 1) % 100 == 0:
            print(f"    t={t + 1}/{T} {(time.time() - t0) / (t + 1) * 1000:.0f} ms/step "
                  f"contact {arena.contacted.float().mean():.2f}", flush=True)
    if learner is not None:
        learner.end_episode(lif, ret)
    dist1, bear1 = arena.nearest()
    stats = arena.episode_stats()
    stats.update(return_mean=ret.mean().item(), dn_rates={k: round(v, 1) for k, v in dn_acc.items()},
                 forward_mean=fwd_acc, turn_abs_mean=turn_acc, turn_mean=turn_signed, steps=T,
                 sec_per_gen=time.time() - t0, bearing_abs_start=bear0, bearing_abs_end=bear1,
                 contacted=arena.contacted.clone(), dist_start=dist0, dist_end=dist1,
                 bearing_abs_2s=bear_2s, dist_2s=dist_2s,
                 back_contact_rate=(back_steps > 0).float().mean().item(),      # fraction of envs with any back/side touch
                 sustained_fraction=(front_steps / T).mean().item(),           # fraction of steps in front contact
                 advance_2s=((dist_2s < dist0 - 0.05).float().mean().item()),  # dist to nearest human fell >0.05 m by 2 s
                 first_contact_t=arena.first_contact_t.clone(),
                 group_rates={k: round(v, 1) for k, v in grp_acc.items()}, group_peaks={k: round(v, 1) for k, v in grp_peak.items()})
    if learner is not None:
        stats.update(learner.stats())
    if explorer is not None:
        stats.update(explorer.stats(T))
    if record is not None:
        meta = dict(N=parts["N"], viz_neurons=viz_idx.tolist(), arena_L=arena.L[0].item(), dt=arena.dt,
                    rover_lw=[2 * arena.hl, 2 * arena.hw], stage=arena.stage, gain=parts["gain"])
        with open(record, "w") as f:
            json.dump(dict(meta=meta, frames=frames), f)
        print(f"recorded {len(frames)} frames -> {record}")
    return stats


def make_learner(kind: str, parts: dict, args: object) -> object | None:
    if kind == "none":
        return None
    if kind == "three_factor":
        from learn.three_factor import ThreeFactor
        return ThreeFactor(parts["lif"], parts["groups"], plastic=args.plastic, eta=args.eta,
                           r_target=args.r_target, eta_h=args.eta_h)
    if kind == "es":
        from learn.es import ES
        return ES(parts["lif"], parts["groups"], plastic=args.plastic, sigma=args.sigma, top_frac=args.top_frac)
    raise ValueError(kind)


def parse_range(spec: str) -> list[float]:
    a, b, n = spec.split(":")
    a, b, n = float(a), float(b), int(n)
    if n == 1:
        return [a]
    if a > 0 and b / a > 20:   # log spaced for wide ranges
        return [a * (b / a) ** (i / (n - 1)) for i in range(n)]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def sweep(args: object) -> list[dict]:
    values = parse_range(args.sweep[1])
    what = args.sweep[0]
    parts = build(args.envs, gain=args.gain, device=args.device, engine=args.engine, stage=args.stage,
                  seed=args.seed, episode_s=args.seconds or 20.0, k_t=args.k_t, k_f=args.k_f, amps=args.amps, retinotopy=args.retinotopy, anterior_sign=args.anterior_sign, explore=args.explore)
    rows = []
    print(f"sweep {what} over {[round(v, 4) for v in values]} (stage {args.stage}, {args.envs} envs)")
    if what == "gain":
        per = args.envs // len(values)
        assert per >= 1, "need envs >= number of sweep values"
        gain_b = torch.tensor([v / args.gain for v in values]).repeat_interleave(per)
        gain_b = torch.cat([gain_b, gain_b[-1:].expand(args.envs - gain_b.numel())])
        st = run_episode(parts, seconds=args.seconds, seed=args.seed, gain_b=gain_b.to(parts["device"]))
        for i, v in enumerate(values):
            sl = slice(i * per, (i + 1) * per)
            c = st["contacted"][sl].float().mean().item()
            imp = (st["bearing_abs_end"][sl] < st["bearing_abs_start"][sl] - 0.02).float().mean().item()
            rows.append(dict(value=v, contact_rate=c, bearing_improved=imp,
                             dist_change=(st["dist_end"][sl] - st["dist_start"][sl]).mean().item()))
    else:
        for v in values:
            if what in ("k_t", "k_f"):
                setattr(parts["motor"], what, v)
            elif what.startswith("amp_"):
                parts["senses"].amp[what[4:]] = v
            else:
                raise ValueError(what)
            st = run_episode(parts, seconds=args.seconds, seed=args.seed)
            imp = (st["bearing_abs_end"] < st["bearing_abs_start"] - 0.02).float().mean().item()
            rows.append(dict(value=v, contact_rate=st["contact_rate"], bearing_improved=imp,
                             dist_change=(st["dist_end"] - st["dist_start"]).mean().item(),
                             dn=st["dn_rates"], forward=st["forward_mean"]))
    for r in rows:
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/sweep_{what}_{time.strftime('%m%d_%H%M')}.json", "w") as f:
        json.dump(rows, f, indent=1)
    return rows


def train(args: object) -> None:
    run = args.run or f"{args.stage}_{args.learn}_{time.strftime('%m%d_%H%M')}"
    os.makedirs("logs", exist_ok=True); os.makedirs("checkpoints", exist_ok=True)
    parts = build(args.envs, gain=args.gain, device=args.device, engine=args.engine, stage=args.stage,
                  seed=args.seed, episode_s=args.seconds or 20.0, k_t=args.k_t, k_f=args.k_f,
                  checkpoint=args.checkpoint, amps=args.amps, retinotopy=args.retinotopy, anterior_sign=args.anterior_sign, explore=args.explore)
    parts["pun_gain"] = args.pun_gain
    learner = make_learner(args.learn, parts, args)
    stage = args.stage
    window = deque(maxlen=200)
    csv_path = f"logs/{run}.csv"
    fields = ["gen", "stage", "wall_s", "sec_per_gen", "contact_rate", "window_contact_rate", "mean_time_to_contact",
              "track_fraction", "acquire_rate", "return_mean", "forward_mean", "turn_abs_mean", "dw_abs", "dopamine",
              *DN_KEYS]
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    wr = csv.DictWriter(f, fieldnames=fields)
    if new:
        wr.writeheader()
    t_start = time.time()
    print(f"run {run}: stage {stage} envs {args.envs} gain {args.gain} learn {args.learn} "
          f"device {parts['device']} engine {parts['lif'].engine} budget {args.minutes} min", flush=True)
    gen = 0
    while (time.time() - t_start) / 60 < args.minutes and gen < args.max_gens:
        st = run_episode(parts, seconds=args.seconds, learner=learner, seed=args.seed + gen, stage=stage,
                         progress=args.progress)
        window.extend(st["contacted"].tolist())
        wcr = sum(window) / len(window)
        row = dict(gen=gen, stage=stage, wall_s=round(time.time() - t_start), sec_per_gen=round(st["sec_per_gen"], 1),
                   contact_rate=round(st["contact_rate"], 3), window_contact_rate=round(wcr, 3),
                   mean_time_to_contact=round(st["mean_time_to_contact"], 2), track_fraction=round(st["track_fraction"], 3),
                   acquire_rate=round(st["acquire_rate"], 3), return_mean=round(st["return_mean"], 3),
                   forward_mean=round(st["forward_mean"], 3), turn_abs_mean=round(st["turn_abs_mean"], 3),
                   dw_abs=st.get("dw_abs", 0.0), dopamine=st.get("dopamine", 0.0), **st["dn_rates"])
        wr.writerow(row); f.flush()
        print(f"gen {gen:4d} {stage} contact {st['contact_rate']:.2f} (win {wcr:.2f}) ttc {st['mean_time_to_contact']:.1f}s "
              f"track {st['track_fraction']:.2f} ret {st['return_mean']:.2f} fwd {st['forward_mean']:.2f} "
              f"dw {st.get('dw_abs', 0):.2e} D {st.get('dopamine', 0):.3f} DN {st['dn_rates']} [{st['sec_per_gen']:.0f}s]", flush=True)
        w_full = learner.full_w(parts["lif"]) if learner is not None else parts["lif"].w
        ck = dict(w=w_full.detach().cpu(), gen=gen, stage=stage, gain=args.gain, run=run,
                  k_t=args.k_t, k_f=args.k_f, learn=args.learn, amps=args.amps,
                  forward_source=parts["motor"].forward_source, retinotopy=parts["retinotopy"],
                  pun_gain=args.pun_gain, eta=args.eta, r_target=args.r_target, eta_h=args.eta_h, explore=args.explore)
        torch.save(ck, f"checkpoints/{run}_latest.pt")
        if gen % args.ckpt_every == 0:
            torch.save(ck, f"checkpoints/{run}_gen{gen:04d}.pt")
        if len(window) >= 200 and wcr >= STAGE_TARGET[stage] and stage != "C" and not args.no_promote:
            print(f"*** stage {stage} target {STAGE_TARGET[stage]} met over {len(window)} episodes -> stage {NEXT_STAGE[stage]}", flush=True)
            stage = NEXT_STAGE[stage]; window.clear()
        gen += 1
    f.close()
    print(f"done: {gen} generations, final stage {stage}, log {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B", "C"])
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--learn", default="none", choices=["none", "three_factor", "es"])
    ap.add_argument("--plastic", default="kc_mbon", choices=["kc_mbon", "dn_in", "lc_dn", "both"])
    ap.add_argument("--eta", type=float, default=3e-4)
    ap.add_argument("--pun_gain", type=float, default=5.0, help="PPL1 (punishment) channel weight vs PAM")
    ap.add_argument("--r_target", type=float, default=150.0, help="synaptic scaling target rate, Hz (three_factor)")
    ap.add_argument("--eta_h", type=float, default=0.02, help="synaptic scaling rate (0 = off)")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--top-frac", type=float, default=0.2)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-gens", type=int, default=10 ** 9)
    ap.add_argument("--seconds", type=float, default=None, help="episode length (default 20)")
    ap.add_argument("--k_t", type=float, default=0.02)
    ap.add_argument("--k_f", type=float, default=0.3)
    ap.add_argument("--amp_small", type=float, default=1.2)
    ap.add_argument("--amp_track", type=float, default=20.0)
    ap.add_argument("--amp_loom", type=float, default=2.0)
    ap.add_argument("--amp_heat", type=float, default=0.8)
    ap.add_argument("--amp_tonic", type=float, default=0.0, help="FALLBACK tonic walking current into DNa01 (0 = off)")
    ap.add_argument("--retinotopy", default="rank", choices=["rank", "angle", "index"])
    ap.add_argument("--no-promote", action="store_true", help="stay in the starting stage (demo-brain runs)")
    ap.add_argument("--explore", action="store_true", help="exploratory internal state when no human is in view (brain/explore.py)")
    ap.add_argument("--anterior-sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--run", default=None)
    ap.add_argument("--sweep", nargs=2, metavar=("WHAT", "A:B:N"), default=None)
    ap.add_argument("--record", default=None, help="write env-0 replay JSON and exit")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    args.amps = dict(amp_small=args.amp_small, amp_track=args.amp_track, amp_loom=args.amp_loom, amp_heat=args.amp_heat,
                     amp_tonic=args.amp_tonic)
    if args.record:
        parts = build(args.envs, gain=args.gain, device=args.device, engine=args.engine, stage=args.stage,
                      seed=args.seed, episode_s=args.seconds or 20.0, k_t=args.k_t, k_f=args.k_f,
                      checkpoint=args.checkpoint, amps=args.amps, retinotopy=args.retinotopy, anterior_sign=args.anterior_sign, explore=args.explore)
        st = run_episode(parts, seconds=args.seconds, seed=args.seed, record=args.record)
        print({k: v for k, v in st.items() if not torch.is_tensor(v)})
    elif args.sweep:
        sweep(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
