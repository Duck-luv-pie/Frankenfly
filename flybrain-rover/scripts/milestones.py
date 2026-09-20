"""
milestones.py -- pass/fail checks for M1, M2, M3 (see CLAUDE.md).

    python scripts/milestones.py m1
    python scripts/milestones.py m2 [--gains 0.275,0.1,0.05,...] [--drive 1.5] [--ms 300]
    python scripts/milestones.py m3 [--envs 64] [--gain 0.05] [--seconds 2]

m2 runs the whole gain sweep as ONE batch (per-env gain multiplier), so it costs one simulation.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402

REPORT = ["LC10a_L", "LC10a_R", "LC11_L", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF",
          "KC", "MBON", "PAM", "PPL1", "THERMO_L"]


def m1(args: object) -> bool:
    d, N, groups = load_brain(args.brain)
    nnz = int(d["W_values"].shape[0])
    ok = 3000 <= N <= 15000 and 300_000 <= nnz <= 3_500_000
    print(f"N={N} nnz={nnz:,}")
    for k, v in groups.items():
        print(f"  {k:10s} {len(v):6d}")
        ok &= len(v) > 0
    print("M1", "PASS" if ok else "FAIL")
    return ok


def m2(args: object, quiet: bool = False) -> tuple[list, list]:
    d, N, groups = load_brain(args.brain)
    gains = [float(x) for x in args.gains.split(",")]
    B = len(gains)
    base = 1.0
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, B,
              device=args.device, engine=args.engine, g=base)
    lif.set_gain_b(gains)
    groups = {k: v.to(lif.device) for k, v in groups.items()}
    I = torch.zeros(B, N, device=lif.device)
    I[:, groups["LC10a_L"]] = args.drive
    t0 = time.time()
    checks_200 = None
    for t in range(args.ms):
        lif.step(I)
        if t == 199:
            checks_200 = (lif.rates(groups["DNa02_L"]) > lif.rates(groups["DNa02_R"])).cpu()
    dt_ms = (time.time() - t0) / args.ms * 1000
    all_groups = {k: v for k, v in groups.items() if len(v)}
    peak = torch.stack([lif.rates(v) for v in all_groups.values()], 1)  # (B, G)
    peak_rate, peak_arg = peak.max(1)
    keys = list(all_groups)
    if not quiet:
        print(f"device {lif.device} engine {lif.engine} N {N} nnz {lif.nnz:,} drive {args.drive} "
              f"{args.ms} ms  ({dt_ms:.1f} ms/step at B={B})")
        print(f"{'gain':>7s} " + " ".join(f"{k:>8s}" for k in REPORT) + "   peak(group)  M2")
    results = []
    for b, g in enumerate(gains):
        row = {k: float(lif.rates(groups[k])[b]) for k in REPORT}
        L, R = row["DNa02_L"], row["DNa02_R"]
        silent = peak_rate[b].item() < 1.0
        seize = peak_rate[b].item() > 200
        ok = (L > R) and not seize and not silent and bool(checks_200[b])
        results.append((g, ok, row))
        if not quiet:
            print(f"{g:7.4f} " + " ".join(f"{row[k]:8.1f}" for k in REPORT) +
                  f"  {peak_rate[b].item():6.0f} {keys[peak_arg[b].item()]:8s} {'PASS' if ok else 'FAIL'}")
    passed = [g for g, ok, _ in results if ok]
    print("M2", f"PASS at gains {passed}" if passed else "FAIL at all gains")
    return passed, results


def _parts(args, stage="A"):
    from train import build  # noqa: E402
    parts = build(B=args.envs, gain=args.gain, device=args.device, engine=args.engine, stage=stage,
                  brain_path=args.brain, seed=args.seed, k_f=args.k_f, k_t=args.k_t,
                  retinotopy=args.retinotopy, anterior_sign=args.anterior_sign)
    parts["senses"].amp.update(small=args.amp_small, track=args.amp_track, loom=args.amp_loom, heat=args.amp_heat,
                               tonic=args.amp_tonic)
    return parts


def center_human(arena, dist=2.0, bearing_deg=0.0):
    """Stage A env, but the (single) human is placed at `dist` m and `bearing_deg` (+ = right), frozen."""
    import torch as _t
    ang = arena.ryaw - math.radians(bearing_deg)
    arena.hxy[:, 0] = arena.rxy + dist * _t.stack([_t.cos(ang), _t.sin(ang)], -1)
    arena.hspeed.zero_()


def m2c(args):
    """Lateral symmetry: human at 0 / -30 / +30 deg, 2 m, 300 ms. Pass = centred |L-R| < 25% of (L+R) and
    the ipsilateral DNa02 more than double the other side at +-30 deg, with the turn sign right."""
    parts = _parts(args)
    lif, retina, senses, motor, arena, groups = (parts[k] for k in ("lif", "retina", "senses", "motor", "arena", "groups"))
    res = {}
    for name, bdeg in (("centre", 0.0), ("left30", -30.0), ("right30", 30.0)):
        arena.reset(seed=args.seed, stage="A"); lif.reset(); retina.reset(); lif.set_gain_b(torch.ones(args.envs, device=lif.device))
        center_human(arena, 2.0, bdeg)
        for _ in range(int(round(args.ms / 1000 / arena.dt))):
            r = retina.from_state(*arena.retina_inputs(), dt=arena.dt)
            I = senses.inject(r, torch.zeros(args.envs, 2, device=lif.device))
            for _ in range(20):
                lif.step(I)
        fwd, turn = motor.decode(lif)
        res[name] = dict(L=lif.rates(groups["DNa02_L"]).mean().item(), R=lif.rates(groups["DNa02_R"]).mean().item(),
                         LC_L=lif.rates(groups["LC10a_L"]).mean().item(), LC_R=lif.rates(groups["LC10a_R"]).mean().item(),
                         turn=turn.mean().item(), fwd=fwd.mean().item())
    print(f"brain {args.brain} retinotopy {parts['retinotopy']} anterior_sign {parts['anterior_sign']} envs {args.envs} "
          f"gain {args.gain} amp_track {args.amp_track} {args.ms} ms")
    for k, v in res.items():
        print(f"  {k:8s} LC10a L/R {v['LC_L']:5.1f}/{v['LC_R']:5.1f}  DNa02 L/R {v['L']:6.1f}/{v['R']:6.1f}  turn {v['turn']:+.2f}  fwd {v['fwd']:.2f}")
    c, l, r = res["centre"], res["left30"], res["right30"]
    sym = abs(c["L"] - c["R"]) < 0.25 * max(c["L"] + c["R"], 1e-6)
    lat = l["L"] > 2 * l["R"] and r["R"] > 2 * r["L"] and l["turn"] < 0 < r["turn"]
    print("M2c", "PASS" if (sym and lat) else "FAIL", f"(centre symmetric: {sym}; lateral correct: {lat})")
    return sym and lat


def m3(args: object) -> tuple[bool, bool]:
    from train import run_episode  # noqa: E402
    parts = _parts(args)
    stats = run_episode(parts, seconds=args.seconds, learner=None, log_bearing=True, freeze_humans=True)
    b0, b1 = stats["bearing_abs_start"], stats["bearing_abs_end"]
    d0, d1 = stats["dist_start"], stats["dist_end"]
    improved = (b1 < b0 - 0.02)
    advanced = (d1 < d0 - 0.05)
    frac, adv = improved.float().mean().item(), advanced.float().mean().item()
    print(f"brain {args.brain} envs {args.envs} seed {args.seed} gain {args.gain} amp_track {args.amp_track} amp_tonic {args.amp_tonic} k_f {args.k_f} {args.seconds}s stage A, humans frozen:")
    print(f"  M3  turn-toward: mean|bearing| {b0.mean():.3f} -> {b1.mean():.3f} rad, |bearing| decreased >0.02 rad in {100*frac:.0f}% of envs")
    print(f"  M3b advance:     mean dist {d0.mean():.2f} -> {d1.mean():.2f} m, dist decreased >0.05 m in {100*adv:.0f}% of envs")
    print(f"  DN rates (Hz, episode mean) {stats['dn_rates']}  fwd {stats['forward_mean']:.3f} |turn| {stats['turn_abs_mean']:.2f}")
    print("M3", "PASS" if frac > 0.70 else "FAIL", "(>70% turn toward)")
    print("M3b", "PASS" if adv > 0.50 else "FAIL", "(>50% advance)")
    return frac > 0.70, adv > 0.50


def m2b(args: object) -> bool:
    """Centered human at 2 m vs no human, 300 ms. Pass = DNp09 or DNa01 > 5 Hz with human and < 2 Hz without."""
    parts = _parts(args)
    lif, retina, senses, motor, arena, groups = (parts[k] for k in ("lif", "retina", "senses", "motor", "arena", "groups"))
    out = {}
    for cond in ("human", "none"):
        arena.reset(seed=args.seed, stage="A"); lif.reset(); retina.reset(); lif.set_gain_b(torch.ones(args.envs, device=lif.device))
        center_human(arena, 2.0)
        if cond == "none":
            arena.hvalid.zero_()
        steps = int(round(args.ms / 1000 / arena.dt))
        for _ in range(steps):
            r = retina.from_state(*arena.retina_inputs(), dt=arena.dt)
            I = senses.inject(r, torch.zeros(args.envs, 2, device=lif.device))
            for _ in range(20):
                lif.step(I)
        fwd, turn = motor.decode(lif)
        out[cond] = {k: round(lif.rates(groups[k]).mean().item(), 1) for k in
                     ["LC10a_L", "LC10a_R", "LC4_L", "LC4_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF"]}
        if "DN_ALL" in groups:
            out[cond]["DN_ALL"] = round(lif.rates(groups["DN_ALL"]).mean().item(), 2)
        out[cond]["forward"] = round(fwd.mean().item(), 3)
    print(f"brain {args.brain} envs {args.envs} seed {args.seed} gain {args.gain} amp_track {args.amp_track} amp_loom {args.amp_loom} amp_tonic {args.amp_tonic} k_f {args.k_f}, {args.ms} ms")
    for cond in ("human", "none"):
        print(f"  {cond:6s} " + " ".join(f"{k} {v}" for k, v in out[cond].items()))
    h, n = out["human"], out["none"]
    fwd_on = max(h["DNp09"], h["DNa01_L"], h["DNa01_R"])
    fwd_off = max(n["DNp09"], n["DNa01_L"], n["DNa01_R"])
    ok = fwd_on > 5 and fwd_off < 2
    print("M2b", "PASS" if ok else "FAIL", f"(DNa01/DNp09 {fwd_on} Hz with human, {fwd_off} Hz without)")
    if "DN_ALL" in h:
        okp = h["DN_ALL"] >= 2 * max(n["DN_ALL"], 0.25) and h["forward"] > 0.1
        print("M2b-pop", "PASS" if okp else "FAIL",
              f"(DN_ALL mean {h['DN_ALL']} Hz with human vs {n['DN_ALL']} without; forward {h['forward']} at k_f {args.k_f})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["m1", "m2", "m2b", "m2c", "m3"])
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--engine", default="auto")
    ap.add_argument("--gains", default="0.275,0.15,0.1,0.07,0.05,0.035,0.025,0.015,0.01,0.005")
    ap.add_argument("--drive", type=float, default=1.5)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp_small", type=float, default=1.2)
    ap.add_argument("--amp_track", type=float, default=20.0)
    ap.add_argument("--amp_loom", type=float, default=2.0)
    ap.add_argument("--amp_heat", type=float, default=0.8)
    ap.add_argument("--k_f", type=float, default=0.3)
    ap.add_argument("--k_t", type=float, default=0.02)
    ap.add_argument("--amp_tonic", type=float, default=0.0, help="FALLBACK tonic walking current into DNa01 (0 = off)")
    ap.add_argument("--retinotopy", default="rank", choices=["rank", "angle", "index"])
    ap.add_argument("--anterior-sign", type=int, default=1, choices=[1, -1])
    a = ap.parse_args()
    {"m1": m1, "m2": m2, "m2b": m2b, "m2c": m2c, "m3": m3}[a.which](a)
