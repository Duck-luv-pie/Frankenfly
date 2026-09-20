"""
ablations.py -- lesion controls: does the behaviour come from the wiring?

    python scripts/ablations.py [--brain data/brain_v2.npz] [--envs 256] [--seconds 2] [--device cpu]

Untrained brain, stage A, humans frozen (the M3 protocol). Rows:
  baseline; LC10a lesioned (both sides); DNa02_L lesioned; LC4+LPLC2 lesioned; degree-preserving shuffle.
Lesion = lif.lesion(group, mode="both") (incoming and outgoing synapses zeroed, i.e. the neurons are
removed), restored after each row. Shuffle = random permutation of the POST index of every edge: every
neuron keeps its exact out-degree and in-degree and every synapse keeps its sign/weight, only targets move.
Looming row: human starts 3 m dead ahead walking straight at the rover at 1.4 m/s for 2 s; reports GF rate.
Metrics: turn-toward = fraction of envs whose |bearing| to the human fell by >0.02 rad over the episode;
advance = fraction whose distance fell by >0.05 m; turn_mean = signed mean motor turn (+ = right).
Writes logs/ablations.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402
from train import build, run_episode  # noqa: E402


def frozen_stage_a(parts: dict, seconds: float, seed: int) -> dict:
    st = run_episode(parts, seconds=seconds, seed=seed, stage="A", freeze_humans=True)
    b0, b1, d0, d1 = st["bearing_abs_start"], st["bearing_abs_end"], st["dist_start"], st["dist_end"]
    return dict(turn_toward=(b1 < b0 - 0.02).float().mean().item(), advance=(d1 < d0 - 0.05).float().mean().item(),
                turn_mean=st["turn_mean"], DNa02_L=st["dn_rates"]["DNa02_L"], DNa02_R=st["dn_rates"]["DNa02_R"],
                GF=st["dn_rates"]["GF"], forward=st["forward_mean"])


def looming(parts: dict, seconds: float, seed: int) -> float:
    """Human 3 m dead ahead, walking straight at the rover at 1.4 m/s. Returns GF episode-mean rate."""
    arena = parts["arena"]
    lif, retina, senses, motor, groups = (parts[k] for k in ("lif", "retina", "senses", "motor", "groups"))
    arena.reset(seed=seed, stage="A"); lif.reset(); retina.reset(); lif.set_gain_b(arena.gain)
    ang = arena.ryaw
    arena.hxy[:, 0] = arena.rxy + 3.0 * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
    arena.hhead[:, 0] = (ang + torch.pi + torch.pi) % (2 * torch.pi) - torch.pi
    arena.hspeed[:, 0] = 1.4; arena.htimer[:] = 1e9; arena.hpaused.zero_()
    T = int(round(seconds / arena.dt)); gf = 0.0
    for t in range(T):
        r = retina.from_state(*arena.retina_inputs(), dt=arena.dt)
        I = senses.inject(r, arena.heat())
        for _ in range(20):
            lif.step(I)
        fwd, turn = motor.decode(lif)
        arena.step(torch.zeros_like(fwd), torch.zeros_like(turn))      # rover holds still, human approaches
        gf += lif.rates(groups["GF"]).mean().item() / T
    return gf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain_v2.npz")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--amp_tonic", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--out", default="logs/ablations.csv")
    ap.add_argument("--checkpoint", default=None, help="evaluate a trained checkpoint instead of the untrained brain")
    a = ap.parse_args()
    amps = {"amp_tonic": a.amp_tonic} if a.amp_tonic > 0 else None
    parts = build(a.envs, gain=a.gain, device=a.device, engine=a.engine, stage="A", brain_path=a.brain,
                  seed=a.seed, episode_s=a.seconds, amps=amps, checkpoint=a.checkpoint)
    lif, g = parts["lif"], parts["groups"]
    rows = []

    def row(name, **kw):
        t0 = time.time()
        m = frozen_stage_a(parts, a.seconds, a.seed)
        m.update(kw); m["condition"] = name; m["wall_s"] = round(time.time() - t0)
        rows.append(m)
        print(f"{name:26s} turn-toward {m['turn_toward']:.2f}  advance {m['advance']:.2f}  turn_mean {m['turn_mean']:+.3f}  "
              f"DNa02 L/R {m['DNa02_L']:.1f}/{m['DNa02_R']:.1f}  GF {m['GF']:.1f}  [{m['wall_s']}s]", flush=True)

    print(f"brain {a.brain} checkpoint {a.checkpoint} envs {a.envs} seed {a.seed} {a.seconds}s stage A humans frozen gain {a.gain} amp_tonic {a.amp_tonic}")
    row("baseline")
    snap = lif.lesion(torch.cat([g["LC10a_L"], g["LC10a_R"]])); row("lesion LC10a (L+R)"); lif.restore(snap)
    snap = lif.lesion(g["DNa02_L"]); row("lesion DNa02_L"); lif.restore(snap)
    gf_base = looming(parts, a.seconds, a.seed)
    snap = lif.lesion(torch.cat([g["LC4_L"], g["LC4_R"], g["LPLC2_L"], g["LPLC2_R"]]))
    row("lesion LC4+LPLC2"); gf_les = looming(parts, a.seconds, a.seed); lif.restore(snap)
    rows[-1]["GF_looming"] = round(gf_les, 1); rows[0]["GF_looming"] = round(gf_base, 1)
    print(f"looming (human walks at rover from 3 m, 1.4 m/s): GF baseline {gf_base:.1f} Hz, LC4+LPLC2 lesioned {gf_les:.1f} Hz")
    # degree-preserving shuffle: permute post index of every edge, weights stay with their pre neuron
    d, N, _ = load_brain(a.brain)
    Wi = torch.stack([lif.post_idx.cpu(), lif.pre_idx.cpu()]); Wv = lif.w.detach().cpu()   # current (trained) weights
    gen = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(Wi.shape[1], generator=gen)
    Wi_sh = torch.stack([Wi[0][perm], Wi[1]])
    parts["lif"] = LIF(Wi_sh, Wv, N, a.envs, device=parts["device"], engine=a.engine, g=lif.g)
    row("shuffled (degree-preserving)")
    os.makedirs("logs", exist_ok=True)
    fields = ["condition", "turn_toward", "advance", "turn_mean", "DNa02_L", "DNa02_R", "GF", "GF_looming", "forward", "wall_s"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k, "")) for k in fields})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
