"""
evaluate.py -- run one fixed eval episode per requested stage for a brain/checkpoint and log the result.

    python scripts/evaluate.py --checkpoint checkpoints/tfA_both_latest.pt --stages A,B,C
    python scripts/evaluate.py --brain data/brain.npz --name brain_untrained   # untrained baseline

Builds once (stage = the first requested stage; --checkpoint, if given, wins over --gain and every
other trained-with parameter -- see train.build). Then for each requested stage i, runs one
no-learner episode with seed = --seed + i and appends a CSV row to --out and to logs/eval_all.csv.
No training, no checkpoint writes.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import torch  # noqa: F401  (imported for parity with train.py / to fail fast if torch is missing)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build, run_episode  # noqa: E402

FIELDS = ["name", "brain", "checkpoint", "stage", "envs", "seed", "seconds", "gain", "k_t", "k_f",
          "amp_track", "amp_tonic", "front_contact_rate", "mean_time_to_contact", "sustained_fraction",
          "back_contact_rate", "mean_abs_bearing_2s", "advance_2s", "DNa02_L", "DNa02_R", "DNp09", "GF",
          "track_fraction", "return_mean", "acquire_rate", "mean_time_to_first_sight", "explore", "wall_s"]


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _append_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)


def evaluate(brain: str = "data/brain.npz", checkpoint: str | None = None, name: str | None = None,
             envs: int = 256, seed: int = 1000, seconds: float = 25.0,
             stages: str = "A,B,C", device: str = "cpu", engine: str = "event", out: str | None = None,
             gain: float = 0.05, amp_tonic: float = 0.0, explore: bool = False) -> list[dict]:
    stage_list = [s.strip() for s in stages.split(",") if s.strip()]
    assert stage_list, "no stages requested"
    if not name:
        name = _stem(checkpoint) if checkpoint else f"{_stem(brain)}_untrained"
    if not out:
        out = f"logs/eval_{name}.csv"
    amps = {"amp_tonic": amp_tonic} if amp_tonic > 0 else None

    parts = build(envs, gain=gain, device=device, engine=engine, stage=stage_list[0], brain_path=brain,
                  seed=seed, episode_s=seconds, checkpoint=checkpoint, amps=amps, explore=explore)

    # checkpoints carry their own gain / k_t / k_f / amps and build() restores them over the CLI
    # defaults above, so the values actually in effect have to be read back off the built parts.
    gain_v = float(parts["lif"].g)
    k_t_v = float(parts["motor"].k_t)
    k_f_v = float(parts["motor"].k_f)
    amp_track_v = float(parts["senses"].amp["track"])
    amp_tonic_v = float(parts["senses"].amp["tonic"])

    print(f"{name}: brain={brain} checkpoint={checkpoint or '-'} envs={envs} seed={seed} seconds={seconds} "
          f"gain={gain_v:.4f} k_t={k_t_v:.4f} k_f={k_f_v:.4f} amp_track={amp_track_v:.2f} "
          f"amp_tonic={amp_tonic_v:.2f} device={parts['device']} engine={parts['lif'].engine} "
          f"stages={stage_list}")
    header = (f"{'stage':>5s} {'front':>6s} {'ttc':>6s} {'sustn':>6s} {'back':>6s} {'bear2s':>7s} "
              f"{'adv2s':>6s} {'DNa02_L':>8s} {'DNa02_R':>8s} {'DNp09':>7s} {'GF':>6s} {'track':>6s} "
              f"{'ret':>7s} {'wall_s':>7s}")
    print(header)

    rows = []
    for i, stage in enumerate(stage_list):
        seed_i = seed + i
        st = run_episode(parts, seconds=seconds, seed=seed_i, stage=stage)
        dn = st["dn_rates"]
        row = dict(
            name=name, brain=brain, checkpoint=checkpoint or "", stage=stage, envs=envs, seed=seed_i,
            seconds=round(float(seconds), 3), gain=round(gain_v, 3), k_t=round(k_t_v, 3),
            k_f=round(k_f_v, 3), amp_track=round(amp_track_v, 3), amp_tonic=round(amp_tonic_v, 3),
            front_contact_rate=round(st["contact_rate"], 3),
            mean_time_to_contact=round(st["mean_time_to_contact"], 3),
            sustained_fraction=round(st["sustained_fraction"], 3),
            back_contact_rate=round(st["back_contact_rate"], 3),
            mean_abs_bearing_2s=round(st["bearing_abs_2s"].mean().item(), 3),
            advance_2s=round(st["advance_2s"], 3),
            DNa02_L=round(dn["DNa02_L"], 1), DNa02_R=round(dn["DNa02_R"], 1),
            DNp09=round(dn["DNp09"], 1), GF=round(dn["GF"], 1),
            track_fraction=round(st["track_fraction"], 3),
            return_mean=round(st["return_mean"], 3),
            acquire_rate=round(st["acquire_rate"], 3),
            mean_time_to_first_sight=round(st["mean_time_to_first_sight"], 3),
            explore=int(parts.get("explore", False)),
            wall_s=round(st["sec_per_gen"], 3),
        )
        rows.append(row)
        print(f"{stage:>5s} {row['front_contact_rate']:6.3f} {row['mean_time_to_contact']:6.2f} "
              f"{row['sustained_fraction']:6.3f} {row['back_contact_rate']:6.3f} "
              f"{row['mean_abs_bearing_2s']:7.3f} {row['advance_2s']:6.3f} {row['DNa02_L']:8.1f} "
              f"{row['DNa02_R']:8.1f} {row['DNp09']:7.1f} {row['GF']:6.1f} {row['track_fraction']:6.3f} "
              f"{row['return_mean']:7.3f} {row['wall_s']:7.2f}")

    _append_csv(out, rows)
    _append_csv("logs/eval_all.csv", rows)
    print(f"wrote {len(rows)} row(s) -> {out} (+ logs/eval_all.csv)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--name", default=None, help="default: checkpoint stem, else <brain stem>_untrained")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--stages", default="A,B,C")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--out", default=None, help="default: logs/eval_<name>.csv")
    ap.add_argument("--gain", type=float, default=0.05, help="only used when --checkpoint is not given")
    ap.add_argument("--amp_tonic", type=float, default=0.0)
    ap.add_argument("--explore", action="store_true", help="exploratory internal state (brain/explore.py)")
    args = ap.parse_args()
    evaluate(brain=args.brain, checkpoint=args.checkpoint, name=args.name, envs=args.envs, seed=args.seed,
              seconds=args.seconds, stages=args.stages, device=args.device, engine=args.engine,
              out=args.out, gain=args.gain, amp_tonic=args.amp_tonic, explore=args.explore)
