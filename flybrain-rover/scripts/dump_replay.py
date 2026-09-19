"""
dump_replay.py -- one stage A episode as JSON for Ducks's Three.js viz (format: replay/FORMAT.md).

    python scripts/dump_replay.py --seed 7                                   # untrained brain
    python scripts/dump_replay.py --seed 7 --checkpoint checkpoints/X_latest.pt --tag trained
    python scripts/dump_replay.py --seed 7 --lesion LC10a_L,LC10a_R --tag lesioned
    python scripts/dump_replay.py --seed 7 --amp_tonic 0.6 --tag tonic         # FALLBACK forward drive

Writes replay/episode_<seed>[_<tag>].json. Env 0 of a small batch is recorded at 50 Hz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build, run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--amp_tonic", type=float, default=0.0)
    ap.add_argument("--lesion", default=None, help="comma-separated group keys to lesion (in+out synapses zeroed)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    a = ap.parse_args()
    amps = {"amp_tonic": a.amp_tonic} if a.amp_tonic > 0 else None
    parts = build(a.envs, gain=a.gain, device=a.device, engine=a.engine, stage=a.stage, brain_path=a.brain,
                  seed=a.seed, episode_s=a.seconds, checkpoint=a.checkpoint, amps=amps)
    lesioned = []
    if a.lesion:
        keys = [k.strip() for k in a.lesion.split(",") if k.strip()]
        parts["lif"].lesion(torch.cat([parts["groups"][k] for k in keys]), mode="both")
        lesioned = keys
    tag = a.tag or ("trained" if a.checkpoint else ("lesioned" if lesioned else "untrained"))
    out = f"replay/episode_{a.seed}_{tag}.json"
    os.makedirs("replay", exist_ok=True)
    st = run_episode(parts, seconds=a.seconds, seed=a.seed, stage=a.stage, record=out)
    with open(out) as f:
        d = json.load(f)
    d["meta"].update(brain=a.brain, checkpoint=a.checkpoint, lesion=lesioned, amp_tonic=a.amp_tonic, seed=a.seed,
                     stage=a.stage, seconds=a.seconds, hz=50, env_index=0,
                     summary=dict(contact_rate=st["contact_rate"], forward_mean=round(st["forward_mean"], 3),
                                  dn_rates=st["dn_rates"]))
    with open(out, "w") as f:
        json.dump(d, f)
    print(f"{out}: {len(d['frames'])} frames, contact {st['contact_rate']:.2f}, fwd {st['forward_mean']:.2f}, DN {st['dn_rates']}")


if __name__ == "__main__":
    main()
