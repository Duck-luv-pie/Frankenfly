"""
lock_demo.py -- evaluate a checkpoint on the fixed demo criteria and, if it passes, lock it as the demo brain.

    python scripts/lock_demo.py --checkpoint checkpoints/X_latest.pt [--lock]
    python scripts/lock_demo.py --brain data/brain.npz --lock            # untrained brain as the demo brain

Fixed evaluation set (M6): stage A, 128 envs, 20 s, seed 1000, no learning, CPU event engine.
Pass = front contact >= 0.90 AND sustained fraction >= 0.50 AND mean DNa02 rate <= 160 Hz AND no named
group's episode-mean rate > 200 Hz. Peaks (max over env steps of the batch-mean rate) are reported too.
--lock copies the checkpoint to checkpoints/demo_brain.pt with the evaluation, source path and git commit
stored inside it, and rewrites the "Demo brain" section of DEMO.md with the exact bridge command.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import build, run_episode  # noqa: E402

CRIT = dict(contact=0.90, sustained=0.50, dna02_max=160.0, group_max=200.0)


def evaluate_demo(brain: str, checkpoint: str | None, envs: int, seconds: float, seed: int, device: str, engine: str,
                  explore: bool = False) -> dict:
    parts = build(envs, device=device, engine=engine, stage="A", brain_path=brain, seed=seed, episode_s=seconds,
                  checkpoint=checkpoint, explore=explore)
    st = run_episode(parts, seconds=seconds, seed=seed, stage="A")
    g = st["group_rates"]; pk = st["group_peaks"]
    dna02 = 0.5 * (g["DNa02_L"] + g["DNa02_R"])
    worst = max(g, key=g.get)
    res = dict(brain=brain, checkpoint=checkpoint, envs=envs, seconds=seconds, seed=seed, stage="A",
               front_contact=round(st["contact_rate"], 3), sustained=round(st["sustained_fraction"], 3),
               time_to_contact=round(st["mean_time_to_contact"], 2), back_contact=round(st["back_contact_rate"], 3),
               advance_2s=round(st["advance_2s"], 3), track=round(st["track_fraction"], 3),
               dna02_mean=round(dna02, 1), max_group=worst, max_group_rate=g[worst], max_group_peak=max(pk.values()),
               group_rates=g, group_peaks=pk,
               gain=parts["lif"].g, k_f=parts["motor"].k_f, k_t=parts["motor"].k_t, forward_source=parts["motor"].forward_source,
               retinotopy=parts["retinotopy"], amps=dict(parts["senses"].amp), explore=explore)
    res["pass"] = bool(res["front_contact"] >= CRIT["contact"] and res["sustained"] >= CRIT["sustained"]
                       and res["dna02_mean"] <= CRIT["dna02_max"] and res["max_group_rate"] <= CRIT["group_max"])
    return res


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def write_demo_md(res: dict, path: str = "DEMO.md") -> None:
    ck = "checkpoints/demo_brain.pt"
    section = f"""## Demo brain (locked {time.strftime('%Y-%m-%d %H:%M')}, commit {git_commit()})
Source: `{res['checkpoint'] or res['brain'] + ' (untrained)'}` → `{ck}`.
Fixed evaluation (stage A, {res['envs']} envs, {res['seconds']} s, seed {res['seed']}, no learning): front contact **{res['front_contact']}**,
sustained {res['sustained']}, time to contact {res['time_to_contact']} s, back contact {res['back_contact']}, advance@2s {res['advance_2s']},
DNa02 mean {res['dna02_mean']} Hz, highest group {res['max_group']} {res['max_group_rate']} Hz (peak of any group {res['max_group_peak']} Hz).
Criteria: contact ≥ 0.90, sustained ≥ 0.50, DNa02 ≤ 160 Hz, no group > 200 Hz → **{'PASS' if res['pass'] else 'FAIL'}**.
Config carried inside the checkpoint: gain {res['gain']}, k_f {res['k_f']} (forward = {res['forward_source']}), k_t {res['k_t']},
retinotopy {res['retinotopy']}, amps {json.dumps(res['amps'])}.

Bridge command (laptop next to the robot; the SDK daemon runs in the Python 3.8 venv):
```
.venv38/bin/python scripts/robot_daemon.py --conn ap                       # terminal 1: robot camera + wheels
.venv/bin/python scripts/demo.py --checkpoint {ck} --robot-daemon 127.0.0.1:9500 --viz-ws 8765   # terminal 2
```
Dry run without the robot: `.venv/bin/python scripts/demo.py --checkpoint {ck} --source 0 --show --dry-run`.
"""
    txt = open(path).read() if os.path.exists(path) else "# DEMO.md — FlyBrain Rover demo\n\n"
    if "## Demo brain" in txt:
        head = txt[: txt.index("## Demo brain")]
        rest = txt[txt.index("## Demo brain"):]
        nxt = rest.find("\n## ", 1)
        tail = rest[nxt + 1:] if nxt != -1 else ""
        txt = head + section + "\n" + tail
    else:
        txt = txt.rstrip() + "\n\n" + section
    open(path, "w").write(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--lock", action="store_true", help="on pass: write checkpoints/demo_brain.pt and DEMO.md")
    ap.add_argument("--force", action="store_true", help="lock even on FAIL (say so in DEMO.md)")
    a = ap.parse_args()
    res = evaluate_demo(a.brain, a.checkpoint, a.envs, a.seconds, a.seed, a.device, a.engine, a.explore)
    print(json.dumps({k: v for k, v in res.items() if k not in ("group_rates", "group_peaks")}, indent=1))
    print("group means (Hz):", res["group_rates"])
    print("group peaks (Hz):", res["group_peaks"])
    print("M6", "PASS" if res["pass"] else "FAIL")
    if a.lock and (res["pass"] or a.force):
        os.makedirs("checkpoints", exist_ok=True)
        if a.checkpoint:
            ck = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        else:
            parts = build(1, device="cpu", engine="event", brain_path=a.brain)
            ck = dict(w=parts["lif"].w.cpu(), gen=-1, stage="A", gain=parts["lif"].g, run="untrained",
                      k_t=parts["motor"].k_t, k_f=parts["motor"].k_f, learn="none", amps=dict(parts["senses"].amp),
                      forward_source=parts["motor"].forward_source, retinotopy=parts["retinotopy"])
        ck["demo_eval"] = json.dumps({k: v for k, v in res.items() if k not in ("group_rates", "group_peaks")})
        ck["demo_source"] = a.checkpoint or a.brain
        ck["git"] = git_commit()
        torch.save(ck, "checkpoints/demo_brain.pt")
        write_demo_md(res)
        print("locked -> checkpoints/demo_brain.pt, DEMO.md updated")


if __name__ == "__main__":
    main()
