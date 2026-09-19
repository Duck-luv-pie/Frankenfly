"""
watch_queue.py -- poll checkpoints/*_latest.pt for new generations and auto-evaluate them.

    python scripts/watch_queue.py                        # loop forever, scan every 15 min
    python scripts/watch_queue.py --once                  # evaluate whatever is new right now, then exit
    python scripts/watch_queue.py --interval 300 --envs 128 --stages A,B

For every checkpoints/<run>_latest.pt this loads the checkpoint (CPU, weights_only) to read its
run name and generation, and evaluates it with scripts.evaluate.evaluate(...) whenever that
generation is new -- i.e. the checkpoint was never evaluated, or its gen advanced since the last
scan. Progress is tracked per checkpoint path in logs/eval_queue_state.json so a restart does not
re-evaluate old generations. Results are logged to logs/eval_queue.csv (and logs/eval_all.csv, via
evaluate()). Checkpoints do not record which brain.npz they were trained on, so the brain defaults
to data/brain.npz unless the run name contains "v2", in which case data/brain_v2.npz is used.
Never deletes anything. Errors on one checkpoint are printed and skipped; the scan continues.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/ itself -> `evaluate`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from evaluate import evaluate  # noqa: E402

STATE_PATH = "logs/eval_queue_state.json"
QUEUE_OUT = "logs/eval_queue.csv"


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_state(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _brain_for(run):
    return "data/brain_v2.npz" if "v2" in (run or "") else "data/brain.npz"


def scan_once(args, state):
    """Evaluate every checkpoints/*_latest.pt whose generation is new. Mutates and persists state."""
    n_done = 0
    for path in sorted(glob.glob("checkpoints/*_latest.pt")):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            run = ck.get("run") or os.path.basename(path)[: -len("_latest.pt")]
            gen = int(ck["gen"])
            last = state.get(path)
            if last is not None and gen <= last:
                continue
            name = f"{run}_gen{gen:04d}"
            brain = _brain_for(run)
            print(f"[{_ts()}] {path}: run={run} gen={gen} (last evaluated {last}) "
                  f"-> evaluating as {name} (brain {brain})", flush=True)
            evaluate(brain=brain, checkpoint=path, name=name, envs=args.envs, seconds=args.seconds,
                      stages=args.stages, device=args.device, engine=args.engine, out=QUEUE_OUT)
            state[path] = gen
            _save_state(STATE_PATH, state)
            n_done += 1
            print(f"[{_ts()}] {path}: done ({name})", flush=True)
        except Exception as e:
            print(f"[{_ts()}] {path}: ERROR ({type(e).__name__}: {e}) -- skipping", flush=True)
            continue
    if n_done == 0:
        print(f"[{_ts()}] scan: nothing new", flush=True)
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=900, help="seconds between scans")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--stages", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--once", action="store_true", help="scan once for new checkpoints and exit")
    args = ap.parse_args()
    os.makedirs("logs", exist_ok=True)
    print(f"[{_ts()}] watch_queue: interval={args.interval}s envs={args.envs} seconds={args.seconds} "
          f"stages={args.stages} device={args.device} engine={args.engine} once={args.once}", flush=True)
    state = _load_state(STATE_PATH)
    while True:
        scan_once(args, state)
        if args.once:
            break
        print(f"[{_ts()}] sleeping {args.interval:.0f}s", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
