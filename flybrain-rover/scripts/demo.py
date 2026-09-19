"""
demo.py -- the judge-facing demo: the robot bridge plus five hotkeys and the live brain feed.

    .venv38/bin/python scripts/robot_daemon.py --conn ap                                   # terminal 1 (SDK, py3.8)
    .venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --robot-daemon 127.0.0.1:9500   # terminal 2
    .venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run      # webcam rehearsal

Hotkeys (press in the --show window, or type the digit + Enter in the terminal):
    1  baseline: undo every lesion and restore learned synapses (the trained demo brain as loaded)
    2  lesion LC10a (both sides): the tracking neurons are removed; the robot stops turning toward you
    3  restore the lesion
    4  lobotomize: reset every plastic synapse to the raw connectome (learning wiped, wiring intact)
    5  restore plasticity (the learned synapses come back, bit-identical)
    r / p  reward / punish the fly (PAM / PPL1 burst)   l  silence the brain   space  E-STOP   q  quit
The live feed is always on: ws://localhost:8765 (Three.js) and logs/demo_live.jsonl. All other flags are
scripts/robot_bridge.py flags (--explore for the search state, --yolo-device, --v-max, --w-max, ...).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain import plastic as plastic_mod  # noqa: E402
from scripts import robot_bridge  # noqa: E402

STATE = {"lesion": None, "lobotomy": None}


def _lc10a(brain):
    return torch.cat([brain.groups["LC10a_L"], brain.groups["LC10a_R"]])


def k_baseline(brain, rover):
    if STATE["lobotomy"] is not None:
        plastic_mod.restore(brain.lif, STATE["lobotomy"]); STATE["lobotomy"] = None
    if STATE["lesion"] is not None:
        brain.lif.restore(STATE["lesion"]); STATE["lesion"] = None
    brain.lobotomy = False


def k_lesion(brain, rover):
    if STATE["lesion"] is None:
        STATE["lesion"] = brain.lif.lesion(_lc10a(brain), mode="both")


def k_restore_lesion(brain, rover):
    if STATE["lesion"] is not None:
        brain.lif.restore(STATE["lesion"]); STATE["lesion"] = None


def k_lobotomize(brain, rover):
    if STATE["lobotomy"] is None:
        STATE["lobotomy"] = plastic_mod.lobotomize(brain.lif)          # plastic synapses -> connectome values


def k_restore_plasticity(brain, rover):
    if STATE["lobotomy"] is not None:
        plastic_mod.restore(brain.lif, STATE["lobotomy"]); STATE["lobotomy"] = None


HOTKEYS = {
    "1": ("baseline (all restored)", k_baseline),
    "2": ("lesion LC10a L+R", k_lesion),
    "3": ("restore LC10a", k_restore_lesion),
    "4": ("lobotomize (plastic synapses -> connectome)", k_lobotomize),
    "5": ("restore learned synapses", k_restore_plasticity),
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--viz-ws" not in argv:
        argv += ["--viz-ws", "8765"]
    if "--viz-jsonl" not in argv:
        argv += ["--viz-jsonl", "logs/demo_live.jsonl"]
    print(__doc__)
    robot_bridge.main(argv, hotkeys=HOTKEYS)
