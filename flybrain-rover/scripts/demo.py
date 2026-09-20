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
    6  wiring shuffle on/off: the same neurons with the same in/out degrees and weights, rewired at random
    7  remove another quarter of LC10a (press four times to delete the population a quarter at a time)
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
from scripts.voice import Voice  # noqa: E402

STATE = {"sets": [], "snaps": [], "quarter": 0, "lobotomy": None, "voice": None}


def say(key):
    """The fly narrates what was just done to it, from cached audio (scripts/voice.py)."""
    if STATE["voice"] is not None:
        STATE["voice"].say(key)


def _lc10a(brain):
    return torch.cat([brain.groups["LC10a_L"], brain.groups["LC10a_R"]])


def _apply_lesions(brain):
    """Lesions are stored as neuron sets, not edge snapshots, because edge indices are only valid for
    the brain they came from: the shuffled twin coalesces to a different edge count, so a snapshot taken
    on one and restored on the other is a crash waiting for the worst possible moment."""
    STATE["snaps"] = [brain.lif.lesion(idx, mode="both") for idx in STATE["sets"]]


def _undo_lesions(brain):
    for snap in reversed(STATE["snaps"]):
        brain.lif.restore(snap)
    STATE["snaps"] = []


def _undo_lobotomy(brain):
    if STATE["lobotomy"] is not None:
        plastic_mod.restore(brain.lif, STATE["lobotomy"])
        STATE["lobotomy"] = None
        brain.learning_wiped = False


def k_baseline(brain, rover):
    _undo_lobotomy(brain)
    _undo_lesions(brain)
    STATE["sets"], STATE["quarter"] = [], 0
    if brain.shuffled:
        brain.use_shuffled(False)
    brain.lobotomy = False
    say("baseline")


def k_lesion(brain, rover):
    if any(t.numel() == _lc10a(brain).numel() for t in STATE["sets"]):
        return
    _undo_lesions(brain)
    STATE["sets"].append(_lc10a(brain))
    _apply_lesions(brain)
    say("lesion_lc10a")


def k_restore_lesion(brain, rover):
    _undo_lesions(brain)
    STATE["sets"], STATE["quarter"] = [], 0
    say("restore")


def k_lobotomize(brain, rover):
    if STATE["lobotomy"] is None:
        STATE["lobotomy"] = plastic_mod.lobotomize(brain.lif)      # plastic synapses -> connectome values
        brain.learning_wiped = True                                 # what the viewer's HUD shows
        say("lobotomy")


def k_restore_plasticity(brain, rover):
    if STATE["lobotomy"] is not None:
        _undo_lobotomy(brain)
        say("restore_plasticity")


def k_shuffle(brain, rover):
    """Swapping the wiring swaps the whole edge array, so every lesion has to come off the outgoing
    brain and go back on to the incoming one."""
    _undo_lobotomy(brain)                                          # plastic weights are per-brain too
    _undo_lesions(brain)
    on = brain.use_shuffled(not brain.shuffled)
    _apply_lesions(brain)
    print("wiring: SHUFFLED (same neurons, random targets)" if on else "wiring: the real connectome", flush=True)
    say("shuffle_on" if on else "shuffle_off")


def k_lesion_quarter(brain, rover):
    """Delete LC10a a quarter at a time, so a judge can watch tracking degrade rather than switch off."""
    if STATE["quarter"] >= 4:
        print("LC10a already fully removed (3 restores it)", flush=True)
        return
    idx = _lc10a(brain)
    order = torch.randperm(idx.numel(), generator=torch.Generator().manual_seed(0)).to(idx.device)
    lo = (STATE["quarter"] * idx.numel()) // 4
    hi = ((STATE["quarter"] + 1) * idx.numel()) // 4
    _undo_lesions(brain)
    STATE["sets"].append(idx[order[lo:hi]])
    _apply_lesions(brain)
    STATE["quarter"] += 1
    print(f"LC10a: {25 * STATE['quarter']}% removed ({hi} of {idx.numel()} cells)", flush=True)
    say("quarter")


HOTKEYS = {
    "1": ("baseline (all restored)", k_baseline),
    "2": ("lesion LC10a L+R", k_lesion),
    "3": ("restore LC10a", k_restore_lesion),
    "4": ("lobotomize (plastic synapses -> connectome)", k_lobotomize),
    "5": ("restore learned synapses", k_restore_plasticity),
    "6": ("wiring shuffle on/off", k_shuffle),
    "7": ("remove another quarter of LC10a", k_lesion_quarter),
    "l": ("silence the brain on/off (the viewer's pause)", lambda brain, rover: setattr(brain, "lobotomy", not brain.lobotomy)),
}

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--viz-ws" not in argv:
        argv += ["--viz-ws", "8765"]
    if "--shuffle" not in argv:
        argv += ["--shuffle"]
    if "--viz-jsonl" not in argv:
        argv += ["--viz-jsonl", "logs/demo_live.jsonl"]
    if "--voice" in argv:
        argv = [x for x in argv if x != "--voice"]
        STATE["voice"] = Voice()
    print(__doc__)
    robot_bridge.main(argv, hotkeys=HOTKEYS)
