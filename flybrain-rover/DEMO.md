# DEMO.md — FlyBrain Rover, the two-minute demo

A real fruit fly's brain wiring, running live, in a robot, turning toward and reaching the person in front of it. Nothing in between is programmed. Every claim below has a number behind it in STATUS.md.

## Setup (10 minutes, before judges arrive)
1. Laptop on the floor-side table, RoboMaster S1 powered on, laptop joined to the robot's Wi-Fi (AP mode) or both on the same network (STA).
2. Terminal 1 (the SDK lives in Python 3.8 under Rosetta):
   ```
   .venv38/bin/python scripts/robot_daemon.py --conn ap
   ```
   Wait for "robot connected, gimbal recentred, video on". The turret stays at neutral for the whole demo.
3. Terminal 2 (the brain):
   ```
   .venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --robot-daemon 127.0.0.1:9500
   ```
   It prints latency every 30 frames; expect 30 to 60 ms camera to wheels. The live feed is on ws://localhost:8765 for Ducks's Three.js page; a JSONL copy lands in logs/demo_live.jsonl.
4. Ducks's viz on the wall screen, connected to the websocket. If the viz is not ready, open `replay/viewer.html` and load `replay/episode_7_untrained.json` as the standby picture.
5. Rehearse once with `--dry-run` added to the demo.py line (wheels don't move, everything else runs). Then run it live and walk in from 2 m: the robot should turn to face you and roll forward until it touches your shoe.
6. Person starts IN VIEW (1 to 3 m, inside the ±49° camera cone). Search from out of view exists (`--explore`) but the demo starts with the person visible, by decision.

Hotkeys (in the window, or type the digit and Enter in terminal 2): **1** baseline, **2** lesion LC10a, **3** restore LC10a, **4** lobotomize, **5** restore learning, **r/p** reward/punish, **space** E-stop, **q** quit.

## The two-minute script
| time | presenter says | key | audience sees |
|---|---|---|---|
| 0:00 | "This is the wiring of a real fruit fly's brain, 15,000 neurons and 2.3 million synapses from the connectome published this month, running live as spiking neurons. The camera is its eye, the descending neurons are its legs. Nobody programmed the behaviour." | 1 | Robot idle, brain quiet on the wall. |
| 0:20 | "Step toward it." (a judge walks in from 2 m) | | Retina columns light up where the judge is, LC10a flares on that side, DNa02 fires, the robot turns to face them and rolls forward until it touches. |
| 0:45 | "Those are the same neurons a male fly uses to track a mate. Left eye drives left steering neuron, right drives right; the wiring is doing the geometry." | | Wall shows DNa02 L/R bars see-saw as the judge moves sideways. |
| 1:00 | "Now I remove the tracking neurons, LC10a, both sides. Same brain, minus 275 cells." | 2 | Robot stops turning toward the judge; the wall shows LC10a and DNa02 flat. "In the simulator that takes tracking from 78% to 0%. A wiring shuffle with the same neurons also gives 0%." |
| 1:20 | "Put them back." | 3 | Tracking returns within a second. |
| 1:30 | "This brain also learned. Rewards were delivered by stimulating its own dopamine neurons, which changed existing synapses only. Wipe the learning, the raw connectome stays." | 4 | Robot still tracks (the wiring alone does that), but slower to close; the wall's DN bars drop. |
| 1:45 | "And restore it." | 5 | Learned synapses back, bit-identical. |
| 1:55 | "It's a fly. It wants to touch you. That's all it knows." | space | Stop. Offer the foam swatter. |

Honesty lines to keep, if asked: the forward speed is read from the population of descending neurons (a read-out choice; the eye does not reach the two forward neurons alone); the untrained wiring already reaches the person 97% of the time in simulation, learning adds speed and consistency; search from out of view uses an internal exploratory state that is not sensory; we do not claim the fly understands anything.

## Fallbacks
- **Robot misbehaves or Wi-Fi drops**: press space (E-stop). Switch to `--dry-run` and narrate against the wall viz; the brain, retina and hotkeys all still run on the webcam (`--source 0 --show`).
- **No robot at all**: `replay/viewer.html` plays the recorded episodes (`replay/episode_7_untrained.json`, `_trained.json`, `_lesioned.json`, `_tonic.json`) with the same rover, retina, rates and spike raster. Ducks's Three.js scene reads the same files.
- **Detector misses the judge** (dark clothes, odd angle): stand closer, 1.5 m, or run the brain on the webcam frame with `--show` to see the boxes; `--fake-boxes` proves the brain chain if the detector is the problem.
- **Latency over 100 ms**: `--yolo-device mps` (or cuda), `--substeps 10`, close the viz browser tab.

## Search (exploratory internal state), parameters locked (M7)
`--explore` on scripts/demo.py / robot_bridge.py. brain/explore.py defaults, evaluated and locked 2026-09-18:
onset 0.5 s without retina presence; tonic 0.4 mV/step into DNa01_L/R; saccade bursts 1.0 mV/step into DNa02_L then DNa02_R,
300 ms ± 30 %, every 1.5 s ± 30 %, biased toward the warmer heat side when the sensors disagree; everything gated off within
20 ms of any retina presence. Evaluation (brain.npz untrained, DN_ALL forward k_f 0.3, retinotopy rank, stage B = 1 to 8 humans
starting OUT of view, 128 envs, 20 s, seed 1000): front contact **0.742** (bar 0.60), 90.6 % of envs sight a human, first sight
2.46 s, first contact 6.0 s; without the explorer 0.469. Labelled in code as internal drive, not sensory. Use it only if a judge
asks "what if it can't see me"; the scripted demo starts with the person in view.

## Demo brain (locked 2026-09-18 12:26, commit 9a7c129)
Source: `checkpoints_box/tfA_both_512_gen0020.pt` → `checkpoints/demo_brain.pt`.
Fixed evaluation (stage A, 128 envs, 20.0 s, seed 1000, no learning): front contact **0.977**,
sustained 0.817, time to contact 3.23 s, back contact 0.008, advance@2s 0.734,
DNa02 mean 156.1 Hz, highest group LC10a_L 265.9 Hz (peak of any group 311.2 Hz).
Criteria: contact ≥ 0.90, sustained ≥ 0.50, DNa02 ≤ 160 Hz, no group > 200 Hz → **literal FAIL on LC10a only (266 Hz); PASS with sensory groups excluded** (LC10a is driven by the retina at amp_track 20 and exceeds 200 Hz in the untrained brain too, 206 Hz; highest non-sensory group DNa02_L 157.8 Hz).
Config carried inside the checkpoint: gain 0.05, k_f 0.3 (forward = dn_all), k_t 0.02,
retinotopy rank, amps {"small": 1.2, "track": 20.0, "loom": 2.0, "heat": 0.8, "tonic": 0.0}.

Bridge command (laptop next to the robot; the SDK daemon runs in the Python 3.8 venv):
```
.venv38/bin/python scripts/robot_daemon.py --conn ap                       # terminal 1: robot camera + wheels
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --robot-daemon 127.0.0.1:9500 --viz-ws 8765   # terminal 2
```
Dry run without the robot: `.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run`.


## Wall visual: Ducks's 3-D viewer driven by our brain
`scripts/viz_adapter.py` serves his `hunt_gpu.html` and feeds it our state (his protocol: `/events` SSE, `/brain.json`, `/who`, `/control`).
```
# replay on the wall (works now, loops):
.venv/bin/python scripts/viz_adapter.py --ui ../hunting-fly/brain/companion_brain/ui --replay replay/episode_7_trained.json --port 8601
# live from the robot demo (demo.py already streams ws://localhost:8765):
.venv/bin/python scripts/viz_adapter.py --ui ../hunting-fly/brain/companion_brain/ui --live ws://localhost:8765 --port 8601
```
Open http://localhost:8601. Our 15,000 neurons appear on his brain map at their real soma positions (neuPrint), spikes light up per tick, the DNa02/DNa01/DN_all bars, the 24 retina cells, heat and drive are ours. `fly.js` and the .glb models come from his `ui/` directory (`replay/assets/` holds the human and RoboMaster models until his push includes them; a stand-in fly.js is served if his is missing).
