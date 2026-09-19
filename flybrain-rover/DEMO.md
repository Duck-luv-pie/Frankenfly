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

Add `--voice` to have the fly say what was just done to it (cached audio, no network needed; generate once with `python scripts/voice.py --generate` and an ElevenLabs key). Add `--brain data/brain_tiny.npz` to run the 2,211-neuron circuit instead of the full one.

Hotkeys (in the window, or type the digit and Enter in terminal 2): **1** baseline, **2** lesion LC10a, **3** restore, **4** lobotomize, **5** restore learning, **6** wiring shuffle on/off, **7** remove another quarter of LC10a, **r/p** reward/punish, **space** E-stop, **q** quit.

## The two-minute script
Lead with the controls, not the chase. The chase is only the readout that makes a lesion visible, and nobody has to be chased for the point to land.

| time | presenter says | key | audience sees |
|---|---|---|---|
| 0:00 | "This is the wiring of a real fruit fly's brain. 15,000 neurons, 2.3 million synapses, from the connectome published this month, running live as spiking neurons. The camera is its eye, the descending neurons are its legs. Nobody trained it to do anything." | 1 | Robot idle, brain quiet on the wall. |
| 0:15 | "Walk toward it." (a judge steps in from 2 m) | | Retina columns light on their side, LC10a flares, DNa02 fires, the robot turns and closes until it touches. |
| 0:35 | "Those are LC10a, the same neurons a male fly uses to track a mate. Left eye drives the left steering neuron. And this is before any learning: untrained, straight out of the microscope, it reaches you 97 times out of 100." | | DNa02 L/R bars see-saw as they move sideways. |
| 0:55 | "Here is the control. I keep every neuron, every synapse, every connection strength, and every neuron's exact number of inputs and outputs. I randomize only which cell connects to which." | 6 | The robot goes still. |
| 1:10 | "The eye is still working. LC10a is firing at 70 Hz, same as a second ago. But nothing arrives at the steering neurons: 158 hertz to zero. Same parts, same wiring diagram statistics, different map." | | LC10a lit, DNa02 flat. |
| 1:20 | "Put the real wiring back." | 6 | Tracking returns immediately. |
| 1:30 | "And it fails the way a population of neurons should, not like a switch." (press 7 four times, a beat between each) | 7 | DNa02 158 to 137 to 83 to 40 to 0 Hz; it keeps turning until three quarters are gone, then goes blind. |
| 1:45 | "All 275 cells back." | 3 | Identical behaviour returns, bit for bit. |
| 1:55 | "It is a fly. It wants to find you. Nobody programmed that." | space | Stop. |

Optional beats if a judge asks: **4** wipes the learning back to the raw connectome and **5** restores it; `--explore` makes it search when nobody is in view; a foam swatter is available and entirely unnecessary.

What to say if asked "why not just train a neural net": a net with random weights does nothing, and a trained one cannot be lesioned into a named cell type and predicted. Ours can. We predicted that removing one DNa02 would bias the turn to the other side before we ran it, and it did.

Honest lines to keep: the forward read-out (population descending activity) and the search state are our engineering, and we say so. The steering, which is the claim, is untouched wiring.

### The numbers behind those beats
`checkpoints/demo_brain.pt`, a person box at the left or right of a 320x240 frame, 45 frames at 30 Hz, CPU event engine:

| condition | turn | forward | DNa02 L / R (Hz) | LC10a L / R (Hz) |
|---|---|---|---|---|
| real connectome, person left | -1.00 | +0.81 | **158.4** / 0.0 | 66.6 / 0.0 |
| real connectome, person right | +1.00 | +1.00 | 0.0 / **251.5** | 0.0 / 64.7 |
| shuffled wiring, person left | 0.00 | +0.05 | **0.0** / 0.0 | **69.9** / 0.0 |
| shuffled wiring, person right | 0.00 | +0.20 | 0.0 / **0.0** | 0.0 / **69.5** |

Graded LC10a lesion (real wiring, person left): 0% removed 158.4 Hz and turn -1.00; 25% 137.5 and -1.00; 50% 83.4 and -1.00; 75% 40.3 and -0.81; 100% 0.0 and 0.00. Restoring returns exactly 158.4 and -1.00. The turn command saturates while the drive degrades, which is the point: the population is redundant, so half of it still steers.

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

## Second way to drive the S1: S-Bus (Ducks's path, no SDK, no gimbal needed)
Ducks drives the S1 through its **S-Bus receiver pins** (under the rear cover, see his `docs/wiring.md`): laptop USB → spare ESP32 flashed with his `firmware/sbus-bridge` (inverter) → S1 S-Bus Signal + GND. Nothing comes back, so the eye is the laptop webcam (or his ESP32-CAM stream URL). Our bridge can use his driver directly:
```
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --s1-sbus /dev/tty.usbserial-XXXX --local-eye --source 0 --dry-run   # check signs
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --s1-sbus /dev/tty.usbserial-XXXX --local-eye --source 0 --v-max 0.5 --w-max 90
```
`--sign-yaw -1` if it turns the wrong way. His measured S1 constants (slow preset): 0.85 m/s and 90°/s at full stick; our forward/turn map onto those. Needs `pyserial` (installed) and his `companion_brain` package on disk (auto-found in `../hunting-fly-s1/brain`, `../hunting-fly/brain` or `--companion-dir`). Tested against a fake serial port: 70 Hz frames, failsafe centres the sticks 0.5 s after the last packet.
