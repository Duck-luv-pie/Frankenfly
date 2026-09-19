# HANDOFF — Saturday morning, read this first

Written ~07:00 Saturday, after an overnight session. Taka is asleep. Everything below is measured, and every number has its protocol next to it.

## Where it stands

The fly brain works and the demo is built. A real male-Drosophila connectome subcircuit, 15,000 neurons and 2,334,959 signed synapses, runs as spiking neurons at 15 ms per camera frame on the laptop. **Untrained, it turns toward a person in 75 to 78% of simulated trials and reaches them 97.7% of the time**, and it trails a walking person within 1 m for 87 to 92% of the time. Learning uses the fly's own dopamine neurons and buys speed, not the behaviour: contact in 3.2 s instead of 4.3 s. Two controls run live on a keypress and they are the point of the demo, not the chase.

The robot half is wired but untested on the real chassis. Ducks's S-Bus link to the S1 works, so the S1's own dead camera no longer blocks anything.

## What works right now, with the exact commands

Everything runs from `~/Downloads/filess/flybrain-rover` (cd there first; the venv is `.venv`).

```bash
# the demo on a laptop webcam, no robot, wheels printed instead of sent
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run

# the 3-D viewer (Ducks's page) driven by our brain, from a recorded episode
.venv/bin/python scripts/viz_adapter.py --ui ../hunting-fly-s1/brain/companion_brain/ui \
  --replay replay/episode_7_trained.json --port 8601      # then open http://localhost:8601

# the whole test suite
.venv/bin/python -m pytest tests -q        # 125 pass
```

Demo hotkeys (in the window, or type the digit and Enter in the terminal):
**1** baseline · **2** remove the tracking neurons (LC10a) · **3** restore · **4** wipe the learning · **5** restore it · **6** wiring shuffle on/off · **7** delete a quarter of LC10a · **space** stop · **q** quit.

## Driving the real robot

Two independent halves: **eyes** come from `--source`, **wheels** from one of two links. Pick whichever hardware is ready.

```bash
# wheels over S-Bus (Ducks's ESP32 bridge, no DJI SDK, no gimbal needed) + laptop webcam as the eye
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt \
  --s1-sbus /dev/cu.usbserial-XXXX --local-eye --source 0 --dry-run     # check the turn sign first
# then drop --dry-run and add --v-max 0.5 --w-max 90

# or the ESP32-CAM as the eye instead of the webcam
--source http://companion-cam.local:81/stream
```

Find the serial port with `ls /dev/cu.*` once the ESP32 is plugged into the laptop. **Always run `--dry-run` first**: stand left of the camera, then right, and check the printed turn flips sign. If it turns the wrong way, add `--sign-yaw -1`. There is a 300 ms watchdog and space is an E-stop.

## Blocked, and on whom

| what | who | note |
|---|---|---|
| S1 gimbal and camera boards are offline | whoever has the robot | chassis answers, gimbal/camera/blaster do not. Charge the battery (it was at 5%) and reseat the gimbal cable. **Not on the critical path**: the S-Bus link plus any other camera is enough. |
| serial port of the ESP32 S-Bus bridge | Ducks | plug it into the demo laptop, then the command above works |
| 60-second video for Devpost | anyone with a phone | shot list and voiceover are written, see below |
| project name on Devpost | team | the repo is `hunting-fly`, so that is probably the name; needs one decision |
| sponsor tracks | Taka | **locks 2:00 PM Saturday** |

## Do these first, in order

1. **Charge the S1 battery** and plug the ESP32 bridge into the demo laptop. Then run the S-Bus dry-run above and check the turn sign. That is the last untested link in the chain.
2. **Record the 60-second video.** Shot list: 0-8 s the brain map with spikes moving; 8-22 s a person walks in and the robot turns and drives to their feet; 22-38 s press 6 and it sits still while the person moves; 38-48 s press 6 again and it tracks; 48-60 s press 7 four times and the rates fade out. If the robot is not ready, record the same script against the 3-D viewer and say so.
3. **Pick the tracks before 2 PM.** Finalists is the real target. Everything else is in the tracks notes; most sponsor tracks have hard requirements we do not meet (QNX needs their OS, Baseten needs a model served there, Bracket Bot needs their hardware).

## The demo, in two minutes

Lead with the controls, not the chase. "A robot that follows a person" is a contest we lose: a three-line controller does it better. What a neural net cannot do is fail on command in a named way.

1. "This is the wiring of a real fruit fly's brain, running live as spiking neurons. Nobody trained it." Someone walks in, the robot turns and reaches them.
2. "Those are LC10a, the same neurons a male fly uses to track a mate."
3. Press **6**. "Same neurons, same number of connections, same strengths. I only randomize which cell connects to which." The robot goes still. **The eye still fires at 70 Hz; the steering neurons go from 158 Hz to zero.**
4. Press **6** again. Tracking returns.
5. Press **7** four times. The steering drive fades 158, 137, 83, 40, 0 Hz, and the robot keeps turning until three quarters of the population is gone. Press **3** and it comes back exactly.

Say out loud what we engineered: the forward read-out (we read it from the whole descending population, because the eye does not drive the two classic forward neurons) and the search behaviour. The steering, which is the claim, is untouched wiring.

## Numbers you can quote, with their protocol

- Untrained turn-toward **75 to 78%**, 64 to 256 arenas, 2 s, humans frozen.
- Untrained front contact **0.977**, trained **0.977** but in **3.23 s** against 4.25 s; stage A, 128 arenas, 20 s, seed 1000.
- Trailing a walking person: within 1 m **87 to 92%** of the time, never lost, 32 arenas, walkers at 1.0 to 1.4 m/s.
- Lesions, 256 arenas, seed 2000: remove LC10a → **0%**; shuffle the wiring keeping every degree → **11%** trained, **0%** untrained; remove the looming cells → giant fibre **130 Hz to 0.1 Hz** while steering stays at **75%**; remove one steering neuron → rightward bias.
- Live shuffle on the demo brain: LC10a **70 Hz** either way, DNa02 **158 Hz to 0**.
- Search from out of view: **74%** contact, first sight in **2.5 s**, 128 arenas, 20 s.
- Latency: **15 ms** median camera to command (detector 8 to 11 ms on the laptop GPU, brain 4 ms). Budget was 100 ms.

## Where things live

- Shared repo: `github.com/Duck-luv-pie/hunting-fly`. Our half is `flybrain-rover/` on `main`; Ducks's full project is the `s1-fly-brain` branch.
- Locally: `~/Downloads/filess/flybrain-rover` (ours) and `~/Downloads/filess/hunting-fly-s1` (his branch, checked out so we can import his S1 and camera drivers rather than rewriting them).
- The demo brain is `checkpoints/demo_brain.pt`. The Devpost story is `DEVPOST.md`, the form fields are `DEVPOST_FIELDS.md`, the runbook is `DEMO.md`, the figures are in `devpost/`.
- Training ran on a rented A100. It may be stopped now; ask Taka for the ssh line if you need the remaining checkpoints. **The control loop never uses it.**

## Gotchas that cost us time

- `cd` into `flybrain-rover` before any command; the venv path is relative.
- The DJI SDK only has Python 3.6 to 3.8 wheels, so it lives in a second venv, `.venv38`, and talks to the brain through `scripts/robot_daemon.py`. Do not try to `pip install robomaster` into the main venv.
- Keep the gimbal in chassis-lead mode, otherwise the camera stays pointed at the wall while the body turns.
- `turn > 0` means clockwise everywhere: in the sim, in the bridge, and as positive `z` in `chassis.drive_speed`.
- Gain 0.05 and amp_track 20 are calibrated. Changing them silently breaks the milestone numbers above.
