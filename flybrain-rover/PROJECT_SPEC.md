# PROJECT_SPEC

What the project is, which surface holds what, how each piece is built and run. Written 2026-09-19. Devpost closes 8:00 AM Sunday; the build cutoff is **07:30 Sunday**, after which nothing new ships.

Cross-references, not copies: hotkeys and fallbacks in `DEMO.md`, numbers with protocols in `RESULTS.md`, the game in `badge/GAME_SPEC.md`, the badge API in `badge/SDK_NOTES.md`, the pitches in `devpost/runbook.html`, the rules in `CLAUDE.md`.

---

## 1. What this is, and the one claim

The male *Drosophila* CNS connectome (neuPrint `male-cns:v1.0`, Berg et al., *Cell*, 2026) simulated as leaky integrate-and-fire neurons on the real wiring: 15,000 neurons, 2,334,959 signed synapses (RESULTS 1). A camera frame becomes current on the fly's eye neurons; its descending neurons, which would drive its legs, become wheel commands on a RoboMaster S1. Nothing between the camera and the wheels is programmed.

**The claim: the behaviour is in the wiring, and we can break it on command in the ways the anatomy predicts.** Untrained, it turns toward a person in 75% of arenas (64 arenas, 2 s, people frozen, RESULTS 2). Randomize only which cell connects to which, keeping every sign, strength and degree, and the eye still fires at 69.9 Hz while steering drops 158.4 Hz to 0.0 (demo brain, person at the left of frame, 45 frames at 30 Hz, RESULTS 3).

---

## 2. The four surfaces

Every piece of content has one primary surface. Everywhere else it is a pointer, not a copy.

| surface | who it is for | the one sentence it lands |
|---|---|---|
| **The floor** (robot, laptop, wall screen) | finalist and sponsor judges | "The wiring is doing this, and watch it stop when I break the wiring and nothing else." |
| **The badge** (in a pocket) | every attendee, the Best Badge Hack judge | "A real fly's escape reflex is running in your hand, and it is faster than you." |
| **The website** (one page at the domain) | the attendee who held a badge, the GoDaddy judge | "Paste this and your badge has a fruit fly's escape circuit in it." |
| **The Devpost** | judges reading away from the table, and the record | "Every claim here has a protocol behind it, including the ones that failed." |


### Content map, one primary surface each

| content | primary surface |
|---|---|
| the chase and control beats (`DEMO.md` hotkeys) | floor |
| `devpost/runbook.html` | floor, presenter only. Internal |
| `scripts/viz_adapter.py` plus Ducks's 3-D viewer | floor, wall screen, default |
| `replay/viewer.html`, the four episodes | floor, robot-down fallback. Not hosted |
| `scripts/talk.py`, `scripts/live_chart.py` | floor, the ElevenLabs and Tiger beats |
| `badge/flybadge_app.lua`, the B lesion button | badge (handout), website (Copy payload) |
| `badge/swat_app.lua` (SWAT v1) | badge, second unit, table toy |
| `devpost/*.png`, the video | Devpost gallery and video slot |
| `RESULTS.md`, `README.md`, `reproduce.py`, tests | repo |
| `CLAUDE.md`, `STATUS.md`, `TODAY.md`, `HANDOFF*.md` | none, by design. Internal |

---

## 3. The robot demo

Four minutes for a finalist judge, wrapped around the two-minute script in `DEMO.md`. Lead with the controls: the chase is only the read-out that makes a lesion visible.

- **0:00** badge into the judge's hand before any explanation, while the terminal comes up. Then the opening line, the walk-in from 2 m, LC10a named, and 97.7% contact with its protocol (128 arenas, 20 s, seed 1000, people walking, RESULTS 2).
- **0:55** the **judge** presses **6**, not the presenter. A control you performed yourself is a control you believe. Then **7** four times with a beat between each: 158.4, 137.5, 83.4, 40.3, 0.0 Hz (RESULTS 3), and **3** to restore.
- **2:00** back to the badge in their hand (section 4), then the caveats in section 8. Leave them holding the badge, and speak the domain.

**Table:** laptop and badges against the wall under the screen, a taped lane about 2.5 m by 1.5 m in front, the USB webcam on the S1 shell at the calibrated 0.231 m. P1 talks and owns every hotkey and the E-stop; P2 stands at the lane's far end, walks in, catches the robot, resets between judges.

**Wall screen:** replay loop between judges, live mode during a live run, and say once that the fly's position is dead-reckoned from the motor command because the robot reports none, while the spikes and bars are the live brain. Live when the sentence is about neurons, replay when it is about the chase, never the present tense over a recording.

**Wheels, in order of preference:** the S-Bus adapter on laptop USB with a USB webcam as the eye, because nothing in that loop touches a network and voice and Tiger keep the venue Wi-Fi; then the Pi relay; then the DJI SDK daemon; then `--dry-run`, identical everywhere except the wheels. Sign check before judges arrive: stand left, stand right, the printed `turn` must flip sign, else `--sign-yaw -1`.

**Drop order** (one twenty-second fix, then drop a tier): Tiger chart, voice agent, live viewer, camera, wheels, detector, laptop. What is left is the badge, which needs none of it.

---

## 4. The badge

### 4a. FlyBadge, as it exists

`slug=flybadge`, built from `badge/app_template.lua` plus the generated `badge/flybadge_circuit.lua` into one 17,077-byte file. 150 connectome neurons and 900 measured synapses (LC4 and LPLC2 converging on the giant fibre), fixed-point LIF at 100 Hz on the ESP32-C3, verified against the Python reference spike for spike by `tests/test_badge_app.py`. No RESULTS section covers the badge circuit yet, so cite the test, not a results table. There is no light sensor, so a swat is read through the accelerometer. **B** lesions the eye's wiring, **A** restores it, **UP** and **DOWN** set the threshold. The idle screen carries one line, the domain, from the moment the domain exists.

**No build since 16:28 has been confirmed loading on a real badge.** The one hardware session we have is in `badge/SDK_NOTES.md`: the escaped-decimal build opened and bounced straight back to the launcher, which the serial console traced to `badge.radio.enable()` exhausting the heap and panicking the device. Everything after that (base64, the chunked decoder, the LC10a removal, the STARTLE visual) is verified in software only, against the Python reference and the desktop Lua harness, and the pushes since have timed out. Treat 'it works on the badge' as unproven until someone opens it and swats it. The radio is cut permanently regardless.

### 4b. The game

`badge/swat_app.lua` (SWAT v1, `slug=swat`) is built and tested in software, a reaction test against the reflex. **SWATTER** (`badge/GAME_SPEC.md`, `slug=swatter`) is the go/no-go game: animals land, you swat flies and spare the rest, every animal is the same dark blob for 180 ms, and the default fly's giant fibre fires at 470 ms from that instant (ramp 40, `badge/sim_fixed.py`, GAME_SPEC 3, a Python reference number, not a stopwatch on hardware). **SWATTER is specified, not built**, and ships only if `tests/test_swatter_app.py` passes before 07:30. Until then SWAT v1 is the table toy on the second badge. Neither game goes on the website, and neither enables the radio.

### 4c. The live link to the robot

**Chosen channel: USB serial, not BLE.** The badge logs one line with `badge.sys.log` when `step()` returns a giant fibre spike; a laptop script reads it with pyserial and sends one UDP datagram to `127.0.0.1:9600`, the hotkey port `scripts/robot_bridge.py` already binds and `scripts/talk.py` already uses. BLE is rejected twice over: the heap panic above, and `badge.radio` filtering receive to its own `LUA1` frames with no documented way for a laptop to join. NFC is reader-only, the wrong direction.

**What a judge sees:** they swat the badge, six LEDs flash white, and the rover flinches, because the same swat arrived as looming current on its own LC4 and LPLC2 and its 15,000 neurons decided to escape. Press **B** and nothing happens on either device: no spike, no message.

**What keeps it honest.** The swat enters at the sensory layer through `brain/senses.py`, the channel the arena's looming stimulus already uses, so it never touches the retina path or the wheels (CLAUDE rules 3 and 5 hold), and it is labelled in code and out loud as a transplanted stimulus, the way `amp_tonic` is: the flinch is the connectome's, the trigger ours. The escape read-out in `brain/motor.py` is armed only briefly after a swat, because the rover's giant fibre already responds to a person walking in; that gate is engineering we name rather than hide.

**Build, then the kill rule.** Three log calls in `badge/swat_template.lua`, `scripts/badge_link.py` (parser searching each line for `@FLY`, 400 ms debounce, loopback only), hotkey **`w`** in `scripts/demo.py` with the loom injection in `Brain.step`, `tests/test_badge_link.py` over `os.openpty()`. (This said **8** until the merge: 8, 9 and 0 are now lobotomize, wake and stop/resume, which are demo-critical and already wired to the viewer's buttons, so the badge swat moved to a letter.) Timebox: 45 minutes. Two traps: the badge IDE holds the serial port exclusively (`[Errno 16] Resource busy`, fixed by clicking Disconnect), and nobody has confirmed the badge runs an app on USB power with the battery switch off. If it misses the window, delete it: nothing depends on it, and **`w`** simply goes unbound. The loom dose numbers behind the stimulus are one probe run on one seed, not in `RESULTS.md`, so do not quote them.

---

## 5. The website

**One page, and it is the install page.** The domain does not become a project site: the rover cannot be demonstrated in a browser, the video tells that story better, and a second job halves the install rate of the first. `site/index.html` already is this page. Bands 0 to 3 are built and stay as they are; 4 and 5 are the only additions, below the install steps, never above them.

**Band 0, hero.** Headline: *Put a fly's brain on your badge.* Body: "The complete wiring diagram of a male fruit fly's brain was published this month: every neuron, every synapse. Your badge can run 150 of those neurons, the escape reflex that fires when you swat at a fly and miss. Swat the one below, then put it on your own badge in four steps." Buttons: `Copy the app` (the real copy button, scrolling to the steps) and `What the robot does`.

**Band 1, try it here first.** The browser circuit as built: SWAT, B, A, the raster and its caption, auto-swatting after load. One line added under the readouts: "Press B and the eye keeps firing at the same rate. Nothing reaches the giant fibre, because the only thing we removed was the wiring out of it. That is the whole argument of this project, in 150 neurons."

**Band 2, install it on your badge.** The four steps as built, keeping the three traps that cost real time: battery switch off first, a data cable, the port named **USB JTAG/serial debug unit**. Keep the troubleshooting strip, above all "it opens and jumps straight back to the home screen: press **Reboot** in the IDE, not just reload" (`badge/SDK_NOTES.md`).

**Band 3, what is actually in it.** The honesty block, as built: real neuron and synapse counts with their signs and the fixed-point LIF, against an uncalibrated swat threshold and an escape slowed so a human can play against it.

**Band 4, new: the same wiring, bigger.** "The badge circuit is the escape reflex. The full project is the rest of the fly: 15,000 neurons and 2,334,959 signed synapses driving a robot toward a person. We wrote the encoder and the read-out. We wrote no controller, and no rule anywhere says turn toward a person." Then four tiles with their conditions: **75%** of arenas turn toward a person untrained (64 arenas, 2 s, people frozen); **97.7%** reach them and touch them (128 arenas, 20 s, people walking, seed 1000); **15 ms** median camera to wheel command, 24 ms at the 95th percentile (dry run, detector on the laptop GPU, budget 100 ms); **0** lines of steering logic. Then the caveat: "The robot has not been driven on a real chassis yet."

**Band 5, new:** three links, the video, the Devpost, the repo. **Banned:** results tables, figures, the runbook, hotkeys, bios, a hosted viewer, a second Copy button.

**Before deploying, re-embed the payload, but know what you are choosing.** The page embeds the older build; `badge/flybadge_app.lua` is now 17,077 bytes with the STARTLE visual and the chunked decoder, which cut peak decode heap from 17.7 KB to 3.5 KB (`badge/SDK_NOTES.md`). The newer build is better on every measurement we can take from a laptop. It is also the one that has never been confirmed loading on a badge. Do not deploy it to strangers until someone has opened it on hardware; until then the page should carry whichever build a person has actually watched run.

---

## 6. The Devpost

`DEVPOST.md` whole as the body, `DEVPOST_FIELDS.md` for the field-by-field paste, video at the top. Gallery in control-first order: `control.png`, `silence.png`, `pathway.png` (captioned as anatomy, not causation), `lesions.png`, `graded.png`, `dropout.png`, the viewer screenshot, two table photos. Sponsor paragraphs under their prizes. Keep the negative results: four flat training runs and a refuted pathway prediction are why the positive numbers are believable. Keep the honesty lines: the forward read-out and search state are ours, the steering is the connectome's. "Try it out" links, domain first because it does something in one click, then the repo. No install steps, no hotkeys, no number without its protocol.

---

## 7. Build and run

All commands from `/Users/takatoshilee/Downloads/filess/flybrain-rover`. The venv is `.venv`; the DJI SDK path needs `.venv38`.

```bash
# demo, no robot: brain, retina, hotkeys, viewer feed, wheels printed not sent
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run

# demo, live wheels over the S-Bus adapter, USB webcam as the eye
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt \
  --s1-sbus /dev/cu.usbserial-XXXX --local-eye --source 1 --voice --v-max 0.5 --w-max 90

# wall screen: Ducks's viewer on our brain, at http://localhost:8601
.venv/bin/python scripts/viz_adapter.py --ui ~/Downloads/filess/hunting-fly-s1/brain/companion_brain/ui \
  --live ws://localhost:8765

# voice agent, then the Tiger stream and wall chart at http://localhost:8790
.venv/bin/python scripts/talk.py
.venv/bin/python scripts/export_timeseries.py --live ws://localhost:8765 --run-id demo_$(date +%H%M)
.venv/bin/python scripts/live_chart.py

# badge: connectome to Lua, Lua to the file the IDE imports, verified against Python
.venv/bin/python badge/export_badge_circuit.py
.venv/bin/python badge/build_app.py
.venv/bin/python -m pytest tests/test_badge_app.py -q

# the suite, and the eight headline claims
.venv/bin/python -m pytest tests -q          # 212 collected
.venv/bin/python scripts/reproduce.py --quick
```

Pushing the badge app is the five-step IDE flow in `badge/SDK_NOTES.md`, with **Reboot** rather than reload on a first push. Deploying the site is `site/README.md`: `docs/index.html` on `github.com/Duck-luv-pie/hunting-fly`, Pages on branch `main` folder `/docs`, four A records at `185.199.108-111.153` on the apex, the custom domain field, Enforce HTTPS. Only a repo admin can do the Pages half.

---

## 8. What is real, and what is not

**Real, verified, reproducible.** The brain and its controls: the wiring shuffle, the LC10a lesion (78% to 0%, 256 arenas, seed 2000, 2 s, RESULTS 3), the graded quarter-removal. The pruned circuit that still works: 2,211 neurons, 10,387 synapses, 81% turn-toward (RESULTS 4). The voice agent reading live brain state and pressing the demo's hotkeys. Tiger Data live ingest, a continuous aggregate, 92 to 93% compression (`export_timeseries.py --report`, HANDOFF_mlh). Two viewers. 248 tests (`pytest -q`).

**Real, but ours and labelled as ours.** Forward speed is the mean rate across all 241 descending neurons, because the eye never drives the two textbook forward-walking neurons even after forcing every neuron on the shortest anatomical path into the circuit (RESULTS 6). Also the search state, and the badge trigger if it ships. Say so unprompted.

**Not a bottleneck, whatever the figure looks like.** AOTU and LAL carry 80.6% of the bottleneck weight between eye and steering (RESULTS 4b), which reads as load-bearing and is not: lesion the tubercle and tracking is still at 61% against 71% for the same number of random cells (128 arenas, one seed, stage A, untrained, RESULTS 4b). A connectome says where the wiring went, a lesion says what it is for, and here the two disagree. Never call it load-bearing.

**Does not exist.** The robot has never been driven by this brain on a real chassis, so every motion figure here is dry run or simulation. No video. SWATTER (the go/no-go design) specified, not built, and superseded: **Swat** (`badge/swatgame_*.lua`) is the badge game that shipped, and **the badge link is built** (`scripts/badge_link.py`, B and RIGHT over USB serial into the hotkey port). Neither has run on real hardware: the game is verified in a desktop harness and the link over a pty. The domain not registered, the site not deployed. The name in section 9 is a default, not a team decision.

**Stale numbers.** The test count is 248. `CLAUDE.md`, `README.md` and section 7 now all say 248; `RESULTS.md` 8 and `devpost/runbook.html` still carry older figures and are the two left to fix. Publish one number everywhere or none. The runbook also still points the DNS at the replay viewer or the Devpost; it points at the install page.

---

## 9. Owners, and what is still open

| piece | owner |
|---|---|
| brain, demo, hotkeys, badge build, site copy | Taka |
| Pages on `hunting-fly`, the viewer, the S-Bus and Pi hardware | Ducks, the only repo admin |
| the domain at the GoDaddy booth | whoever walks over first; the badge screen cannot print it until then |
| the 60 second video (tonight, not Sunday), P1 presenter, P2 operator | unassigned |

**Decided here, do not reopen.** The domain serves the install page and nothing else. FlyBadge is the handout and the website payload. The replay viewer is not hosted. No badge app enables the radio. The badge link goes over USB serial, not BLE. Nothing printed. The project name defaults to **Hunting Fly**, the repo a judge opens, and the domain to **swatme.tech**, because that URL is an instruction and matches what the page does. The team can override either tonight; the Devpost title, site header and video caption block on the name.

**Open, and each needs one person and one answer.**

1. **Is Best Badge Hack separable from Solana?** The runbook says confirmed on the prize page, `TRACKS.md` says confirm at the booth. Ask, then correct whichever is stale.
2. **Does SWATTER get built before 07:30?** If not, nothing public mentions it.
3. **Does the badge link survive its timebox?** Two ten-minute checks gate it: does the badge run an app on USB power with the battery switch off, and can pyserial hold the console with the IDE tab closed.
4. **Which wheels path is live on Sunday**, and is the turn sign checked on the real chassis. Nothing below the cutoff is decidable without one live drive.
5. **Who re-embeds the badge payload** into `site/index.html` after a badge rebuild. It belongs in the badge build step, not a checklist someone forgets.
