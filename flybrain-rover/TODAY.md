# TODAY.md — Friday Sept 18, in order (read after CONTEXT.md)

Goal by tonight: a brain that turns AND advances toward a person from wiring alone if possible, a non-saturated trained checkpoint, a body-correct retina, and a robot bridge ready to plug into the RoboMaster the moment we have it. Hard stop when Taka leaves for the shuttle.

## Task 1 — Forward drive from population descending activity (30 min, do first)

Hypothesis (CONTEXT.md section 6): Ducks reads forward from all descending neurons; we read it from DNa01 + DNp09, which the eye does not drive. Test:
1. Add a DN_ALL group to brain.npz and brain_v2.npz: every neuron in the subcircuit whose type starts with "DN", plus GF. Report its size.
2. motor.py: forward = k_f * mean rate over DN_ALL. turn unchanged (DNa02 R minus L). Keep the GF override in dodge mode.
3. Sweep k_f over 5 values with amp_tonic 0. Run M2b (centred human) and M3b (advance). Report advance fraction and closing distance.
Decision: if advance > 70% with amp_tonic 0, this replaces the tonic fallback; set it as default, note it in STATUS.md as "forward = population descending drive, a readout choice, biologically defensible", and skip the fallback in the writeup. If not, keep amp_tonic 0.6 as the labelled fallback (CONTEXT.md section 3 wording) and move on.

## Task 2 — Real retinotopy (1 to 2 h)

Symptom: a centred human drives DNa02_R but not DNa02_L, because senses.py assigns LC neurons to columns by index order. Fix: for each LC10a/LC11/LC12/LC15 neuron, get its lobula column ROI centroid from neuPrint (LO_* column ROIs, or the neuron's mean presynaptic position in the lobula), sort within each hemisphere by azimuthal position, and assign columns by that order. Store the assignment in brain.npz as col_of_<group>. Verify: centred human gives DNa02_L and DNa02_R within 20% of each other; human at +30 degrees gives right dominant; at -30, left dominant. Re-run M3. This must be done before hardware because it decides whether the robot veers when someone stands dead ahead.

## Task 3 — Dopamine balance (30 min setup, 60 min run)

The overnight tonic run saturated DNa02 at 300 Hz because +5 on contact reaches PAM but -0.02 per step never reaches PPL1, so plasticity is potentiation-only. Allowed changes within the existing rules: weight the PPL1 channel 3 to 5x relative to PAM, lower eta 4x, lower dopamine normalisation. Also allowed (Taka approved): mild homeostatic synaptic scaling on the plastic set so mean rates stay near their untrained values; this is textbook biology, not a new learning rule, and it stays inside Dale's law and the bounds. Then re-run the three-factor "both" job for 60 min with whichever forward mechanism Task 1 chose, 64 envs, stage A. Pass: contact rate above 0.9 AND DNa02 mean under 150 Hz AND |dw| still moving. That checkpoint is the demo brain.

## Task 4 — Robot bridge bring-up, dry (30 min)

On this laptop: pip install opencv-python ultralytics robomaster. Run the bridge with --fake-boxes --no-camera --dry-run and confirm latency. Then --source 0 --show with the webcam: walk left and right of the laptop and confirm turn sign and magnitude in the printed commands, and that the retina columns track you. Fix anything that is off before the real robot exists.

## Task 5 — Detector check on Ducks's frames (when he sends them)

Run yolo11n on his sim frames. If it misses his procedural humans, fine-tune for ten minutes on 200 frames auto-labelled from his ground-truth positions (he was asked to dump them). Verify from_boxes agrees with from_state within one column.

## Task 6 — Demo assets (if time)

- Brain visualization feed: a small websocket or file tail that streams per-frame group rates and a spike raster from the bridge so Ducks's Three.js can show the brain live next to the robot. Use the replay/FORMAT.md schema.
- ElevenLabs voice: one line when contact happens and one when lobotomized. Only if trivial.
- Tiger Data: write episode metrics and spike rates to a Postgres/Tiger table instead of only CSV. Only if trivial. Tracks lock 2:00 PM Saturday.

## Decisions already made (do not re-open)

- Gain stays 0.05. amp_track 20.
- Search (stage B) is out of scope for the demo; the demo starts with the person in view.
- No new neural networks, ever. Plastic weights only on existing synapses, Dale's law, bounds.
- The Vast box is gone; everything runs on the Mac.
- Foam swatter.

## What to hand Taka at each checkpoint

After each task, three lines in STATUS.md: what changed, the number that proves it, and what it means for the demo. Taka pastes those to Claude chat when he wants a second opinion.

## Updates (Taka, Sept 18 morning)
- Tasks 1, 2, 3 are DONE (see STATUS.md "morning session"): forward = DN_ALL population rate (k_f 0.3, no fallback; advance 100%); retinotopy from lobula footprints (`--retinotopy rank`, veer fixed, aim improved); PPL1 x5, eta 3e-4, homeostatic scaling 150 Hz; the 60-min three-factor "both" run with all three started 10:16 on the Mac.
- **Compute change: from now on training runs on a Vast.ai GPU box, not the Mac.** The Mac keeps only the robot bridge/demo (CPU event engine at batch 1). Repo copy to the box is a one-time rsync (needs Taka's approval or a permission rule); then jobs run under nohup/tmux there with the CUDA sparse engine and 512 envs. This supersedes "The Vast box is gone; everything runs on the Mac" in this file and CONTEXT.md section 8.
