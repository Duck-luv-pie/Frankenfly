# PITCH.md — RoboMaster S1 request, QNX Makerspace, Sat 1:00 AM (requests close 12:50 AM)

Goal: get one DJI RoboMaster S1. Points are awarded for originality of presentation, so the pitch is a live act, not a speech: the fly brain runs on the laptop, reacts to the organizer, and we tell them it only needs a body.

## Logistics (do in this order)
- 12:30 AM: be in the Makerspace. Submit the request before 12:50 (form / desk, whatever they use). Ask where pitches queue.
- Laptop: charged, brightness up, `caffeinate -dims` running, Wi-Fi off is fine (nothing needs the network). Foam swatter in hand.
- Terminal ready BEFORE you walk up (takes ~15 s to load the brain):
  ```
  cd ~/Downloads/filess/flybrain-rover
  .venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run --yolo-device mps
  ```
  Window "flybrain" shows the webcam with the person box, forward/turn and latency; the terminal prints the wheel commands it WOULD send. Test it in the room first: walk left and right of the laptop, turn should flip sign, box should follow you. If the room is too dark for the detector, add `--fake-boxes` (the brain still runs, the box sweeps on its own) and say so.
- Second window: `replay/viewer.html` in a browser with `replay/episode_7_trained.json` loaded and playing (the arena top-down + neuron bars). If Ducks's Three.js viz is ready, use that instead on ws://localhost:8765.
- Two presenters max: one talks (Taka), one runs the laptop and turns it toward the organizer (Ducks). Max/Senthil stand behind.

## The 60-second version (word for word)
"Two weeks ago Janelia published the complete wiring of a fruit fly's nervous system. 166,000 neurons. We downloaded it."

(turn the laptop so the webcam sees the organizer)

"This is 15,000 of those neurons running live, right now, as spiking neurons. That box is you. The camera is its eye. Watch the bars: LC10a, the neurons a male fly uses to track a mate, are firing on the side you're standing on. And those two, DNa02 left and right, are its steering. Step to your left."

(they move; the turn sign flips on screen)

"Nobody programmed that. It's the wiring. We proved it: knock out LC10a," (press 2) "and it goes blind. Put it back," (press 3) "and it tracks you again."

"Here's our problem. It has a brain and no body. Those numbers at the bottom are wheel commands: forward, turn, in metres per second and degrees per second. They're going nowhere. Give us a RoboMaster and they go to the wheels. It will turn toward you, roll up, and touch your shoe. That's the whole project: a real fly brain that wants to find you."

(hold up the foam swatter)

"This is the safety system. Also an E-stop, a 300 ms watchdog, and a 0.7 m/s speed cap. But mostly this."

"We've already written the S1 driver. Camera stream in, drive_speed out, 40 milliseconds camera to command. We measured it. We just need the body."

## The 20-second version (if they cut you off)
"Real fruit fly connectome, 15,000 neurons, running live on this laptop. That box is you; those bars are the neurons a fly uses to chase a mate, firing on your side. The wheel commands are already being computed. We need the RoboMaster to turn them into a robot that finds you and touches your shoe. Swatter included."

## What makes it original (say none of this, do it)
1. The organizer is in the demo: the brain reacts to them, not to a slide.
2. Two keypresses that visibly break and repair a brain (2 and 3). Let THEM press 2 if they're close enough.
3. The swatter beat lands the safety question before they ask it.
4. The ask is literal: the commands are on screen, going nowhere.

## Likely questions, crib
- Why the S1 and not an RC car? "We need the camera stream and a velocity API in one SDK. The S1 gives both, plus mecanum wheels so turn and forward are independent. Our model already assumes its camera geometry, 23 cm off the floor, 98° field of view."
- Is it safe? "Speed cap 0.7 m/s, E-stop key, watchdog zeroes the wheels if a frame is late by 300 ms, foam swatter for morale."
- What if it fails to see people? "We have a search state: when nobody is in view it scans and creeps forward; in simulation it finds someone in 2.5 s. And the demo starts with the person in view anyway."
- Did it learn? "It tracks out of the box from the wiring. Learning is the fly's own dopamine neurons changing existing synapses; it made contact 25 % faster in simulation. We never train a separate network."
- How fast is the loop? "About 40 ms camera to command on this laptop. Budget is 100."
- Who else has done this? "People ran this connectome on Doom and Mario. Nobody put it in a body that walks up to you, and nobody shows the lesion controls."
- What do you do with it after? "Return it Sunday, or buy it if we win something."

## Fallbacks
- Webcam/detector dead: `--fake-boxes` (say "the detector is off, the brain is real"), or play `replay/viewer.html` and narrate the same script over the recorded episode.
- Laptop dead: phone with the replay viewer (it's a single HTML file; AirDrop `handoff/replay/viewer.html` + `episode_7_trained.json` to your phone before you go), swatter, and the 20-second version.
- Time: rehearse the 60-second version twice on the shuttle; it must run under 75 s with the movement beats.
