# CONTEXT.md — the full picture (read after CLAUDE.md, before doing anything)

This file is the context Claude Code did not have. It explains why the project exists, who is on the team, what the judges reward, what the science says, what the other simulator does, and how we talk about results honestly. CLAUDE.md has the rules; TODAY.md has the plan; STATUS.md has the results.

## 1. The event and what wins it

Hack the North 2026, University of Waterloo, Sept 18 to 20. 36 hours of hacking starting Friday night. About 260 submissions. Main award: 12 finalists chosen on WOW factor, technical ability, originality, and design. Judging is a live demo at a table, roughly five minutes per judge, not slides and not a product pitch. Sponsor prizes are separate and must be selected by 2:00 PM EDT Saturday; one project can win several.

What past finalists at this event look like: physical, interactive, memorable demos a judge can walk up to and engage with (a robotic arm, a motorized chess board, a Nerf gun as a game controller). Our project fits that lane exactly.

Tracks we are entering: Finalists (the real target), MLH Tiger Data (spike trains and episode logs are high-frequency time-series), MLH MongoDB Atlas (sweep results and checkpoints), MLH ElevenLabs (give the fly a voice). Baseten only if a model is actually served there. Not Rox (they want LLM agents on messy data; we have no LLM). Not Bracket Bot (requires their hardware; we use a RoboMaster S1).

## 2. The project in one paragraph

A real fruit fly brain drives a wheeled robot. We use the male Drosophila central nervous system connectome (Berg et al., Cell, Sept 3 2026: 166,700 neurons, ~125M synapses, brain plus optic lobes plus ventral nerve cord, served on neuPrint as male-cns:v1.0). We keep a ~15,000-neuron subcircuit from eye to descending neurons, simulate it as leaky integrate-and-fire on the real wiring, translate a camera image into current on the fly's eye neurons, read the fly's descending (leg-command) neurons as wheel commands, and put that on a RoboMaster S1 that finds, faces, and touches a person in a room. Nothing in the middle is programmed. Learning, where we do it, uses the fly's own mechanism: reward and punishment delivered by stimulating its dopamine neurons (PAM for reward, PPL1 for punishment), which gate plasticity at existing synapses only.

## 3. Why this is not just another fly-brain demo

The connectome went viral the week it was released: people trained it to play Doom (with damage delivered to two PPL101 dopamine cells), Mario 64, Beat Saber; there are browser pets, a drone-pilot repo, and connectome-pilot (a rover and arm controller with plasticity on a synaptic subset). Judges may have seen the Doom clip. Our differentiation, in order:
1. A physical body interacting with a live human in the room, not a game or a sim.
2. Lesion controls that prove the behaviour is the wiring: knock out LC10a and turning goes to 0%; shuffle the wiring (same neurons, same degrees) and it goes to 0%; knock out LC4/LPLC2 and the escape response vanishes while steering survives. Nobody else shows the negative control.
3. Dopamine-gated plasticity you can toggle, and a lobotomize button that resets it live.
4. A live brain visualization next to the robot, so judges see cause and effect: person approaches, eye neurons flare, descending neurons fire, robot moves.

Pitch, one sentence: "This is a real fruit fly's brain wiring, running live, in a robot, and it turns toward you using the same neurons a real fly uses to track a mate. Nobody programmed that."

What we do NOT claim: consciousness, understanding, biological fidelity beyond the wiring, or "the fly learned to hunt" unless learning is demonstrated with a non-saturated circuit and a control. connectome-pilot's own numbers show a 3-line hand-coded controller can beat the connectome; our story is emergence from real wiring, not superiority.

## 4. The team and who owns what

- Ducks (Ducks_luv_pie): hardware, the RoboMaster S1, the Three.js arena and visualization. He built the first fly sim (see section 6). He is decisive and ships.
- Max (RatWeeb): general help, likely hardware and demo.
- Senthil (sensei): limited availability (interview after the event).
- Taka: this repo. The brain, the sensory and motor mapping, the arena, the training, the robot bridge, and the sponsor track strategy. Taka does ML research at UofT (AI guardrails). He joined this team two days before the event; the previous member dropped out.

Ducks's requirements for the controller, verbatim in spirit: (1) find the human, which his version is bad at; (2) track the human as they move, humans roam randomly and never evade; (3) fly barely faster than humans; (4) reward for finding and touching with the front of the model, then continuous reward for sustained front contact; (5) punish failure to find or touch; (6) do not train a separate neural net, train the fly's own neurons via plasticity, make sure it is the real fly brain; (7) a dashboard.

## 5. The science we are standing on

- LC10a: visual projection neurons a male fly uses to track a female during courtship. In our circuit they are the ONLY eye population that reaches DNa02 (steering). This is the backbone of the demo.
- LC11, LC12, LC15: small-object detectors. They respond to small moving things and are inhibited when the object gets large (Keles and Frye 2017). This is why the earlier sim turned 180 degrees right before contact: the small-object channel switches off as the human fills the view. senses.py drives them with presence x (1 - size) so they fade near contact, and LC10a with presence x size so it takes over.
- LC4 and LPLC2: looming detectors (angular velocity and angular size). They converge on the giant fibre (GF), the escape neuron. In our circuit they drive GF and nothing else. That is the dodge channel and the escape lesion.
- DNa02 (left and right): descending neurons for steering. Turning velocity is roughly linear in the right-minus-left difference ("see-saw"). DNa01: slower sustained turns. DNp09: forward locomotion and courtship pursuit in males, but in our circuit its strongest inputs are inhibitory and eye-driven, so it does not fire.
- Mushroom body: Kenyon cells (KC) to MBON synapses are where fly learning lives, gated by dopamine (PAM reward, PPL1 punishment). Overnight finding: KCs are silent in our task because nothing sensory reaches them at a stable gain, so KC-to-MBON plasticity has nothing to act on. Eye-to-DN plasticity ("both" mask) is where learning actually moved weights.
- Neuron model: leaky integrate-and-fire with Shiu et al. 2024 constants. One global gain scales synapse counts to current; 0.05 is sane, 0.1 lets thermosensory input seize the mushroom body, 0.275 (Shiu's value) seizes everything. Stay at 0.05.
- The forward-drive result: shortest anatomical paths from LC10a and LC4 to DNp09/DNa01 exist (1 to 3 hops, via AOTU and LAL neurons), 739 of the 795 path neurons were already in the circuit, and forcing all of them in (brain_v2) changed nothing. The path exists but carries no drive. This is a real scientific finding and goes in the writeup.

## 6. Ducks's simulator, and staying compatible with it

Ducks built a browser sim (Vite/Three.js) with an 8,503-neuron "hunter subcircuit" (LC10a, LC11, LC12, LC15 plus central and descending neurons), a retina, heat cells, reward training over ~575 episodes, and a lobotomize/restore toggle. His trained fly reached 96% touch but only 19% tracking; he says it is bad at search. His screenshot shows decode bars for DNa02, DNa01, and DN_all, so he very likely derives forward speed from population descending activity, not from DNa01/DNp09 alone. That is the leading hypothesis for why his fly advances and ours only turns; see TODAY.md task 1.

His camera, confirmed: 320 x 240 RGB, pinhole, no distortion, 82 degrees vertical, 98.43 horizontal (focal length 138 px), turret-mounted 23.1 cm above ground, one frame per sim step with dt. He outputs raw pixels only; the retina encoding is ours (brain/retina.py, 24 angular columns, width not height because a person's head leaves the frame inside ~1.75 m). He also simulates an HC-SR501 PIR: single digital output, 120 degree cone, 7 m, 5 s hold. We asked him to keep pan/tilt at 0, dump ground-truth human positions per frame, provide two PIR floats left/right, and tell us the arena size and which laptop drives the robot. Humans in his sim are procedural and seeded (height, build, skin, clothing vary by seed), so domain randomization of appearance is already built in on his side; our analytic retina never sees pixels in training anyway.

Contract between the two sims: same column semantics (presence, size, motion per column; loom L/R), same motor sign (turn > 0 = clockwise = positive z in chassis.drive_speed), same camera constants.

## 7. Hardware and the demo

Robot: DJI RoboMaster S1 (request the moment hardware inventory opens at the event; 10 were listed). Python SDK `robomaster`, chassis.drive_speed(x, y, z), mecanum wheels. Backup: an RC car with ESP32. Compute: the brain runs on a laptop next to the robot, never in the cloud; latency budget under 100 ms camera to wheels (measured 15 ms without the detector, YOLO adds 20 to 40 ms on CPU). A Raspberry Pi 5 is backup compute, not the plan.

Demo choreography: split screen, robot on the floor, brain on the wall. A judge walks toward the robot; retina columns light, LC10a flares, DNa02 fires, the robot turns and closes. Then the lesion button: knock out LC10a and it goes still. Restore and it comes back. Foam swatter, not electric (organizers will shut down an electric one). Start the demo with the person in view; search is not solved and we say so.

## 8. Practical constraints

- Event starts Friday evening; the team takes a shuttle from Toronto Friday night.
- Taka's Claude quota must last the weekend: use the cheaper model for mechanical work, subagents for grunt tasks, no open-ended loops.
- Local git only. Never push, publish, or send anything anywhere.
- The Vast.ai box was destroyed; everything runs on the M2 Pro (MPS dense engine for training at 64 envs, CPU event engine for the robot at batch 1).
- Cite every external repo and paper in README (Eon fly-brain, Shiu 2024, connectome-pilot, Imperol3/flybrain, Keles and Frye 2017, Berg 2026, neuprint-python, Ultralytics, RoboMaster SDK).
