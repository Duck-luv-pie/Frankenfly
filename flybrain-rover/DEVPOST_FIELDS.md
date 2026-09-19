# DEVPOST_FIELDS.md — every submission field, ready to paste

Deadline: Sunday. Form is at Hack the North 2026 > My projects > the draft. Five steps; only the last one submits.
Team decision needed on the project name, because the shared repo is called `hunting-fly` and this half is `flybrain-rover`.

---

## Step 1, Project overview

### Project name (60 characters)
Recommended, because it is the repo the judges will open:

```
Hunting Fly
```

Alternatives: `Hunting Fly: a real connectome driving a robot` (46), `The Fly That Finds You` (22), `FlyBrain Rover` (14).

### Elevator pitch (200 characters)
Recommended, 185 characters:

```
A real fruit fly's connectome, 15,000 neurons spiking live, drives a robot that finds you. Nobody trained it. Delete 275 neurons and it goes blind. Shuffle the same wiring and it stops.
```

Alternative, 177: `We put a real fruit fly's wiring diagram in a robot. Untrained, it turns toward you and reaches you. Remove 275 named neurons and it goes blind. Put them back and it sees again.`

### Thumbnail
`devpost/control.png` (3:2 crop) or a still of the robot facing a person. Not the screenshot of a terminal.

---

## Step 2, Project details

### About the project
Paste `DEVPOST.md` whole. It is already Markdown with the nine Devpost headings in order, every number carries its protocol, and it states plainly what we engineered versus what came out of the connectome.

### Built with (up to 25 tags)
```
python, pytorch, numpy, opencv, ultralytics, yolo, neuprint, connectome, spiking-neural-network,
computational-neuroscience, drosophila, robomaster, dji, esp32, sbus, three.js, javascript,
websockets, matplotlib, vast.ai, cuda, raspberry-pi
```
Drop `raspberry-pi` if the Pi does not end up in the loop, and `vast.ai` / `cuda` if you would rather not advertise cloud training (we used it only for training, never for the control loop).

### Try it out links
```
https://github.com/Duck-luv-pie/hunting-fly
```
Add a second link only if the 3-D viewer gets hosted. The repo README explains both halves; ours is `flybrain-rover/`.

### Image gallery (up to 15, 3:2 works best)
In this order, because the story is the controls:

1. `devpost/control.png` — the wiring-shuffle control. The single strongest image: the eye still fires at 70 Hz, the steering neurons go from 158 Hz to zero.
2. `devpost/lesions.png` — every lesion against the intact circuit, as turn-toward fraction.
3. `devpost/graded.png` — deleting the tracking population a quarter at a time.
4. `logs/viz_his_ui_page.jpg` — the live 3-D viewer: robot, person, 24 retina columns, descending-neuron bars, and the 15,000-neuron brain map with spikes.
5. A photo of the robot facing a person, taken at the table.
6. A photo of the laptop screen mid-demo with the brain map lit.

Captions matter more than the images. Suggested caption for 1: "Same neurons, same number of connections, same synaptic strengths. Only which cell connects to which is randomized. The eye is untouched; nothing reaches the steering neurons."

### Video demo (YouTube or Vimeo, embedded at the top)
60 seconds, shot on a phone, landscape. Shot list:

| seconds | shot | voiceover |
|---|---|---|
| 0-8 | laptop screen: the brain map, spikes moving | "This is the wiring of a real fruit fly's brain, 15,000 neurons, running live as spiking neurons." |
| 8-22 | wide: person walks in, robot turns and drives to their feet | "The camera is its eye. The descending neurons, the ones that drive its legs, drive the wheels. Nothing in between is programmed, and nobody trained it." |
| 22-38 | screen: press 6, then the robot sitting still while a person moves | "Now the control. Same neurons, same connection counts, same strengths. I only randomize which cell connects to which. The eye still fires. The steering neurons get nothing." |
| 38-48 | press 6 again, robot resumes tracking | "Put the real wiring back and it finds you again." |
| 48-60 | screen: press 7 four times, rates dropping | "Delete the tracking neurons a quarter at a time and it fades out like a population, not a switch. This is a fly. It wants to find you." |

Record it twice and keep the better one. If the robot is unavailable, record the same script against the 3-D viewer and say so in the caption.

---

## Step 3, Additional info
Tracks to select are in `TRACKS.md`. Finalists is the target. The MLH tracks (Tiger Data, MongoDB Atlas, ElevenLabs) only if the integration is actually there by 2 PM Saturday; do not claim one we did not build.

## Step 4, Manage team
Add Ducks, Max and Senthil so it appears in everyone's portfolio.

## Before you hit Submit
- The repo link works from a logged-out browser (it is public).
- Every number in the story still matches `STATUS.md`.
- The video plays, is not private, and is under a minute.
- The honesty lines survived editing: the forward read-out and the search state are ours; the steering is the connectome's.
