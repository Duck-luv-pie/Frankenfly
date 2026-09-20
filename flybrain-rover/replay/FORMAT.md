# Replay JSON format (scripts/dump_replay.py / train.py --record)

One file = one episode of one environment (env 0 of the batch), sampled at 50 Hz (every env step, 20 ms).
Units: metres, radians, seconds, Hz. World frame: x right, y up (2D top view), yaw CCW-positive, 0 = +x.
Robot-relative azimuth/bearing: + = robot's RIGHT (matches image x). Motor turn: + = turn right (clockwise).

```
{
  "meta": {
    "N": 15000,                 // neurons in the simulated brain
    "viz_neurons": [..512 ids], // fixed random sample of neuron indices used by frames[].spikes
    "arena_L": 6.3,             // square arena side (m), centred at the origin: walls at +-L/2
    "dt": 0.02,                 // seconds between frames
    "rover_lw": [0.32, 0.17],   // rover box length (along heading) and width (m)
    "stage": "A", "gain": 0.05,
    "brain": "data/brain.npz", "checkpoint": null, "lesion": [], "amp_tonic": 0.0,
    "seed": 7, "seconds": 15.0, "hz": 50, "env_index": 0,
    "summary": {"contact_rate": 0.25, "forward_mean": 0.0, "dn_rates": {...}}
  },
  "frames": [ { ...one object per step, fields below... } ]
}
```

Per-frame fields
| field | type | meaning |
|---|---|---|
| `t` | float | seconds since episode start |
| `rxy` | [x, y] | rover centre position (m) |
| `ryaw` | float | rover heading (rad, CCW from +x). Rover box: length 0.32 along heading, width 0.17; front strip = last 0.06 m |
| `hxy` | [[x, y], ...] | positions of the valid humans (m); list length = number of humans this episode |
| `hr` | [r, ...] | cylinder radius of each human (m), same order as `hxy` |
| `forward` | float in [-1, 1] | motor forward command (× v_max ≈ 1.15 × human speed, ~1–1.6 m/s) |
| `turn` | float in [-1, 1] | motor turn command, + = right; × 2.5 rad/s |
| `reward` | float | reward delivered this step (+1 acquire, +5 first front contact, +0.5 sustained, −0.02 no contact, −3 timeout) |
| `heat` | [L, R] | left/right heat detector outputs in [0, 1] (60° half-cones at ±30°, 7 m range, leaky) |
| `pres` | [24 floats] | retina columns: fraction of each 4.1° azimuth column covered by a human, left→right (column 0 = −49°, column 23 = +49°) |
| `size` | [24 floats] | angular width of the covering human / 98.43° hfov, per column |
| `mot` | [24 floats] | d(pres)/dt per column, clipped to [−1, 1] |
| `loom` | [L, R] | positive rate of width expansion per hemisphere (looming), 1/s |
| `rates` | {group: Hz} | mean firing rate of each named neuron group at this step: LC10a_L/R, LC11_L/R, LC12_L/R, LC15_L/R, LC4_L/R, LPLC2_L/R, THERMO_L/R, DNa02_L/R, DNa01_L/R, DNp09, GF, PAM, PPL1, KC, MBON |
| `spikes` | [int, ...] | indices into `meta.viz_neurons` of sampled neurons that spiked at least once during this 20 ms step |

Rendering hints: draw the rover as an oriented box at (`rxy`, `ryaw`); humans as cylinders of radius `hr`;
the camera FOV as a ±49.2° wedge; colour the 24 `pres` columns as a fan in front of the rover; plot `rates`
as bars (DNa02_L vs DNa02_R shows steering, DNa01/DNp09 forward, GF escape, PAM/PPL1 dopamine); flash
`spikes` on a 512-dot raster. Episode files: `episode_<seed>_untrained.json`, `_trained.json` (best
checkpoint), `_lesioned.json` (LC10a removed), `_tonic.json` (fallback forward drive).
