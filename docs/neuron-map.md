# Neuron map

All neuron groups are selected from the FlyWire community annotations
(`Supplemental_file1_neuron_annotations.tsv`, Schlegel et al. 2024) by `cell_type`, regex, or
neurotransmitter, then split by the `side` column. Counts below are what the v783 model contains.
Edit `brain/configs/default.yaml` to change any of this.

## Senses → input neurons

| Robot signal | Group (cell types) | Count L/R | Why |
|---|---|---|---|
| Fast expansion of a visible moving region | `LC4` | 54/50 | LC4 encodes looming speed and drives the Giant Fiber (von Reyn et al. 2017) |
| same | `LPLC2` | 108/102 | LPLC2 detects radial expansion, synapses onto the Giant Fiber (Klapoetke et al. 2017) |
| Slow, sustained expansion | `LC16` | 77/74 | LC16 activation drives backward walking (Wu et al. 2016) |
| Small dark moving object | `LC11` | 66/61 | small-object detector (Keleş & Frye 2017) |
| same | `LC10a` | 115/119 | small-object tracking, courtship pursuit (Ribeiro et al. 2018) |
| Moving vertical edge | `LC12`, `LC15` | 198/182, 52/54 | bar / edge detectors |
| PIR rising edge (warm body arrived) | `JO_wind` = `JO-C*`, `JO-E*` | 229/204 | Johnston's organ C/E neurons respond to wind and static antennal deflection (Yorozu et al. 2009); A/B are the sound channels |

Each group is driven as a Poisson process at `feature × gain × 150 Hz`, on the side of the
hemifield where the feature was seen.

## Readouts → behavior

| Behavior | Groups (cell types) | Count | Evidence |
|---|---|---|---|
| **escape** | `GF` = `DNp01` (Giant Fiber) | 1/1 | escape takeoff command neuron |
| **backward** | `MDN` | 2/2 | moonwalker descending neurons, backward walking (Bidaye et al. 2014) |
| **freeze** | `DNp09` | 1/1 | freezing on looming (Zacarias et al. 2018) |
| **threat** | `DNp02`, `DNp04`, `DNp11`, `pC1_aggression` = `pC1d`, `pC1e` | 1/1 each, 2/2 | looming-responsive DNs and the female aggression cluster (Schretter et al. 2020) |
| **groom** | `DNg11`, `DNg12*` | 3/3, 21/21 | front-leg grooming descending neurons (Guo et al. 2022) |
| **land** | `DNp07`, `DNp10` | 1/1 | landing-associated DNs (Ache et al. 2019) |
| **social** | `pC1_social` = `pC1a`, `pC1b`, `pC1c` | 3/3 | female receptivity / social cluster (the FlyWire brain is female, so P1 does not exist) |
| **track** | `DNa01`, `DNa02` | 1/1 | steering descending neurons (Rayshubskiy et al. 2020); right minus left drives turning |
| *forward walking (motor channel only)* | `DN_all` = every `descending` neuron | 337/337 | descending population activity tracks walking (Aymanns et al. 2022); used as the continuous walking drive, not as a state |

Modulators (continuous, not states):

| Modulator | Plus | Minus | Expression |
|---|---|---|---|
| valence | `MBON_approach` = MBON07, 09, 11, 12, 14 | `MBON_avoid` = MBON01, 02, 03, 05 | iris tint green ↔ red (approach/avoid assignment approximates Aso et al. 2014) |
| arousal | `octopamine` (top_nt) | | pupil dilation |
| reward | `DAN_reward` = `PAM*` | `DAN_punish` = `PPL1*` | logged, free for future use |

All readouts are z-scores against the brain's resting activity measured at startup; see
[architecture.md](architecture.md#reading-a-noisy-brain-baselines-and-z-scores).

## What the circuit does on its own

`companion test-gf`: with no input the network is silent. Driving LC4 + LPLC2 on one side at
100 Hz makes the Giant Fiber spike within ~10 ms and fire at ~175 Hz, and also recruits DNp02,
DNp04, DNp11 (threat), DNa02 (steering) and a little MDN — none of which is programmed; it is
what the wiring does.

## Naming notes

- `GF` is annotated as cell type `DNp01` (hemibrain type "Giant Fiber").
- Antennal grooming neurons called `aDN1/2` in the literature are not labelled in the
  annotations; `DNg11`/`DNg12` cover grooming.
- Courtship song neurons (`pIP10`, `vPR6`, `P1`) are male-specific and absent from this female
  brain; `pC1` is the female counterpart.
