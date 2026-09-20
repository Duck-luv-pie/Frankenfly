# SWATTER, badge game design spec

**One line:** animals land in front of you, you swat the flies and spare everything else, and the flies are
hard to hit because a real fly's escape reflex is running on the badge deciding whether it gets away.

This replaces the earlier pure reaction test (`GAME_SPEC_swat_v1.md`). That version raced the fly's reflex,
which was true but thin. This one adds the thing that makes it a game: **a decision**. You cannot just be
fast, because being fast at the wrong target costs you.

Build target: one badge app (`swatter`). Reuses the connectome and the fixed-point LIF verbatim from
`badge/app_template.lua`. Around 350 lines of new Lua, most of it UI and the state machine.

---

## 1. Why this design

Two things are true about a fly, and both are in the game:

1. **A fly is hard to swat**, because LC4 and LPLC2 detect your hand looming and the giant fibre launches an
   escape before your hand arrives. That is the circuit we already have running. It is the difficulty.
2. **You should not swat everything.** The cost of being trigger-happy is what turns a reaction test into a
   game. So other animals land too, and hitting them costs you a life.

The result is a speed-versus-accuracy squeeze where **the speed limit is an animal's real reflex**. You get
about 180 ms before you can tell what landed, and the fly escapes at 470 ms from the same instant. That
290 ms window is exactly where human go/no-go reaction sits. Those numbers are measured off the circuit,
not tuned until it felt right.

Then the twist no other badge game can do: **press B and you lesion the fly's eye.** Flies stop escaping
and the game becomes trivial. The difficulty knob is a neuroscience experiment.

---

## 2. The core loop

```
IDLE ──A──► SPAWN ──(random 400-1200 ms)──► TARGET ──┬── swat, was a fly, GF had not fired ──► HIT       +1
  ▲                                                   ├── swat, was a fly, GF fired first   ──► ESCAPED   0
  │                                                   ├── swat, was NOT a fly               ──► FOUL     -1 life
  │                                                   ├── waited, was NOT a fly             ──► SPARED   +1
  │                                                   └── waited 2500 ms, was a fly         ──► FLEW OFF  0
  └───────────────────── 900 ms result card ◄─────────┘
```

A round is one animal. Three lives. The run ends at zero lives; the score is flies swatted.

**The looming clock starts the instant the animal becomes visible.** From the fly's point of view something
is approaching from above (you), so current ramps into LC4 and LPLC2 from that moment. Your reaction time
is therefore the fly's exposure: the slower you are, the more of your hand it has seen.

---

## 3. Measured constants, do not invent these

`RAMP` is added to the injected current each 10 ms brain step, capped at 4000. Reproduce with
`python badge/export_badge_circuit.py` then `python badge/sim_fixed.py`.

| ramp | giant fibre fires at | use |
|---|---|---|
| 60 | 320 ms | `HORNET`, round 15+, effectively unwinnable, which is the joke |
| 50 | 380 ms | `QUICK` fly, rounds 8 to 14 |
| **40** | **470 ms** | **the default fly, rounds 4 to 7. Human go/no-go is 350 to 450 ms.** |
| 32 | 580 ms | `SLOW` fly, rounds 1 to 3, the tutorial |
| lesioned | never | 0 giant fibre spikes in 3 s at full drive, verified |

Difficulty follows the round number, never a menu. Tell the player only through the animal's name.

---

## 4. The animals

Four types, told apart by **colour first, size second, word third**. The word is a fallback for someone
watching over a shoulder, not the primary channel.

| animal | swat? | colour | body | word | why it is there |
|---|---|---|---|---|---|
| fly | **yes** | `0x2A2A2A` near-black, `0xFF715B` eyes | 44 x 30 | `fly` | the only one with the circuit |
| bee | no | `0xE8B339` amber | 52 x 34 | `bee` | close enough to a fly to punish mashing |
| chicken | no | `0xF2E6D8` pale, `0xD94F3A` comb | 96 x 78 | `hen` | unmistakable, a free point if you wait |
| ladybird | no | `0xD94F3A` red | 40 x 32 | `bug` | fly-sized and tempting |

Spawn weights: fly 55%, bee 20%, ladybird 15%, chicken 10%, with at least one non-fly in every four rounds
so mashing cannot pay.

**The resolve animation is the discrimination challenge.** For the first **180 ms** every animal is drawn
as the same dark blob at the same size. Then it snaps to its real colour, size and word. So the earliest
honest decision point is 180 ms, and the fly escapes at 470 ms from the same instant.

---

## 5. Screen, 320 x 240, exact

```
┌────────────────────────────────────────────────────┐
│ SWATTER                              12   ***      │  y=6   title left, score and lives right
│                                                    │
│                  ▓▓▓▓▓▓▓▓▓▓                        │  the animal, centred on (160,116)
│                ▓▓▓  o   o  ▓▓▓                     │
│                  ▓▓▓▓▓▓▓▓▓▓                        │
│                      fly                           │  y=150, only after 180 ms
│                                                    │
│  ████████████░░░░░░░░░░░░░░░░░░░░░░░░              │  y=196  the fly's alarm: LC4 + LPLC2 firing
│  swat it                                           │  y=218  one line, changes per state
└────────────────────────────────────────────────────┘
```

Eleven widgets, created once in `on_enter` and reused:

| id | factory | notes |
|---|---|---|
| `bg` | `box(root,320,240)` | `bg_color 0x0A0A0A`, `border_width 0` |
| `body` | `box(root,44,30)` | the animal; `set_size`, `set_pos`, `style` per round; `radius` half the height |
| `eyeL`, `eyeR` | `box(root,7,7)` | `radius 4`, hidden for the chicken, repositioned per type |
| `word` | `label` | centred at `(160,150)`, hidden until 180 ms |
| `title` | `label` | coral, 18px, `align("top_left",10,6)` |
| `score` | `label` | 18px, `align("top_right",-58,6)` |
| `lives` | `label` | `align("top_right",-10,6)`, text `"***"` trimmed to the life count |
| `alarm` | `bar(root,0,100,0)` | `set_size(300,10)`, `set_pos(10,196)`, coral on `0x221E1B` |
| `hint` | `label` | dim, 14px, `align("top_left",10,218)` |
| `flash` | `box(root,320,240)` | result tint, `bg_opa` 0 normally |

Worst-case per tick: the alarm value, the body size and position, two eyes. **At most 6 native UI calls per
tick.** Colours and words change only on transitions.

**The alarm bar is the fly's own eye**, driven by the live LC4 plus LPLC2 population rate, scaled so it
fills as the giant fibre nears threshold. It is the tell: nearly full means you are too late. It is also
what makes the lesion legible, because lesioned the bar still fills and nothing ever happens.

---

## 6. LEDs

The six LEDs mirror the fly's alarm so someone across the table sees it too. Write only on change, one
`show()` per tick at most.

- **fly on screen:** `n = floor(6 * eye_rate / threshold_rate)` lit in coral, filling clockwise from upper
  left. They visibly charge as the fly becomes more frightened.
- **anything else on screen:** LEDs stay dark. Their silence is a fair hint: a bee has no giant fibre here.
- **ESCAPED:** six white, 200 ms. **HIT:** six green `(40,200,90)`, 200 ms. **FOUL:** six red `(200,30,30)`, 400 ms.
- **LESIONED:** LED 1 dim red throughout, visible from a distance.
- **Game over:** one slow clockwise red chase.

---

## 7. Controls

| button | IDLE | TARGET | result card |
|---|---|---|---|
| **A** | start a run | swat | skip |
| **physical swat** | start a run | swat, same as A | skip |
| **B** | toggle lesion | toggle lesion, immediately | toggle |
| **START** | reset score and lives | abandon the run | reset |
| **HOME** | exits, default behaviour, do not intercept | | |

Both A and a real swat strike, because swatting the badge is the verb the game is named after and the
accelerometer is already read for the looming input. Debounce a physical strike to one per 400 ms.

---

## 8. Scoring

- **HIT** a fly: `+1`, plus a bonus point for every third in a row without a foul, shown as a combo.
- **ESCAPED**: 0 and no penalty. Losing to the fly is the normal case, not a punishment.
- **SPARED** a non-fly: `+1`. Restraint pays the same as speed.
- **FOUL**: `-1 life`, combo resets.
- **FLEW OFF**: 0, combo resets.

Best score in `badge.store.set_int("best", n)`, shown on the idle screen.

---

## 9. What is real and what is not

**Real:** 150 neurons and 900 synapses of the male *Drosophila* connectome with their measured counts and
signs, the fixed-point LIF at 100 Hz, every latency in section 3, and the fact that cutting LC4 and LPLC2
abolishes escape entirely.

**Not real:** the bee, chicken and ladybird have no circuit, they are cardboard. The looming ramp is a
number we chose, not a real hand. A real giant fibre escape is far faster than 470 ms; we slowed the
stimulus so a human can play against it at all.

Say this on the idle screen and in the Devpost. The game is more impressive with the caveat than without,
because the caveat shows which parts are measurements.

---

## 10. Code reuse, and the one rule

**Copy verbatim from `badge/app_template.lua`:** the injected `--@CIRCUIT@` block, the chunked `b64`
decoder, the fixed-point constants, `reset_brain()` and `step()`. Do not retune the constants. Do not touch
`badge/flybadge_circuit.lua`.

**The rule:** nothing may make a fly escape except `step()` returning a giant-fibre spike. If you write
`if elapsed > 470 then escape()` the entry is dead, because the claim is that a measured circuit decides
it. A test enforces this.

**Do not enable the radio.** It panics the badge with this circuit loaded; see `badge/SDK_NOTES.md`.

---

## 11. Build and install

```
cd ~/Downloads/filess/flybrain-rover
python badge/export_badge_circuit.py
python badge/build_app.py --template badge/swatter_template.lua --out badge/swatter_app.lua
lua badge/test_swatter.lua
python -m pytest tests/test_swatter_app.py -q
```

Manifest: `slug=swatter`, `name=Swatter`, `icon=SWT`, `api=2`, `heap_kb=96`, `wake_lock=1`. A different slug
from `flybadge` so both live on the badge. **Reboot after the first push**, not just `reload`, or `heap_kb`
is ignored.

---

## 12. Pass criteria

1. Escape latency per difficulty matches section 3 within 20 ms.
2. Lesioned: zero escapes across 30 rounds at ramp 60.
3. Un-lesioning restores the latency within 10 ms.
4. Worst-case tick under 20 native calls and under 10 ms.
5. Widgets created exactly once; `show()` at most once per tick.
6. All four types are pixel-identical for the first 180 ms and differ after, asserted in the harness.
7. Swatting a non-fly always fouls, whatever the circuit is doing.
8. No `if` anywhere decides whether a fly escapes.
9. Survives a power cycle with the best score kept.
