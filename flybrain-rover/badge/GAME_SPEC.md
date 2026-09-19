# SWAT — badge game design spec

**One line:** you race a real fruit fly's escape reflex, in milliseconds, and you lose until you start
removing its neurons.

Build target: a second badge app (`swat`), alongside the existing `flybadge`. Reuses the connectome and the
LIF verbatim from `badge/app_template.lua`, so the hard part is already written and tested. Estimated 250 to
320 lines of new Lua, most of it UI and the state machine.

---

## 1. Why this game and not another game

A generic badge game is judged against every other badge game. This one cannot be written by anyone else at
the event, because the opponent is 150 neurons and 900 synapses of a real *Drosophila* connectome running at
100 Hz on the badge. Every design decision below exists to make that legible in ten seconds without a word
of explanation.

The hook a judge will repeat afterwards: **"I lost a reaction test to a fruit fly, and then I lesioned it and
won."**

---

## 2. The core loop

```
IDLE ──A──► ARMED ──(random 800-2200 ms)──► LOOMING ──┬── you press A first ──► HIT
  ▲                                                    │
  │                                                    └── giant fibre fires first ──► ESCAPED
  └──────────────── 1400 ms result screen ◄────────────┘
```

During **LOOMING**, a dark disc grows from the centre of the screen. Its expansion rate is injected as
current into LC4 and LPLC2, exactly as `flybadge` injects a swat. The real circuit decides when the giant
fibre spikes. You press **A** to swat. Whoever is first wins the round.

The round always reports two numbers side by side: **your reaction time** and **the fly's escape latency**,
both in milliseconds, both measured with `badge.sys.ms()` from the same instant (the moment looming starts).

---

## 3. Measured constants (do not guess these, they were measured)

Run `python badge/export_badge_circuit.py` then the sim to reproduce. `RAMP` is added to the injected
current every 10 ms brain step, capped at 4000.

| Difficulty | `RAMP` | GF fires at | Feels like |
|---|---|---|---|
| `SLOW` | 45 | 420 ms | you win most rounds |
| `NORMAL` | 80 | 250 ms | genuine coin flip |
| `FAST` | 120 | 170 ms | you lose most rounds |
| `INSANE` | 200 | 110 ms | you cannot win |

Lesioned (LC4 + LPLC2 outgoing synapses zeroed): the giant fibre **never** fires, even at maximum drive for
three seconds. Verified: 0 spikes.

Start the game on `NORMAL`. Human visual reaction time is about 250 ms, so `NORMAL` is deliberately the
knife edge.

---

## 4. Screen layout, 320 x 240, exact

All coordinates are top-left origin, integers, as `badge.ui` requires.

```
┌────────────────────────────────────────────────────┐
│ SWAT                                    07 / 10    │  y=6   title (coral, 20px) | score right (18px)
│ 150 neurons · NORMAL                               │  y=30  subtitle (dim, 14px)
│                                                    │
│                    ████████                        │  the looming disc: a box, centred on (160,118),
│                  ████████████                      │  grown from 10x10 to 210x210 over the round
│                  ████ ** ████                      │  the fly sprite sits at (150,110), on top
│                  ████████████                      │
│                    ████████                        │
│                                                    │
│  you 231 ms          fly 250 ms                    │  y=186  both times (20px); winner in coral
│  A swat   B lesion   UP/DOWN speed   START reset   │  y=216  hint (dim, 14px)
└────────────────────────────────────────────────────┘
```

Widget budget: 8 widgets total, far under the 512 cap.

| id | factory | position / size | notes |
|---|---|---|---|
| `bg` | `box(root,320,240)` | `set_pos(0,0)` | `bg_color 0x000000`, `border_width 0` |
| `disc` | `box(root,10,10)` | recentred each tick | `bg_color 0x2A0E0A`, `radius 105` makes it a circle |
| `fly` | `label(root,"**")` | `set_pos(150,110)` | white, 24px, jumps to y=74 on escape |
| `title` | `label` | `align("top_left",10,6)` | coral `0xFF715B`, 20px |
| `sub` | `label` | `align("top_left",10,30)` | dim `0x8A8580`, 14px, shows difficulty |
| `score` | `label` | `align("top_right",-10,6)` | 18px, `07 / 10` |
| `times` | `label` | `align("top_left",10,186)` | 20px, the two numbers |
| `hint` | `label` | `align("top_left",10,216)` | dim, 14px, static |

**Disc geometry.** Create it once. Each tick during LOOMING:
```lua
local d = 10 + math.floor(200 * progress)      -- progress 0..1, see below
disc:set_size(d, d)
disc:set_pos(160 - d // 2, 118 - d // 2)       -- keep it centred
disc:style({ radius = d // 2 })                -- a box with radius = half its size is a circle
```
`progress` is **not** wall time: it is `cur / CUR_MAX` where `cur` is the current being injected. That way
the picture and the neurons are the same signal, which is the whole point. Bring `disc` to front once at
creation, then `fly:bring_to_front()` so the sprite sits on top.

---

## 5. LEDs — the fly getting nervous

This is the second thing a judge notices and it costs almost nothing. Map the eye's population firing rate
to how many of the 6 LEDs are lit, so the badge visibly tenses before it escapes.

- **IDLE / ARMED:** all off.
- **LOOMING:** `n = math.min(6, math.floor(eye_spikes_this_tick / 3))` LEDs lit in coral `(255,113,91)`,
  filled clockwise from upper left using the index order `{1,2,3,4,5,6}`. Write only when `n` changes.
- **ESCAPED:** all six white `(255,255,255)` for 180 ms, then off. The fly sprite jumps to y=74 for 280 ms.
- **HIT:** all six green `(40,200,90)` for 180 ms.
- **LESIONED (any state):** LED 1 dim red `(60,0,0)` so the state is visible across a table.

Stage with `set`/`set_all`, then one `show()`. Never call `show()` more than once per tick.

---

## 6. State machine, exact

Keep every timestamp from `badge.sys.ms()`. Never count ticks; `on_tick` is nominal 20 ms and not guaranteed.

| state | entered when | what runs each tick | leaves when |
|---|---|---|---|
| `IDLE` | boot, or 1400 ms after a result | brain stepped with `cur = 0` | **A** pressed → `ARMED` |
| `ARMED` | from IDLE | brain stepped with `cur = 0`; disc hidden | `now >= arm_until` → `LOOMING`. **A** pressed → `FALSE_START` |
| `LOOMING` | from ARMED | `loom = loom + RAMP` per 10 ms step, `cur = min(4000, loom)`; 2 steps per tick; disc grows | GF spikes → `ESCAPED`. **A** pressed → `HIT`. 3000 ms elapsed → `ESCAPED` (safety) |
| `HIT` | A before GF | brain frozen | 1400 ms → `IDLE` |
| `ESCAPED` | GF before A | brain frozen | 1400 ms → `IDLE` |
| `FALSE_START` | A during ARMED | brain frozen | 1400 ms → `IDLE`, round counts as a loss |

`arm_until = now + 800 + badge.sys.random(1400)`.

Timing, precisely:
- `loom_t0 = badge.sys.ms()` at the instant `LOOMING` is entered.
- Your reaction: `press_ms = badge.sys.ms() - loom_t0`, captured inside `on_button`, not `on_tick`.
- The fly's latency: `gf_ms = steps_since_loom * 10`, where `steps_since_loom` counts brain steps, because
  the brain's clock is exact and the tick clock is not. This is the honest number and it must match the
  table in section 3.

Round ends on whichever happens first. If both land in the same tick, the brain wins (the fly is faster
than the button scan); say so on screen as `fly 250 ms · same tick`.

---

## 7. Controls

| button | IDLE | ARMED | LOOMING | result |
|---|---|---|---|---|
| **A** | start the round | false start | swat | ignored |
| **B** | toggle lesion | toggle lesion | toggle lesion (takes effect immediately) | toggle |
| **UP / DOWN** | difficulty up / down | same | ignored | same |
| **START** | reset score to 0/0 | same | ignored | same |
| **HOME** | exits (default behaviour, do not intercept) | | | |

Difficulty cycles `SLOW → NORMAL → FAST → INSANE`. Persist both difficulty and lesion state with
`badge.store.set_int("diff", n)` and read them in `on_enter`; save in `on_exit`.

**The lesion is the teaching moment.** When B is pressed, the subtitle changes to
`LC4 + LPLC2 CUT · it cannot see you` in red, and the fly never escapes again. When restored, back to
`150 neurons · NORMAL` in dim. Do not explain it in words anywhere else; the score doing the talking is
better.

---

## 8. What to copy, verbatim, and what to write

**Copy unchanged from `badge/app_template.lua`** (these are tested against the Python reference spike for
spike, do not modify them):
- the whole injected `--@CIRCUIT@` block,
- the constants `V_REST, V_TH, SYN_DECAY, GAIN_Q, REFRAC, CUR_MAX`,
- `reset_brain()`,
- `step(cur, flirt)` — call it as `step(cur, false)`.

**Write new:** the state machine, the UI, the LED mapping, scoring, persistence.

**Do not** re-derive the circuit, change the fixed-point constants, or add any rule that makes the fly
escape. If you find yourself writing `if swat_near then escape()`, stop: the escape must only ever come out
of `step()` returning a giant-fibre spike. That property is the entire entry.

---

## 9. Build and install

```
cd ~/Downloads/filess/flybrain-rover
python badge/export_badge_circuit.py                                   # if the circuit changed
python badge/build_app.py --template badge/swat_template.lua --out badge/swat_app.lua
lua badge/test_lua.lua                                                 # sanity: the LIF still matches Python
```
Then paste `badge/swat_app.lua` into `badge.hackthenorth.com/ide` → Import app → Connect → Push.

Manifest header, first lines of `badge/swat_template.lua`:
```lua
--[==[badge-app
slug=swat
name=SWAT
icon=><
api=2
heap_kb=96
wake_lock=1
version=1.0
author=FlyBrain Rover
]==]
```
`slug=swat` must differ from `flybadge` so both apps coexist on the badge. `heap_kb=96` is required: the
badge compiles `main.lua` in RAM and the circuit is 6.8 KB of source.

---

## 10. Pass criteria

Ship it when all seven are true:

1. On `NORMAL`, ten rounds against an intact fly give between 2 and 7 hits for an average player. If it is
   0 or 10, the ramp constant is wrong.
2. The reported `fly` latency on each difficulty matches section 3 within 20 ms. If it does not, the brain
   is not being stepped twice per tick or the ramp is being applied per tick instead of per step.
3. Press **B**: the fly never escapes again, across at least 20 rounds at `INSANE`.
4. Press **B** again: escape returns at the same latency as before, within 20 ms.
5. `ms/tick` stays under 10 at all times (add a temporary debug label; the existing app shows how).
6. The app survives a power cycle with score reset but difficulty and lesion state restored.
7. No `if` statement anywhere decides whether the fly escapes.

---

## 11. Stretch, only if 1 to 7 are done

**Two badges, one fly each, over `badge.radio`.** Both badges broadcast `SWAT1:<seed>` and use the same
`badge.sys.random` seed so the arm delay and ramp are identical. Each human races their own fly; after the
round, exchange `SWAT1:<press_ms>` and show both on screen: `you 231 · them 198 · fly 250`. Keep payloads
under 44 bytes, drain at most 4 frames per tick, and fall back silently to single player if
`badge.radio.enable()` returns false.

**Streak LEDs.** Three hits in a row: a clockwise chase in coral before the next round arms.

---

## 12. What this is honestly

The fly's escape latency here is the real circuit's response to a ramp we chose. A real *Drosophila* giant
fibre escape is faster than any of these numbers, and the looming stimulus is current we inject rather than
an image on the fly's eye. What is real: the neurons, the synapses, their counts and signs, and the fact
that removing LC4 and LPLC2 abolishes the escape. Say that if a judge asks, and say it before they ask.
