# The control vocabulary

Two robot stacks, two servers, one set of words. This exists because we grew three lobotomies with
three different names, and a control that one server silently ignores is the kind of thing nobody
notices until a judge is standing in front of it.

## The words

Every control goes over `POST /control` as JSON, on either server.

| key | meaning | who already spoke it |
|---|---|---|
| `lobotomy` | cut the visual pathway. The fly can no longer be told a person is there. | Ducks — his page, his `/status`, and his GPIO button on the second Pi |
| `rover` | `false` detaches the wheels, `true` reattaches. The brain keeps running either way. | Ducks — `--rover-off`, the page's rover button |
| `paused` | hold the tick loop | Ducks |
| `heat` | the thermal channel on or off | Ducks |
| `speed` | playback or fly speed | ours |
| `halted` | **alias for `rover` inverted.** `{"halted": true}` == `{"rover": false}` | ours, kept for `scripts/talk.py` and the operator panel |

`halted` is an alias, not a second concept. `scripts/viz_adapter.py` translates `rover` into it before
anything reads it, an explicit `halted` wins if both are sent, and every frame reports both names.
`tests/test_lobotomy.py::test_rover_is_an_accepted_alias_for_halted` holds that.

## The two lobotomies are genuinely different, and that is fine

Both answer `{"lobotomy": true}`. What they then do is not the same thing, and neither is wrong:

- **Ducks's** (`brain/companion_brain/hunt_gpu/real.py`, `viewer.py`) keeps a second `SpikingHunter`
  with `sever_senses()` applied and swaps which brain drives. A whole-brain substitution.
- **Ours** (`scripts/robot_bridge.py`, `Brain.set_blind`) lesions LC10a, LC4 and LPLC2 — zeroing their
  synapses, so the cells keep firing and nothing downstream hears them — and lets the internal search
  state take over. Measured on the demo brain with a person in view: forward `+0.278 → +0.039`, and the
  turn starts alternating sides instead of holding.

Say "the projection is cut", not "the neurons are gone". The cells are still spiking in both cases.

## Read-only, and who gets to press things

Both servers serve the same page, `brain/companion_brain/ui/hunt_gpu.html`, which asks `/config` on
load and hides every control when `read_only` is true.

- **Senthil's `serve()`** defaults `--public` **on**, so spectators get a read-only page and every POST
  returns 403. That is right for anyone watching over the network.
- **`viz_adapter.py`** answers `read_only: false`, because the person in front of it is the presenter.

Do not "unify" these to one default. They are different jobs and the difference is deliberate.

## Keyboard, on our stack

`scripts/demo.py`. Also accepted as single bytes over UDP `127.0.0.1:9600`, which is how
`scripts/talk.py` drives it.

| key | |
|---|---|
| `1` | baseline: undo every lesion, restore learning, clear lobotomy and stop |
| `2` / `3` | lesion / restore LC10a both sides |
| `4` / `5` | wipe learning / restore it |
| `6` | wiring shuffle on/off |
| `7` | remove another quarter of LC10a |
| `8` / `9` | **lobotomize / wake up** |
| `0` | **stop / resume the wheels** (`rover` off/on) |
| — | the badge does not use a keyboard hotkey. `scripts/badge_link.py` presses `8`/`9`/`0` over UDP when you press **B** or **RIGHT** on the badge itself. An earlier plan bound `w` for this; nothing binds it now. |
| `l` | silence the brain entirely |
| `r` / `p` | reward / punish (PAM / PPL1 burst) |
| `space` | E-stop |
