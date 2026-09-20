# Hacker Badge SDK notes

Source: the badge app guide served at `badge.hackthenorth.com` (the IDE's own agent instructions), plus
the public badge page and the instruction manual. Written up 2026-09-19 for FlyBadge.

## The four questions

**1. What chip, and how much RAM.** ESP32-C3. The docs give app budgets rather than chip totals, and the
budgets are what actually bind:

| Limit | Value |
|---|---|
| Lua heap per app | 48 KiB default, 96 KiB with `heap_kb=96` |
| `main.lua` source | 64 KiB |
| Native widgets | 512 |
| Filesystem quota per app | 64 KiB total, 16 KiB per file |
| Persisted config | 32 keys, strings 128 bytes |
| Radio payload | 1 to 44 bytes |

`heap_kb` is a ceiling, not a reservation. The important trap: a source file under the 64 KiB cap can
still run the heap out **while compiling**, because the badge compiles `main.lua` in RAM. That is the
reason FlyBadge stores its connectome as two Lua strings decoded with `string.byte` instead of as a table
literal. A table of 900 numbers costs far more than a 900-byte string.

**2. What language.** Lua, sandboxed, single file, `api=2`. Present: `base`, `table`, `string`, `math`,
`utf8`. Absent: `os`, `io`, `package`, `debug`, `coroutine`, and from base also `pcall`, `xpcall`, `load`,
`loadfile`, `dofile`, `require` (a restricted `require` is reinstalled for app-local modules), and
`setmetatable`. Bytecode is refused; everything loads in text mode. Apps are event driven:

```lua
function on_enter(root) end    -- 3,000 ms budget, build the UI here
function on_tick() end         -- nominal 20 ms cadence, 250 ms budget, aim for a few ms
function on_button(b, kind) end-- 1,000 ms budget
function on_exit() end         -- 1,000 ms budget, save state
```

Three consecutive `on_tick` failures suspend ticks. There is no `sleep`; use `badge.sys.ms()`.

**3. How you build and load an app.** No CLI, no compile step, no dashboard upload. It is a browser flow:

1. Open `badge.hackthenorth.com/ide` in Chrome or Edge (WebSerial; Safari and Firefox will not work).
2. **Import app**, and paste the whole single file including its manifest header.
3. Battery switch **OFF**, then plug the badge in with a USB cable that carries **data**, not power only.
4. **Connect**, and choose the port named **USB JTAG/serial debug unit** (Espressif).
5. **Push**, and keep the cable attached until it finishes.

Files land in `/littlefs/apps/<slug>/`. Push adds and replaces but never deletes, so a renamed file leaves
the old one behind; remove it from the console with `rm /littlefs/apps/<slug>/<file>` then `reload`.
Changing a runtime manifest option (`heap_kb`, `wake_lock`, `home_button`) on an already-installed slug
needs a **Reboot**; code-only changes just need Push and reopening the app. The Sync tab on the Hacker
Dashboard is for backing up contacts, not for installing apps.

The manifest is a Lua long comment at the top of the same file, and the importer splits it out:

```lua
--[==[badge-app
slug=flybadge
name=FlyBadge
icon=FLY
api=2
heap_kb=96
wake_lock=1
]==]
```

`slug` must match the folder name and be `[a-z0-9][a-z0-9_-]{0,31}`.

**4. The sensor, LED, screen, button and IR APIs.** Two of those do not exist, which changed the FlyBadge
design:

- **There is no light sensor.** The only sensor exposed to Lua is an accelerometer. The scanner pattern at
  the bottom of the badge is the **NFC reader**, which reads cards and NDEF text, not brightness.
- **There is no IR API at all.** Connect and bump run over Bluetooth, and Lua cannot see those system
  frames: `badge.radio` is a separate restricted channel that prefixes every frame with `LUA1` and filters
  receive to the same. Two badges can talk to each other only if both run an app that uses it.
- **There is no audio API**, so no beep.
- Widgets have no Lua click handlers. All input goes through `on_button`.

What does exist:

```lua
-- screen: 320x240, badge.ui.screen_width / screen_height
local l   = badge.ui.label(root, "hello")
local b   = badge.ui.box(root, 292, 196)
local bar = badge.ui.bar(root, 0, 100, 40)            -- parent, min, max, value
-- also: arc slider image line button switch checkbox roller textarea
l:align("top_left", 10, 8)   l:set_pos(x, y)   l:set_size(w, h)
l:set_text("hi")   bar:set_value(40)   l:style({ text_color = 0xFFFFFF, text_font = 20 })
-- style keys: bg_color bg_opa color opa radius border_* text_color text_font text_align arc_* line_*
-- text_font: 14 16 18 20 22 24, or "small" "default" "large"
-- `root` is read-only: parent your widgets to it, never style or resize it.

badge.led.set(1, 255, 0, 0)    -- 6 LEDs, 1-indexed, R G B as three 0-255 integers, not 0xRRGGBB
badge.led.set_all(255, 255, 255)   badge.led.clear()   badge.led.show()   -- writes stage, show() latches
-- left side top to bottom {1,6,5}; right side top to bottom {2,3,4}; clockwise from upper left {1..6}

badge.input.BUTTON     -- A B HOME DOWN LEFT RIGHT UP AUX1 START
badge.input.KIND       -- PRESSED RELEASED
badge.input.is_down(badge.input.BUTTON.UP)    badge.input.held()   -- bitmask

local x, y, z = badge.sensor.accel()   -- milligravity, cached at 50 Hz, nil + err if unavailable
badge.sensor.shake()   badge.sensor.tap()      -- true once per event, with a refractory period
badge.sensor.orientation()                     -- flat_up flat_down left_edge right_edge top_edge ...

badge.sys.ms()   badge.sys.uptime()   badge.sys.log("...")   badge.sys.random(n)
badge.sys.stats()      -- { lua_used lua_peak lua_limit widgets uptime_ms free_heap }
badge.store.set("best", 12)   badge.store.get("best", 0)     -- also set_int/get_int, set_str/get_str
badge.fs.write("appdata/save.dat", "...")   badge.fs.read(...)   badge.fs.list()
badge.nfc.enable()   badge.nfc.card()   badge.nfc.read_text()  -- off by default, power hungry
badge.radio.enable()   badge.radio.send("ping")                -- 1 to 44 bytes
badge.radio.on_recv(function(mac, rssi, payload) end)          -- 8-slot ring, 4 frames per tick
badge.me.name()   badge.me.badge_id()   badge.contacts.count()   badge.contacts.get(1)
badge.app.exit()
```

## The trap that cost us an hour: badge.radio and heap

`badge.radio.enable()` can panic the whole badge, and it looks exactly like your app refusing to open.

Measured on real hardware with FlyBadge (16 KB source, 150 neurons, 9 widgets) loaded:

```
app_reg: heap after exit Launcher: free=77760 largest=63488
hal_radio: init_once: free heap 32840 largest 22528
E BLE_INIT: Malloc failed
BLE assert emi.c 164, param 00000000 00001000
Guru Meditation Error: Core 0 panic'ed (Interrupt wdt timeout on CPU0)
Rebooting...
```

The app itself loads and `on_enter` runs correctly up to the radio call. Entering it costs about 45 KB of
the badge's roughly 78 KB of free heap, the BLE controller then wants around 22 KB contiguous, the malloc
fails, and the BLE driver asserts and takes the device down. The badge reboots into the launcher, so from
the outside the app "opens and immediately goes back to the home screen" with no Lua error anywhere.

Rules we now follow:
- A large app and Bluetooth do not fit together. If you need the radio, keep the rest of the app small.
- `badge.radio.enable()` returning true is not the end of it; the panic happens inside BLE init afterwards.
- If an app bounces to the launcher, read the serial console before touching the Lua. A `Guru Meditation
  Error` is a firmware-level panic, not a script error, and no amount of Lua debugging will find it.
- The same applies to the two-badge stretch in `badge/GAME_SPEC.md`: it needs a much smaller circuit.

## Heap, measured: what actually costs memory on this badge

The badge has roughly 78 KB of free system heap when an app launches, and `heap_kb` is a ceiling, not a
reservation. Two things we got wrong, both found by measurement rather than reading:

**Decoding into a per-byte table is the most expensive thing in our startup.** Building a 900-entry table
one `string.char` at a time costs **17.7 KB of peak heap** to keep 900 bytes, because Lua doubles a
table's array part and holds the old array alive during the copy. Flushing into a 64-byte buffer and
`table.concat`ing chunks costs **3.5 KB**, an 80% saving for four extra lines. If you decode anything at
load, chunk it.

**Escaped-decimal string literals are cheaper than they look.** `"\65\66..."` costs about 2.4 KB and needs
no runtime decode at all, because the lexer strips the escape digits from its buffer as it reads. Base64 is
a smaller *file* (4 characters per 3 bytes, no backslashes for a serial push to mangle) but it needs a
runtime decoder, and the decoder is where the heap goes. We ship base64 with a chunked decoder: smaller to
push, and the decode peak is now under the cost of the escapes it replaced.

**Runtime manifest options need a Reboot, not a reload.** Changing `heap_kb`, `wake_lock`, `api`,
`home_button` or `confirm_home` on an already-installed slug has no effect after Push plus `reload`; only
a Reboot applies them. A launcher entry showing no icon when the manifest sets one is a sign the installed
`manifest.cfg` is not the one you just pushed. Check with `cat /littlefs/apps/<slug>/manifest.cfg`.

## What FlyBadge does with this

`badge/app_template.lua` plus the generated `badge/flybadge_circuit.lua` build into
`badge/flybadge_app.lua`, one 16 KB file to paste into the IDE.

- **Looming** comes from the accelerometer, since there is no light sensor. A swat moves the badge, and
  the per-axis change in acceleration is the stimulus. A leaky accumulator keeps the drive up for a few
  ticks after the hit, the way an expanding edge would. The gain is tunable live with UP and DOWN and
  saved in `badge.store`, because none of it could be calibrated without the hardware in hand.
- **The flirt** uses `badge.radio`, not IR. Both badges broadcast `FLY1` once a second; receiving one
  drives the LC10a cells for three seconds and turns the LEDs pink. Honest caveat: in this subcircuit
  LC10a has no path to the giant fibre, so that is a display of courtship-pathway activity, not a
  behaviour the wiring produces.
- **The escape** is real. Fixed-point LIF at 100 Hz over 900 measured synapses, and the giant fibre's
  spike is what flashes the LEDs.
