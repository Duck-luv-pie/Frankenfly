# Wiring and bring-up

Two ESP32 boards. **They are not wired to each other.** Each has its own USB power and talks to
the Mac over WiFi. The Mac, the body and the cam must all be on the same WiFi network.

| Board | Role | Wired to it | Talks to |
|---|---|---|---|
| ESP32 DevKit V1 = **body** | face, voice, presence sense | 2× GC9A01 eyes, DFPlayer Mini + speaker, HC-SR501 PIR | Mac, UDP |
| ESP32-CAM + carrier = **cam** | the fly's eye | nothing (OV2640 is on the board) | Mac, HTTP MJPEG |
| Mac = **brain** | connectome simulation | — | both |

Why the camera is not on the DevKit: the ESP32-CAM has only ~5 free GPIOs once the camera and SD
pins are used, and its WiFi is brownout-prone, so it gets its own board and its own 5 V supply.

## Body: ESP32 DevKit V1 (30-pin ESP-WROOM-32)

All pins are defined in `firmware/body/include/config.h`. Every device shares GND with the
DevKit. Both displays share the SPI clock, data, DC and reset lines; only chip select differs.

### Left eye, GC9A01 1.28" 240×240 (7-pin module)

| GC9A01 pin | DevKit pin | Note |
|---|---|---|
| VCC | 3V3 | 3.3 V only, never 5 V |
| GND | GND | |
| SCL | GPIO 18 | SPI clock, shared |
| SDA | GPIO 23 | SPI MOSI, shared |
| RES | GPIO 26 | reset, shared |
| DC | GPIO 27 | data/command, shared |
| CS | GPIO 5 | left select |
| BLK (if present) | 3V3 | backlight on. Or GPIO 32 and set `PIN_TFT_BL 32` for dimming |

### Right eye, GC9A01

Identical, except **CS → GPIO 25**.

### DFPlayer Mini (16-pin; with the SD slot facing you, pin 1 = VCC is top-left)

| DFPlayer pin | DevKit pin | Note |
|---|---|---|
| VCC | VIN (5 V) | from the DevKit's USB 5 V |
| GND | GND | |
| RX | GPIO 17 (TX2) **via 1 kΩ resistor** | the resistor stops the hiss from 3.3 V logic driving the 5 V module |
| TX | GPIO 16 (RX2) | 3.3 V level, direct |
| BUSY | GPIO 35 | LOW while playing (input-only pin) |
| SPK_1, SPK_2 | 8 Ω speaker ≤ 3 W | on-board amplifier, no extra board |

Micro SD: FAT32, folder `/mp3/`, files `0001.mp3` … `0005.mp3` (see `assets/sounds/README.md`).

### HC-SR501 PIR

| PIR pin | DevKit pin | Note |
|---|---|---|
| VCC | VIN (5 V) | needs 5 V; its regulator makes 3.3 V internally |
| OUT | GPIO 34 | 3.3 V logic already, input-only pin, no resistor |
| GND | GND | |

Jumper to **H** (retriggerable). Time-delay pot fully counter-clockwise (~3 s hold). Sensitivity
mid. The sensor needs ~60 s after power-on; firmware ignores it for the first 60 s.

### Power

| Load | Rail | Current |
|---|---|---|
| 2 × GC9A01 | 3V3 (on-board regulator, ~500 mA max) | ~40 mA each |
| DFPlayer + speaker | 5 V (VIN) | up to ~200 mA at full volume |
| PIR | 5 V (VIN) | ~1 mA |

One 5 V / ≥1 A phone charger into the DevKit micro-USB covers it. Pins avoided on purpose:
0, 2, 12, 15 (boot strapping), 6–11 (flash).

## Cam: AI-Thinker ESP32-CAM on its carrier board

Nothing to wire. The carrier (ESP32-CAM-MB or the CH340 variant) supplies 5 V and USB serial.
Power from its own charger or USB port, 5 V ≥ 1 A. The bright flash LED on GPIO 4 stays off.

## Flashing

```sh
cp firmware/secrets.example.ini firmware/secrets.ini    # WiFi SSID / password
uv tool install platformio                              # once
```

Cam: connect USB, **hold IO0, tap RST, release IO0**, then

```sh
cd firmware/cam && pio run -t upload && pio device monitor
```

Tap RST after the upload. Body:

```sh
cd firmware/body && pio run -t upload && pio device monitor
```

## Network protocol

| Link | Transport | Rate | Content |
|---|---|---|---|
| cam → brain | HTTP `GET http://companion-cam.local:81/stream` | ~15 fps QVGA JPEG | multipart MJPEG; `/status` on port 80 gives JSON |
| brain → body | UDP to `companion-body.local:4210` | 20 Hz | eye targets, sound, arm stub |
| body → brain | UDP to the brain's IP, port 4211 | 2 Hz + on each PIR edge | `pir`, `busy`, `rssi` |

Both boards announce themselves with mDNS. If `.local` names fail on your network, put the IPs
in `brain/configs/default.yaml` (`senses.camera.url`, `body.host`).

Brain → body:

```json
{"t": 1234567, "state": "escape",
 "eyes": {"l": {"px": -0.3, "py": 0.1, "pr": 0.55, "ut": 0.9, "lt": 0.0, "tint": [255, 80, 80]},
          "r": {"px": -0.3, "py": 0.1, "pr": 0.55, "ut": 0.9, "lt": 0.0, "tint": [255, 80, 80]}},
 "blink": true, "sound": {"track": 1, "vol": 20}, "arms": {"l": 0.0, "r": 0.0}}
```

`px`, `py` pupil position −1..1; `pr` pupil radius 0..1; `ut`/`lt` upper/lower lid openness 0..1;
`tint` iris RGB; `sound.track` 0 = silent; `arms` reserved.

Body → brain: `{"t": 1234567, "pir": 1, "busy": 0, "rssi": -58}`

## RoboMaster S1 (the legs)

The S1 is driven through the **S-Bus port on its motion controller**: lift the lid on top of the
chassis behind the gimbal (part 6, "chassis rear cover"); beside the micro USB port is a block of
pins three wide and eight rows tall. The row at the rear-hinge end is S-Bus (Signal, 5 V, GND,
Signal on the USB side), the six middle rows are PWM 1-6, the row at the gimbal end is the
UART (RX, TX, GND). Only Signal and GND are connected; leave the 5 V pin alone. The stock S1
has no SDK; the Pi is the "receiver". Channel map and the driver: `brain/companion_brain/body/s1.py`.

S-Bus is an *inverted* serial signal, and the Pi's own UART cannot invert, so one of:

**A. ESP32 as the inverter (jumper wires only).** Flash `firmware/sbus-bridge` onto a spare
ESP32 DevKit (`pio run -t upload`). Pi USB -> ESP32 USB cable. ESP32 GPIO17 -> S1 S-Bus Signal,
ESP32 GND -> S1 S-Bus GND. On the Pi the ESP32 is `/dev/ttyUSB0`; the brain runs
`companion run --s1 /dev/ttyUSB0`. The ESP32's blue LED is on while frames flow; if they stop
it centres the sticks itself.

**B. A transistor on the Pi's header.** Pi GPIO14 (pin 8) -> 1 kOhm -> NPN base (2N2222 / BC547);
emitter -> GND; collector -> S1 S-Bus Signal with 10 kOhm to 3.3 V. Pi GND (pin 6) -> S1 GND.
Enable the UART with `dtparam=uart0=on` in `/boot/firmware/config.txt`; the port is
`/dev/ttyAMA0` (the default in `body.s1.port`).

**C. An FTDI FT232RL adapter** with TX inverted in its EEPROM (`ftx_prog --invert-txd`): adapter TX
-> Signal, GND -> GND, port `/dev/ttyUSB0`.

Bench test before the brain: `python3 tools/s1_sbus.py --port <PORT> --neutral` (does the chassis
stiffen with the app disconnected?), then without `--neutral` for w/s/a/d/q/e driving, or hands-off
`--pulse forward:0.4:1` (3 s centred, then the axis for that long, then centred).

**The fly driving the S1** (verified 2026-09-19, Mac -> ESP32 bridge -> S1, robot on the floor):

```sh
cd brain
uv run --with pyserial companion hunt-brain --watch --load data/cache/hunter_brain.npz --s1 /dev/cu.usbserial-210 --open
```

The spiking connectome hunter runs in its arena at http://localhost:8601 and the rover mirrors the fly's
real speed and heading rate each tick (`body.s1.speed_mps_full` / `yaw_dps_full` convert them to sticks;
the fly's 1.6 m/s saturates the rover's 0.85, so the rover follows the same path more slowly). The page
shows the fly's heading and yaw rate and what the rover is being told; the camera button cycles to a
locked bird's-eye view for comparing turns. `--set body.s1.rotation_only=true` (a global option: it goes
*before* the subcommand) makes the rover pivot only. Lessons: a dead S1 battery looks exactly like a
broken link (the motion controller ignores S-Bus, the ESP32 still reports frames flowing) - check it
first; the hunter wags its heading several times a second, which the rover cannot follow tick by tick,
so on turns it shivers until the fly commits (heading tracking, not yet built, is the fix). If an axis
is mirrored, flip `sign_forward` / `sign_strafe` / `sign_yaw` under `body.s1` in the config. The S1 yaws
slowly per stick: `--pulse yaw:1:2` at full stick, count the degrees turned, and set `stick_yaw` /
`stick_forward` so the robot's turn-to-speed ratio matches the fly's (hunt.turn_max / hunt.speed_max).

## Bring-up order

1. **Cam first, nothing wired.** Flash, open `http://companion-cam.local:81/stream` in a
   browser. Serial monitor prints the IP and a status line every 10 s.
2. **One eye.** Wire the left eye, flash the body. On boot each display shows a test pattern
   (white disc, cyan iris, black pupil) for 0.8 s, then the idle animation. Add the right eye.
3. **DFPlayer + SD card.** Serial shows `DFPlayer online` and track 1 plays once at boot.
   `DFPlayer not responding` means: card missing/not FAT32, RX/TX swapped, or no 5 V.
4. **PIR.** After 60 s the serial monitor prints `PIR 1` / `PIR 0` as you move; the eyes widen
   on their own on each rising edge.
5. **Brain.** On the Mac: `cd brain && uv run companion run -v`. The body's serial monitor
   goes quiet about idle animation, the eyes follow the brain, and the brain log shows
   `state=` changing as you move a hand toward the camera (escape), wave (track) or step into
   the PIR's view (startle then groom).

## Tuning

- Eyes too jumpy or too slow: `EASE_PER_FRAME` in `config.h`.
- Escape too easy/hard: `decode.behaviors.escape.ref_hz` and `senses.features.loom_fast.gain`.
- Camera sees the wrong way: `set_hmirror` / `set_vflip` in `firmware/cam/src/main.cpp`.
