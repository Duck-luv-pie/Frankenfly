# Hacker Badge -> Raspberry Pi radio probe

This directory is a **transport test only**. It answers one question before the robot is changed:

> Can the Raspberry Pi receive an A/B message sent by the stock Hacker Badge Lua radio API?

It does not call the robot's `/control` endpoint and cannot move, lobotomize, or restore the robot.

## Files

- `badge_radio_probe.lua` - import this as a new app in the Hacker Badge IDE.
- `pi_badge_receiver.py` - run this on the Raspberry Pi; it scans BLE advertisements for the probe messages.
- `requirements.txt` - Python dependency for the Pi receiver.

The test payload is deliberately short:

- A sends `FBT1|A|NNNN` (future meaning: lobotomize).
- B sends `FBT1|B|NNNN` (future meaning: restore).
- `NNNN` is a request sequence number.

The badge sends each press six times because advertisements can be lost. The Pi deduplicates those copies.
The badge saying **queued** only means its firmware accepted the transmission; it does not prove the Pi received it.

## 1. Install the probe on the badge

1. Open <https://badge.hackthenorth.com/ide/> in desktop Chrome or Edge.
2. Save the current IDE app first. The existing app in the separate badge workspace is Tilt Dodger.
3. Click **Import app** and import the complete `badge_radio_probe.lua` file.
4. Confirm the new slug is `fly_pi_probe`, then choose **Replace editor files**.
5. Turn the badge off, connect it with a USB data cable, and turn it on normally. Do not hold Start.
6. Click **Connect**, choose the Espressif **USB JTAG/serial debug unit**, then click **Push**.
7. Open **Fly Pi Probe** from the badge launcher. It should show `Radio ready`.

If the IDE has no **Import app** button, copy the lines inside the `badge-app` header into
`manifest.cfg`, and everything after the header into `main.lua`, then Connect and Push.

If opening the app immediately reboots to the launcher, capture the serial console output. A previous,
much larger FlyBadge app exhausted the heap while starting BLE; this probe is intentionally small, but
the real badge and firmware still decide whether it works.

## 2. Put the receiver on the Pi

If this repository is already deployed to the robot Pi, sync it from the Mac:

```sh
tools/pi_sync.sh companion@companion-pi.local
```

On the Pi:

```sh
sudo rfkill unblock bluetooth
sudo systemctl enable --now bluetooth
sudo apt update
sudo apt install -y bluez python3-venv

cd ~/companion/tools/badge_controller
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python pi_badge_receiver.py --self-test
```

`parser self-test: PASS` checks the receiver code only. It is not a radio test.

### Using a Raspberry Pi OS microSD card as the transfer device

If this folder was copied to the card's `bootfs` partition from a Mac, shut the Pi down, put the card
back in the Pi's microSD slot, and boot it. On current Raspberry Pi OS the transfer folder appears at:

```text
/boot/firmware/badge-controller-transfer
```

After completing Raspberry Pi OS's first-boot user and Wi-Fi setup, open Terminal and run:

```sh
bash /boot/firmware/badge-controller-transfer/install_on_pi.sh
```

The script copies the probe to `~/badge-controller`, creates a private Python environment, and runs the
parser self-test. A prepared boot-card bundle can include the Python packages for an offline install. It
does not configure autostart or connect anything to the robot.

## 3. Run the real radio test

On the Pi:

```sh
~/badge-controller/run_receiver.sh
```

If the repository-sync method above was used instead of the microSD transfer, use the original
`cd ~/companion/tools/badge_controller` and `.venv/bin/python pi_badge_receiver.py` commands.

Keep the badge close to the Pi for the first test. In the foreground probe app:

1. Press **A** once. The Pi should print `A / LOBOTOMIZE TEST` with the same sequence shown by the badge.
2. Press **B** once. The Pi should print `B / RESTORE TEST`.
3. Repeat from a few metres away.
4. Stop the receiver with Ctrl-C.

Success is at least one decoded A packet and one decoded B packet. Repeated transmissions with the same
sequence are expected and are suppressed by the receiver.

On a headless card prepared with the included web service, the same output is available from a Mac
connected to the Pi's `companion` hotspot at `http://10.42.0.1:8080/`. The page also displays setup and
Bluetooth errors, so it remains useful when SSH is unavailable.

## If the Pi does not decode a message

First run the receiver's advertisement dump:

```sh
.venv/bin/python pi_badge_receiver.py --show-all
```

Press A and B again. This prints the byte fields BlueZ exposes. Look for `LUA1`, `FBT1`, or the ASCII/hex
form of the payload. Avoid leaving `--show-all` on in a crowded room because it is intentionally noisy.

In two more Pi terminals, collect the lower-level evidence:

```sh
sudo btmon
```

```sh
bluetoothctl scan on
```

Press A and B, then stop both tools. If `btmon` reports the badge but the Python receiver does not decode
it, save that output: the packet layout needs to be added to the parser. If neither tool sees the badge,
check `bluetoothctl show` and `btmgmt info`. The Pi 5 advertises Bluetooth 5 support, but extended
advertising is an optional controller feature, so its built-in adapter may not support the badge's exact
mode.

Fallbacks, in order:

1. Try a Linux-supported USB BLE 5 adapter with extended-advertising support.
2. Use a second Hacker Badge connected to the Pi over USB as the BLE receiver; the handheld controller
   remains wireless badge-to-badge.
3. Reflash custom badge firmware only as a separate, explicitly accepted project. The stock Lua API does
   not expose Wi-Fi or HTTP even though the ESP32-C3 hardware supports Wi-Fi.

## What comes after this probe

Do not connect the probe directly to robot controls. After badge -> Pi reception is proven, the next
version should add Pi -> badge acknowledgements, retries and stale-message protection. Only then should a
Pi bridge map explicit A/B commands to the selected runtime's real state:

- On `s1-fly-brain`, POST `{"lobotomy": true|false}` to the existing local `:8601/control` endpoint and
  confirm it through `/status`.
- On `main` / `flybrain-rover`, add a dedicated command path that sets the drive-gating
  `brain.lobotomy` flag. Do not use demo keys 4/5, which change learned synapses instead.

Robot integration must first be tested with wheel output disabled.
