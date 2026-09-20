#!/bin/sh
# Stop the robot Pi from disturbing the body ESP32's USB serial port: ModemManager probes and grabs new
# serial devices (I/O errors, "multiple access on port"), so it is disabled and CP210x devices are marked
# for it to ignore. Also shows USB power trouble (dmesg, throttling) if the port drops for that reason.
ssh -o ConnectTimeout=8 companion@10.42.0.1 '
echo "== ModemManager"; systemctl is-active ModemManager 2>/dev/null; sudo systemctl disable --now ModemManager 2>/dev/null && echo "disabled" || echo "(not installed)"
echo "== udev: ignore CP210x for ModemManager, stable /dev/body link"
printf "ACTION==\"add\", SUBSYSTEM==\"tty\", ATTRS{idVendor}==\"10c4\", ATTRS{idProduct}==\"ea60\", ENV{ID_MM_DEVICE_IGNORE}=\"1\", SYMLINK+=\"body\", MODE=\"0666\"\n" | sudo tee /etc/udev/rules.d/90-companion-body.rules >/dev/null
sudo udevadm control --reload && sudo udevadm trigger && sleep 1 && ls -la /dev/body 2>/dev/null || echo "(no /dev/body yet: replug the ESP32)"
echo "== USB / serial events (last 15)"; sudo dmesg 2>/dev/null | grep -iE "cp210|ttyUSB|over-current|undervolt|usb 1-|usb 3-" | tail -15
echo "== power"; vcgencmd get_throttled 2>/dev/null; sudo dmesg 2>/dev/null | grep -i "voltage" | tail -2
echo "== who holds the port now"; P=$(ls /dev/serial/by-id/*CP210* 2>/dev/null | head -1); sudo fuser -v "$P" 2>&1 | tail -3
'
