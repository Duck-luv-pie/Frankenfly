#!/bin/sh
# The Pi 5 caps its USB ports at 600 mA unless a 5 A PD supply is negotiated; the ESP32 with two displays
# trips that ("over-current change" in dmesg, the serial port resets). Allow the full 1.6 A (fine on a bank
# or supply rated 3 A+), show temperature and throttling, and reboot. Run from the Mac on "companion".
ssh -o ConnectTimeout=8 companion@10.42.0.1 '
C=/boot/firmware/config.txt
grep -q "^usb_max_current_enable=1" $C || echo "usb_max_current_enable=1" | sudo tee -a $C >/dev/null
echo "== config:"; grep usb_max_current $C
echo "== temperature / throttling:"; vcgencmd measure_temp; vcgencmd get_throttled
echo "== over-current trips so far:"; sudo dmesg | grep -c "over-current"
echo "== rebooting the Pi (about a minute; the fly starts by itself)"; sudo reboot
' 2>&1 | grep -v "closed by remote"
