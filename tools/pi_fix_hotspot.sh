#!/bin/sh
# Make the robot Pi's Wi-Fi "companion" plain WPA2 (no management-frame protection, CCMP only) so ESP32
# boards can join it, and restart it. Run from the Mac while on "companion"; the Mac drops for ~20 s
# and rejoins by itself. Afterwards the ESP32s join within their 30 s retry.
ssh -o ConnectTimeout=8 companion@10.42.0.1 '
echo "== before"; nmcli -f 802-11-wireless-security.proto,802-11-wireless-security.pairwise,802-11-wireless-security.group,802-11-wireless-security.pmf,802-11-wireless.band con show companion
sudo nmcli con modify companion 802-11-wireless-security.proto rsn 802-11-wireless-security.pairwise ccmp 802-11-wireless-security.group ccmp 802-11-wireless-security.pmf disable 802-11-wireless.band bg 802-11-wireless.channel 6
echo "== after"; nmcli -f 802-11-wireless-security.proto,802-11-wireless-security.pairwise,802-11-wireless-security.group,802-11-wireless-security.pmf con show companion
nohup sh -c "sleep 1; sudo nmcli con up companion" >/dev/null 2>&1 &
echo "== restarting the hotspot; rejoin companion in ~20 s, then run tools/pi_check.sh"
'
