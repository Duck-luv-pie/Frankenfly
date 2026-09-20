#!/bin/sh
# One-shot diagnosis of the fly on the robot Pi. Run from the Mac while on the Pi's Wi-Fi "companion".
PI=companion@10.42.0.1
ssh -o ConnectTimeout=8 "$PI" '
echo "== settings";        cat ~/companion-hunt.env 2>/dev/null || echo "(no env file: never pushed with settings)"
echo "== fly service";     systemctl is-active companion-hunt; journalctl -u companion-hunt --no-pager -o cat -n 500 2>/dev/null | grep -E "companion-hunt\]|\[body\]|hunt-real\] the|LOBOTOM|trained brain|Traceback|Error" | tail -6
echo "== devices on companion (leases)"; cat /var/lib/NetworkManager/dnsmasq-wlan0.leases 2>/dev/null || echo "(none)"
echo "== body ESP32 by name"; getent hosts companion-body.local || echo "companion-body.local: NOT resolving (the ESP32 is not on this network)"
echo "== what the fly is telling the rover right now"
curl -s -m 3 http://localhost:8601/status | python3 -c "import json,sys; d=json.load(sys.stdin); print(\"tick_hz\", d.get(\"tick_hz\"), \"(20 = trained pace) | rover_on\", d[\"rover_on\"], \"| rover\", d[\"rover\"], \"| drive\", d[\"drive\"], \"| people in view\", len(d[\"boxes\"]), \"| camera_ok\", d[\"camera_ok\"], \"| lobotomized\", d[\"lobotomized\"])" 2>/dev/null || echo "(fly not answering on :8601 yet)"
'
