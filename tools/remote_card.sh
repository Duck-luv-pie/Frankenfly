#!/bin/sh
# Build the boot files for the REMOTE Pi's card (cloud-init): the remote joins the robot's Wi-Fi "companion" and runs
# tools/pi/remote_button.py as a service from first boot, no ssh or clone needed. Image the card as in tools/pi_image.md,
# then, with its boot partition mounted at /Volumes/bootfs:   tools/remote_card.sh [/Volumes/bootfs]
# Extra Wi-Fi networks (for apt to reach the internet once, if gpiozero is missing): put them in tools/pi/extra-wifi.yaml
# (git-ignored; access-points entries at 8 spaces of indent, e.g. '        "Home":\n          password: "..."').
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/Volumes/bootfs}"
[ -d "$OUT" ] || { echo "no directory $OUT (mount the card, or pass an output directory)" >&2; exit 1; }
HASH='$6$DFN.pNFMrJXRVDdA$BPgSLiJ/ToYAd/33wY63DY8rMKdRchUQxVwkvHv8sa8c8YCLWwKweTiIEpOjmsOE6AB32yVne.cVxjr4Peruc.'   # 'hunting-fly'
indent() { sed 's/^/      /'; }
{
  cat <<HEAD
#cloud-config
hostname: companion-remote
manage_etc_hosts: true
timezone: America/Toronto
ssh_pwauth: true
users:
- name: companion
  groups: users,adm,dialout,audio,netdev,video,plugdev,cdrom,games,input,gpio,spi,i2c,render,sudo
  shell: /bin/bash
  lock_passwd: false
  passwd: $HASH
  sudo: ALL=(ALL) NOPASSWD:ALL
write_files:
- path: /home/companion/remote_button.py
  permissions: '0755'
  defer: true
  content: |
HEAD
  indent < "$HERE/pi/remote_button.py"
  cat <<MID
- path: /etc/systemd/system/remote-button.service
  permissions: '0644'
  content: |
MID
  indent < "$HERE/pi/remote-button.service"
  cat <<TAIL
runcmd:
- chown companion:companion /home/companion/remote_button.py
- [sh, -c, "python3 -c 'import gpiozero' 2>/dev/null || for i in 1 2 3 4 5 6; do apt-get install -y python3-gpiozero python3-lgpio && break; sleep 20; done || true"]
- systemctl daemon-reload
- systemctl enable --now remote-button
TAIL
} > "$OUT/user-data"
{
  cat <<NET
network:
  version: 2
  wifis:
    wlan0:
      dhcp4: true
      optional: true
      regulatory-domain: "CA"
      access-points:
        "companion":
          password: "hunting-fly"
NET
  [ -f "$HERE/pi/extra-wifi.yaml" ] && cat "$HERE/pi/extra-wifi.yaml"
} > "$OUT/network-config"
touch "$OUT/ssh"
echo "wrote $OUT/user-data ($(wc -l < "$OUT/user-data") lines), $OUT/network-config, $OUT/ssh"
echo "eject the card (diskutil eject), boot the remote Pi near the robot; the READY light comes on once the robot's brain is up."
