#!/bin/bash
# Forced one-shot setup launched by systemd.run= from cmdline.txt. This is the
# fallback for images whose cloud-init has already disabled itself.
set -u

TRANSFER=
for _ in $(seq 1 120); do
  if [ -d /boot/firmware/badge-controller-transfer ]; then
    TRANSFER=/boot/firmware/badge-controller-transfer
    break
  fi
  if [ -d /boot/badge-controller-transfer ]; then
    TRANSFER=/boot/badge-controller-transfer
    break
  fi
  sleep 1
done

if [ -z "$TRANSFER" ] || [ ! -d "$TRANSFER" ]; then
  exit 1
fi

WEB="$TRANSFER/web"
mkdir -p "$WEB"
exec > >(tee -a "$WEB/force-setup.log" "$WEB/receiver.log") 2>&1

echo
echo "=== forced FlyBadge setup $(date -Is) ==="
echo "kernel: $(uname -a)"
echo "python: $(python3 --version 2>&1)"

if ! id companion >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash --groups sudo,adm,netdev companion
fi
mkdir -p /home/companion/.ssh
cat >/home/companion/.ssh/authorized_keys <<'KEY'
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILoBnDNBNppp44hvzfALJFfkatBfakx5CIWbVE9NLwy8 m43sun@uwaterloo.ca
KEY
chmod 700 /home/companion/.ssh
chmod 600 /home/companion/.ssh/authorized_keys
chown -R companion:companion /home/companion/.ssh
echo 'companion ALL=(ALL) NOPASSWD:ALL' >/etc/sudoers.d/90-companion
chmod 440 /etc/sudoers.d/90-companion

cat >/usr/local/sbin/start-flybadge-probe.sh <<'SCRIPT'
#!/bin/bash
TRANSFER=/boot/firmware/badge-controller-transfer
[ -d "$TRANSFER" ] || TRANSFER=/boot/badge-controller-transfer
mkdir -p "$TRANSFER/web"
exec >>"$TRANSFER/web/receiver.log" 2>&1
exec /bin/bash "$TRANSFER/boot_receiver.sh"
SCRIPT
chmod 755 /usr/local/sbin/start-flybadge-probe.sh

cat >/usr/local/sbin/start-flybadge-web.sh <<'SCRIPT'
#!/bin/bash
TRANSFER=/boot/firmware/badge-controller-transfer
[ -d "$TRANSFER" ] || TRANSFER=/boot/badge-controller-transfer
exec /usr/bin/python3 -m http.server 8080 --bind 0.0.0.0 --directory "$TRANSFER/web"
SCRIPT
chmod 755 /usr/local/sbin/start-flybadge-web.sh

cat >/etc/systemd/system/flybadge-probe.service <<'UNIT'
[Unit]
Description=Hacker Badge BLE reception probe (read-only)
After=bluetooth.service
Wants=bluetooth.service
StartLimitIntervalSec=300
StartLimitBurst=3

[Service]
Type=simple
ExecStart=/usr/local/sbin/start-flybadge-probe.sh
Restart=on-failure
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT

cat >/etc/systemd/system/flybadge-web.service <<UNIT
[Unit]
Description=Browser output for Hacker Badge probe
After=local-fs.target

[Service]
Type=simple
ExecStart=/usr/local/sbin/start-flybadge-web.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable flybadge-web.service flybadge-probe.service
systemctl unmask bluetooth.service 2>/dev/null || true
systemctl enable bluetooth.service 2>/dev/null || true

# SSH is optional now that the browser log exists, but enable either unit when
# the image includes OpenSSH.
systemctl unmask ssh.service ssh.socket 2>/dev/null || true
systemctl enable ssh.socket 2>/dev/null || systemctl enable ssh.service 2>/dev/null || true

{
  echo "Forced setup completed: $(date -Is)"
  echo "web enabled: $(systemctl is-enabled flybadge-web.service 2>&1 || true)"
  echo "receiver enabled: $(systemctl is-enabled flybadge-probe.service 2>&1 || true)"
  echo "bluetooth enabled: $(systemctl is-enabled bluetooth.service 2>&1 || true)"
} >"$WEB/force-status.txt"

# Remove this one-time setup from the kernel command line. Returning success
# makes systemd-run-generator reboot; the next boot is an ordinary multi-user
# boot where the two newly enabled services start.
CMDLINE=/boot/cmdline.txt
[ -f "$CMDLINE" ] || CMDLINE=/boot/firmware/cmdline.txt
if [ -f "$CMDLINE" ]; then
  python3 - "$CMDLINE" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
tokens = path.read_text().strip().split()
tokens = [
    token for token in tokens
    if not token.startswith("systemd.run=")
    and not token.startswith("systemd.run_success_action=")
    and not token.startswith("systemd.run_failure_action=")
    and token != "systemd.unit=kernel-command-line.target"
]
path.write_text(" ".join(tokens) + "\n")
PY
fi

echo "=== forced setup complete ==="
sync
