#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ ! -x "$HERE/.venv/bin/python" ]; then
  echo "The receiver is not installed yet."
  echo "Run install_on_pi.sh first."
  exit 1
fi

exec "$HERE/.venv/bin/python" "$HERE/pi_badge_receiver.py" "$@"
