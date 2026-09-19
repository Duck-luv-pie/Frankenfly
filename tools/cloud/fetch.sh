#!/usr/bin/env bash
# Pull the cloud runs' checkpoints and logs back to this Mac:  tools/cloud/fetch.sh user@host [port]
set -euo pipefail
HOST="${1:?usage: fetch.sh user@host [ssh-port]}"
PORT="${2:-22}"
DST="$(cd "$(dirname "$0")/../../brain" && pwd)/data/cache/"
rsync -avz --progress -e "ssh -p $PORT" "$HOST:~/companion/brain/data/cache/cloud_*" "$DST"
