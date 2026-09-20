#!/usr/bin/env bash
# Export this repo into the team repo's flybrain-rover/ subdirectory.
#
#     bash scripts/export_to_hunting_fly.sh --dry-run            # show what would change, touch nothing
#     bash scripts/export_to_hunting_fly.sh                      # into ../hunting-fly-integration by default
#     bash scripts/export_to_hunting_fly.sh --into ../hunting-fly
#
# Until now this was a hand-copy: someone selected files, copied them across and re-committed with a
# reworded message. That is why the two trees had drifted by 5 commits and 7 files, and why a `.env`
# once landed in the public checkout. This reproduces the same curation mechanically, refuses to run if
# a secret would travel, and tells you about files it has not seen before instead of quietly shipping
# them.
#
# The rule: everything git tracks here, minus DENY below. Tracked-ness does the heavy lifting, because
# .gitignore already excludes the venvs, the caches, the logs and the 2 GB of connectome .npz files.
set -euo pipefail

cd "$(dirname "$0")/.."
SRC="$PWD"
DEST_ROOT="../hunting-fly-integration"
DRY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY="--dry-run"; shift ;;
    --into)    DEST_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

DEST="$DEST_ROOT/flybrain-rover"
[ -d "$DEST_ROOT" ] || { echo "no such directory: $DEST_ROOT" >&2; exit 1; }

# ---------------------------------------------------------------- what does not travel
# Internal working documents (the team repo is what a judge opens), the archived handoff snapshot, the
# training logs, the big weight files, and site/index.html which is published as docs/index.html at the
# team repo's root rather than duplicated here.
DENY='
^CLAUDE\.md$
^CONTEXT\.md$
^DEVPOST_FIELDS\.md$
^HANDOFF_(docs|mlh|qnx|robot)\.md$
^(PITCH|PLAN|STATUS|TODAY|TRACKS)\.md$
^docs/
^handoff/
^logs_box/
^data/brain_.*\.npz$
^site/index\.html$
'
DENY_RE=$(printf '%s' "$DENY" | grep -v '^$' | paste -sd'|' -)

# ---------------------------------------------------------------- the manifest
git ls-files | grep -Ev "$DENY_RE" | sort > /tmp/fbr_export_manifest.txt
COUNT=$(wc -l < /tmp/fbr_export_manifest.txt | tr -d ' ')

# ---------------------------------------------------------------- the guard
# A deny-list is only as good as the person maintaining it, so this checks the actual payload rather
# than trusting the rules above. It has caught a real mistake before.
FAIL=0
if grep -qE '(^|/)\.env$|\.(key|pem|p12|pfx)$|(^|/)id_rsa|credentials\.json$' /tmp/fbr_export_manifest.txt; then
  echo "REFUSING: a secret-shaped file is in the manifest:" >&2
  grep -E '(^|/)\.env$|\.(key|pem|p12|pfx)$|(^|/)id_rsa|credentials\.json$' /tmp/fbr_export_manifest.txt >&2
  FAIL=1
fi
# and the contents, for a key pasted into a tracked file.
# Two files are exempt and both have to be: .env.example is the documented placeholder, and this script
# carries the patterns below as literal text, so scanning itself is a guaranteed false positive. It
# fired on exactly that the first time it ran.
SCAN=$(grep -vE '^\.env\.example$|^scripts/export_to_hunting_fly\.sh$' /tmp/fbr_export_manifest.txt | tr '\n' ' ')
SECRETS=$(grep -lE 'sk-[A-Za-z0-9]{20,}|xoxb-|postgres(ql)?://[^ ]*:[^ @]{8,}@|AKIA[0-9A-Z]{16}' \
            $SCAN 2>/dev/null || true)
if [ -n "$SECRETS" ]; then
  echo "REFUSING: these tracked files look like they contain a live credential:" >&2
  printf '  %s\n' $SECRETS >&2
  FAIL=1
fi
[ "$FAIL" -eq 0 ] || exit 1

# ---------------------------------------------------------------- report before acting
if [ -d "$DEST" ]; then
  ( cd "$DEST" && find . -type f -not -path './.git/*' | sed 's|^\./||' | sort ) > /tmp/fbr_export_dest.txt
else
  : > /tmp/fbr_export_dest.txt
fi

NEW=$(comm -23 /tmp/fbr_export_manifest.txt /tmp/fbr_export_dest.txt || true)
GONE=$(comm -13 /tmp/fbr_export_manifest.txt /tmp/fbr_export_dest.txt || true)

echo "exporting $COUNT files -> $DEST"
[ -n "$NEW" ]  && { echo "  new here:";  printf '    + %s\n' $NEW; }
# Files present at the destination and absent from the manifest. checkpoints/demo_brain.pt is the
# expected one: it is gitignored here (8.9 MB of weights) but deliberately committed over there, so it
# must survive. Anything else on this list is someone's work about to be deleted -- look before you run.
if [ -n "$GONE" ]; then
  echo "  at the destination but NOT in this export (left untouched):"
  printf '    ? %s\n' $GONE
fi

# ---------------------------------------------------------------- copy
# No --delete: removing a file over there is a decision, not a side effect of an export.
mkdir -p "$DEST"
rsync -a $DRY --files-from=/tmp/fbr_export_manifest.txt "$SRC/" "$DEST/"

if [ -n "$DRY" ]; then
  echo "dry run: nothing was written."
else
  echo "done. Now, from $DEST_ROOT:"
  echo "    git add flybrain-rover"
  # The .gitignore we just copied says checkpoints/, so plain `git add` drops the demo brain without a
  # word. It is 8.9 MB and the demo does not run without it.
  if [ -f "$DEST/checkpoints/demo_brain.pt" ]; then
    echo "    git add -f flybrain-rover/checkpoints/demo_brain.pt   # ignored by the copied .gitignore"
  fi
  echo "    git commit -m 'flybrain-rover: <what changed>'"
fi
