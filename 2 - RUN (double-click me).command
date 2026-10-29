#!/bin/bash
# Starts the market desk. Keep this window open; close it (or press Ctrl+C) to stop.
cd "$(dirname "$0")" || exit 1
[ -d .venv ] || { echo "Run '1 - SETUP' first."; read -r -p "Press Return to close." _; exit 1; }
source .venv/bin/activate
set -a; source config.env; set +a
if [[ "$NTFY_TOPIC" == change-me* ]]; then
  echo "Heads up: NTFY_TOPIC in config.env is still the placeholder, so phone alerts won't reach you."
fi
echo "Starting the market desk. Your Mac will stay awake while this window is open."
( sleep 25; [ -f desk.html ] && open desk.html ) &
caffeinate -i python desk.py "$@"
