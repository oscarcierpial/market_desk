#!/bin/bash
# Downloads ~2 years of 1-minute history from Alpaca for the day-trading backtest.
# Takes ~30 minutes the first time; later runs only add new days. Safe to stop and restart.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
set -a; source config.env; set +a
caffeinate -i python history.py
read -r -p "Done. Press Return to close." _
