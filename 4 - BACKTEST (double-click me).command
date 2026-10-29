#!/bin/bash
# Replays every downloaded day through the live scanner code and opens the report.
# Results are saved per day, so a stopped run picks up where it left off.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
set -a; source config.env; set +a
caffeinate -i python daytrade_bt.py && open daytrade_backtest.html
read -r -p "Done. Press Return to close." _
