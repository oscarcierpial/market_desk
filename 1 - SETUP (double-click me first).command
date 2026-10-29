#!/bin/bash
# One-time setup: builds a private Python environment and installs the libraries.
cd "$(dirname "$0")" || exit 1
echo "Setting up the market desk in: $(pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 isn't installed. Get it from https://www.python.org/downloads/ then run this again."
  read -r -p "Press Return to close." _; exit 1
fi
python3 -m venv .venv && source .venv/bin/activate || { echo "Could not create the environment."; read -r -p "Press Return to close." _; exit 1; }
pip install --upgrade pip -q
pip install -r requirements.txt -q || { echo "Library install failed (check your internet connection)."; read -r -p "Press Return to close." _; exit 1; }
chmod +x "2 - RUN (double-click me).command"
echo
echo "Running a full test (about 2-3 minutes: it scans ~550 stocks)..."
[ -f config.env ] || cp config.example.env config.env   # first run on a fresh clone
set -a; source config.env; set +a
python desk.py --test --no-claude && echo "Test passed." && open desk.html
echo
echo "Setup complete. Next: fill in config.env (opening it now), save it, then double-click '2 - RUN'."
open -e config.env
read -r -p "Press Return to close." _
