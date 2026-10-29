"""Shared helpers: settings, file paths, time, state files, logging, phone alerts."""
import json, os, sys, threading
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
BASE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(BASE, "state")
os.makedirs(STATE, exist_ok=True)
YF_LOCK = threading.Lock()          # yfinance isn't reliably thread-safe
DRY_RUN = False                      # set by desk.py --dry-run


def env(name, default):
    """Read a setting from config.env / the environment, cast to the default's type."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return type(default)(raw)
    except (TypeError, ValueError):
        return default


# ----------------------------- settings -----------------------------
BUDGET = env("BUDGET", 500.0)                    # max dollars per trade
MIN_ALERT_SCORE = env("MIN_ALERT_SCORE", 6)      # setups scoring below this are logged, not pushed
MIN_RR = env("MIN_RR", 1.5)                      # minimum reward:risk to first target
WATCHLIST_SIZE = env("WATCHLIST_SIZE", 20)
BRIEF_MINUTES = env("BRIEF_MINUTES", 15)


def path(name):
    return os.path.join(STATE, name)


def load(name, default):
    try:
        with open(path(name), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save(name, obj):
    tmp = path(name) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, path(name))


def now():
    return datetime.now(ET)


def hhmm(dt=None):
    dt = dt or now()
    return dt.hour * 100 + dt.minute


def in_window(start, end, dt=None):
    """start/end as HHMM ints, e.g. in_window(945, 1130). Weekdays only (holidays not handled)."""
    dt = dt or now()
    return dt.weekday() < 5 and start <= hhmm(dt) <= end


def log(msg):
    line = f"{now():%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    try:
        with open(path("desk.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def notify(title, body, priority="default", tags=""):
    """Push to phone via ntfy, Discord and/or Telegram (whichever are configured)."""
    print("\n" + "-" * 66 + f"\n{title}\n{body}\n" + "-" * 66, flush=True)
    if DRY_RUN:
        return
    try:
        if topic := os.environ.get("NTFY_TOPIC"):
            if not topic.startswith("change-me"):
                requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
                              headers={"Title": title[:200].encode("utf-8"), "Priority": priority,
                                       "Tags": tags}, timeout=10)
        if hook := os.environ.get("DISCORD_WEBHOOK"):
            requests.post(hook, json={"content": f"**{title}**\n{body}"[:1900]}, timeout=10)
        if (tok := os.environ.get("TELEGRAM_TOKEN")) and (chat := os.environ.get("TELEGRAM_CHAT_ID")):
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": f"{title}\n{body}"[:4000]}, timeout=10)
    except Exception as e:
        print(f"[notify error] {e}", file=sys.stderr)
