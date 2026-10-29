#!/usr/bin/env python3
"""
Historical data for backtesting the day-trading scanner, from Alpaca's free market-data API.

Rebuilds what the desk would have seen each trading day, without peeking ahead:
  1. Daily bars for the whole universe          -> yesterday's levels (high/low/close, ATR, avg volume)
  2. The first 15 minutes (9:30-9:44) of every stock -> that morning's watchlist: the biggest movers
     and heaviest volume by 9:45, plus the pre-market "ideas" scoring, exactly what was knowable then
  3. Full-day 1-minute bars for the watchlist plus SPY, QQQ, IWM and the sector ETFs

Saved to state/history/ (one file per day). Safe to stop and restart: finished days are skipped.

  python history.py            download (or top up) the last BT_YEARS years (default 2)
  python history.py --check    just test the keys

Needs ALPACA_KEY_ID and ALPACA_SECRET in config.env (free paper-trading account keys).
"""
import os, pickle, sys, time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from common import ET, env, load, log, path
import discovery

DATA = "https://data.alpaca.markets/v2"
TRADE = "https://paper-api.alpaca.markets/v2"
HIST = path("history")
DAYS = os.path.join(HIST, "days")
os.makedirs(DAYS, exist_ok=True)
YEARS = env("BT_YEARS", 2.0)
WATCH_N = env("WATCHLIST_SIZE", 20)
SECTORS = ["XLK", "SMH", "XLY", "XLC", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]
CONTEXT = ["SPY", "QQQ", "IWM"] + SECTORS
_last_call = [0.0]


# ------------------------------- Alpaca API -------------------------------
def _headers():
    k, s = os.environ.get("ALPACA_KEY_ID", ""), os.environ.get("ALPACA_SECRET", "")
    if not k or not s:
        sys.exit("ALPACA_KEY_ID / ALPACA_SECRET are missing from config.env")
    return {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}


def _get(url, params):
    """GET with the free plan's 200 requests/minute pacing and polite retries."""
    for attempt in range(6):
        wait = 0.31 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = requests.get(url, params=params, headers=_headers(), timeout=60)
        except requests.RequestException as e:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(20)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Alpaca request kept failing: {url}")


def _to_alpaca(sym):
    return sym.replace("-", ".")


def _from_alpaca(sym):
    return sym.replace(".", "-")


def calendar(start, end):
    rows = _get(f"{TRADE}/calendar", {"start": str(start), "end": str(end)})
    return [{"date": r["date"], "open": r["open"], "close": r["close"]} for r in rows]


def bars(symbols, timeframe, start, end, chunk=200):
    """{symbol: DataFrame(Open, High, Low, Close, Volume)} indexed in Eastern time. Split-adjusted."""
    out = {}
    for i in range(0, len(symbols), chunk):
        part = [_to_alpaca(s) for s in symbols[i:i + chunk]]
        params = {"symbols": ",".join(part), "timeframe": timeframe, "start": start, "end": end,
                  "limit": 10000, "adjustment": "split", "feed": "sip"}
        rows = {}
        while True:
            j = _get(f"{DATA}/stocks/bars", params)
            for s, bl in (j.get("bars") or {}).items():
                rows.setdefault(s, []).extend(bl)
            if not j.get("next_page_token"):
                break
            params["page_token"] = j["next_page_token"]
        for s, bl in rows.items():
            df = pd.DataFrame(bl)
            df.index = pd.to_datetime(df.t, utc=True).dt.tz_convert(ET)
            df.index.name = None
            out[_from_alpaca(s)] = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close",
                                                      "v": "Volume"})[["Open", "High", "Low", "Close", "Volume"]]
    return out


def _et(day, hhmm):
    return pd.Timestamp(f"{day} {hhmm}", tz=ET).isoformat()


# ------------------------------- watchlist rebuild -------------------------------
def _ideas_score(lv):
    """Same pre-market scoring as discovery.premarket_scan."""
    score = 0
    if lv["from_high_pct"] > -3 and lv["coil"] < 3.0:
        score += 3
    if lv["from_low_pct"] < 3 and lv["coil"] < 3.0:
        score += 2
    if lv["vol_ratio_yday"] > 2:
        score += 2
    if abs(lv["ret5"]) > 8:
        score += 2
    if lv["nr7"]:
        score += 1
    if lv["atr_pct"] < 1.2:
        score -= 2
    return score if score >= 3 else 0


def watchlist(day, levels, opening):
    """Rank by what the live watchlist rewards (movers list, relative volume, pre-market ideas),
    using only bars up to 9:44."""
    cands = []
    for tk, df in opening.items():
        lv = levels.get(tk)
        if tk in CONTEXT or lv is None or df.empty:
            continue
        chg = 100 * (float(df.Close.iloc[-1]) / lv["prev_close"] - 1)
        rvol = float(df.Volume.sum()) / (lv["avg_vol"] * 15 / 390) if lv["avg_vol"] else 0
        cands.append({"ticker": tk, "chg": chg, "rvol": rvol, "dv": float((df.Close * df.Volume).sum()),
                      "ideas": _ideas_score(lv)})
    if not cands:
        return []
    by_move = {c["ticker"] for c in sorted(cands, key=lambda c: -abs(c["chg"]))[:50]}
    by_dv = {c["ticker"] for c in sorted(cands, key=lambda c: -c["dv"])[:50]}
    for c in cands:
        on_list = c["ticker"] in by_move or c["ticker"] in by_dv
        c["score"] = (min(c["rvol"], 5) + min(abs(c["chg"]) / 2, 4) if on_list else 0) + c["ideas"] / 2
    ranked = sorted((c for c in cands if c["score"] > 0), key=lambda c: -c["score"])
    return ranked[:WATCH_N]


# ------------------------------- build -------------------------------
def build():
    t0 = time.time()
    end_day = datetime.now(ET).date() - timedelta(days=1)      # Eastern date; the free plan can't see today
    start = os.environ.get("BT_START", "")                  # e.g. 2016-01-04 to go back further than BT_YEARS
    first_day = date.fromisoformat(start) if start else end_day - timedelta(days=int(365 * YEARS))
    cal = [c for c in calendar(first_day - timedelta(days=60), end_day)]
    days = [c for c in cal if c["date"] >= str(first_day)]
    uni = sorted(set(load("universe.json", {}).get("tickers", {})) | set(CONTEXT))
    uni = [u for u in uni if "^" not in u and "=" not in u]

    daily_path = os.path.join(HIST, "daily.pkl")
    daily = pickle.load(open(daily_path, "rb")) if os.path.exists(daily_path) else {}
    need_from = date.fromisoformat(cal[0]["date"]) + timedelta(days=10)
    if (not daily or max(df.index.max() for df in daily.values()).date() < date.fromisoformat(days[-1]["date"])
            or min(df.index.min() for df in daily.values()).date() > need_from):
        log(f"[history] daily bars for {len(uni)} symbols...")
        safe_end = min(pd.Timestamp(f"{end_day} 23:59", tz=ET), pd.Timestamp.now(tz=ET) - pd.Timedelta(minutes=20))
        daily = bars(uni, "1Day", _et(cal[0]["date"], "00:00"), safe_end.isoformat())   # free plan: nothing from the last 15 min
        for df in daily.values():
            df.index = df.index.tz_localize(None).normalize()
        pickle.dump(daily, open(daily_path, "wb"))
        log(f"[history] daily bars saved ({len(daily)} symbols)")

    idx_cache = {tk: df.index.values for tk, df in daily.items()}
    todo = [c for c in days if not os.path.exists(os.path.join(DAYS, c["date"] + ".pkl"))]
    log(f"[history] {len(days)} trading days, {len(todo)} to download")
    for n, c in enumerate(todo, 1):
        d = date.fromisoformat(c["date"])
        levels = {}
        for tk, df in daily.items():
            pos = int(idx_cache[tk].searchsorted(np.datetime64(d)))      # bars strictly before day d
            if pos < 25:
                continue
            past = df.iloc[max(0, pos - 126):pos]                         # ~6 months, like the live desk
            lv = discovery._levels_from_daily(past, d)
            if lv and (tk in CONTEXT or (lv["prev_close"] >= discovery.MIN_PRICE and lv["avg_vol"] >= discovery.MIN_AVG_VOLUME)):
                levels[tk] = lv
        try:
            opening = bars(sorted(levels), "1Min", _et(d, c["open"]), _et(d, "09:44:59"))
            wl = watchlist(d, levels, opening)
            syms = sorted({w["ticker"] for w in wl} | set(CONTEXT))
            full = bars(syms, "1Min", _et(d, c["open"]), _et(d, c["close"]))
        except Exception as e:
            log(f"[history] {d} failed ({e}); will retry next run")
            continue
        rec = {"date": c["date"], "close": c["close"], "watch": wl,
               "levels": {s: levels[s] for s in syms if s in levels}, "bars": full}
        tmp = os.path.join(DAYS, c["date"] + ".tmp")
        pickle.dump(rec, open(tmp, "wb"))
        os.replace(tmp, os.path.join(DAYS, c["date"] + ".pkl"))
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"[history] {n}/{len(todo)} days ({c['date']}) · {el / 60:.0f} min elapsed, "
                f"~{el / n * (len(todo) - n) / 60:.0f} min left")
    log(f"[history] done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    if "--check" in sys.argv:
        j = _get(f"{DATA}/stocks/bars", {"symbols": "SPY", "timeframe": "1Day", "limit": 1, "feed": "sip",
                                         "start": "2025-01-02", "end": "2025-01-03"})
        print("Alpaca keys work." if j.get("bars") else f"Unexpected reply: {j}")
    else:
        build()
