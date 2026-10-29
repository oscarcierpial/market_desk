"""
Stage 1 of the scanner: find candidates across the whole market.

premarket_scan()   once a day: daily bars for the full universe (~560 stocks) -> key levels for
                   every stock, plus a list of setups-in-waiting (coiled near highs, unusual
                   volume yesterday, big recent movers).
refresh_watchlist() every 20 minutes: market-wide most-active / gainers / losers screens,
                   Stocktwits trending, today's news catalysts and the pre-market list, filtered
                   for quality and ranked into the ~20 stocks the setup scanner watches closely.
"""
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from common import YF_LOCK, load, save, now, log, env, WATCHLIST_SIZE
import universe

MIN_PRICE = env("MIN_PRICE", 5.0)
MIN_AVG_VOLUME = env("MIN_AVG_VOLUME", 1_000_000.0)
MIN_MARKET_CAP = env("MIN_MARKET_CAP", 1_000_000_000.0)
CORE = ["SPY", "QQQ"]


def _download_daily(tickers, period="6mo"):
    frames = {}
    for i in range(0, len(tickers), 150):          # chunks keep Yahoo happy
        chunk = tickers[i:i + 150]
        try:
            with YF_LOCK:
                d = yf.download(chunk, period=period, interval="1d", progress=False,
                                group_by="ticker", threads=False, auto_adjust=False)
        except Exception as e:
            log(f"[data] daily download failed for a chunk: {e}")
            time.sleep(3)
            continue
        for t in chunk:
            try:
                multi = isinstance(d.columns, pd.MultiIndex)
                df = (d[t] if multi and t in d.columns.get_level_values(0) else d).dropna()
                if isinstance(df.columns, pd.MultiIndex):        # flatten any leftover ticker level
                    df.columns = df.columns.get_level_values(-1)
                if len(df) > 25 and {"High", "Low", "Close", "Volume"} <= set(map(str, df.columns)):
                    frames[t] = df
            except Exception:
                continue
        time.sleep(1)
    return frames


def _levels_from_daily(df, today):
    """Key levels using the last COMPLETED session (drops today's partial bar if present)."""
    done = df[df.index.date < today] if df.index[-1].date() == today else df
    if len(done) < 21:
        return None
    last = done.iloc[-1]
    tr = pd.concat([done.High - done.Low, (done.High - done.Close.shift()).abs(),
                    (done.Low - done.Close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())
    avg_vol = float(done.Volume.tail(20).mean())
    hi6, lo6 = float(done.High.max()), float(done.Low.min())
    rng10 = float(done.High.tail(10).max() - done.Low.tail(10).min())
    return {
        "prev_high": float(last.High), "prev_low": float(last.Low), "prev_close": float(last.Close),
        "atr": atr, "atr_pct": 100 * atr / float(last.Close), "avg_vol": avg_vol,
        "vol_ratio_yday": float(last.Volume) / avg_vol if avg_vol else 0,
        "from_high_pct": 100 * (float(last.Close) / hi6 - 1), "from_low_pct": 100 * (float(last.Close) / lo6 - 1),
        "coil": rng10 / atr if atr else 99,                  # 10-day range in ATRs; low = tight
        "nr7": bool((done.High - done.Low).tail(7).idxmin() == done.index[-1]),
        "ret5": 100 * (float(last.Close) / float(done.Close.iloc[-6]) - 1),
        "sma20": float(done.Close.tail(20).mean()), "sma50": float(done.Close.tail(50).mean()),
    }


def premarket_scan():
    log("[premarket] scanning ~550 stocks (1-2 minutes)...")
    t0 = time.time()
    today = now().date()
    uni = list(universe.build().keys())
    frames = _download_daily(uni)
    levels, ideas = {}, []
    for t, df in frames.items():
        lv = _levels_from_daily(df, today)
        if not lv or lv["prev_close"] < MIN_PRICE or lv["avg_vol"] < MIN_AVG_VOLUME:
            continue
        levels[t] = lv
        reasons, score = [], 0
        if lv["from_high_pct"] > -3 and lv["coil"] < 3.0:
            reasons.append("coiled near 6-month high"); score += 3
        if lv["from_low_pct"] < 3 and lv["coil"] < 3.0:
            reasons.append("coiled near 6-month low"); score += 2
        if lv["vol_ratio_yday"] > 2:
            reasons.append(f"{lv['vol_ratio_yday']:.1f}x volume yesterday"); score += 2
        if abs(lv["ret5"]) > 8:
            reasons.append(f"{lv['ret5']:+.0f}% in 5 days"); score += 2
        if lv["nr7"]:
            reasons.append("narrowest range in 7 days"); score += 1
        if lv["atr_pct"] < 1.2:
            score -= 2                                   # too slow for day trades
        if score >= 3:
            ideas.append({"ticker": t, "score": score, "reasons": reasons, "price": lv["prev_close"]})
    ideas.sort(key=lambda x: x["score"], reverse=True)
    save("levels.json", {"date": today.isoformat(), "levels": levels})
    save("premarket.json", {"date": today.isoformat(), "ideas": ideas[:40]})
    log(f"[premarket] {len(frames)} downloaded, {len(levels)} pass filters, {len(ideas)} ideas "
        f"({time.time() - t0:.0f}s)")
    return ideas


def _screens():
    """Market-wide lists from Yahoo's screener."""
    out = {}
    for name in ("most_actives", "day_gainers", "day_losers"):
        try:
            with YF_LOCK:
                quotes = yf.screen(name, count=50).get("quotes", [])
            for q in quotes:
                out.setdefault(q["symbol"], {**q, "lists": []})["lists"].append(name.replace("_", " "))
        except Exception as e:
            log(f"[discovery] screen {name} failed: {e}")
    return out


def _stocktwits():
    try:
        r = requests.get("https://api.stocktwits.com/api/2/trending/symbols.json",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
        return {s["symbol"]: (s.get("trends") or {}).get("summary", "")
                for s in r.get("symbols", []) if s.get("instrument_class") == "Stock"}
    except Exception:
        return {}


def _day_fraction(t):
    mins = (t.hour * 60 + t.minute) - (9 * 60 + 30)
    return min(max(mins / 390, 0.05), 1.0)


def _has_weeklies(ticker, cache):
    if ticker in cache:
        return cache[ticker]
    ok = False
    try:
        with YF_LOCK:
            exps = yf.Ticker(ticker).options
        ok = bool(exps) and (pd.Timestamp(exps[0]) - pd.Timestamp(now().date())).days <= 8
    except Exception:
        pass
    cache[ticker] = ok
    return ok


def refresh_watchlist():
    log("[watchlist] refreshing...")
    t0, t = time.time(), now()
    today = t.strftime("%Y-%m-%d")
    lv = load("levels.json", {})
    levels = lv.get("levels", {}) if lv.get("date") == today else {}
    pre = load("premarket.json", {})
    ideas = {i["ticker"]: i for i in pre.get("ideas", [])} if pre.get("date") == today else {}
    cats = {k: v for k, v in load("catalysts.json", {}).items() if v.get("date") == today}
    screens = _screens()
    trending = _stocktwits()
    frac = _day_fraction(t)

    cands = {}
    def add(tk, pts, why):
        c = cands.setdefault(tk, {"ticker": tk, "score": 0.0, "why": []})
        c["score"] += pts
        c["why"].append(why)

    for tk, q in screens.items():
        price = q.get("regularMarketPrice") or 0
        avgv = q.get("averageDailyVolume3Month") or 0
        cap = q.get("marketCap") or 0
        if price < MIN_PRICE or avgv < MIN_AVG_VOLUME or cap < MIN_MARKET_CAP or q.get("quoteType") != "EQUITY":
            continue
        rvol = (q.get("regularMarketVolume") or 0) / (avgv * frac) if avgv else 0
        chg = q.get("regularMarketChangePercent") or 0
        add(tk, min(rvol, 5) + min(abs(chg) / 2, 4), f"{', '.join(q['lists'])} ({chg:+.1f}%, {rvol:.1f}x vol)")
    for tk, c in cats.items():
        add(tk, 4 + min(c["score"] / 5, 3), f"news: {c['headline'][:80]}")
    for tk, why in trending.items():
        add(tk, 1.5, "trending on Stocktwits")
    for tk, i in ideas.items():
        add(tk, i["score"] / 2, "pre-market: " + ", ".join(i["reasons"]))

    # Quality filters for names we didn't see in the screens (price/volume from daily levels)
    uni = universe.tickers()
    ranked = []
    for c in sorted(cands.values(), key=lambda c: c["score"], reverse=True):
        tk = c["ticker"]
        if tk in CORE or "." in tk or "^" in tk:
            continue
        if tk not in screens:
            l = levels.get(tk)
            if l is None and tk not in uni:
                continue                     # unknown and unfiltered: skip rather than risk junk
            if l and (l["prev_close"] < MIN_PRICE or l["avg_vol"] < MIN_AVG_VOLUME):
                continue
        ranked.append(c)

    cache = load("optionable.json", {})
    cache = cache if cache.get("_date") == today else {"_date": today}
    watch = []
    for c in ranked:
        if len(watch) >= WATCHLIST_SIZE:
            break
        if _has_weeklies(c["ticker"], cache):
            watch.append(c)
    save("optionable.json", cache)

    # Make sure every watched stock has key levels (intraday discoveries may be outside the universe)
    missing = [c["ticker"] for c in watch if c["ticker"] not in levels] + [x for x in CORE if x not in levels]
    if missing:
        for tk, df in _download_daily(missing, period="3mo").items():
            l2 = _levels_from_daily(df, t.date())
            if l2:
                levels[tk] = l2
        save("levels.json", {"date": today, "levels": levels})

    wl = [{"ticker": x, "score": 99, "why": ["core index"]} for x in CORE] + watch
    save("watchlist.json", {"time": t.isoformat(), "items": wl})
    log(f"[watchlist] {len(watch)} stocks + SPY/QQQ from {len(cands)} candidates "
        f"({time.time() - t0:.0f}s): {', '.join(c['ticker'] for c in watch)}")
    return wl
