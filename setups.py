"""
Stage 2 of the scanner: every minute, check the watchlist's 1-minute bars for the setups we
traded by hand, as rules:

  ORB        5-min close outside the 9:30-9:45 range, volume up, same side of VWAP
  VWAP       trending stock pulls back to VWAP, holds, and a bounce candle closes above the prior high
  LEVEL      yesterday's high (or low) breaks on a volume spike
  HOOK       higher low + 1-min RSI turning up from 30-42 + strong candle (the BB pattern)

Each has a mirror-image put version. Setups must pass: market direction (brief regime and SPY vs
VWAP), time windows, reward:risk >= MIN_RR. Every setup is logged; the good ones get pushed.
"""
import time

import numpy as np
import pandas as pd
import yfinance as yf

from common import YF_LOCK, ET, load, now, log, env, hhmm, MIN_RR

WINDOWS = [(env("WINDOW1_START", 945), env("WINDOW1_END", 1130)),
           (env("WINDOW2_START", 1330), env("WINDOW2_END", 1530))]
ORB_END = env("ORB_END", 1100)          # opening range breakouts only count until this time


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _clean(df, t):
    df = df.dropna().copy()
    if df.empty:
        return None
    df.index = df.index.tz_convert(ET)
    df = df[(df.index.hour * 100 + df.index.minute >= 930) & (df.index.date == t.date())]
    if len(df) and df.index[-1].replace(second=0) >= t.replace(second=0, microsecond=0):
        df = df.iloc[:-1]                              # drop the minute that is still forming
    return df if len(df) >= 16 else None


def fetch_bars(tickers):
    """1-minute bars for today. Batch download first, then retry any the batch missed one at a time
    (Yahoo often drops a few symbols from big batch requests)."""
    t, out = now(), {}
    try:
        with YF_LOCK:
            d = yf.download(tickers, period="1d", interval="1m", progress=False, group_by="ticker",
                            threads=False, prepost=False)
        for tk in tickers:
            try:
                df = _clean(d[tk] if len(tickers) > 1 else d, t)
                if df is not None:
                    out[tk] = df
            except Exception:
                continue
    except Exception as e:
        log(f"[data] batch download failed: {e}")
    missing = [tk for tk in tickers if tk not in out]
    for tk in missing:
        try:
            with YF_LOCK:
                h = yf.Ticker(tk).history(period="1d", interval="1m", prepost=False)
            df = _clean(h, t)
            if df is not None:
                out[tk] = df
            time.sleep(0.3)
        except Exception:
            continue
    still = [tk for tk in tickers if tk not in out]
    if still and t.hour * 100 + t.minute >= 950:
        log(f"[data] no usable 1-min data for {len(still)}/{len(tickers)}: {', '.join(still[:12])}")
    return out


def indicators(df):
    tp = (df.High + df.Low + df.Close) / 3
    df["vwap"] = (tp * df.Volume).cumsum() / df.Volume.cumsum()
    df["rsi"] = _rsi(df.Close)
    df["vol_avg"] = df.Volume.rolling(20, min_periods=5).mean().shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - df.Close.shift()).abs(), (df.Low - df.Close.shift()).abs()], axis=1).max(axis=1)
    df["atr1"] = tr.rolling(14, min_periods=5).mean()
    return df


def _swings(df, look=30, k=3, low=True):
    s = df.Low if low else df.High
    s = s.iloc[-look:]
    v, idx = s.to_numpy(dtype=float), s.index        # numpy for speed; same logic as before
    pts = []
    for i in range(k, len(v) - 1):
        win = v[max(0, i - k): i + k + 1]
        if (low and v[i] == win.min()) or (not low and v[i] == win.max()):
            pts.append((idx[i], float(v[i])))
    return pts


def detect(tk, df, lv):
    """Return a list of raw setups for one stock."""
    df = indicators(df)
    last, prev = df.iloc[-1], df.iloc[-2]
    px, vwap, t_now = float(last.Close), float(last.vwap), hhmm(df.index[-1])
    hod, lod = float(df.High.max()), float(df.Low.min())
    atr1 = float(last.atr1) if last.atr1 == last.atr1 else px * 0.002
    vol_ratio = float(last.Volume / last.vol_avg) if last.vol_avg and last.vol_avg == last.vol_avg else 1.0
    found = []

    def add(name, direction, trigger, stop, t1, t2, why):
        # levels must make sense for the direction: stop on the losing side, targets on the winning side
        if direction == "call" and not (stop < trigger < t1):
            return
        if direction == "put" and not (stop > trigger > t1):
            return
        found.append({"ticker": tk, "setup": name, "direction": direction, "price": px,
                      "trigger": round(trigger, 2), "stop": round(stop, 2), "t1": round(t1, 2),
                      "t2": round(t2, 2), "why": why, "bar_vol_ratio": round(vol_ratio, 1),
                      "vwap": round(vwap, 2), "hod": hod, "lod": lod})

    # --- Opening range breakout (5-minute closes) ---
    orng = df.between_time("09:30", "09:44")
    if len(orng) >= 10 and 945 <= t_now <= ORB_END:
        orh, orl = float(orng.High.max()), float(orng.Low.min())
        rng = orh - orl
        b5 = df.resample("5min", label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
        b5 = b5[b5.index + pd.Timedelta(minutes=5) <= df.index[-1] + pd.Timedelta(minutes=1)]  # complete bars only
        if len(b5) >= 5 and rng > 0:
            c, p = b5.iloc[-1], b5.iloc[-2]
            v_ok = c.Volume > 1.2 * float(b5.Volume.iloc[:-1].median())   # median ignores the opening spike
            if c.Close > orh >= p.Close and v_ok and c.Close > vwap:
                add("ORB", "call", float(c.Close), orl + rng / 2, orh + rng, orh + 1.5 * rng,
                    f"5-min close above opening range high {orh:.2f} on volume, above VWAP")
            if c.Close < orl <= p.Close and v_ok and c.Close < vwap:
                add("ORB", "put", float(c.Close), orh - rng / 2, orl - rng, orl - 1.5 * rng,
                    f"5-min close below opening range low {orl:.2f} on volume, below VWAP")

    # --- VWAP pullback in a trend ---
    above = float((df.Close > df.vwap).mean())
    recent = df.iloc[-6:-1]
    day_chg = 100 * (px / lv["prev_close"] - 1) if lv else 0
    if above >= 0.7 and day_chg > 0.8:
        touched = recent.Low.min() <= vwap * 1.0015 and recent.Low.min() >= vwap * 0.996
        if touched and px > vwap and px > float(prev.High) and last.Close > last.Open and hod > px * 1.002:
            stop = float(recent.Low.min()) - 0.1 * atr1
            add("VWAP pullback", "call", px, stop, hod, hod + 0.5 * (hod - vwap),
                f"up {day_chg:.1f}% and above VWAP {above:.0%} of the day; pulled back to VWAP {vwap:.2f} and bounced")
    if above <= 0.3 and day_chg < -0.8:
        touched = recent.High.max() >= vwap * 0.9985 and recent.High.max() <= vwap * 1.004
        if touched and px < vwap and px < float(prev.Low) and last.Close < last.Open and lod < px * 0.998:
            stop = float(recent.High.max()) + 0.1 * atr1
            add("VWAP rejection", "put", px, stop, lod, lod - 0.5 * (vwap - lod),
                f"down {day_chg:.1f}% and below VWAP {1 - above:.0%} of the day; rallied to VWAP {vwap:.2f} and got rejected")

    # --- Prior-day level break ---
    if lv:
        pdh, pdl, datr = lv["prev_high"], lv["prev_low"], lv["atr"]
        before = df.iloc[-4:-1]
        if px > pdh and (before.Close <= pdh).all() and vol_ratio > 2:
            add("Prior-day high break", "call", px, min(float(before.Low.min()), pdh - 0.1 * datr),
                px + 0.35 * datr, px + 0.6 * datr, f"broke yesterday's high {pdh:.2f} on {vol_ratio:.1f}x volume")
        if px < pdl and (before.Close >= pdl).all() and vol_ratio > 2:
            add("Prior-day low break", "put", px, max(float(before.High.max()), pdl + 0.1 * datr),
                px - 0.35 * datr, px - 0.6 * datr, f"broke yesterday's low {pdl:.2f} on {vol_ratio:.1f}x volume")

    # --- Higher low + RSI hook (and the mirror) ---
    rsi_now, rsi_prev = float(last.rsi), float(prev.rsi)
    rsi_min5, rsi_max5 = float(df.rsi.iloc[-6:].min()), float(df.rsi.iloc[-6:].max())
    lows = _swings(df, low=True)
    if len(lows) >= 2 and px > vwap:
        (t1_, l1), (t2_, l2) = lows[-2], lows[-1]
        fresh = (df.index[-1] - t2_) <= pd.Timedelta(minutes=8)
        if l2 > l1 and fresh and rsi_min5 <= 42 and rsi_now > rsi_prev and rsi_now - rsi_min5 >= 3 and px > float(prev.High) and vol_ratio > 1.1:
            swing_hi = float(df.High.iloc[-30:].max())
            tgt = swing_hi if swing_hi > px * 1.002 else hod
            add("Higher-low hook", "call", px, l2 - 0.1 * atr1, tgt, max(hod, tgt + (tgt - l2)),
                f"higher low {l2:.2f} > {l1:.2f}, RSI turned up from {rsi_min5:.0f}, strong candle on {vol_ratio:.1f}x volume")
    highs = _swings(df, low=False)
    if len(highs) >= 2 and px < vwap:
        (t1_, h1), (t2_, h2) = highs[-2], highs[-1]
        fresh = (df.index[-1] - t2_) <= pd.Timedelta(minutes=8)
        if h2 < h1 and fresh and rsi_max5 >= 58 and rsi_now < rsi_prev and rsi_max5 - rsi_now >= 3 and px < float(prev.Low) and vol_ratio > 1.1:
            swing_lo = float(df.Low.iloc[-30:].min())
            tgt = swing_lo if swing_lo < px * 0.998 else lod
            add("Lower-high rollover", "put", px, h2 + 0.1 * atr1, tgt, min(lod, tgt - (h2 - tgt)),
                f"lower high {h2:.2f} < {h1:.2f}, RSI rolled over from {rsi_max5:.0f}, weak candle on {vol_ratio:.1f}x volume")
    return found


def market_bias(bars):
    """SPY's position vs its VWAP: the 'trade in SPY's direction' rule."""
    spy = bars.get("SPY")
    if spy is None:
        return "unknown"
    spy = indicators(spy)
    return "up" if spy.Close.iloc[-1] > spy.vwap.iloc[-1] else "down"


def score(s, regime, bias, catalyst, rvol_day, contract):
    """1-10. Also returns the list of reasons so the alert explains itself."""
    pts, notes = 5, []
    risk = abs(s["trigger"] - s["stop"])
    rr = abs(s["t1"] - s["trigger"]) / risk if risk else 0
    s["rr"] = round(rr, 2)
    r = (regime or "").lower()
    aligned = (s["direction"] == "call" and r.startswith("bull")) or (s["direction"] == "put" and r.startswith("bear"))
    against = (s["direction"] == "call" and r.startswith("bear")) or (s["direction"] == "put" and r.startswith("bull"))
    if aligned: pts += 1; notes.append(f"market read {regime}")
    if against: pts -= 3; notes.append(f"against market read {regime}")
    if (s["direction"] == "call") == (bias == "up"): pts += 1; notes.append(f"SPY {'above' if bias == 'up' else 'below'} VWAP")
    elif bias in ("up", "down"): pts -= 2; notes.append(f"SPY {'above' if bias == 'up' else 'below'} VWAP (against)")
    if rvol_day >= 2: pts += 1; notes.append(f"{rvol_day:.1f}x normal volume")
    if rvol_day >= 3: pts += 1
    if catalyst: pts += 1; notes.append("news catalyst")
    if rr >= 2: pts += 1
    if contract: pts += 1
    else: notes.append("no option passes the filter")
    return max(1, min(10, pts)), notes


def allowed_now(t=None):
    t = t or now()
    return t.weekday() < 5 and any(a <= hhmm(t) <= b for a, b in WINDOWS)


def scan(watch_items):
    """Run detection on the watchlist. Returns setups that pass the hard filters (not yet scored)."""
    tickers = sorted({w["ticker"] for w in watch_items} | {"SPY", "QQQ"})
    bars = fetch_bars(tickers)
    lv_all = load("levels.json", {}).get("levels", {})
    bias = market_bias(bars)
    out = []
    for tk, df in bars.items():
        lv = lv_all.get(tk)
        try:
            for s in detect(tk, df, lv):
                risk = abs(s["trigger"] - s["stop"])
                if risk < s["trigger"] * 0.0015:          # stop too tight to survive noise
                    continue
                if abs(s["t1"] - s["trigger"]) / risk < MIN_RR:
                    continue
                elapsed = max(len(df), 1) / 390
                s["rvol_day"] = round(float(df.Volume.sum()) / (lv["avg_vol"] * elapsed), 1) if lv and lv.get("avg_vol") else 1.0
                out.append(s)
        except Exception as e:
            log(f"[setups] {tk}: {e}")
    return out, bias, bars
