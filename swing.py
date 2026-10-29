#!/usr/bin/env python3
"""
Swing trades: 30-60 DTE options on daily-chart setups, sized for a $500-$1,000 account.

Process (stock first, option second):
  market regime (SPY/QQQ/IWM/VIX + breadth) -> sector ranks -> daily setups on ~550 stocks
  -> stock score (100) -> earnings check -> option chain -> option score (100)
  -> final = 60% stock + 40% option -> size by grade -> top 1-3 or NO TRADE

Four setups: bullish pullback (call), breakout (call), breakdown (put), rally rejection (put).

Linear regression is used twice:
  1. A 50-day regression channel on log price: slope (trend direction/speed), R^2 (trend
     quality) and z (how many standard deviations price sits from the line). Feeds the trend and
     setup scores.
  2. A ridge regression fitted on the backtest predicts each setup's R-multiple from its
     features. It is tested on the most recent 30% of history it never saw, and that
     out-of-sample result is shown next to every prediction so you can see whether it has any edge.

Win rates come only from the backtest (stock-level: target before stop within 15 trading days,
5 years of daily bars). Fewer than 30 matching observations = "Win rate not validated".

Every pick is paper-tracked to an outcome in state/swing.db, and near-misses are tracked at
the stock level, so the live record builds alongside the backtest.

  python swing.py scan          run the daily scan now (prints the report)
  python swing.py backtest      rebuild the backtest + regression (5-10 minutes)
  python swing.py update        grade/mark open picks
  python swing.py report        print the track record
  python swing.py take ID PRICE [CONTRACTS]   record that you actually bought pick ID
  python swing.py close ID PRICE              record your actual exit price
"""
import json, math, sqlite3, sys, threading, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from common import YF_LOCK, ET, load, save, now, log, notify, env, path
import universe

# ------------------------------- settings -------------------------------
ACCOUNT = env("SWING_ACCOUNT", 750.0)
RISK = {"A+": env("SWING_RISK_APLUS", 0.15), "A": env("SWING_RISK_A", 0.10), "B": env("SWING_RISK_B", 0.05)}
MAX_RISK = min(env("SWING_MAX_RISK", 0.20), 0.20)          # hard ceiling: never above 20%
MIN_DTE, MAX_DTE, MAX_DTE_HIGH = env("SWING_MIN_DTE", 30), env("SWING_MAX_DTE", 60), env("SWING_MAX_DTE_HIGH", 75)
MIN_DELTA, MAX_DELTA = env("SWING_MIN_DELTA", 0.55), env("SWING_MAX_DELTA", 0.75)
PREF_SPREAD = env("SWING_MAX_SPREAD_PCT", 0.05)            # preferred bid/ask as % of mid
HARD_SPREAD = env("SWING_HARD_SPREAD_PCT", 0.10)           # single options wider than this: rejected
HARD_SPREAD_VERT = env("SWING_HARD_SPREAD_VERTICAL", 0.15)  # debit spreads (two legs) wider than this: rejected
MIN_OI = env("SWING_MIN_OI", 100)
USE_SPREADS = env("SWING_DEBIT_SPREADS", 1)
MIN_PRICE = env("SWING_MIN_PRICE", 20.0)
MIN_DOLLAR_VOL = env("SWING_MIN_DOLLAR_VOL", 100_000_000.0)
MAX_HOLD = env("SWING_MAX_HOLD", 15)                       # trading days
MIN_STOCK, MIN_OPTION = env("SWING_MIN_STOCK_SCORE", 80), env("SWING_MIN_OPTION_SCORE", 75)
MIN_RR = env("SWING_MIN_RR", 1.5)
MIN_SAMPLE = 30
STOP_FLOOR = env("SWING_STOP_FLOOR_ATR", 1.0)              # stop at least this many ATRs from entry
TARGET_CAP = env("SWING_TARGET_CAP_ATR", 4.0)              # breakout/breakdown targets at most this far
RATE = env("SWING_RATE", 0.04)
SCAN_TIME = env("SWING_SCAN_TIME", 1000)                   # HHMM ET
CHAIN_CANDIDATES = 12                                      # option chains checked per scan

SECTOR_ETF = {"Information Technology": "XLK", "Health Care": "XLV", "Financials": "XLF",
              "Consumer Discretionary": "XLY", "Communication Services": "XLC", "Industrials": "XLI",
              "Consumer Staples": "XLP", "Energy": "XLE", "Utilities": "XLU", "Real Estate": "XLRE",
              "Materials": "XLB"}
ETF_NAME = {v: k for k, v in SECTOR_ETF.items()}
MARKET = ["SPY", "QQQ", "IWM", "^VIX"]
SETUP_NAME = {"PB": "Bullish pullback", "BO": "Breakout", "BD": "Bearish breakdown", "RJ": "Bearish rally rejection"}
SETUP_DIR = {"PB": "call", "BO": "call", "BD": "put", "RJ": "put"}
FEATURES = ["regime", "trend", "rs", "setup_pts", "volume", "momentum", "rr_pts",
            "lr_r2", "lr_slope_dir", "lr_z_dir", "rvol", "atr_pct", "is_PB", "is_BO", "is_BD"]
_job_lock = threading.Lock()


# ------------------------------- data -------------------------------
def _download(tickers, period, chunk=60):
    """Daily bars in small chunks so the minute scanner isn't blocked for long."""
    frames = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            with YF_LOCK:
                d = yf.download(part, period=period, interval="1d", progress=False, group_by="ticker",
                                threads=True, auto_adjust=True)
        except Exception as e:
            log(f"[swing] download failed for a chunk: {e}")
            time.sleep(3)
            continue
        for t in part:
            try:
                multi = isinstance(d.columns, pd.MultiIndex)
                df = d[t] if multi and t in d.columns.get_level_values(0) else d
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(-1)
                df = df.dropna(subset=["Close"])
                if len(df) > 60:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    frames[t] = df
            except Exception:
                continue
        time.sleep(0.5)
    return frames


def sectors():
    """GICS sector per S&P 500 stock (Wikipedia), cached weekly."""
    s = load("sectors.json", {})
    if s.get("map") and time.time() - s.get("built", 0) < 7 * 86400:
        return s["map"]
    try:
        t = universe._wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["symbol", "gics sector"])
        col = [c for c in t.columns if "gics sector" in str(c).lower()][0]
        m = {str(r["Symbol"]).replace(".", "-"): str(r[col]) for _, r in t.iterrows()}
        save("sectors.json", {"map": m, "built": time.time()})
        return m
    except Exception as e:
        log(f"[swing] sector list failed ({e}); using saved copy")
        return s.get("map", {})


# ------------------------------- indicators -------------------------------
def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def linreg(y, n=50):
    """Rolling least-squares line through the last n points of y.
    Returns slope per bar, R^2, and z = distance of the last point from the line in residual SDs."""
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    m_t, m_y = t.rolling(n).mean(), y.rolling(n).mean()
    var_t = (t * t).rolling(n).mean() - m_t ** 2
    var_y = (y * y).rolling(n).mean() - m_y ** 2
    cov = (t * y).rolling(n).mean() - m_t * m_y
    slope = cov / var_t
    fit_end = m_y + slope * (t - m_t)
    r2 = (cov ** 2 / (var_t * var_y)).clip(0, 1)
    resid = np.sqrt((var_y * (1 - r2)).clip(lower=1e-12))
    return slope, r2, (y - fit_end) / resid


def indicators(df):
    c, h, l, o, v = df.Close, df.High, df.Low, df.Open, df.Volume
    d = pd.DataFrame(index=df.index)
    d["c"], d["h"], d["l"], d["o"], d["v"] = c, h, l, o, v
    d["sma20"], d["sma50"], d["sma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d["sma50_slope"] = d.sma50 - d.sma50.shift(10)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["atr_pct"] = 100 * d.atr / c
    d["rsi"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macdh"] = macd - macd.ewm(span=9, adjust=False).mean()
    d["vol50"] = v.shift(1).rolling(50).mean()
    d["rvol"] = v / d.vol50
    d["vol_quiet"] = v.shift(1).rolling(5).mean() / d.vol50      # volume during the pullback / base
    d["ret63"] = c / c.shift(63) - 1
    slope, r2, z = linreg(np.log(c))
    d["lr_slope_ann"], d["lr_r2"], d["lr_z"] = np.exp(slope * 252) - 1, r2, z
    d["hh20"], d["ll20"] = h.rolling(20).max(), l.rolling(20).min()
    d["res20"], d["sup20"] = h.shift(1).rolling(20).max(), l.shift(1).rolling(20).min()
    d["rng10"] = h.shift(1).rolling(10).max() - l.shift(1).rolling(10).min()
    d["low5"], d["high5"] = l.rolling(5).min(), h.rolling(5).max()
    d["low3"], d["high3"] = l.rolling(3).min(), h.rolling(3).max()
    d["hi252"], d["lo252"] = h.rolling(252, min_periods=120).max(), l.rolling(252, min_periods=120).min()
    d["clv"] = (c - l) / (h - l).replace(0, np.nan)
    d["dollar_vol"] = (c * v).rolling(20).mean()
    d["hv20"] = np.log(c).diff().rolling(20).std() * math.sqrt(252)
    return d


def market_context(frames):
    """Regime points for calls and puts on every date, plus breadth, RS ranks and sector ranks."""
    close = pd.DataFrame({t: f.Close for t, f in frames.items()}).sort_index().ffill(limit=5)
    spy, qqq, vix = close["SPY"], close["QQQ"], close["^VIX"]
    stocks = close.drop(columns=[c for c in close.columns if c in MARKET or c in ETF_NAME], errors="ignore")
    breadth = (stocks > stocks.rolling(50).mean()).sum(axis=1) / stocks.rolling(50).mean().notna().sum(axis=1)
    ctx = pd.DataFrame({"spy": spy, "spy50": spy.rolling(50).mean(), "spy200": spy.rolling(200).mean(),
                        "qqq": qqq, "qqq50": qqq.rolling(50).mean(), "vix": vix, "vix10": vix.rolling(10).mean(),
                        "breadth": breadth})
    if "IWM" in close:
        ctx["iwm"], ctx["iwm50"] = close["IWM"], close["IWM"].rolling(50).mean()
    vixp = np.where(ctx.vix < 20, 3, np.where(ctx.vix < 25, 1, 0))
    ctx["reg_call"] = (4 * (ctx.spy > ctx.spy50) + 3 * (ctx.spy > ctx.spy200) + 3 * (ctx.qqq > ctx.qqq50)
                       + vixp + 2 * (ctx.breadth > 0.55))
    ctx["reg_put"] = (4 * (ctx.spy < ctx.spy50) + 2 * (ctx.spy < ctx.spy200) + 3 * (ctx.qqq < ctx.qqq50)
                      + 3 * (ctx.vix > ctx.vix10) + 3 * (ctx.breadth < 0.45))
    ret63 = stocks / stocks.shift(63) - 1
    rs_pct = ret63.rank(axis=1, pct=True)
    etfs = [e for e in ETF_NAME if e in close]
    sec_rank = (close[etfs] / close[etfs].shift(63) - 1).rank(axis=1, ascending=False) if etfs else pd.DataFrame()
    return ctx, rs_pct, sec_rank


def _pts(x, cuts):
    """Step function: cuts = [(threshold, points), ...] checked in order with >=."""
    out = np.zeros(len(x))
    done = np.zeros(len(x), dtype=bool)
    for th, p in cuts:
        hit = (x >= th) & ~done
        out[hit] = p
        done |= hit
    return out


def signals(tk, d, ctx, rs_pct, sec_rank, sector_etf=None):
    """Every setup on every bar for one stock, with its stock score (0-100) and features."""
    ctx = ctx.reindex(d.index)
    rs = rs_pct[tk].reindex(d.index) if tk in rs_pct else pd.Series(np.nan, index=d.index)
    sr = sec_rank[sector_etf].reindex(d.index) if sector_etf and sector_etf in sec_rank else pd.Series(np.nan, index=d.index)
    c, h, l, o = d.c, d.h, d.l, d.o
    ok = (c >= MIN_PRICE) & (d.dollar_vol >= MIN_DOLLAR_VOL) & d.sma200.notna() & d.atr.notna() & d.lr_r2.notna()

    r2p = np.where(d.lr_r2 >= 0.7, 4, np.where(d.lr_r2 >= 0.5, 2.5, np.where(d.lr_r2 >= 0.3, 1, 0)))
    trend_call = (3 * (c > d.sma50) + 3 * (d.sma50 > d.sma200) + 2 * (d.sma50_slope > 0)
                  + r2p * (d.lr_slope_ann > 0) + np.where(d.lr_slope_ann >= 0.2, 3, np.where(d.lr_slope_ann > 0, 1.5, 0)))
    trend_put = (3 * (c < d.sma50) + 3 * (d.sma50 < d.sma200) + 2 * (d.sma50_slope < 0)
                 + r2p * (d.lr_slope_ann < 0) + np.where(d.lr_slope_ann <= -0.2, 3, np.where(d.lr_slope_ann < 0, 1.5, 0)))
    sec_call = np.where(sr.isna(), 1.5, np.where(sr <= 3, 3, np.where(sr <= 5, 2, np.where(sr <= 8, 1, 0))))
    sec_put = np.where(sr.isna(), 1.5, np.where(sr >= 9, 3, np.where(sr >= 7, 2, np.where(sr >= 4, 1, 0))))
    rs_call = 12 * ((rs - 0.5) / 0.4).clip(0, 1).fillna(0) + sec_call
    rs_put = 12 * ((0.5 - rs) / 0.4).clip(0, 1).fillna(0) + sec_put
    vol = (_pts(d.rvol.fillna(0).values, [(2, 6), (1.5, 5), (1.2, 3.5), (1.0, 2)])
           + np.where(d.vol_quiet < 1.0, 4, np.where(d.vol_quiet < 1.1, 2, 0)))
    mom_call = 3 * (d.rsi > d.rsi.shift(1)) + 4 * d.rsi.between(45, 70) + 3 * (d.macdh > d.macdh.shift(1))
    mom_put = 3 * (d.rsi < d.rsi.shift(1)) + 4 * d.rsi.between(30, 55) + 3 * (d.macdh < d.macdh.shift(1))
    a = d.atr

    setups = {}
    # 1. Bullish pullback: uptrend, controlled pullback into the 20/50-day average, reversal candle
    depth = (d.hh20 - d.low5) / a
    sup_dist = np.minimum((d.low3 - d.sma20).abs(), (d.low3 - d.sma50).abs()) / a
    rev = (c > h.shift(1)) | ((c > o) & (d.clv >= 0.6) & (c > c.shift(1)))
    m = ((c > d.sma50) & (d.sma50 > d.sma200) & (d.sma50_slope > 0) & (depth >= 1.5) & (d.low3 <= d.sma20 * 1.01)
         & (d.low3 >= d.sma50 * 0.98) & rev & (d.rvol >= 0.8) & (c < d.hh20 - 0.5 * a))
    pts = (np.where(depth.between(2, 4), 6, np.where(depth.between(1.5, 5), 4, 2))
           + np.where(sup_dist <= 0.3, 4, np.where(sup_dist <= 0.7, 2, 1))
           + np.where((d.clv >= 0.8) & (c > h.shift(1)), 5, np.where(d.clv >= 0.6, 3, 1))
           + np.where(d.lr_z.rolling(5).min().between(-2.5, -0.5), 5, np.where(d.lr_z.rolling(5).min().between(-0.5, 0), 2, 0)))
    setups["PB"] = (m, c, d.low5 - 0.25 * a, d.hh20, pts)
    # 2. Breakout: tight base under resistance, quiet volume, then a close through it on heavy volume
    base_h = (d.res20 - d.sup20)
    ext = (c - d.res20) / a
    m = ((c > d.res20) & (c > d.sma50) & (base_h / d.res20 <= 0.15) & (d.rng10 <= 4 * a)
         & (d.v.shift(1).rolling(10).mean() / d.vol50 < 1.0) & (d.rvol >= 1.5) & (ext <= 1.0) & (d.clv >= 0.5))
    pts = (np.where(d.rng10 / a <= 2.5, 6, np.where(d.rng10 / a <= 3.25, 4, 2))
           + np.where(d.clv >= 0.75, 5, 3) + np.where(ext <= 0.5, 5, 2)
           + np.where(c >= 0.95 * d.hi252, 4, np.where(c >= 0.85 * d.hi252, 2, 0)))
    setups["BO"] = (m, c, d.res20 - 1.0 * a, d.res20 + base_h, pts)
    # 3. Bearish breakdown: below the averages, closes through support on heavy volume
    ext = (d.sup20 - c) / a
    m = ((c < d.sup20) & (c < d.sma50) & ((c < d.sma200) | (d.sma50 < d.sma200)) & (base_h / d.sup20 <= 0.20)
         & (d.rvol >= 1.5) & (ext <= 1.0) & (d.clv <= 0.5))
    pts = (np.where(d.rng10 / a <= 2.5, 6, np.where(d.rng10 / a <= 3.25, 4, 2))
           + np.where(d.clv <= 0.25, 5, 3) + np.where(ext <= 0.5, 5, 2)
           + np.where(c <= 1.05 * d.lo252, 4, np.where(c <= 1.15 * d.lo252, 2, 0)))
    setups["BD"] = (m, c, d.sup20 + 1.0 * a, (d.sup20 - base_h).clip(lower=0.01), pts)
    # 4. Rally rejection: downtrend, bounce into the 20/50-day average, bearish rejection candle
    rally = (d.high5 - d.ll20) / a
    res_dist = np.minimum((d.high3 - d.sma20).abs(), (d.high3 - d.sma50).abs()) / a
    rej = (c < l.shift(1)) | ((c < o) & (d.clv <= 0.4) & (c < c.shift(1)))
    m = ((c < d.sma50) & (d.sma50 < d.sma200) & (d.sma50_slope < 0) & (rally >= 1.5) & (d.high3 >= d.sma20 * 0.99)
         & (d.high3 <= d.sma50 * 1.02) & rej & (d.rvol >= 0.8) & (c > d.ll20 + 0.5 * a))
    pts = (np.where(rally.between(2, 4), 6, np.where(rally.between(1.5, 5), 4, 2))
           + np.where(res_dist <= 0.3, 4, np.where(res_dist <= 0.7, 2, 1))
           + np.where((d.clv <= 0.2) & (c < l.shift(1)), 5, np.where(d.clv <= 0.4, 3, 1))
           + np.where(d.lr_z.rolling(5).max().between(0.5, 2.5), 5, np.where(d.lr_z.rolling(5).max().between(0, 0.5), 2, 0)))
    setups["RJ"] = (m, c, d.high5 + 0.25 * a, d.ll20, pts)

    rows = []
    for key, (mask, entry, stop, target, spts) in setups.items():
        call = SETUP_DIR[key] == "call"
        # Stops closer than STOP_FLOOR ATRs sit inside normal daily noise: push them out to the floor
        stop = np.minimum(stop, entry - STOP_FLOOR * a) if call else np.maximum(stop, entry + STOP_FLOOR * a)
        if key in ("BO", "BD"):                         # measured moves capped at TARGET_CAP ATRs
            target = np.minimum(target, entry + TARGET_CAP * a) if call else np.maximum(target, entry - TARGET_CAP * a)
        risk = (entry - stop) if call else (stop - entry)
        rr = ((target - entry) if call else (entry - target)) / risk
        mask = (mask & ok & (risk > 0) & (rr >= MIN_RR)).fillna(False).values
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        rrv = rr.values[idx]
        part = pd.DataFrame({
            "i": idx, "date": d.index[idx], "ticker": tk, "setup": key, "direction": SETUP_DIR[key],
            "entry": entry.values[idx], "stop": stop.values[idx], "target": target.values[idx], "rr": rrv,
            "regime": np.asarray(ctx.reg_call if call else ctx.reg_put, dtype=float)[idx],
            "trend": np.asarray(trend_call if call else trend_put, dtype=float)[idx],
            "rs": np.asarray(rs_call if call else rs_put, dtype=float)[idx], "setup_pts": np.asarray(spts, dtype=float)[idx], "volume": vol[idx],
            "momentum": np.asarray(mom_call if call else mom_put, dtype=float)[idx], "rr_pts": _pts(rrv, [(3, 15), (2.5, 12), (2, 10), (1.5, 5)]),
            "lr_r2": d.lr_r2.values[idx], "lr_slope_ann": d.lr_slope_ann.values[idx], "lr_z": d.lr_z.values[idx],
            "rvol": d.rvol.values[idx], "atr_pct": d.atr_pct.values[idx], "atr": a.values[idx],
            "rs_pct": rs.values[idx], "sector_rank": sr.values[idx], "hv20": d.hv20.values[idx], "rsi": d.rsi.values[idx]})
        rows.append(part)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["stock_score"] = out[["regime", "trend", "rs", "setup_pts", "volume", "momentum", "rr_pts"]].sum(axis=1).round(1)
    sgn = np.where(out.direction == "call", 1, -1)
    out["lr_slope_dir"] = out.lr_slope_ann * sgn
    out["lr_z_dir"] = out.lr_z * sgn
    for k in ("PB", "BO", "BD"):
        out[f"is_{k}"] = (out.setup == k).astype(float)
    return out


# ------------------------------- backtest + regression -------------------------------
def _simulate(o, h, l, c, i, call, entry, stop, target):
    """Daily bars after signal bar i. Stop is checked before target on the same bar (conservative);
    a gap through the stop fills at the open. Returns (outcome, R, bars held, exit index) or None."""
    risk = (entry - stop) if call else (stop - entry)
    n = len(c)
    if i + MAX_HOLD >= n:
        return None
    for j in range(i + 1, i + MAX_HOLD + 1):
        if call:
            if o[j] <= stop:
                return "loss", (o[j] - entry) / risk, j - i, j
            if l[j] <= stop:
                return "loss", -1.0, j - i, j
            if h[j] >= target:
                return "win", (target - entry) / risk, j - i, j
        else:
            if o[j] >= stop:
                return "loss", (entry - o[j]) / risk, j - i, j
            if h[j] >= stop:
                return "loss", -1.0, j - i, j
            if l[j] <= target:
                return "win", (entry - target) / risk, j - i, j
    j = i + MAX_HOLD
    return "timeout", ((c[j] - entry) if call else (entry - c[j])) / risk, MAX_HOLD, j


def _stats(df):
    if df.empty:
        return {"n": 0}
    w = df[df.outcome == "win"]
    lo = df[df.outcome == "loss"]
    return {"n": int(len(df)), "win_rate": round(100 * len(w) / len(df), 1),
            "loss_rate": round(100 * len(lo) / len(df), 1),
            "timeout_rate": round(float(100 * (df.outcome == "timeout").mean()), 1),
            "avg_r": round(float(df.r.mean()), 3), "avg_win_r": round(float(w.r.mean()), 2) if len(w) else None,
            "avg_timeout_r": round(float(df[df.outcome == "timeout"].r.mean()), 2) if (df.outcome == "timeout").any() else None,
            "days_to_win": float(w.days.median()) if len(w) else None,
            "days_to_loss": float(lo.days.median()) if len(lo) else None,
            "first": str(df.date.min())[:10], "last": str(df.date.max())[:10]}


def _fit_ridge(X, y, lam=1.0):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"mu": mu.tolist(), "sd": sd.tolist(), "coef": coef.tolist(), "b0": float(y.mean())}


def _predict(model, X):
    Z = (np.asarray(X, dtype=float) - np.array(model["mu"])) / np.array(model["sd"])
    return Z @ np.array(model["coef"]) + model["b0"]


def regression(trades):
    """Ridge regression of R-multiple on setup features. Train on the oldest 70%, test on the newest 30%."""
    df = trades.dropna(subset=FEATURES).sort_values("date")
    if len(df) < 200:
        return None
    X, y = df[FEATURES].values.astype(float), df.r.clip(-3, 5).values.astype(float)
    cut = int(len(df) * 0.7)
    m = _fit_ridge(X[:cut], y[:cut])
    pred = _predict(m, X[cut:])
    test = df.iloc[cut:].assign(pred=pred)
    corr = float(np.corrcoef(pred, y[cut:])[0, 1]) if np.std(pred) > 0 else 0.0
    q_hi, q_lo = np.quantile(pred, 2 / 3), np.quantile(pred, 1 / 3)
    top, bot = test[test.pred >= q_hi], test[test.pred <= q_lo]
    final = _fit_ridge(X, y)
    weights = sorted(zip(FEATURES, final["coef"]), key=lambda x: -abs(x[1]))
    return {**final, "features": FEATURES, "n_train": cut, "n_test": len(df) - cut,
            "test_from": str(test.date.min())[:10], "oos_corr": round(corr, 3),
            "top_third_avg_r": round(float(top.r.mean()), 3), "bottom_third_avg_r": round(float(bot.r.mean()), 3),
            "top_third_win": round(100 * float((top.outcome == "win").mean()), 1),
            "bottom_third_win": round(100 * float((bot.outcome == "win").mean()), 1),
            "weights": [(f, round(w, 3)) for f, w in weights]}


def backtest(years=5):
    t0 = time.time()
    log(f"[swing] backtest: downloading {years}y of daily bars (several minutes)...")
    uni = [t for t in universe.build().keys() if "^" not in t]
    secmap = sectors()
    frames = _download(sorted(set(uni) | set(MARKET) | set(ETF_NAME)), f"{years}y")
    if not all(t in frames for t in ("SPY", "QQQ", "^VIX")):
        log("[swing] backtest aborted: market data missing")
        return None
    ctx, rs_pct, sec_rank = market_context(frames)
    trades = []
    for tk, f in frames.items():
        if tk in MARKET or tk in ETF_NAME:
            continue
        d = indicators(f)
        s = signals(tk, d, ctx, rs_pct, sec_rank, SECTOR_ETF.get(secmap.get(tk, "")))
        if s.empty:
            continue
        o, h, l, c = d.o.values, d.h.values, d.l.values, d.c.values
        busy = {"call": -1, "put": -1}                   # one open trade per direction per stock
        for r in s.sort_values("i").itertuples():
            if r.i <= busy[r.direction]:
                continue
            res = _simulate(o, h, l, c, r.i, r.direction == "call", r.entry, r.stop, r.target)
            if res is None:
                continue
            busy[r.direction] = res[3]
            trades.append({**r._asdict(), "outcome": res[0], "r": res[1], "days": res[2]})
    tr = pd.DataFrame(trades)
    if tr.empty:
        log("[swing] backtest found no trades")
        return None
    tr = tr.drop(columns=["Index"], errors="ignore")
    res = {"built": time.time(), "built_at": now().isoformat(), "years": years, "n_stocks": len(frames),
           "max_hold": MAX_HOLD, "by_setup": {}}
    for k in SETUP_NAME:
        g = tr[tr.setup == k]
        res["by_setup"][k] = {"all": _stats(g), "score70": _stats(g[g.stock_score >= 70]),
                              "score80": _stats(g[g.stock_score >= 80]),
                              "score80_by_year": {str(y): _stats(x) for y, x in
                                                  g[g.stock_score >= 80].groupby(pd.to_datetime(g[g.stock_score >= 80].date).dt.year)}}
    res["all"] = _stats(tr)
    res["score80"] = _stats(tr[tr.stock_score >= 80])
    res["model"] = regression(tr)
    save("swing_backtest.json", res)
    m = res["model"]
    log(f"[swing] backtest done: {len(tr)} trades, {res['score80'].get('n', 0)} with stock score 80+ "
        f"({time.time() - t0:.0f}s). Regression out-of-sample corr {m['oos_corr'] if m else 'n/a'}")
    return res


def bt_stats(setup):
    bt = load("swing_backtest.json", {})
    return bt.get("by_setup", {}).get(setup, {}).get("score80", {"n": 0}), bt


# ------------------------------- options math -------------------------------
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, iv, call):
    if T <= 0 or iv <= 0:
        return max(S - K, 0) if call else max(K - S, 0)
    sq = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (RATE + 0.5 * iv * iv) * T) / sq
    d2 = d1 - sq
    if call:
        return S * _ncdf(d1) - K * math.exp(-RATE * T) * _ncdf(d2)
    return K * math.exp(-RATE * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(S, K, T, iv, call):
    if T <= 0 or iv <= 0:
        return (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
    d1 = (math.log(S / K) + (RATE + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return _ncdf(d1) if call else _ncdf(d1) - 1


def implied_vol(price, S, K, T, call):
    lo, hi = 0.01, 5.0
    if price <= (max(S - K, 0) if call else max(K - S, 0)) + 1e-4 or price <= 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(S, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _num(x):
    try:
        x = float(x)
        return x if x == x else 0.0
    except (TypeError, ValueError):
        return 0.0


# ------------------------------- tracking DB -------------------------------
def db():
    con = sqlite3.connect(path("swing.db"), timeout=30)
    con.execute("""CREATE TABLE IF NOT EXISTS picks (
      id INTEGER PRIMARY KEY, date TEXT, ts TEXT, signal_date TEXT, ticker TEXT, direction TEXT, setup TEXT,
      kind TEXT, rank INTEGER, entry REAL, stop REAL, target REAL, rr REAL, stock_score REAL, option_score REAL,
      final_score REAL, tier TEXT, contract TEXT, opt_entry REAL, contracts INTEGER, max_loss REAL,
      est_value_target REAL, hist_win REAL, hist_n INTEGER, pred_r REAL, earnings TEXT, regime TEXT, note TEXT,
      status TEXT DEFAULT 'open', last_px REAL, last_opt REAL, last_update TEXT, days_held INTEGER DEFAULT 0,
      outcome TEXT, r_multiple REAL, opt_exit REAL, opt_return REAL, closed_date TEXT,
      taken INTEGER DEFAULT 0, fill REAL, fill_contracts INTEGER, exit_fill REAL, exit_date TEXT)""")
    con.execute("CREATE TABLE IF NOT EXISTS iv_history (date TEXT, ticker TEXT, iv REAL, PRIMARY KEY (date, ticker))")
    con.commit()
    return con


def iv_rank(con, ticker, iv_now):
    rows = [r[0] for r in con.execute("SELECT iv FROM iv_history WHERE ticker=? ORDER BY date DESC LIMIT 252", (ticker,))]
    if len(rows) < 60 or iv_now is None:
        return None, None, len(rows)
    lo, hi = min(rows), max(rows)
    rank = 100 * (iv_now - lo) / (hi - lo) if hi > lo else 50.0
    pct = 100 * sum(1 for x in rows if x < iv_now) / len(rows)
    return round(rank), round(pct), len(rows)


# ------------------------------- option selection -------------------------------
def next_earnings(ticker):
    try:
        with YF_LOCK:
            cal = yf.Ticker(ticker).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            ed = ed if isinstance(ed, (list, tuple)) else [ed]
            future = sorted(pd.Timestamp(x).date() for x in ed if pd.Timestamp(x).date() >= now().date())
            return future[0] if future else None
    except Exception:
        pass
    try:
        with YF_LOCK:
            e = yf.Ticker(ticker).get_earnings_dates(limit=6)
        fut = sorted(x.date() for x in e.index if x.date() >= now().date())
        return fut[0] if fut else None
    except Exception:
        return None


def _chain_rows(df, S, T, call):
    out = []
    for _, r in df.iterrows():
        bid, ask = _num(r.bid), _num(r.ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2
        K = float(r.strike)
        iv = implied_vol(mid, S, K, T, call) or (_num(r.impliedVolatility) or None)
        if not iv or iv < 0.03:
            continue
        out.append({"strike": K, "bid": bid, "ask": ask, "mid": mid, "spread": ask - bid, "spread_pct": (ask - bid) / mid,
                    "iv": iv, "delta": bs_delta(S, K, T, iv, call), "oi": int(_num(r.openInterest)),
                    "volume": int(_num(r.volume)), "symbol": str(r.contractSymbol)})
    return out


def _value(legs, S, days_left, call):
    T = max(days_left, 0) / 365
    return sum(sign * bs_price(S, leg["strike"], T, leg["iv"], call) for sign, leg in legs)


def _option_score(st, cand, ivinfo):
    """Option score 0-100: liquidity 20, spread 15, DTE 15, delta 15, IV 15, OI 10, expected return 10."""
    p = {}
    vol = st["volume"]
    p["liquidity"] = (10 if vol >= 1000 else 7 if vol >= 300 else 4 if vol >= 100 else 2 if vol >= 20 else 0) + \
                     (10 if cand["dollar_vol"] >= 1e9 else 7 if cand["dollar_vol"] >= 3e8 else 4)
    sp = st["spread_pct"]
    if st["kind"] == "single":
        p["spread"] = 15 if sp <= 0.03 else 11 if sp <= PREF_SPREAD else 6 if sp <= 0.075 else 2
    else:                                                     # two legs: judged on the combined natural spread
        p["spread"] = 15 if sp <= 0.03 else 11 if sp <= PREF_SPREAD else 6 if sp <= 0.10 else 2
    dte = st["dte"]
    p["dte"] = 15 if 35 <= dte <= 55 else 13 if MIN_DTE <= dte <= MAX_DTE else 9
    dl = abs(st["delta"])
    p["delta"] = 15 if 0.60 <= dl <= 0.70 else 12
    rank, ratio = ivinfo.get("rank"), st["iv"] / cand["hv20"] if cand["hv20"] else None
    if rank is not None:
        p["iv"] = 15 if rank <= 30 else 12 if rank <= 50 else 7 if rank <= 70 else 2
    elif ratio is not None:
        p["iv"] = 15 if ratio <= 1.0 else 12 if ratio <= 1.2 else 8 if ratio <= 1.4 else 4 if ratio <= 1.7 else 0
    else:
        p["iv"] = 5
    oi = st["oi"]
    p["oi"] = 10 if oi >= 5000 else 8 if oi >= 2000 else 6 if oi >= 1000 else 4 if oi >= 500 else 1
    er = st["exp_return"]
    p["exp_return"] = 10 if er >= 1.0 else 8 if er >= 0.6 else 6 if er >= 0.4 else 3 if er >= 0.25 else 0
    return sum(p.values()), p


def _tier(final):
    return "A+" if final >= 90 else "A" if final >= 85 else "B"


def choose_option(cand, con):
    """Best affordable structure for a stock candidate, or (None, reason)."""
    S, call = cand["price"], cand["direction"] == "call"
    today = now().date()
    try:
        with YF_LOCK:
            tk = yf.Ticker(cand["ticker"])
            exps = list(tk.options)
    except Exception as e:
        return None, f"no option chain ({e.__class__.__name__})"
    ok_exps = [(x, (pd.Timestamp(x).date() - today).days) for x in exps]
    ok_exps = [(x, d) for x, d in ok_exps if MIN_DTE <= d <= MAX_DTE_HIGH][:5]
    if not ok_exps:
        return None, f"no expiration between {MIN_DTE} and {MAX_DTE_HIGH} days"
    st80, _ = bt_stats(cand["setup"])
    d_win = (st80.get("days_to_win") or 6) * 7 / 5            # trading -> calendar days
    d_loss = (st80.get("days_to_loss") or 4) * 7 / 5
    d_hold = MAX_HOLD * 7 / 5
    ivinfo, structs, reasons = {}, [], set()
    for exp, dte in ok_exps:
        T = dte / 365
        try:
            with YF_LOCK:
                ch = tk.option_chain(exp)
        except Exception:
            continue
        rows = _chain_rows(ch.calls if call else ch.puts, S, T, call)
        if not rows:
            reasons.add("no live quotes (market closed?)")
            continue
        if "atm_iv" not in ivinfo and dte <= 50:          # record ATM IV for future IV rank
            atm = min(rows, key=lambda r: abs(r["strike"] - S))
            ivinfo["atm_iv"] = atm["iv"]
            con.execute("INSERT OR REPLACE INTO iv_history VALUES (?,?,?)", (today.isoformat(), cand["ticker"], atm["iv"]))
            con.commit()
            ivinfo["rank"], ivinfo["pct"], ivinfo["days"] = iv_rank(con, cand["ticker"], atm["iv"])
        longs = [r for r in rows if MIN_DELTA <= abs(r["delta"]) <= MAX_DELTA and r["oi"] >= MIN_OI]
        if not longs:
            reasons.add(f"no {MIN_DELTA:.2f}-{MAX_DELTA:.2f} delta strike with open interest {MIN_OI}+")
        for lg in longs:
            base = {"expiry": exp, "dte": dte, "delta": lg["delta"], "iv": lg["iv"]}
            if lg["spread_pct"] <= HARD_SPREAD:
                structs.append({**base, "kind": "single", "legs": [(1, lg)], "mid": lg["mid"],
                                "bid": lg["bid"], "ask": lg["ask"], "spread_pct": lg["spread_pct"],
                                "oi": lg["oi"], "volume": lg["volume"], "width": None})
            else:
                reasons.add(f"bid/ask over {HARD_SPREAD:.0%} of mid")
            if not USE_SPREADS:
                continue
            beyond = [r for r in rows if (r["strike"] > lg["strike"] if call else r["strike"] < lg["strike"])
                      and r["oi"] >= MIN_OI]
            if not beyond:
                continue
            # Every short strike from the next one out up to the one nearest the stock target:
            # narrow spreads fit a small account, wider ones keep more of the move.
            beyond.sort(key=lambda r: abs(r["strike"] - lg["strike"]))
            far = min(beyond, key=lambda r: abs(r["strike"] - cand["target"]))
            for sh in beyond:
                if abs(sh["strike"] - lg["strike"]) > abs(far["strike"] - lg["strike"]):
                    break
                net = lg["mid"] - sh["mid"]
                width = abs(sh["strike"] - lg["strike"])
                if net <= 0.05 or net >= width:
                    continue
                nat = (lg["spread"] + sh["spread"]) / net
                if nat > HARD_SPREAD_VERT:
                    reasons.add("debit spread too wide to fill fairly")
                    continue
                structs.append({**base, "kind": "spread", "legs": [(1, lg), (-1, sh)], "mid": net,
                                "bid": lg["bid"] - sh["ask"], "ask": lg["ask"] - sh["bid"], "spread_pct": nat,
                                "oi": min(lg["oi"], sh["oi"]), "volume": min(lg["volume"], sh["volume"]), "width": width})
    if not structs:
        return None, "; ".join(sorted(reasons)) or "no contract passed the liquidity filters"

    best, cheapest = None, None
    for st in structs:
        v_t = _value(st["legs"], cand["target"], st["dte"] - d_win, call)
        v_s = _value(st["legs"], cand["stop"], st["dte"] - d_loss, call)
        v_0 = _value(st["legs"], S, st["dte"] - d_hold, call)
        if st["width"]:
            v_t, v_s, v_0 = (min(max(x, 0), st["width"]) for x in (v_t, v_s, v_0))
        st.update(v_target=v_t, v_stop=max(v_s, 0), v_flat=max(v_0, 0), exp_return=v_t / st["mid"] - 1,
                  opt_rr=(v_t - st["mid"]) / max(st["mid"] - max(v_s, 0), 0.01))
        if st["exp_return"] <= 0:
            continue
        score, parts = _option_score(st, cand, ivinfo)
        final = 0.6 * cand["stock_score"] + 0.4 * score
        tier = _tier(final)
        budget = min(RISK[tier], MAX_RISK) * ACCOUNT
        cost = st["mid"] * 100
        n = int(budget // cost)
        st.update(option_score=round(score, 1), parts=parts, final=round(final, 1), tier=tier, budget=budget,
                  cost=cost, contracts=n)
        if cheapest is None or cost < cheapest["cost"]:
            cheapest = st
        if n < 1 or score < MIN_OPTION:
            continue
        if best is None or (final, -st["spread_pct"]) > (best["final"], -best["spread_pct"]):
            best = st
    if best is None:
        if cheapest is None:
            return None, "no contract makes money at the stock target"
        if cheapest["contracts"] < 1:
            return None, (f"cheapest workable contract costs ${cheapest['cost']:.0f}, over the "
                          f"${cheapest['budget']:.0f} {cheapest['tier']} risk budget")
        return None, f"best option score {max(s.get('option_score', 0) for s in structs):.0f} < {MIN_OPTION}"
    best["ivinfo"] = ivinfo
    return best, ""


# ------------------------------- the daily scan -------------------------------
def _regime_text(row):
    bits = [f"SPY {'>' if row.spy > row.spy50 else '<'} 50 SMA", f"{'>' if row.spy > row.spy200 else '<'} 200 SMA",
            f"QQQ {'>' if row.qqq > row.qqq50 else '<'} 50 SMA", f"VIX {row.vix:.1f}{' rising' if row.vix > row.vix10 else ''}",
            f"{100 * row.breadth:.0f}% of stocks above 50 SMA"]
    if "iwm" in row and row.iwm == row.iwm:
        bits.insert(3, f"IWM {'>' if row.iwm > row.iwm50 else '<'} 50 SMA")
    lean = "Bullish" if row.reg_call >= 11 else "Bearish" if row.reg_put >= 11 else "Mixed"
    return lean, ", ".join(bits)


def scan(push=True):
    t0, t = time.time(), now()
    log("[swing] daily scan: downloading 1 year of daily bars...")
    uni = [x for x in universe.build().keys() if "^" not in x]
    secmap = sectors()
    frames = _download(sorted(set(uni) | set(MARKET) | set(ETF_NAME)), "1y")
    if not all(x in frames for x in ("SPY", "QQQ", "^VIX")):
        log("[swing] scan aborted: market data missing")
        return None
    live = {k: float(f.Close.iloc[-1]) for k, f in frames.items()}      # today's price so far
    today = pd.Timestamp(t.date())
    market_open = t.weekday() < 5 and 930 <= t.hour * 100 + t.minute < 1600
    # Signals use COMPLETED daily bars only: drop today's bar while it is still forming.
    done = {k: (f[f.index < today] if market_open else f) for k, f in frames.items()}
    ctx, rs_pct, sec_rank = market_context(done)
    last = ctx.dropna(subset=["spy50", "reg_call"]).iloc[-1]
    lean, reg_text = _regime_text(last)
    sig_date = str(ctx.index[-1].date())
    sec_order = sec_rank.iloc[-1].sort_values() if not sec_rank.empty else pd.Series(dtype=float)
    sector_table = [{"etf": e, "sector": ETF_NAME[e], "rank": int(r),
                     "ret63": round(100 * (done[e].Close.iloc[-1] / done[e].Close.iloc[-64] - 1), 1)}
                    for e, r in sec_order.items() if e in done]

    cands = []
    for tk, f in done.items():
        if tk in MARKET or tk in ETF_NAME or len(f) < 210:
            continue
        d = indicators(f)
        s = signals(tk, d, ctx, rs_pct, sec_rank, SECTOR_ETF.get(secmap.get(tk, "")))
        if s.empty:
            continue
        s = s[s.i == len(d) - 1]                       # setups on the latest completed bar only
        for r in s.to_dict("records"):
            call = r["direction"] == "call"
            px = live.get(tk, r["entry"]) if market_open else r["entry"]
            # Don't chase: price must still be on the right side of the stop and within half an ATR of the signal close
            if (call and (px <= r["stop"] or px >= r["target"] or px > r["entry"] + 0.5 * r["atr"])) or \
               (not call and (px >= r["stop"] or px <= r["target"] or px < r["entry"] - 0.5 * r["atr"])):
                continue
            risk = (px - r["stop"]) if call else (r["stop"] - px)
            rr = ((r["target"] - px) if call else (px - r["target"])) / risk
            if rr < MIN_RR:
                continue
            r.update(price=px, rr=rr, rr_pts=float(_pts(np.array([rr]), [(3, 15), (2.5, 12), (2, 10), (1.5, 5)])[0]),
                     dollar_vol=float(d.dollar_vol.iloc[-1]), sector=secmap.get(tk, "n/a"))
            r["stock_score"] = round(sum(r[k] for k in ("regime", "trend", "rs", "setup_pts", "volume", "momentum", "rr_pts")), 1)
            cands.append(r)
    bt = load("swing_backtest.json", {})
    model = bt.get("model")
    for r in cands:
        r["pred_r"] = float(_predict(model, [[r[f] for f in FEATURES]])[0]) if model else None
    cands.sort(key=lambda r: (r["stock_score"], r.get("pred_r") or 0), reverse=True)
    qualified = [r for r in cands if r["stock_score"] >= MIN_STOCK]
    log(f"[swing] {len(cands)} setups on {sig_date} bars, {len(qualified)} with stock score {MIN_STOCK}+")

    con = db()
    picks, rejected = [], []
    used = set()
    for r in qualified[:CHAIN_CANDIDATES]:
        key = (r["sector"], r["direction"])
        if r["sector"] != "n/a" and key in used:
            rejected.append({**_brief(r), "why": f"already have a {r['direction']} in {r['sector']} (correlated)"})
            continue
        ed = next_earnings(r["ticker"])
        r["earnings"] = ed.isoformat() if ed else None
        if ed and (ed - t.date()).days <= MAX_HOLD * 7 / 5 + 1:
            rejected.append({**_brief(r), "why": f"EARNINGS RISK: reports {ed:%b %d}, inside the holding period"})
            continue
        opt, why = choose_option(r, con)
        if not opt:
            rejected.append({**_brief(r), "why": why})
            continue
        ev = _ev(r["setup"], opt)
        if ev is not None and ev <= 0:
            rejected.append({**_brief(r), "why": f"negative expected value: ${ev:+.0f} per contract using backtest "
                                                 f"win/loss/timeout rates for this setup"})
            continue
        if ed and ed <= pd.Timestamp(opt["expiry"]).date():
            r["earnings_note"] = f"Earnings {ed:%b %d} is after the planned exit but before expiry: exit before it"
        r["option"] = opt
        r["_regime"] = f"{lean}: {reg_text}"
        r["final"] = opt["final"]
        picks.append(r)
        if r["sector"] != "n/a":
            used.add(key)
    picks.sort(key=lambda r: (r["final"], r.get("pred_r") or 0), reverse=True)
    picks = picks[:3]

    report = {"time": t.isoformat(), "date": t.strftime("%Y-%m-%d"), "signal_date": sig_date, "lean": lean,
              "regime_text": reg_text, "reg_call": float(last.reg_call), "reg_put": float(last.reg_put),
              "sectors": sector_table, "n_setups": len(cands), "n_qualified": len(qualified),
              "account": ACCOUNT, "market_open": market_open,
              "picks": [_card(r, bt) for r in picks], "rejected": rejected[:8],
              "watch": [_brief(r) for r in cands[:10]],
              "model": {k: model[k] for k in ("oos_corr", "top_third_avg_r", "bottom_third_avg_r", "n_test", "test_from")} if model else None,
              "backtest_built": bt.get("built_at")}
    report["text"] = format_report(report)
    save("swing.json", report)
    _log_picks(con, report, picks, qualified, rejected, sig_date)
    con.close()
    log(f"[swing] {len(picks)} trade(s) ({time.time() - t0:.0f}s)" + ("" if picks else ": NO TRADE"))
    if push:
        if picks:
            p = report["picks"][0]
            notify(f"Swing: {p['Ticker']} {p['Direction']} · final {p['Final score']}",
                   "\n".join(f"{k}: {p[k]}" for k in ("Setup", "Option", "Entry", "Stop", "Target", "Option price",
                                                       "Maximum loss", "Position size", "Historical setup win rate"))
                   + (f"\n+{len(picks) - 1} more on the desk page" if len(picks) > 1 else ""), tags="chart_with_upwards_trend")
        else:
            notify("Swing scan: NO TRADE", f"{len(qualified)} stock setups scored {MIN_STOCK}+; none passed the option, "
                   f"earnings and budget rules. Regime: {lean}.", priority="low", tags="zzz")
    return report


def _ev(setup, o, contracts=1):
    """Option expected value ($) from the backtest's win/loss/timeout mix, or None if not validated."""
    st, _ = bt_stats(setup)
    if st.get("n", 0) < MIN_SAMPLE:
        return None
    pw, pl, pt = st["win_rate"] / 100, st["loss_rate"] / 100, st["timeout_rate"] / 100
    return 100 * contracts * (pw * (o["v_target"] - o["mid"]) + pl * (o["v_stop"] - o["mid"]) + pt * (o["v_flat"] - o["mid"]))


def _brief(r):
    return {"ticker": r["ticker"], "direction": r["direction"], "setup": SETUP_NAME[r["setup"]],
            "stock_score": r["stock_score"], "price": round(r["price"], 2), "stop": round(r["stop"], 2),
            "target": round(r["target"], 2), "rr": round(r["rr"], 2), "pred_r": None if r.get("pred_r") is None else round(r["pred_r"], 2)}


def _win_text(setup, bt):
    st = bt.get("by_setup", {}).get(setup, {}).get("score80", {"n": 0})
    if not bt:
        return "Win rate not validated (backtest not built yet)", "none", st
    if st.get("n", 0) < MIN_SAMPLE:
        return (f"Win rate not validated (only {st.get('n', 0)} past {SETUP_NAME[setup].lower()} signals "
                f"with stock score 80+)"), f"{st.get('n', 0)} observations", st
    return (f"This setup historically showed {st['win_rate']:.0f}% wins (target before stop within {MAX_HOLD} "
            f"trading days, stock score 80+) across {st['n']} observations"), \
        f"{st['n']} signals, {st['first'][:4]}-{st['last'][:4]}, {bt.get('n_stocks', '?')} current stocks (survivorship bias)", st


def _card(r, bt):
    o = r["option"]
    call = r["direction"] == "call"
    lg = o["legs"][0][1]
    exp = pd.Timestamp(o["expiry"])
    if o["kind"] == "single":
        contract = f"{r['ticker']} {exp:%b %d %Y} ${lg['strike']:g} {'call' if call else 'put'}"
        strike = f"${lg['strike']:g}"
    else:
        sh = o["legs"][1][1]
        contract = (f"{r['ticker']} {exp:%b %d %Y} ${lg['strike']:g}/${sh['strike']:g} "
                    f"{'call' if call else 'put'} debit spread")
        strike = f"buy ${lg['strike']:g} / sell ${sh['strike']:g}"
    win_txt, sample, st = _win_text(r["setup"], bt)
    ev = _ev(r["setup"], o, o["contracts"])
    stock_ev = st.get("avg_r")
    validated = st.get("n", 0) >= MIN_SAMPLE
    conf = ("High" if validated and st["avg_r"] > 0.2 and st["win_rate"] >= 50 and r["final"] >= 88 and (ev or 0) > 0
            else "Medium" if validated and st["avg_r"] > 0 and (ev or 0) > 0
            else "Low (expected value negative in backtest)" if validated else "Low (win rate not validated)")
    iv = o["iv"]
    ivi = o.get("ivinfo", {})
    ivp = (f"{ivi['pct']}th percentile / rank {ivi['rank']} (1-yr)" if ivi.get("rank") is not None
           else f"n/a: building history ({ivi.get('days', 0)} of 60 days); IV/HV20 = {iv / r['hv20']:.2f}" if r["hv20"] else "n/a")
    risk = abs(r["price"] - r["stop"])
    mid = o["mid"]
    lim_lo, lim_hi = round(mid - 0.01, 2), round(mid + 0.01, 2)
    loss = o["cost"] * o["contracts"]
    exp_move = r["price"] * iv * math.sqrt(o["dte"] / 365)
    model = bt.get("model")
    pred = (f"{r['pred_r']:+.2f}R (ridge regression; out-of-sample corr {model['oos_corr']:+.2f}"
            f"{', weak: treat as a tiebreaker only' if model and abs(model['oos_corr']) < 0.1 else ''})") if r.get("pred_r") is not None else "n/a"
    reg = (f"50-day regression slope {100 * r['lr_slope_ann']:+.0f}%/yr, R² {r['lr_r2']:.2f}, "
           f"price {abs(r['lr_z']):.1f} SD {'above' if r['lr_z'] > 0 else 'below'} the line")
    sec = f"{r['sector']} (#{int(r['sector_rank'])} of 11 by 3-month return)" if r["sector_rank"] == r["sector_rank"] else r["sector"]
    invalid = (f"Daily close {'below' if call else 'above'} ${r['stop']:.2f} (the stock stop), or no progress after "
               f"{MAX_HOLD} trading days")
    not_take = [f"Stock opens beyond ${r['price'] + (0.5 if call else -0.5) * r['atr']:.2f} (chasing)",
                f"Bid/ask wider than {PREF_SPREAD:.0%} when you place the order",
                "SPY breaks its 50-day average " + ("down" if call else "up") + " before entry",
                f"Fill above ${lim_hi:.2f}"]
    if r.get("earnings_note"):
        not_take.append(r["earnings_note"])
    if not r.get("earnings"):
        not_take.append("Earnings date not found: check it yourself first")
    return {
        "Ticker": r["ticker"], "Direction": "Bullish (calls)" if call else "Bearish (puts)",
        "Setup": SETUP_NAME[r["setup"]], "Stock price": f"${r['price']:.2f}", "Option": contract,
        "Expiration": f"{exp:%b %d, %Y}", "Strike": strike, "DTE": o["dte"], "Delta": f"{o['delta']:+.2f}",
        "Entry": f"Stock ~${r['price']:.2f}; limit {lim_lo:.2f}-{lim_hi:.2f} (mid {mid:.2f})",
        "Stop": f"${r['stop']:.2f} on the STOCK (Fidelity Contingent order keyed to the stock price)",
        "Target": f"${r['target']:.2f}; take partial profits around +50%, more at +75-100%, trail the rest",
        "Option price": f"${mid:.2f} mid (${o['cost']:.0f} per contract)",
        "Maximum loss": f"${loss:.0f} ({100 * loss / ACCOUNT:.1f}% of ${ACCOUNT:.0f})",
        "Position size": f"{o['contracts']} contract(s) · {o['tier']} setup, ${o['budget']:.0f} risk budget",
        "Stock R/R": f"{r['rr']:.1f}:1 (risk ${risk:.2f}, reward ${abs(r['target'] - r['price']):.2f})",
        "Estimated option R/R": f"{o['opt_rr']:.1f}:1",
        "IV": f"{100 * iv:.0f}% (HV20 {100 * r['hv20']:.0f}%)", "IV percentile": ivp,
        "Open interest": f"{o['oi']:,}", "Volume": f"{o['volume']:,}",
        "Bid/ask": f"{o['bid']:.2f} / {o['ask']:.2f} ({100 * o['spread_pct']:.1f}% of mid)",
        "Market regime": r.get("_regime", ""), "Sector strength": sec,
        "Historical setup win rate": win_txt, "Backtest sample": sample,
        "Technical score": f"{r['stock_score']:.0f}/100", "Option score": f"{o['option_score']:.0f}/100",
        "Final score": f"{r['final']:.0f}/100",
        "Why this trade": (f"{SETUP_NAME[r['setup']]}; 3-month return "
                           f"{'stronger' if call else 'weaker'} than {100 * (r['rs_pct'] if call else 1 - r['rs_pct']):.0f}% "
                           f"of the universe; {reg}; volume {r['rvol']:.1f}x average; {r['sector']} sector."),
        "What invalidates the trade": invalid,
        "What would make you NOT take the trade": "; ".join(not_take),
        "_details": {
            "Mid": f"{mid:.2f}", "Expected move to expiry": f"±${exp_move:.2f} ({100 * exp_move / r['price']:.1f}%)",
            "Option cost": f"${o['cost']:.0f} per contract",
            "Estimated option value at stock target": f"${o['v_target']:.2f} (after ~{(st.get('days_to_win') or 6):.0f} trading days, same IV)",
            "Estimated option value at stock stop": f"${o['v_stop']:.2f}",
            "Potential percentage return": f"{100 * o['exp_return']:+.0f}%",
            "Expected value": (f"${ev:+.0f} per position (option, from backtest win/loss/timeout rates); stock-level "
                               f"{stock_ev:+.2f}R per trade") if ev is not None else "Not computed: win rate not validated",
            "Regression-predicted R": pred, "Regression channel": reg, "Confidence": conf,
            "Score breakdown": (f"regime {r['regime']:.0f}/15 · trend {r['trend']:.0f}/15 · RS {r['rs']:.0f}/15 · "
                                f"setup {r['setup_pts']:.0f}/20 · volume {r['volume']:.0f}/10 · momentum {r['momentum']:.0f}/10 · "
                                f"R/R {r['rr_pts']:.0f}/15 | option: " + " · ".join(f"{k} {v:.0f}" for k, v in o["parts"].items())),
            "Earnings": r.get("earnings") or "not found",
        },
        "_raw": {"ticker": r["ticker"], "direction": r["direction"], "setup": r["setup"], "entry": r["price"],
                 "stop": r["stop"], "target": r["target"], "rr": r["rr"], "stock_score": r["stock_score"],
                 "option_score": o["option_score"], "final": r["final"], "tier": o["tier"], "kind": o["kind"],
                 "expiry": o["expiry"], "legs": [(s, l["strike"]) for s, l in o["legs"]], "mid": mid,
                 "contracts": o["contracts"], "max_loss": loss, "v_target": o["v_target"],
                 "hist_win": st.get("win_rate"), "hist_n": st.get("n", 0), "pred_r": r.get("pred_r"),
                 "earnings": r.get("earnings")},
    }


ORDER = ["Ticker", "Direction", "Setup", "Stock price", "Option", "Expiration", "Strike", "DTE", "Delta", "Entry",
         "Stop", "Target", "Option price", "Maximum loss", "Position size", "Stock R/R", "Estimated option R/R", "IV",
         "IV percentile", "Open interest", "Volume", "Bid/ask", "Market regime", "Sector strength",
         "Historical setup win rate", "Backtest sample", "Technical score", "Option score", "Final score",
         "Why this trade", "What invalidates the trade", "What would make you NOT take the trade"]
MEDALS = ["🥇 TOP TRADE", "🥈 SECOND TRADE", "🥉 THIRD TRADE"]


def format_report(rep):
    lines = [f"SWING SCAN {rep['time'][:16].replace('T', ' ')} ET · bars through {rep['signal_date']} · account ${rep['account']:.0f}",
             f"Market: {rep['lean']} ({rep['regime_text']})", ""]
    if not rep["picks"]:
        lines += ["NO TRADE", f"{rep['n_setups']} daily setups found, {rep['n_qualified']} with stock score {MIN_STOCK}+."]
        for r in rep["rejected"]:
            lines.append(f"  {r['ticker']} {r['setup']} ({r['stock_score']:.0f}): {r['why']}")
        return "\n".join(lines)
    for medal, p in zip(MEDALS, rep["picks"]):
        lines.append(medal)
        lines += [f"{k}: {p[k]}" for k in ORDER]
        lines += [f"{k}: {v}" for k, v in p["_details"].items()]
        lines.append("")
    lines.append("This system never says a trade will win. Confirm on your own chart before entering. Not investment advice.")
    return "\n".join(lines)


def _log_picks(con, rep, picks, qualified, rejected, sig_date):
    t = now()
    open_picks = {(r[0], r[1]) for r in con.execute("SELECT ticker, direction FROM picks WHERE status='open' AND kind='pick'")}
    open_keys = {(r[0], r[1]) for r in con.execute("SELECT ticker, direction FROM picks WHERE status='open'")}
    for rank, p in enumerate(rep["picks"], 1):
        r = p["_raw"]
        if (r["ticker"], r["direction"]) in open_picks:
            continue
        con.execute("""INSERT INTO picks (date, ts, signal_date, ticker, direction, setup, kind, rank, entry, stop, target, rr,
            stock_score, option_score, final_score, tier, contract, opt_entry, contracts, max_loss, est_value_target,
            hist_win, hist_n, pred_r, earnings, regime, last_px)
            VALUES (?,?,?,?,?,?,'pick',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rep["date"], t.isoformat(), sig_date, r["ticker"], r["direction"], r["setup"], rank, r["entry"],
                     r["stop"], r["target"], r["rr"], r["stock_score"], r["option_score"], r["final"], r["tier"],
                     json.dumps({"kind": r["kind"], "expiry": r["expiry"], "legs": r["legs"]}), r["mid"], r["contracts"],
                     r["max_loss"], r["v_target"], r["hist_win"], r["hist_n"], r["pred_r"], r["earnings"], rep["lean"], r["entry"]))
        open_keys.add((r["ticker"], r["direction"]))
    picked = {p["_raw"]["ticker"] for p in rep["picks"]}
    why = {x["ticker"]: x["why"] for x in rejected}
    for r in qualified:                       # near misses: tracked at the stock level only
        if r["ticker"] in picked or (r["ticker"], r["direction"]) in open_keys:
            continue
        con.execute("""INSERT INTO picks (date, ts, signal_date, ticker, direction, setup, kind, rank, entry, stop, target, rr,
            stock_score, pred_r, earnings, regime, note, last_px) VALUES (?,?,?,?,?,?,'watch',0,?,?,?,?,?,?,?,?,?,?)""",
                    (rep["date"], t.isoformat(), sig_date, r["ticker"], r["direction"], r["setup"], r["price"], r["stop"],
                     r["target"], r["rr"], r["stock_score"], r.get("pred_r"), r.get("earnings"), rep["lean"],
                     why.get(r["ticker"], "not checked (outside top candidates)"), r["price"]))
        open_keys.add((r["ticker"], r["direction"]))
    con.commit()


# ------------------------------- tracking -------------------------------
def _mark_option(row):
    """Current mid of the tracked structure (falls back to last trade price)."""
    try:
        c = json.loads(row["contract"])
        with YF_LOCK:
            ch = yf.Ticker(row["ticker"]).option_chain(c["expiry"])
        df = ch.calls if row["direction"] == "call" else ch.puts
        val = 0.0
        for sign, k in c["legs"]:
            q = df[df.strike == k]
            if q.empty:
                return None
            q = q.iloc[0]
            b, a = _num(q.bid), _num(q.ask)
            val += sign * ((b + a) / 2 if b > 0 and a > 0 else _num(q.lastPrice))
        return round(max(val, 0), 2)
    except Exception:
        return None


def update(push=True):
    con = db()
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM picks WHERE status='open'")]
    if not rows:
        con.close()
        return 0
    frames = _download(sorted({r["ticker"] for r in rows}), "3mo")
    t = now()
    closed = 0
    for r in rows:
        f = frames.get(r["ticker"])
        if f is None:
            continue
        call = r["direction"] == "call"
        bars = f[f.index > pd.Timestamp(r["date"])]           # starts the session after the pick
        risk = (r["entry"] - r["stop"]) if call else (r["stop"] - r["entry"])
        outcome, rm, j = None, None, 0
        for j, (_, b) in enumerate(bars.iterrows(), 1):
            if call:
                if b.Open <= r["stop"]:
                    outcome, rm = "loss", (b.Open - r["entry"]) / risk
                elif b.Low <= r["stop"]:
                    outcome, rm = "loss", -1.0
                elif b.High >= r["target"]:
                    outcome, rm = "win", (r["target"] - r["entry"]) / risk
            else:
                if b.Open >= r["stop"]:
                    outcome, rm = "loss", (r["entry"] - b.Open) / risk
                elif b.High >= r["stop"]:
                    outcome, rm = "loss", -1.0
                elif b.Low <= r["target"]:
                    outcome, rm = "win", (r["entry"] - r["target"]) / risk
            if outcome:
                break
            if j >= MAX_HOLD:
                outcome = "timeout"
                rm = ((b.Close - r["entry"]) if call else (r["entry"] - b.Close)) / risk
                break
        held = j if outcome else min(len(bars), MAX_HOLD)
        px = float(f.Close.iloc[-1])
        if not outcome and r.get("earnings"):
            try:
                if (pd.Timestamp(r["earnings"]).date() - t.date()).days <= 1:
                    outcome, rm = "earnings_exit", ((px - r["entry"]) if call else (r["entry"] - px)) / risk
            except Exception:
                pass
        opt = _mark_option(r) if r["kind"] == "pick" else None
        con.execute("UPDATE picks SET last_px=?, last_opt=COALESCE(?, last_opt), last_update=?, days_held=? WHERE id=?",
                    (px, opt, t.isoformat(), held, r["id"]))
        if outcome:
            oret = (opt / r["opt_entry"] - 1) if (opt is not None and r["opt_entry"]) else None
            con.execute("UPDATE picks SET status='closed', outcome=?, r_multiple=?, opt_exit=?, opt_return=?, closed_date=? WHERE id=?",
                        (outcome, round(rm, 2), opt, None if oret is None else round(oret, 3), t.strftime("%Y-%m-%d"), r["id"]))
            closed += 1
            if push and r["kind"] == "pick":
                notify(f"Swing {'EXIT' if r['taken'] else 'paper'}: {r['ticker']} {outcome}",
                       f"{SETUP_NAME[r['setup']]} {r['direction']}: stock {rm:+.1f}R"
                       + (f", option {100 * oret:+.0f}% (mark ${opt:.2f} vs ${r['opt_entry']:.2f})" if oret is not None else "")
                       + ("\nYou hold this one: close it." if r["taken"] and not r["exit_fill"] else ""),
                       priority="high" if r["taken"] else "default", tags="bell")
    con.commit()
    con.close()
    log(f"[swing] tracker: {len(rows)} open, {closed} closed")
    return closed


def record():
    """Track record for the dashboard: paper picks (options), near-misses (stock only), real trades."""
    con = db()
    df = pd.read_sql("SELECT * FROM picks", con)
    con.close()
    df = df.astype(object).where(df.notna(), None)
    for col in ("r_multiple", "opt_return", "opt_exit", "opt_entry", "contracts", "fill", "exit_fill", "fill_contracts"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        return {"open": [], "picks": {"n": 0}, "watch": {"n": 0}, "taken": {"n": 0}, "recent": []}

    def summ(x):
        c = x[x.status == "closed"]
        if c.empty:
            return {"n": 0, "open": int((x.status == "open").sum())}
        out = {"n": int(len(c)), "open": int((x.status == "open").sum()),
               "win_rate": round(100 * float((c.outcome == "win").mean())), "avg_r": round(float(c.r_multiple.mean()), 2)}
        o = c.dropna(subset=["opt_return"])
        if len(o):
            out["avg_opt"] = round(100 * float(o.opt_return.mean()))
            out["paper_pnl"] = round(float(((o.opt_exit - o.opt_entry) * 100 * o.contracts).sum()))
        return out
    taken = df[df.taken == 1]
    real = taken.dropna(subset=["exit_fill"])
    t_sum = {"n": int(len(real)), "open": int(len(taken) - len(real))}
    if len(real):
        pnl = (real.exit_fill - real.fill) * 100 * real.fill_contracts
        t_sum.update(pnl=round(float(pnl.sum())), win_rate=round(100 * float((pnl > 0).mean())))
    open_rows = df[(df.status == "open") & (df.kind == "pick")].sort_values("date", ascending=False)
    recent = df[(df.status == "closed") & (df.kind == "pick")].sort_values("closed_date", ascending=False).head(10)
    return {"picks": summ(df[df.kind == "pick"]), "watch": summ(df[df.kind == "watch"]), "taken": t_sum,
            "by_setup": [{"setup": SETUP_NAME.get(k, k), **summ(g)} for k, g in df.groupby("setup")],
            "open": open_rows.to_dict("records"), "recent": recent.to_dict("records")}


# ------------------------------- dashboard -------------------------------
def render_html(e):
    """HTML block for desk.html. `e` is html.escape."""
    rep = load("swing.json", {})
    bt = load("swing_backtest.json", {})
    try:
        rec = record()
    except Exception:
        rec = {"open": [], "picks": {"n": 0}, "watch": {"n": 0}, "taken": {"n": 0}, "recent": []}
    out = ["<h2>Swing trades · 30-60 DTE options</h2>"]
    if not rep:
        out.append(f"<div class=empty>The swing scan runs once a day at {SCAN_TIME // 100}:{SCAN_TIME % 100:02d} a.m. ET "
                   "(the first run also builds the 5-year backtest, which takes a few minutes).</div>")
    else:
        when = datetime.fromisoformat(rep["time"]).strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
        out.append(f"<p class=meta>Scanned {e(when)} on daily bars through {e(rep['signal_date'])} · account "
                   f"${rep['account']:.0f} · {rep['n_setups']} setups, {rep['n_qualified']} with stock score {MIN_STOCK}+<br>"
                   f"Market: <b>{e(rep['lean'])}</b> · {e(rep['regime_text'])}</p>")
        if rep["picks"]:
            for medal, p in zip(MEDALS, rep["picks"]):
                d = "call" if p["_raw"]["direction"] == "call" else "put"
                kv = "".join(f"<dt>{e(k)}</dt><dd>{e(str(p[k]))}</dd>" for k in ORDER) + \
                     "".join(f"<dt>{e(k)}</dt><dd>{e(str(v))}</dd>" for k, v in p["_details"].items())
                out.append(f"""<article class='setup {d}'><span class=score>{e(p['Final score'])}</span>
<h3>{medal} · {e(p['Ticker'])} · {e(p['Setup'])}</h3>
<div class=lv><span>{e(p['Option'])}</span></div>
<div class=lv><span>Stock <b>{e(p['Stock price'])}</b></span><span>Stop <b>{p['_raw']['stop']:.2f}</b></span>
<span>Target <b>{p['_raw']['target']:.2f}</b></span><span>Limit <b>{p['_raw']['mid']:.2f}</b></span>
<span>Max loss <b>{e(p['Maximum loss'])}</b></span></div>
<p class=meta>{e(p['Historical setup win rate'])}</p>
<details><summary class=meta style='cursor:pointer'>Full trade plan</summary><dl class=kv>{kv}</dl></details></article>""")
        else:
            rej = "".join(f"<li><b>{e(r['ticker'])}</b> {e(r['setup'])} ({r['stock_score']:.0f}): {e(r['why'])}</li>"
                          for r in rep["rejected"])
            out.append(f"<div class=empty><b>NO TRADE.</b> Nothing passed every rule today; waiting is a position."
                       f"{'<ul style=margin-top:8px>' + rej + '</ul>' if rej else ''}</div>")
        if rep.get("sectors"):
            secs = rep["sectors"]
            out.append("<p class=meta>Sectors by 3-month return: strongest " +
                       ", ".join(f"{e(x['sector'])} {x['ret63']:+.0f}%" for x in secs[:3]) + " · weakest " +
                       ", ".join(f"{e(x['sector'])} {x['ret63']:+.0f}%" for x in secs[-3:]) + "</p>")
    # open paper picks
    if rec["open"]:
        rows = ""
        for r in rec["open"]:
            basis = r.get("fill") if r.get("taken") and r.get("fill") else r.get("opt_entry")
            pnl = f"{100 * (r['last_opt'] / basis - 1):+.0f}%" if r.get("last_opt") and basis else "-"
            mine = " <span class=pill>yours</span>" if r.get("taken") else ""
            rows += (f"<tr><th>#{r['id']} {e(r['ticker'])}{mine}<span class=sub>{e(SETUP_NAME.get(r['setup'], r['setup']))} "
                     f"· {e(r['date'])}</span></th><td>{r['entry']:.2f} → {(r['last_px'] or r['entry']):.2f}"
                     f"<span class=sub>stop {r['stop']:.2f} / tgt {r['target']:.2f}</span></td>"
                     f"<td>{r['days_held'] or 0}/{MAX_HOLD}d</td><td>{pnl}</td></tr>")
        out.append(f"<h2>Open swing picks</h2><table><tr><th>Pick</th><td>stock</td><td>held</td><td>option</td></tr>{rows}</table>")
    # track record
    def line(name, x, opt=False):
        if not x.get("n"):
            n_open = x.get("open", 0)
            note = f" ({n_open} open)" if n_open else ""
            return f"<tr><th>{name}</th><td colspan=4 class=l>none closed yet{note}</td></tr>"
        extra = (f"{x['avg_opt']:+d}% opt · ${x['paper_pnl']:+,}" if opt and "avg_opt" in x else "")
        return (f"<tr><th>{name}</th><td>{x['n']}</td><td>{x['win_rate']}%</td>"
                f"<td class={'up' if x['avg_r'] > 0 else 'dn'}>{x['avg_r']:+.2f}R</td><td>{extra}</td></tr>")
    t = rec["taken"]
    taken = (f"<tr><th>Your real trades</th><td>{t['n']}</td><td>{t.get('win_rate', '-')}{'%' if t.get('n') else ''}</td>"
             f"<td></td><td>${t.get('pnl', 0):+,}</td></tr>") if t.get("n") or t.get("open") else ""
    out.append("<h2>Swing track record (live)</h2><table><tr><th></th><td>n</td><td>win</td><td>avg</td><td></td></tr>"
               + line("Paper picks", rec["picks"], True) + line("Near-misses (stock only)", rec["watch"]) + taken + "</table>"
               + "<p class=meta>Win = stock target before stop within 15 trading days. Trust nothing under 30 closed picks.</p>")
    # backtest
    if bt.get("by_setup"):
        rows = ""
        for k, v in bt["by_setup"].items():
            s80 = v["score80"]
            if not s80.get("n"):
                continue
            tag = "" if s80["n"] >= MIN_SAMPLE else " <span class=sub>not validated (&lt;30)</span>"
            rows += (f"<tr><th>{e(SETUP_NAME[k])}{tag}</th><td>{s80['n']}</td><td>{s80['win_rate']:.0f}%</td>"
                     f"<td class={'up' if s80['avg_r'] > 0 else 'dn'}>{s80['avg_r']:+.2f}R</td></tr>")
        m = bt.get("model")
        mtxt = (f"Regression model (ridge on {len(FEATURES)} features incl. the 50-day regression channel): tested on "
                f"{m['n_test']} signals since {m['test_from']} it never saw, correlation {m['oos_corr']:+.2f}; "
                f"top third of predictions averaged {m['top_third_avg_r']:+.2f}R vs {m['bottom_third_avg_r']:+.2f}R for the bottom third."
                ) if m else "Regression model: not enough data."
        built = datetime.fromisoformat(bt["built_at"]).strftime("%b %d") if bt.get("built_at") else "?"
        out.append(f"<h2>Backtest · stock score 80+ · {bt.get('years')} yrs</h2>"
                   f"<table><tr><th>Setup</th><td>n</td><td>win</td><td>avg</td></tr>{rows}</table>"
                   f"<p class=meta>Stock-level, target vs stop within {MAX_HOLD} days, stop checked first on the same bar. "
                   f"Current index members only (survivorship bias flatters longs). Built {built}, refreshed weekly. {e(mtxt)}</p>")
    return "".join(out)


# ------------------------------- jobs -------------------------------
def daily(force_backtest=False, push=True, run_backtest=True):
    """Backtest if stale, then scan, then update the tracker. Safe to call from a thread."""
    if not _job_lock.acquire(blocking=False):
        return
    try:
        bt = load("swing_backtest.json", {})
        if run_backtest and (force_backtest or time.time() - bt.get("built", 0) > 7 * 86400):
            backtest()
        update(push=push)
        scan(push=push)
    except Exception as e:
        log(f"[swing] job failed: {e!r}")
    finally:
        _job_lock.release()


def start(fn, *a, **k):
    th = threading.Thread(target=fn, args=a, kwargs=k, daemon=True, name="swing")
    th.start()
    return th


def _cli():
    import common
    args = sys.argv[1:] or ["scan"]
    cmd = args[0]
    if cmd == "scan":
        common.DRY_RUN = True
        rep = scan(push=False)
        print("\n" + (rep or {}).get("text", "scan failed"))
    elif cmd == "backtest":
        res = backtest()
        if res:
            print(json.dumps({k: v["score80"] for k, v in res["by_setup"].items()}, indent=1))
            print(json.dumps({k: v for k, v in (res["model"] or {}).items() if k not in ("mu", "sd", "coef")}, indent=1))
    elif cmd == "update":
        common.DRY_RUN = True
        update(push=False)
    elif cmd == "report":
        print(json.dumps({k: v for k, v in record().items() if k not in ("open", "recent")}, indent=1, default=str))
    elif cmd == "take" and len(args) >= 3:
        con = db()
        n = int(args[3]) if len(args) > 3 else None
        con.execute("UPDATE picks SET taken=1, fill=?, fill_contracts=COALESCE(?, contracts) WHERE id=?", (float(args[2]), n, int(args[1])))
        con.commit()
        print(f"Recorded: you bought pick {args[1]} at {args[2]}")
    elif cmd == "close" and len(args) >= 3:
        con = db()
        con.execute("UPDATE picks SET exit_fill=?, exit_date=? WHERE id=?", (float(args[2]), now().strftime("%Y-%m-%d"), int(args[1])))
        con.commit()
        print(f"Recorded: you sold pick {args[1]} at {args[2]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _cli()
