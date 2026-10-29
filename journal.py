"""
Scorekeeping. Every setup the scanner finds is logged here, alerted or not. After the close,
each one is graded against what price actually did next:

  win      first target reached before the stop           (+reward/risk R)
  loss     stop reached first (or both in the same minute) (-1 R)
  timeout  neither by the close; scored at the closing price

Over a few weeks this shows which setups, times of day and market regimes actually work.
"""
import json, sqlite3, time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from common import path, YF_LOCK, ET, now, log

DB = sqlite3.connect(path("journal.db"), check_same_thread=False)
DB.execute("""CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY, date TEXT, ts TEXT, ticker TEXT, setup TEXT, direction TEXT,
  price REAL, trigger REAL, stop REAL, t1 REAL, t2 REAL, rr REAL, score INTEGER, alerted INTEGER,
  regime TEXT, bias TEXT, catalyst TEXT, rvol REAL, why TEXT, contract TEXT,
  outcome TEXT, r_multiple REAL, hit_t2 INTEGER, graded_ts TEXT)""")
DB.commit()
DEDUPE_MIN = 45


def is_duplicate(s):
    since = (now() - timedelta(minutes=DEDUPE_MIN)).isoformat()
    return DB.execute("SELECT 1 FROM signals WHERE ticker=? AND setup=? AND direction=? AND ts>=?",
                      (s["ticker"], s["setup"], s["direction"], since)).fetchone() is not None


def record(s, regime, bias, alerted):
    t = now()
    DB.execute("""INSERT INTO signals (date, ts, ticker, setup, direction, price, trigger, stop, t1, t2, rr,
                  score, alerted, regime, bias, catalyst, rvol, why, contract)
                  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
               (t.strftime("%Y-%m-%d"), t.isoformat(), s["ticker"], s["setup"], s["direction"], s["price"],
                s["trigger"], s["stop"], s["t1"], s["t2"], s.get("rr"), s.get("score"), int(alerted),
                regime, bias, s.get("catalyst") or "", s.get("rvol_day"), s["why"],
                json.dumps(s.get("contract")) if s.get("contract") else ""))
    DB.commit()


def today_signals():
    cur = DB.execute("SELECT * FROM signals WHERE date=? ORDER BY ts DESC", (now().strftime("%Y-%m-%d"),))
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def grade(date=None):
    """Grade every ungraded signal from `date` (default today). Needs 1-minute data (last 7 days)."""
    date = date or now().strftime("%Y-%m-%d")
    rows = DB.execute("SELECT id, ts, ticker, direction, trigger, stop, t1, t2 FROM signals "
                      "WHERE date=? AND outcome IS NULL", (date,)).fetchall()
    if not rows:
        return 0
    tickers = sorted({r[2] for r in rows})
    with YF_LOCK:
        d = yf.download(tickers, period="7d", interval="1m", progress=False, group_by="ticker", threads=True)
    graded = 0
    for sid, ts, tk, direction, trig, stop, t1, t2 in rows:
        try:
            df = (d[tk] if len(tickers) > 1 else d).dropna()
            df.index = df.index.tz_convert(ET)
            start = datetime.fromisoformat(ts)
            df = df[(df.index > start) & (df.index.date == start.date())]
            if df.empty:
                continue
            risk = abs(trig - stop)
            outcome, r_mult, hit2 = None, None, 0
            for _, b in df.iterrows():
                if direction == "call":
                    hit_stop, hit_t1 = b.Low <= stop, b.High >= t1
                else:
                    hit_stop, hit_t1 = b.High >= stop, b.Low <= t1
                if hit_stop:                                   # same-minute tie counts as a loss
                    outcome, r_mult = "loss", -1.0
                    break
                if hit_t1:
                    outcome, r_mult = "win", abs(t1 - trig) / risk
                    after = df[df.index >= b.name]
                    hit2 = int((after.High >= t2).any() if direction == "call" else (after.Low <= t2).any())
                    break
            if outcome is None:
                close = float(df.Close.iloc[-1])
                outcome = "timeout"
                r_mult = ((close - trig) if direction == "call" else (trig - close)) / risk
            DB.execute("UPDATE signals SET outcome=?, r_multiple=?, hit_t2=?, graded_ts=? WHERE id=?",
                       (outcome, round(r_mult, 2), hit2, now().isoformat(), sid))
            graded += 1
        except Exception as e:
            log(f"[journal] grading {tk} failed: {e}")
    DB.commit()
    log(f"[journal] graded {graded} signals for {date}")
    return graded


def grade_pending():
    """Catch up on any past days that were never graded (e.g. the app was closed at 4 p.m.)."""
    dates = [r[0] for r in DB.execute("SELECT DISTINCT date FROM signals WHERE outcome IS NULL").fetchall()]
    today = now().strftime("%Y-%m-%d")
    for dt in dates:
        if dt < today or now().hour * 100 + now().minute >= 1605:
            grade(dt)


def _bucket(ts):
    h = datetime.fromisoformat(ts)
    m = h.hour * 100 + h.minute
    return "9:45-10:30" if m < 1030 else "10:30-11:30" if m < 1130 else "11:30-1:30" if m < 1330 else "1:30-3:30"


def scorecard():
    df = pd.read_sql("SELECT * FROM signals WHERE outcome IS NOT NULL", DB)
    if df.empty:
        return {"n": 0}
    df["bucket"] = df.ts.map(_bucket)
    df["win"] = (df.outcome == "win").astype(int)

    def agg(col):
        g = df.groupby(col).agg(n=("win", "size"), win_rate=("win", "mean"), avg_r=("r_multiple", "mean"))
        g["win_rate"] = (100 * g.win_rate).round(0)
        g["avg_r"] = g.avg_r.round(2)
        return g.sort_values("avg_r", ascending=False).reset_index().to_dict("records")

    alerted = df[df.alerted == 1]
    return {"n": int(len(df)), "win_rate": round(100 * df.win.mean()), "avg_r": round(df.r_multiple.mean(), 2),
            "alerted_n": int(len(alerted)),
            "alerted_avg_r": round(alerted.r_multiple.mean(), 2) if len(alerted) else None,
            "by_setup": agg("setup"), "by_time": agg("bucket"), "by_regime": agg("regime"),
            "by_direction": agg("direction"), "days": int(df.date.nunique())}


def day_summary(date=None):
    date = date or now().strftime("%Y-%m-%d")
    df = pd.read_sql("SELECT * FROM signals WHERE date=? AND outcome IS NOT NULL", DB, params=(date,))
    if df.empty:
        return None
    a = df[df.alerted == 1]
    return {"n": len(df), "wins": int((df.outcome == "win").sum()), "losses": int((df.outcome == "loss").sum()),
            "timeouts": int((df.outcome == "timeout").sum()), "total_r": round(df.r_multiple.sum(), 1),
            "alerted_n": len(a), "alerted_r": round(a.r_multiple.sum(), 1) if len(a) else 0.0}
