#!/usr/bin/env python3
"""
Backtest of the day-trading scanner: replays every saved day (see history.py) minute by minute
through the SAME code the live desk runs (setups.detect, setups.score), then grades each setup with
the SAME rules as the live scorecard (journal.py):

  win      target 1 before the stop                       (+reward/risk R)
  loss     stop first, or both in the same minute          (-1 R)
  timeout  neither by the close, scored at the closing price

What it can't reproduce, so you know where it's optimistic or pessimistic:
  - News catalysts and Stocktwits (no free history): catalyst is always off (-1 point vs live sometimes)
  - Option availability: assumed available (+1 point, as for most live alerts)
  - Market read: the rule-based read from SPY, IWM and sectors every 30 min (no VIX/yields/oil, no AI)
  - The watchlist is rebuilt once at 9:45 from the first 15 minutes; live it refreshes every 20 min
  - It grades the stock move, not option P/L, and assumes you got in at the trigger price

  python daytrade_bt.py              replay everything downloaded, write daytrade_backtest.html
  python daytrade_bt.py --days 20    only the most recent 20 days (quick check)
  python daytrade_bt.py --verify     prove the fast path gives identical setups to the live code
"""
import glob, html, math, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common import BASE, ET, MIN_ALERT_SCORE, MIN_RR, env, log, path
import setups

DAYS = os.path.join(path("history"), "days")
OUT_PKL = os.path.join(path("history"), "bt_signals.pkl")
OUT_HTML = os.path.join(BASE, "daytrade_backtest.html")
CACHE = os.path.join(path("history"), "bt_cache")
os.makedirs(CACHE, exist_ok=True)
CACHE_VERSION = 1                       # bump when replay logic changes so old results are redone
DEDUPE_MIN = 45
RISK_USD = env("BT_RISK_PER_TRADE", 50.0)       # dollars risked per trade for the $ view
MAX_PER_DAY = env("BT_MAX_TRADES_PER_DAY", 2)   # your rule: first N alerts of the day
OPT_DELTA = env("BT_OPTION_DELTA", 0.50)        # assumed option delta for the option estimate
OPT_COST = env("BT_OPTION_COST", 0.05)          # round-trip bid/ask cost per share of option ($0.05 = $5/contract)
BRIEF_MIN = env("BRIEF_MINUTES", 15)
SECTORS = ["XLK", "SMH", "XLY", "XLC", "XLF", "XLI", "XLE", "XLB", "XLV", "XLP", "XLU", "XLRE"]


# ------------------------------- replay one day -------------------------------
def _regime(full, levels, cut):
    """The rule-based market read (market_brief.rule_regime) from what was known at `cut`."""
    import market_brief
    snap = {}
    for tk in ["SPY", "IWM"] + SECTORS:
        df, lv = full.get(tk), levels.get(tk)
        if df is None or lv is None:
            continue
        part = df[df.index < cut]
        if part.empty:
            continue
        last = float(part.Close.iloc[-1])
        row = {"last": last, "chg": 100 * (last / lv["prev_close"] - 1), "prev": lv["prev_close"]}
        if tk == "SPY" and len(part) > 3:
            tp = (part.High + part.Low + part.Close) / 3
            row["vwap"] = float((tp * part.Volume).sum() / part.Volume.sum())
        snap[tk] = row
    return market_brief.rule_regime(snap)["label"] if snap else ""


def _grade(df, t, s):
    """Live scorecard rules: bars strictly after the scan minute, stop checked first."""
    after = df[df.index > t]
    trig, stop, t1, t2 = s["trigger"], s["stop"], s["t1"], s["t2"]
    risk = abs(trig - stop)
    call = s["direction"] == "call"
    if after.empty:
        return None
    lo, hi = after.Low.values, after.High.values
    hit_stop = lo <= stop if call else hi >= stop
    hit_t1 = hi >= t1 if call else lo <= t1
    i_stop = int(np.argmax(hit_stop)) if hit_stop.any() else 10 ** 9
    i_t1 = int(np.argmax(hit_t1)) if hit_t1.any() else 10 ** 9
    if i_stop == i_t1 == 10 ** 9:
        close = float(after.Close.iloc[-1])
        return "timeout", ((close - trig) if call else (trig - close)) / risk, 0, len(after)
    if i_stop <= i_t1:
        return "loss", -1.0, 0, i_stop + 1
    rest = after.iloc[i_t1:]
    hit2 = int((rest.High >= t2).any() if call else (rest.Low <= t2).any())
    return "win", abs(t1 - trig) / risk, hit2, i_t1 + 1


def replay_day(fname, fast=True):
    rec = pickle.load(open(fname, "rb"))
    day = rec["date"]
    levels, watch = rec["levels"], rec["watch"]
    full = {}
    for tk, df in rec["bars"].items():
        hm = df.index.hour * 100 + df.index.minute
        df = df[(hm >= 930) & (hm < int(rec["close"].replace(":", "")))].copy()   # regular session only
        if len(df) >= 16:
            full[tk] = setups.indicators(df) if fast else df
    tickers = sorted({w["ticker"] for w in watch} | {"SPY", "QQQ"})
    close_hm = int(rec["close"].replace(":", ""))
    if fast:
        orig = setups.indicators
        setups.indicators = lambda df: df          # indicators are causal: precomputed once per day
    out, last_seen, regime_cache = [], {}, {}
    try:
        start = pd.Timestamp(f"{day} 09:45", tz=ET)
        end = pd.Timestamp(f"{day} 15:55", tz=ET)
        t = start
        while t <= end and t.hour * 100 + t.minute < close_hm:
            hm = t.hour * 100 + t.minute
            slot = t.floor(f"{BRIEF_MIN}min")
            if slot not in regime_cache:
                regime_cache[slot] = _regime(full, levels, max(slot, pd.Timestamp(f"{day} 09:31", tz=ET)))
            regime = regime_cache[slot]
            spy = full.get("SPY")
            spy_now = spy[spy.index < t] if spy is not None else None
            bias = "unknown" if spy_now is None or spy_now.empty else (
                "up" if spy_now.Close.iloc[-1] > spy_now.vwap.iloc[-1] else "down") if fast else setups.market_bias({"SPY": spy_now})
            in_window = any(a <= hm <= b for a, b in setups.WINDOWS)
            for tk in tickers:
                df = full.get(tk)
                if df is None:
                    continue
                part = df[df.index < t]
                if len(part) < 16:
                    continue
                lv = levels.get(tk)
                try:
                    found = setups.detect(tk, part.copy() if not fast else part, lv)
                except Exception:
                    continue
                for s in found:
                    risk = abs(s["trigger"] - s["stop"])
                    if risk < s["trigger"] * 0.0015 or abs(s["t1"] - s["trigger"]) / risk < MIN_RR:
                        continue
                    key = (tk, s["setup"], s["direction"])
                    if key in last_seen and t - last_seen[key] < pd.Timedelta(minutes=DEDUPE_MIN):
                        continue
                    last_seen[key] = t
                    elapsed = max(len(part), 1) / 390
                    rvol = float(part.Volume.sum()) / (lv["avg_vol"] * elapsed) if lv and lv.get("avg_vol") else 1.0
                    sc, _ = setups.score(s, regime, bias, False, rvol, contract=True)
                    g = _grade(df, t, s)
                    if g is None:
                        continue
                    out.append({"date": day, "ts": t, "ticker": tk, "setup": s["setup"], "direction": s["direction"],
                                "trigger": s["trigger"], "stop": s["stop"], "t1": s["t1"], "t2": s["t2"],
                                "rr": s["rr"], "score": sc, "alerted": int(in_window and sc >= MIN_ALERT_SCORE),
                                "regime": regime, "bias": bias, "rvol": round(rvol, 2),
                                "outcome": g[0], "r": round(g[1], 3), "hit_t2": g[2], "mins": g[3]})
            t += pd.Timedelta(minutes=1)
    finally:
        if fast:
            setups.indicators = orig
    return out


# ------------------------------- report -------------------------------
def _bucket(ts):
    m = ts.hour * 100 + ts.minute
    return "9:45-10:30" if m < 1030 else "10:30-11:30" if m < 1130 else "11:30-1:30" if m < 1330 else "1:30-3:55"


def _agg(df, col):
    g = df.groupby(col).agg(n=("r", "size"), win=("win", "mean"), avg_r=("r", "mean"), total_r=("r", "sum"))
    g["win"] = (100 * g.win).round(0)
    g["avg_r"] = g.avg_r.round(3)
    g["total_r"] = g.total_r.round(1)
    return g.reset_index()


def _ci(p, n):
    """95% Wilson interval for a win rate."""
    if n == 0:
        return 0, 0
    z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(c - h, 0), 100 * min(c + h, 1)


def _spark(cum, w=140, h=30):
    """Tiny inline equity curve."""
    if len(cum) < 2:
        return ""
    v = np.asarray(cum, dtype=float)
    lo, hi = min(v.min(), 0), max(v.max(), 0)
    span = hi - lo or 1
    xs = np.linspace(0, w, len(v))
    ys = h - (v - lo) / span * h
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    zero = h - (0 - lo) / span * h
    col = "#1f7a5a" if v[-1] > 0 else "#b03a2e"
    return (f"<svg width={w} height={h} viewBox='0 0 {w} {h}' aria-hidden=true><line x1=0 x2={w} y1={zero:.1f} y2={zero:.1f} "
            f"stroke=currentColor stroke-opacity=.25 stroke-dasharray=2,2 /><polyline points='{pts}' fill=none stroke={col} stroke-width=1.5 /></svg>")


def _strategy_stats(d, n_days, halves):
    d = d.sort_values("ts")
    usd = d.r * RISK_USD
    opt_r = d.r - OPT_COST / (OPT_DELTA * (d.trigger - d.stop).abs())
    usd_opt = opt_r * RISK_USD
    cum = usd.cumsum()
    dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
    streak = best = 0
    for x in d.r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    gains, losses = usd[usd > 0].sum(), -usd[usd < 0].sum()
    h = {k: d[d.half == k].r.mean() for k in halves}
    return {"n": len(d), "win": 100 * (d.outcome == "win").mean() if len(d) else 0, "avg_r": d.r.mean() if len(d) else 0,
            "total_r": d.r.sum(), "usd": usd.sum(), "usd_opt": usd_opt.sum(), "pf": gains / losses if losses else float("inf"),
            "dd": dd, "streak": best, "per_week": len(d) / max(n_days / 5, 1), "halves": h,
            "spark": _spark(np.concatenate([[0], cum.values]))}


def report(df):
    e = html.escape
    df = df.copy()
    df["win"] = (df.outcome == "win").astype(int)
    df["bucket"] = df.ts.map(_bucket)
    df["setup_dir"] = df.setup + " (" + df.direction + "s)"
    dates = sorted(df.date.unique())
    mid = dates[len(dates) // 2]
    df["half"] = np.where(df.date < mid, f"1st half (to {mid})", f"2nd half ({mid} on)")
    al = df[df.alerted == 1]

    def table(d, col, label):
        g = _agg(d, col)
        rows = "".join(f"<tr><th>{e(str(getattr(r, col)))}</th><td>{r.n:,}</td><td>{r.win:.0f}%</td>"
                       f"<td class={'up' if r.avg_r > 0 else 'dn'}>{r.avg_r:+.3f}R</td><td>{r.total_r:+,.1f}R</td></tr>"
                       for r in g.itertuples())
        return f"<table><tr><th>{label}</th><td>n</td><td>win</td><td>avg</td><td>total</td></tr>{rows}</table>"

    def cross(d):
        g = d.groupby(["setup_dir", "half"]).agg(n=("r", "size"), avg_r=("r", "mean")).reset_index()
        piv = g.pivot(index="setup_dir", columns="half", values="avg_r")
        n = g.pivot(index="setup_dir", columns="half", values="n")
        cols = list(piv.columns)
        rows = ""
        for s in piv.index:
            cells = "".join(f"<td class={'up' if piv.loc[s, c] > 0 else 'dn'}>{piv.loc[s, c]:+.3f}R "
                            f"<span class=sub>n={int(n.loc[s, c]) if n.loc[s, c] == n.loc[s, c] else 0}</span></td>"
                            if piv.loc[s, c] == piv.loc[s, c] else "<td>-</td>" for c in cols)
            stable = all(piv.loc[s, c] > 0 for c in cols if piv.loc[s, c] == piv.loc[s, c])
            rows += f"<tr><th>{e(s)}</th>{cells}<td>{'holds' if stable else '-'}</td></tr>"
        return f"<table><tr><th>Setup</th>{''.join(f'<td>{e(c)}</td>' for c in cols)}<td>both halves +</td></tr>{rows}</table>"

    halves = sorted(df.half.unique())
    capped = al.sort_values("ts").groupby("date").head(MAX_PER_DAY)
    strategies = [("All alerted setups", al), (f"Your rules: first {MAX_PER_DAY} alerts per day", capped),
                  ("Score 9-10 only", al[al.score >= 9]), ("Score 6-8 only", al[al.score <= 8])]
    strategies += [(k, g) for k, g in al.groupby("setup_dir")]
    srows = ""
    for name, d in strategies:
        if d.empty:
            continue
        st = _strategy_stats(d, len(dates), halves)
        hs = "".join(f"<td class={'up' if v > 0 else 'dn'}>{v:+.2f}R</td>" if v == v else "<td>-</td>" for v in st["halves"].values())
        pf = "∞" if st["pf"] == float("inf") else f"{st['pf']:.2f}"
        srows += (f"<tr><th>{e(name)}<span class=sub>{st['n']:,} trades · {st['per_week']:.1f}/week · {st['win']:.0f}% win</span></th>"
                  f"<td>{st['spark']}</td><td class={'up' if st['avg_r'] > 0 else 'dn'}>{st['avg_r']:+.3f}R</td>"
                  f"<td class={'up' if st['usd'] > 0 else 'dn'}>${st['usd']:+,.0f}</td>"
                  f"<td class={'up' if st['usd_opt'] > 0 else 'dn'}>${st['usd_opt']:+,.0f}</td>"
                  f"<td>{pf}</td><td class=dn>${st['dd']:,.0f}</td><td>{st['streak']}</td>{hs}</tr>")
    strat_html = (f"<h2>Hypothetical returns at ${RISK_USD:,.0f} risked per trade</h2>"
                  f"<div style='overflow-x:auto'><table><tr><th>Strategy</th><td>equity curve</td><td>avg</td><td>stock P/L</td>"
                  f"<td>after option costs</td><td>profit factor</td><td>max drawdown</td><td>worst losing streak</td>"
                  + "".join(f"<td>{e(h_)}</td>" for h_ in halves) + f"</tr>{srows}</table></div>"
                  f"<p class=note><b>Stock P/L</b> = result in R × ${RISK_USD:,.0f}. <b>After option costs</b> assumes a "
                  f"{OPT_DELTA:.2f}-delta option sized to the same dollar risk and ${OPT_COST:.2f}/share (${100 * OPT_COST:.0f}/contract) "
                  f"round-trip bid/ask: tight stops make that cost a big share of each trade's risk. Ignores time decay within the day. "
                  f"<b>Profit factor</b> = dollars won ÷ dollars lost (above 1 makes money). <b>Max drawdown</b> = worst peak-to-trough "
                  f"drop. Look for strategies positive in BOTH halves; one good half is usually luck.</p>")
    lo, hi = _ci(al.win.mean() if len(al) else 0, len(al))
    head = (f"{len(dates)} trading days ({dates[0]} to {dates[-1]}) · {len(df):,} setups logged · {len(al):,} would have "
            f"alerted (score {MIN_ALERT_SCORE}+ inside the windows)")
    summary = (f"Alerted setups: {100 * al.win.mean():.0f}% hit target 1 first (95% range {lo:.0f}-{hi:.0f}%), "
               f"average {al.r.mean():+.3f}R per setup, {al.r.sum():+,.0f}R total.") if len(al) else "No alerted setups."
    css = """body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#1c2430;font-variant-numeric:tabular-nums}
@media (prefers-color-scheme:dark){body{background:#141a21;color:#e3e9ef} table{background:#1c242d!important} th,td{border-color:#2e3a46!important}}
main{max-width:1000px;margin:0 auto;padding:20px} h1{margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px;opacity:.75}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:6px;overflow:hidden;margin-bottom:8px}
th,td{padding:6px 10px;border-bottom:1px solid #d5dce3;text-align:right} th{text-align:left;font-weight:600}
.up{color:#1f7a5a} .dn{color:#b03a2e} .sub{opacity:.6;font-size:12px} .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media (max-width:700px){.two{grid-template-columns:1fr}} p.note{opacity:.75;font-size:13px}"""
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Day-trade backtest</title><style>{css}</style></head><body><main>
<h1>Day-trading scanner backtest</h1><p>{e(head)}<br><b>{e(summary)}</b></p>
{strat_html}
<h2>Alerted setups by score (do higher scores do better?)</h2>{table(al, 'score', 'Score')}
<h2>Every logged setup by score</h2>{table(df, 'score', 'Score')}
<h2>Alerted setups by setup</h2>{table(al, 'setup_dir', 'Setup')}
<h2>Does it hold up? Alerted setups, first half vs second half</h2>{cross(al)}
<div class=two><div><h2>By time of day (alerted)</h2>{table(al, 'bucket', 'Time')}</div>
<div><h2>By market read (alerted)</h2>{table(al, 'regime', 'Market read')}</div></div>
<div class=two><div><h2>By SPY vs VWAP (alerted)</h2>{table(al, 'bias', 'SPY')}</div>
<div><h2>By reward/risk (alerted)</h2>{table(al.assign(rrb=pd.cut(al.rr, [0, 2, 3, 5, 1e9], labels=['1.5-2', '2-3', '3-5', '5+'])), 'rrb', 'R/R')}</div></div>
<p class=note>Same detection and scoring code as the live desk, graded with the live scorecard rules (stop checked first;
a same-minute tie is a loss). Not reproduced: news catalysts and Stocktwits (catalyst point always off), option availability
(assumed yes), VIX/yields/oil and the AI in the market read, and intraday watchlist refreshes (list fixed at 9:45 from the first
15 minutes). Stock moves only: no option spreads, decay or slippage, so live results will be worse than this.
Universe is today's index members (survivorship bias).</p>
<p class=note>Built {datetime.now(ET):%b %d %Y %I:%M %p} ET.</p></main></body></html>"""
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(doc)
    return summary


# ------------------------------- main -------------------------------
def _cached(f):
    return os.path.join(CACHE, f"v{CACHE_VERSION}_" + os.path.basename(f))


def _replay_cached(f):
    out = replay_day(f)
    tmp = _cached(f) + ".tmp"
    pickle.dump(out, open(tmp, "wb"))
    os.replace(tmp, _cached(f))
    return len(out)


def run(n_days=None, workers=None, minutes=None):
    """Replay every downloaded day (results cached per day, so it can be stopped and resumed),
    then write the report from all cached days."""
    from concurrent.futures import as_completed
    files = sorted(glob.glob(os.path.join(DAYS, "*.pkl")))
    if n_days:
        files = files[-n_days:]
    if not files:
        sys.exit("No history yet. Run history.py first (the 3 - DOWNLOAD HISTORY launcher).")
    todo = [f for f in files if not os.path.exists(_cached(f))]
    t0 = time.time()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if todo:
        log(f"[backtest] replaying {len(todo)} of {len(files)} days on {workers} cores...")
        done = 0
        ex = ProcessPoolExecutor(max_workers=workers)
        futs = [ex.submit(_replay_cached, f) for f in todo]
        try:
            for fu in as_completed(futs, timeout=minutes * 60 if minutes else None):
                fu.result()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    log(f"[backtest] {done}/{len(todo)} days ({el / 60:.1f} min, ~{el / done * (len(todo) - done) / 60:.0f} min left)")
        except Exception as e:                       # time budget reached: keep what's done
            log(f"[backtest] paused after {done} days ({e.__class__.__name__}); run again to continue")
            ex.shutdown(wait=False, cancel_futures=True)
            return None
        ex.shutdown()
    rows = []
    for f in files:
        rows.extend(pickle.load(open(_cached(f), "rb")))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No setups found.")
    df.to_pickle(OUT_PKL)
    s = report(df)
    log(f"[backtest] {s}")
    log(f"[backtest] report: {OUT_HTML}")
    return df


def verify(n=3):
    """The fast path precomputes indicators once per day. Prove it finds exactly the same setups
    as calling the live code fresh every minute."""
    files = sorted(glob.glob(os.path.join(DAYS, "*.pkl")))[-n:]
    for f in files:
        a = replay_day(f, fast=True)
        b = replay_day(f, fast=False)
        key = lambda r: (r["ts"], r["ticker"], r["setup"], r["direction"], r["trigger"], r["stop"], r["t1"], r["score"])
        same = sorted(map(key, a)) == sorted(map(key, b))
        print(f"{os.path.basename(f)}: fast {len(a)} vs live-code {len(b)} setups -> {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same:
            sa, sb = set(map(key, a)), set(map(key, b))
            print("  only fast:", sorted(sa - sb)[:5]); print("  only live:", sorted(sb - sa)[:5])


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
    else:
        n = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else None
        m = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else None
        run(n, minutes=m)
