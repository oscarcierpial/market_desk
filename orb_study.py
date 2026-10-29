#!/usr/bin/env python3
"""
ORB (opening range breakout) CALL study, tested the honest way.

The idea came from the 2-year backtest (Oct 2024 - Oct 2026): ORB calls signaled by 10:30 on index
ETFs, tech and high-momentum stocks looked profitable. That period can't also be the proof, so:

  DISCOVERY   2024-10-04 .. latest      where the idea came from (shown, not counted as evidence)
  FRESH TEST  2016-03-07 .. 2024-10-03  never looked at before the rule below was written
     TUNE     2016-03-07 .. 2020-12-31  variations are chosen here
     CONFIRM  2021-01-01 .. 2024-10-03  ...and only count if they hold up here

PRE-REGISTERED RULE (written Oct 5 2026, before running on the fresh data):
  Live-desk ORB call logic (15-minute opening range; a 5-minute close above the range high on
  1.2x median volume, above VWAP; stop = range midpoint; target = range high + range height),
  signal at or before 10:30, on SPY/QQQ, S&P 500 Information Technology stocks, or non-S&P
  momentum names from the desk's universe. Graded with the live scorecard rules.

Variations tested on top: 30-minute range, stop at the range low, 1R and 2R targets, 11:00 cutoff.

  python orb_study.py            run everything in state/history/days, write orb_study.html
  python orb_study.py --verify   prove the engine matches the full replay's ORB signals exactly
"""
import glob, html, json, math, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import pandas as pd

from common import BASE, ET, env, load, log, path

DAYS = os.path.join(path("history"), "days")
OUT_HTML = os.path.join(BASE, "orb_study.html")
OUT_PKL = os.path.join(path("history"), "orb_trades.pkl")
CACHE = os.path.join(path("history"), "orb_cache_v1")
os.makedirs(CACHE, exist_ok=True)
RISK_USD = env("BT_RISK_PER_TRADE", 50.0)
DELTA = env("BT_OPTION_DELTA", 0.50)
COST_STOCK = env("BT_OPTION_COST", 0.05)       # round-trip option bid/ask per share, single stocks
COST_INDEX = env("BT_OPTION_COST_INDEX", 0.02)  # SPY/QQQ options are about a penny wide
DISCOVERY_FROM, TUNE_TO = "2024-10-04", "2020-12-31"
INDEX = {"SPY", "QQQ"}
BASE_VARIANT = (15, "mid", "live", 1030)
VARIANTS = list(product([15, 30], ["mid", "low"], ["live", "1R", "2R"], [1030, 1100]))


def period(day):
    return "discovery" if day >= DISCOVERY_FROM else "tune" if day <= TUNE_TO else "confirm"


def group(tk, sectors):
    if tk in INDEX:
        return "Index ETFs (SPY/QQQ)"
    s = sectors.get(tk)
    if s is None:
        return "Momentum (non-S&P)"
    return "Tech (S&P)" if s == "Information Technology" else "Other S&P sectors"


def _prep(df, close_min):
    m = (df.index.hour * 60 + df.index.minute).to_numpy()
    keep = (m >= 570) & (m < close_min)
    df, m = df[keep], m[keep]
    if len(df) < 16:
        return None
    o, h, l, c, v = (df[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close", "Volume"))
    tp = (h + l + c) / 3
    cv = np.cumsum(v)
    vwap = np.cumsum(tp * v) / np.where(cv > 0, cv, np.nan)
    b = pd.DataFrame({"c": c, "v": v}, index=df.index).resample("5min", label="left", closed="left")
    b5 = pd.DataFrame({"c": b.c.last(), "v": b.v.sum(), "n": b.c.count()})
    b5 = b5[b5.n > 0]
    s5 = (b5.index.hour * 60 + b5.index.minute).to_numpy()
    return {"m": m, "h": h, "l": l, "c": c, "vwap": vwap, "s5": s5, "c5": b5.c.to_numpy(), "v5": b5.v.to_numpy()}


def _candidates(p, or_len):
    """Every scan minute where the live ORB call condition is true (before any variant's filters).
    Replicates setups.detect: 5-min close above the range high on 1.2x median volume, above VWAP."""
    m, h, l = p["m"], p["h"], p["l"]
    or_end = 570 + or_len                                   # first minute after the range
    in_or = m < or_end
    if in_or.sum() < (10 if or_len == 15 else 20):
        return None, []
    orh, orl = float(h[in_or].max()), float(l[in_or].min())     # Python floats: same rounding as live
    if orh - orl <= 0:
        return None, []
    s5, c5, v5 = p["s5"], p["c5"], p["v5"]
    out = []
    for T in range(or_end + 1, 662):                        # detect runs while the last bar <= 11:00
        k = int(np.searchsorted(m, T))                       # bars strictly before T
        if k < 16:
            continue
        last_m = m[k - 1]
        if not (or_end <= last_m <= 660):                    # live: 945 <= t_now <= ORB_END (11:00)
            continue
        n5 = int(np.searchsorted(s5 + 5, last_m + 1, side="right"))   # 5-min bars complete by the last bar
        if n5 < 5:
            continue
        cc, pc = c5[n5 - 1], c5[n5 - 2]
        if cc > orh >= pc and v5[n5 - 1] > 1.2 * float(np.median(v5[:n5 - 1])) and cc > p["vwap"][k - 1]:
            out.append((T, float(cc)))
    return (orh, orl), out


def _apply(p, rng_hl, cands, stop_kind, target_kind, cutoff):
    """One variant: levels, scanner filters, 45-min dedupe, live-scorecard grading."""
    orh, orl = rng_hl
    rng = orh - orl
    m, h, l, c = p["m"], p["h"], p["l"], p["c"]
    cut = 661 if cutoff >= 1100 else (cutoff // 100) * 60 + cutoff % 100   # last scan minute allowed
    out, last_sig = [], -10 ** 9
    for T, cc in cands:
        if T > cut:
            break
        trig = round(cc, 2)                                  # the live code rounds levels to the cent
        stop = round(orl + rng / 2 if stop_kind == "mid" else orl, 2)
        if not stop < trig:
            continue
        risk = trig - stop
        if target_kind == "live":
            t1 = round(orh + rng, 2)
            if not trig < t1 or (t1 - trig) / risk < 1.5:
                continue
        else:
            t1 = round(trig + (1 if target_kind == "1R" else 2) * risk, 2)
        if risk < trig * 0.0015:
            continue
        if T - last_sig < 45:                                # same 45-minute dedupe as the journal
            continue
        last_sig = T
        a = int(np.searchsorted(m, T, side="right"))         # grade on bars strictly after the scan minute
        if a >= len(m):
            continue
        hit_s = np.flatnonzero(l[a:] <= stop)
        hit_t = np.flatnonzero(h[a:] >= t1)
        i_s = hit_s[0] if len(hit_s) else 10 ** 9
        i_t = hit_t[0] if len(hit_t) else 10 ** 9
        if i_s == i_t == 10 ** 9:
            outcome, r = "timeout", (c[-1] - trig) / risk
        elif i_s <= i_t:
            outcome, r = "loss", -1.0
        else:
            outcome, r = "win", (t1 - trig) / risk
        out.append({"T": T, "trigger": trig, "stop": stop, "t1": t1, "risk": risk, "outcome": outcome, "r": float(r)})
    return out


def study_day(fname, variants=None):
    rec = pickle.load(open(fname, "rb"))
    day = rec["date"]
    close_min = int(rec["close"][:2]) * 60 + int(rec["close"][3:5])
    tickers = sorted({w["ticker"] for w in rec["watch"]} | INDEX)
    variants = variants or VARIANTS
    rows = []
    for tk in tickers:
        df = rec["bars"].get(tk)
        if df is None or df.empty:
            continue
        p = _prep(df, close_min)
        if p is None:
            continue
        for or_len in sorted({v[0] for v in variants}):
            hl, cands = _candidates(p, or_len)
            if not cands:
                continue
            for var in (v for v in variants if v[0] == or_len):
                for s in _apply(p, hl, cands, *var[1:]):
                    rows.append({"date": day, "ticker": tk, "variant": var, **s})
    return rows


def _cached_day(fname):
    out = os.path.join(CACHE, os.path.basename(fname))
    if not os.path.exists(out):
        rows = study_day(fname)
        pickle.dump(rows, open(out + ".tmp", "wb"))
        os.replace(out + ".tmp", out)
    return out


# ------------------------------- verification -------------------------------
def verify():
    """Engine (base live logic, 11:00 cutoff) vs the full minute-by-minute replay of setups.detect."""
    bt = pd.read_pickle(os.path.join(path("history"), "bt_signals.pkl"))
    ref = bt[(bt.setup == "ORB") & (bt.direction == "call")]
    files = sorted(glob.glob(os.path.join(DAYS, "*.pkl")))
    files = [f for f in files if os.path.basename(f)[:10] >= ref.date.min()]
    mine = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for r in ex.map(study_day, files, [[(15, "mid", "live", 1100)]] * len(files), chunksize=8):
            mine.extend(r)
    mine = pd.DataFrame(mine)
    key = lambda d, t, tr, st, t1, o: (d, t, round(tr, 2), round(st, 2), round(t1, 2), o)
    a = {key(r.date, r.ticker, r.trigger, r.stop, r.t1, r.outcome) for r in mine.itertuples()}
    b = {key(r.date, r.ticker, r.trigger, r.stop, r.t1, r.outcome) for r in ref.itertuples()}
    print(f"engine {len(a)} signals, full replay {len(b)}, matching {len(a & b)}")
    print("only engine:", sorted(a - b)[:5])
    print("only replay:", sorted(b - a)[:5])
    return len(a ^ b) == 0


# ------------------------------- report -------------------------------
def _stats(d):
    if d.empty:
        return None
    r = d.r.to_numpy()
    cost = np.where(d.ticker.isin(INDEX), COST_INDEX, COST_STOCK)
    ro = r - cost / (DELTA * d.risk.to_numpy())
    n = len(r)
    se = r.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    cum = np.cumsum(r * RISK_USD)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if n else 0
    return {"n": n, "win": 100 * float((d.outcome == "win").mean()), "avg": float(r.mean()), "t": float(r.mean() / se) if se else 0,
            "opt": float(ro.mean()), "usd": float(r.sum() * RISK_USD), "usd_opt": float(ro.sum() * RISK_USD), "dd": dd}


def _cell(st, key="avg"):
    if st is None:
        return "<td>-</td>"
    v = st[key]
    return f"<td class={'up' if v > 0 else 'dn'}>{v:+.3f}R<span class=sub> n={st['n']}</span></td>"


def report(tr):
    e = html.escape
    sectors = load("sectors.json", {}).get("map", {})
    tr = tr.copy()
    tr["group"] = tr.ticker.map(lambda t: group(t, sectors))
    tr["period"] = tr.date.map(period)
    tr["year"] = tr.date.str[:4]
    focus_groups = ["Index ETFs (SPY/QQQ)", "Tech (S&P)", "Momentum (non-S&P)"]
    tr["focus"] = tr.group.isin(focus_groups)
    periods = ["tune", "confirm", "discovery"]
    base = tr[tr.variant == BASE_VARIANT]
    rule = base[base.focus]
    fresh = rule[rule.period != "discovery"]
    fs = _stats(fresh)

    # 1. The pre-registered rule
    rows = ""
    for name, d in [("Pre-registered rule (focus groups)", rule), ("Everything else (other S&P sectors)", base[~base.focus])] + \
                   [(g, base[base.group == g]) for g in focus_groups]:
        cells = "".join(_cell(_stats(d[d.period == p])) for p in periods)
        f = _stats(d[d.period != "discovery"])
        cells += _cell(f) + (f"<td class={'up' if f and f['opt'] > 0 else 'dn'}>{f['opt']:+.3f}R</td><td>{f['t']:+.2f}</td>" if f else "<td>-</td><td>-</td>")
        rows += f"<tr><th>{e(name)}</th>{cells}</tr>"
    t1 = (f"<table><tr><th></th><td>tune 2016-2020</td><td>confirm 2021-Sep 2024</td><td>discovery Oct 2024-2026</td>"
          f"<td><b>fresh total</b></td><td>fresh, after option costs</td><td>t-stat (fresh)</td></tr>{rows}</table>")

    # 2. By year (rule)
    yrows = "".join(f"<tr><th>{y}</th>{_cell(_stats(g))}<td>{_stats(g)['win']:.0f}%</td></tr>" for y, g in rule.groupby("year"))
    t2 = f"<table><tr><th>Year</th><td>avg (n)</td><td>win</td></tr>{yrows}</table>"

    # 3. Variations: tune vs confirm, focus groups only
    vrows, picks = "", []
    for var in VARIANTS:
        d = tr[(tr.variant == var) & tr.focus]
        st_t, st_c, st_d = (_stats(d[d.period == p]) for p in periods)
        label = f"{var[0]}-min range · stop {var[1]} · target {var[2]} · by {var[3] // 100}:{var[3] % 100:02d}"
        if st_t:
            picks.append((st_t["opt"], label, var))
        holds = st_t and st_c and st_t["opt"] > 0 and st_c["opt"] > 0
        vrows += (f"<tr{' style=font-weight:600' if var == BASE_VARIANT else ''}><th>{e(label)}{' (rule)' if var == BASE_VARIANT else ''}</th>"
                  + "".join(f"<td class={'up' if s and s['opt'] > 0 else 'dn'}>{s['opt']:+.3f}R<span class=sub> n={s['n']}</span></td>" if s else "<td>-</td>"
                            for s in (st_t, st_c, st_d)) + f"<td>{'holds' if holds else '-'}</td></tr>")
    t3 = (f"<table><tr><th>Variation (focus groups)</th><td>tune</td><td>confirm</td><td>discovery</td><td>+ in tune AND confirm</td></tr>"
          f"{vrows}</table><p class=note>Values are after estimated option costs. Pick on the tune column only; the confirm column is the check.</p>")
    best = max(picks) if picks else None

    verdict = "Not enough trades." if not fs else (
        f"On fresh data the rule made {fs['avg']:+.3f}R per trade before costs and {fs['opt']:+.3f}R after estimated option costs "
        f"over {fs['n']} trades (t = {fs['t']:+.2f}). " +
        ("That clears the usual bar for a real effect (t above 2)." if fs["t"] >= 2 and fs["opt"] > 0 else
         "That does NOT clear the bar for a real effect (needs t above 2 and positive after costs)."))
    css = """body{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f7;color:#1c2430;font-variant-numeric:tabular-nums}
@media (prefers-color-scheme:dark){body{background:#141a21;color:#e3e9ef} table{background:#1c242d!important} th,td{border-color:#2e3a46!important}}
main{max-width:1050px;margin:0 auto;padding:20px} h2{font-size:15px;margin:28px 0 8px;opacity:.75}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:6px;overflow:hidden}
th,td{padding:6px 10px;border-bottom:1px solid #d5dce3;text-align:right} th{text-align:left;font-weight:600}
.up{color:#1f7a5a} .dn{color:#b03a2e} .sub{opacity:.6;font-size:12px;margin-left:4px} p.note{opacity:.75;font-size:13px}
.verdict{background:#fff;border-left:5px solid #8a6d1f;padding:12px 16px;border-radius:0 8px 8px 0}"""
    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ORB call study</title><style>{css}</style></head><body><main><h1>ORB call study</h1>
<p>{tr.date.nunique():,} trading days ({tr.date.min()} to {tr.date.max()}). Rule written before the fresh data was examined.</p>
<p class=verdict><b>{e(verdict)}</b></p>
<h2>1. The pre-registered rule</h2>{t1}
<h2>2. The rule year by year</h2>{t2}
<h2>3. Variations (chosen on 2016-2020, checked on 2021-2024)</h2>{t3}
<p class=note>Best variation on the tune period: {e(best[1]) if best else '-'}.</p>
<p class=note>Graded like the live scorecard: target 1 before the stop, stop checked first, timeout scored at the close. Option estimate
assumes a {DELTA:.2f}-delta option sized to the same risk with ${COST_STOCK:.2f}/share round-trip cost (${COST_INDEX:.2f} for SPY/QQQ).
Watchlist rebuilt each day from the first 15 minutes; universe is today's index members plus the desk's extras, so early years
are missing stocks that later dropped out (survivorship bias) and momentum names that hadn't listed yet.</p>
</main></body></html>"""
    open(OUT_HTML, "w", encoding="utf-8").write(doc)
    return verdict


def run(minutes=None):
    """Study every downloaded day (cached per day, so it can stop and resume), then write the report."""
    from concurrent.futures import as_completed
    files = sorted(glob.glob(os.path.join(DAYS, "*.pkl")))
    todo = [f for f in files if not os.path.exists(os.path.join(CACHE, os.path.basename(f)))]
    t0 = time.time()
    if todo:
        log(f"[orb] studying {len(todo)} of {len(files)} days...")
        ex = ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1))
        futs = [ex.submit(_cached_day, f) for f in todo]
        done = 0
        try:
            for fu in as_completed(futs, timeout=minutes * 60 if minutes else None):
                fu.result()
                done += 1
                if done % 200 == 0:
                    log(f"[orb] {done}/{len(todo)} days ({time.time() - t0:.0f}s)")
        except Exception as e:
            log(f"[orb] paused after {done} days ({e.__class__.__name__}); run again to continue")
            ex.shutdown(wait=False, cancel_futures=True)
            return
        ex.shutdown()
    rows = []
    for f in files:
        rows.extend(pickle.load(open(os.path.join(CACHE, os.path.basename(f)), "rb")))
    tr = pd.DataFrame(rows)
    tr.to_pickle(OUT_PKL)
    v = report(tr)
    log(f"[orb] {v}")
    log(f"[orb] report: {OUT_HTML} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)
    run(float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else None)
