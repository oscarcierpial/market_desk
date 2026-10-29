"""One scanner pass: watchlist -> setups -> contract -> score -> journal -> alert."""
from common import load, save, now, log, notify, MIN_ALERT_SCORE
import contracts, journal, setups

_runs = 0


def run_scan():
    wl = load("watchlist.json", {}).get("items", [])
    if not wl or load("watchlist.json", {}).get("time", "")[:10] != now().strftime("%Y-%m-%d"):
        log("[scan] no watchlist for today yet; skipping (it builds from 9:00)")
        return []
    brief = load("brief.json", {})
    today = now().strftime("%Y-%m-%d")
    regime = brief.get("regime", "") if brief.get("time", "")[:10] == today else ""
    cats = {k: v for k, v in load("catalysts.json", {}).items() if v.get("date") == today}
    raw, bias, bars = setups.scan(wl)
    in_window = setups.allowed_now()
    fired = []
    for s in raw:
        if journal.is_duplicate(s):
            continue
        cat = cats.get(s["ticker"])
        s["catalyst"] = cat["headline"] if cat else ""
        pre, _ = setups.score(s, regime, bias, bool(cat), s.get("rvol_day", 1), contract=True)
        s["contract"] = contracts.pick(s["ticker"], s["direction"], s["trigger"], s["stop"], s["t1"]) \
            if pre >= MIN_ALERT_SCORE else None
        s["score"], notes = setups.score(s, regime, bias, bool(cat), s.get("rvol_day", 1), s["contract"])
        alert = in_window and s["score"] >= MIN_ALERT_SCORE
        journal.record(s, regime, bias, alert)
        log(f"[setup] {s['ticker']} {s['setup']} {s['direction']} score {s['score']} "
            f"{'ALERT' if alert else 'logged'} (R/R {s['rr']})")
        if alert:
            arrow = "calls" if s["direction"] == "call" else "puts"
            body = (f"Trigger {s['trigger']:.2f} · stop {s['stop']:.2f} · targets {s['t1']:.2f} / {s['t2']:.2f} "
                    f"· R/R {s['rr']:.1f}\n{s['why']}\n"
                    f"{contracts.describe(s['contract'])}\n"
                    f"Context: {'; '.join(notes)}\n"
                    + (f"News: {s['catalyst'][:120]}\n" if s["catalyst"] else "")
                    + "Confirm on your own chart first; this data runs about a minute behind.")
            notify(f"{s['ticker']} · {s['setup']} ({arrow}) · {s['score']}/10", body,
                   priority="high", tags="dart")
        fired.append(s)
    global _runs
    _runs += 1
    if fired or _runs % 10 == 1:
        log(f"[scan] watching {len(bars)} of {len({w['ticker'] for w in wl} | {'SPY', 'QQQ'})} stocks, SPY {'above' if bias == 'up' else 'below'} VWAP, "
            f"{len(fired)} new setup(s){'' if in_window else ' (outside alert windows)'}")
    save("scan_status.json", {"time": now().isoformat(), "bias": bias, "regime": regime,
                              "watching": len(bars), "found": len(fired), "in_window": in_window,
                              "prices": {tk: round(float(df.Close.iloc[-1]), 2) for tk, df in bars.items()}})
    return fired
