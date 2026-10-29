#!/usr/bin/env python3
"""
Market brief
------------
(Module version: desk.py calls build(); dashboard.py draws the page.)
Every 15 minutes during market hours:
  1. Pulls headlines from every feed in sources.py (last 3 hours)
  2. Pulls market data: indexes, VIX, 10-year yield, oil, dollar, gold, bitcoin, sectors
  3. Scores conditions with transparent rules -> Bullish / Bearish / Stable
  4. (Optional) Has Claude read it all and explain what the headlines mean
  5. Writes brief.html (open it in a browser; it refreshes itself) and pushes a short
     summary to your phone at set times or whenever the regime changes

Run:   python market_brief.py              (loops during market hours)
Test:  python market_brief.py --once --dry-run
"""
import json, os, re, sys, time
from datetime import datetime

import requests
import yfinance as yf

from common import ET, YF_LOCK, load, save, now, log, notify, BRIEF_MINUTES
from sources import fetch_all, norm
import llm

# ============================== CONFIG ==============================

LOOKBACK_MIN = 180                      # headlines from the last N minutes
PUSH_TIMES = ["09:15", "10:30", "12:00", "14:00", "15:30"]   # scheduled phone summaries (ET)

INDEXES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small caps", "DIA": "Dow"}
MACRO = {"^VIX": "VIX (fear gauge)", "^TNX": "10-yr yield %", "CL=F": "Oil (WTI)",
         "DX-Y.NYB": "US dollar", "GC=F": "Gold", "BTC-USD": "Bitcoin"}
SECTORS = {"XLK": "Tech", "SMH": "Semis", "XLY": "Consumer disc.", "XLC": "Communication",
           "XLF": "Financials", "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
           "XLV": "Health care", "XLP": "Staples", "XLU": "Utilities", "XLRE": "Real estate"}

# =========================== MARKET DATA ===========================

def snapshot():
    tickers = list(INDEXES) + list(MACRO) + list(SECTORS)
    with YF_LOCK:
        intraday = yf.download(tickers, period="2d", interval="5m", progress=False,
                               group_by="ticker", threads=True)
        daily = yf.download(tickers, period="10d", interval="1d", progress=False,
                            group_by="ticker", threads=True)
    snap = {}
    today = datetime.now(ET).date()
    for t in tickers:
        try:
            c = intraday[t]["Close"].dropna()
            d = daily[t]["Close"].dropna()
            prev = d.iloc[-2] if d.index[-1].date() == today else d.iloc[-1]
            last = float(c.iloc[-1])
            row = {"last": last, "chg": 100 * (last / prev - 1), "prev": float(prev)}
            bars = intraday[t][intraday[t].index.tz_convert(ET).date == today].dropna()
            if t in INDEXES and len(bars) > 3:
                tp = (bars.High + bars.Low + bars.Close) / 3
                row["vwap"] = float((tp * bars.Volume).sum() / bars.Volume.sum())
                row["hi"], row["lo"] = float(bars.High.max()), float(bars.Low.min())
                row["chg_30m"] = 100 * (last / float(bars.Close.iloc[-7]) - 1) if len(bars) > 7 else 0
            snap[t] = row
        except Exception:
            continue
    return snap

def stocktwits_trending(n=12):
    try:
        r = requests.get("https://api.stocktwits.com/api/2/trending/symbols.json",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
        return [{"symbol": s["symbol"], "why": (s.get("trends") or {}).get("summary", "")[:300]}
                for s in r.get("symbols", []) if s.get("instrument_class") != "CRYPTO"][:n]
    except Exception:
        return []

# ======================= RULE-BASED REGIME ========================

def rule_regime(s):
    """Transparent scoring so you always know WHY it says bullish or bearish."""
    pts, why = 0, []
    def add(p, reason):
        nonlocal pts
        pts += p; why.append((p, reason))
    spy = s.get("SPY", {})
    if spy:
        c = spy["chg"]
        if c > 0.5: add(2, f"S&P up {c:+.2f}%")
        elif c > 0.1: add(1, f"S&P up {c:+.2f}%")
        elif c < -0.5: add(-2, f"S&P down {c:+.2f}%")
        elif c < -0.1: add(-1, f"S&P down {c:+.2f}%")
        if "vwap" in spy:
            add(1 if spy["last"] > spy["vwap"] else -1,
                f"SPY {'above' if spy['last'] > spy['vwap'] else 'below'} VWAP ({spy['vwap']:.2f})")
    if "IWM" in s and "SPY" in s:
        gap = s["IWM"]["chg"] - s["SPY"]["chg"]
        if abs(gap) > 0.4:
            add(1 if gap > 0 else -1, f"Small caps {'leading' if gap > 0 else 'lagging'} ({gap:+.2f} pts vs S&P)")
    vix = s.get("^VIX")
    if vix:
        if vix["chg"] > 5: add(-2, f"VIX jumping {vix['chg']:+.1f}% to {vix['last']:.1f}")
        elif vix["chg"] < -4: add(1, f"VIX falling {vix['chg']:+.1f}% to {vix['last']:.1f}")
        if vix["last"] > 25: add(-1, f"VIX elevated at {vix['last']:.1f}")
    tnx = s.get("^TNX")
    if tnx:
        bp = 100 * (tnx["last"] - tnx["prev"])
        if bp > 6: add(-1, f"10-yr yield up {bp:.0f} bp to {tnx['last']:.2f}%")
        elif bp < -6: add(1, f"10-yr yield down {abs(bp):.0f} bp to {tnx['last']:.2f}%")
    oil = s.get("CL=F")
    if oil and abs(oil["chg"]) > 3:
        add(-1 if oil["chg"] > 0 else 1, f"Oil {'spiking' if oil['chg'] > 0 else 'dropping'} {oil['chg']:+.1f}%")
    secs = [s[t]["chg"] for t in SECTORS if t in s]
    if secs:
        up = sum(c > 0 for c in secs)
        if up >= 9: add(1, f"Broad rally: {up}/{len(secs)} sectors up")
        elif up <= 3: add(-1, f"Broad selling: only {up}/{len(secs)} sectors up")
        risk_on = [s[t]["chg"] for t in ("XLK", "XLY", "SMH") if t in s]
        defensive = [s[t]["chg"] for t in ("XLP", "XLU", "XLV") if t in s]
        if risk_on and defensive:
            spread = sum(risk_on) / len(risk_on) - sum(defensive) / len(defensive)
            if spread > 0.5: add(1, f"Risk-on: growth beating defensives by {spread:.1f} pts")
            elif spread < -0.5: add(-1, f"Risk-off: defensives beating growth by {abs(spread):.1f} pts")
    label = "Bullish" if pts >= 3 else "Bearish" if pts <= -3 else "Stable / mixed"
    return {"label": label, "score": pts, "reasons": why}

# ========================= HEADLINE PREP ==========================

STOP = set("the a an to of and in on for with as at by from is are be will after over amid says say "
           "said report reports new us u s its it this that how why what".split())

def top_headlines(items, n=45):
    """Merge the same story across outlets (similar words or same topic+action keywords),
    keep the best-scoring version, and count how many outlets ran it."""
    groups = []
    for h in items:
        words = set(norm(h["title"]).split()) - STOP
        hits = set(h["hits"])
        for g in groups:
            jac = len(words & g["words"]) / max(len(words | g["words"]), 1)
            if jac >= 0.35 or len(hits & g["hitset"]) >= 3:
                g["outlets"].add(h["source"]); g["words"] |= words
                if h["score"] > g["score"]:
                    g.update({k: h[k] for k in ("title", "summary", "source", "link", "score", "hits", "ts")})
                break
        else:
            groups.append({**h, "outlets": {h["source"]}, "words": words, "hitset": hits})
    ranked = sorted(groups, key=lambda g: (g["score"] + 2 * min(len(g["outlets"]), 4), g["ts"]), reverse=True)
    return ranked[:n]

# ========================== CLAUDE BRIEF ==========================

def claude_brief(snap, rules, heads, trending):
    if not llm.enabled():
        return None
    mkt = {t: {k: round(v, 3) for k, v in row.items()} for t, row in snap.items()}
    news = "\n".join(f"- [{h['source']}, {len(h['outlets'])} outlets, "
                     f"{datetime.fromtimestamp(h['ts'], ET):%H:%M}] {h['title']}" for h in heads)
    social = "\n".join(f"- {t['symbol']}: {t['why']}" for t in trending if t["why"])
    prompt = f"""You are the morning-desk strategist for a college student day trader with a $500 account
who trades SPY/QQQ and large-cap options. Time now: {datetime.now(ET):%A %b %d, %H:%M} ET.

MARKET DATA (chg = % vs prior close; ^TNX is the 10-yr yield in %):
{json.dumps(mkt)}

RULE-BASED READ: {rules['label']} (score {rules['score']}): {[r for _, r in rules['reasons']]}

HEADLINES, last {LOOKBACK_MIN} min, deduplicated, most important first:
{news}

STOCKTWITS TRENDING:
{social}

Write a brief. Be concrete, plain-English, and skeptical: separate CONFIRMED news from reports and
rumors, say when a headline is NOT moving prices, and do not invent facts beyond the inputs.
Reply with JSON only:
{{"regime": "Bullish|Bearish|Stable|Mixed", "confidence": "low|medium|high",
 "one_liner": "<=20 words: the market in one sentence",
 "summary": "3-4 sentences: what is driving the market right now and how prices are reacting",
 "drivers": [{{"headline": "short", "what_it_means": "1-2 sentences", "direction": "bullish|bearish|mixed",
              "confirmed": true, "affects": ["tickers or sectors"]}}],
 "not_moving": "headlines getting attention that prices are ignoring, 1-2 sentences",
 "watch_next": ["scheduled events, levels, or developing stories to watch in the next few hours"],
 "risks": ["what could flip the regime"],
 "for_day_traders": "2-3 sentences: trend day or chop, which direction has the edge, what to avoid"}}
Give 3-6 drivers."""
    return llm.chat_json(prompt, max_tokens=2500, kind="brief", timeout=120)

# ============================== BUILD ==============================

def build(use_claude=True, force_push=False):
    t0 = time.time()
    snap = snapshot()
    rules = rule_regime(snap)
    heads = top_headlines(fetch_all(LOOKBACK_MIN))
    trending = stocktwits_trending()
    ai = claude_brief(snap, rules, heads, trending) if use_claude else None
    regime = (ai or {}).get("regime") or rules["label"]
    prev = load("brief.json", {})
    for h in heads:
        h["outlets"] = sorted(h["outlets"])
        h.pop("words", None); h.pop("hitset", None)
    save("brief.json", {"time": now().isoformat(), "regime": regime, "snap": snap, "rules": rules,
                        "heads": heads, "trending": trending, "ai": ai})
    log(f"[brief] {regime} (rules {rules['score']:+d}) | {len(heads)} stories | "
        f"{('AI: ' + llm.label()) if ai else 'rules only'} | {time.time() - t0:.0f}s")

    t = now()
    scheduled = any(abs((t - t.replace(hour=int(p[:2]), minute=int(p[3:]))).total_seconds()) < BRIEF_MINUTES * 30
                    for p in PUSH_TIMES)
    changed = prev.get("regime") and prev["regime"] != regime and prev.get("time", "")[:10] == t.strftime("%Y-%m-%d")
    if force_push or scheduled or changed:
        spy = snap.get("SPY", {})
        lines = [(ai or {}).get("one_liner") or "; ".join(r for _, r in rules["reasons"][:3]) or "No strong signals."]
        if spy:
            lines.append(f"SPY {spy['last']:.2f} ({spy['chg']:+.2f}%)" + (f", VWAP {spy['vwap']:.2f}" if "vwap" in spy else ""))
        for d in (ai or {}).get("drivers", [])[:3]:
            lines.append(f"- {d.get('headline')}: {d.get('direction')}")
        if ai and ai.get("for_day_traders"):
            lines.append(ai["for_day_traders"])
        notify(f"{'Regime change: ' if changed else ''}{regime} · {t:%H:%M}", "\n".join(lines),
               tags="chart_with_upwards_trend")
    return regime
