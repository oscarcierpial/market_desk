"""
Catalyst alerts: checks 21 feeds, scores each new headline, merges the same story across
outlets, optionally asks Claude for a second opinion, and pushes big catalysts to your phone.
Also records which stocks have news today so the scanner can watch them.
"""
import difflib, hashlib, json, os, re, sqlite3, sys, time

import requests

from common import path, load, save, now, log, notify
from sources import FEEDS, fetch_feed, entry_time, keyword_score, norm
import universe
import llm

MAX_AGE_MIN = 45             # ignore stories older than this
ALERT_SCORE = 14             # keyword score that alerts on its own
LLM_GATE_SCORE = 9           # score that earns a Claude review (if API key set)
LLM_ALERT_SCORE = 7          # Claude's 1-10 rating needed to alert
CATALYST_SCORE = 7           # score for a stock-specific headline to put that stock on the scanner
CLUSTER_WINDOW_MIN = 20

DB = sqlite3.connect(path("news_seen.db"), check_same_thread=False)
DB.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY, ts REAL)")
DB.execute("CREATE TABLE IF NOT EXISTS alerted (cluster TEXT PRIMARY KEY, ts REAL)")
_recent = []  # (ts, normalized_title, source, cluster_id, keyword_set)


def _seen(item_id):
    if DB.execute("SELECT 1 FROM seen WHERE id=?", (item_id,)).fetchone():
        return True
    DB.execute("INSERT INTO seen VALUES (?,?)", (item_id, time.time()))
    return False


def _cluster(title_n, source, hits):
    t_now, keys = time.time(), set(hits)
    _recent[:] = [r for r in _recent if t_now - r[0] < CLUSTER_WINDOW_MIN * 60]
    cid = None
    for ts, t, src, c, k in _recent:
        overlap = len(keys & k)
        if (difflib.SequenceMatcher(None, t, title_n).ratio() > 0.6
                or (overlap >= 2 and overlap / max(len(keys | k), 1) >= 0.5)):
            cid = c
            break
    cid = cid or hashlib.md5(title_n.encode()).hexdigest()[:10]
    _recent.append((t_now, title_n, source, cid, keys))
    return cid, len({r[2] for r in _recent if r[3] == cid})


def claude_rate(title, summary, source):
    """Second opinion from whichever AI provider is configured (Claude, Gemini, ...)."""
    prompt = ("You are a markets news desk. Rate how likely this headline is to move the S&P 500 "
              "or the named stocks by >1% today. Consider whether it is NEW, CONFIRMED (vs rumor), "
              "and SURPRISING. Reply with JSON only: "
              '{"score": 1-10, "direction": "bullish|bearish|mixed", "tickers": [..], '
              '"confirmed": true|false, "why": "<20 words"}\n\n'
              f"Source: {source}\nHeadline: {title}\nSummary: {summary[:600]}")
    return llm.chat_json(prompt, max_tokens=300, kind="quick", timeout=30)


def _market_line(tickers):
    try:
        import yfinance as yf
        from common import YF_LOCK
        syms = (["SPY", "QQQ"] + [t for t in tickers if t not in ("SPY", "QQQ")])[:5]
        with YF_LOCK:
            d = yf.download(syms, period="1d", interval="1m", progress=False, group_by="ticker")
        out = []
        for t in syms:
            c = d[t]["Close"].dropna()
            if len(c) > 6:
                out.append(f"{t} {c.iloc[-1]:.2f} ({100 * (c.iloc[-1] / c.iloc[-6] - 1):+.2f}% 5m)")
        return " | ".join(out) or "no intraday data"
    except Exception:
        return "price check unavailable"


def _record_catalyst(tickers, title, score, link):
    """Stocks with real news today get handed to the scanner's watchlist."""
    today = now().strftime("%Y-%m-%d")
    cats = load("catalysts.json", {})
    cats = {t: v for t, v in cats.items() if v.get("date") == today}
    for t in tickers:
        if t not in cats or score > cats[t]["score"]:
            cats[t] = {"date": today, "headline": title, "score": score, "link": link,
                       "time": now().strftime("%H:%M")}
    save("catalysts.json", cats)


def run_once(warmup=False, verbose=False):
    candidates = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as pool:
        fetched = list(pool.map(lambda kv: (kv[0], kv[1][1], fetch_feed(kv[1][0])), FEEDS.items()))
    for name, bonus, entries in fetched:
        for e in entries:
            title = (e.get("title") or "").strip()
            if not title:
                continue
            item_id = hashlib.md5((e.get("link", "") + title).encode()).hexdigest()
            if _seen(item_id):
                continue
            if warmup or (time.time() - entry_time(e)) / 60 > MAX_AGE_MIN:
                continue
            summary = re.sub("<[^>]+>", " ", e.get("summary", ""))
            score, hits, tickers = keyword_score(title, summary, bonus)
            stock_tickers = universe.extract_tickers(title)
            cid, n_src = _cluster(norm(title), name, hits)
            score += 2 * min(n_src - 1, 3)
            if stock_tickers and score >= CATALYST_SCORE:
                _record_catalyst(stock_tickers, title, score, e.get("link", ""))
            if verbose and score >= 5:
                print(f"[{score:>2}] {name}: {title} {hits} {stock_tickers}")
            candidates.append((score, title, summary, name, e.get("link", ""), hits,
                               sorted(set(tickers) | set(stock_tickers)), cid, n_src))
    DB.commit()

    for score, title, summary, src, link, hits, tickers, cid, n_src in sorted(candidates, reverse=True):
        if DB.execute("SELECT 1 FROM alerted WHERE cluster=?", (cid,)).fetchone():
            continue
        verdict = claude_rate(title, summary, src) if score >= LLM_GATE_SCORE else None
        if not (score >= ALERT_SCORE or (verdict and verdict.get("score", 0) >= LLM_ALERT_SCORE)):
            continue
        DB.execute("INSERT OR IGNORE INTO alerted VALUES (?,?)", (cid, time.time())); DB.commit()
        lines = [f"Source: {src}" + (f" (+{n_src - 1} other outlets)" if n_src > 1 else ""),
                 f"Keyword score {score}: {', '.join(hits[:8])}"]
        if verdict:
            lines.append(f"{llm.label()}: {verdict.get('score')}/10 {verdict.get('direction')} "
                         f"{'CONFIRMED' if verdict.get('confirmed') else 'unconfirmed'} - {verdict.get('why')}")
            tickers = sorted(set(tickers) | set(verdict.get("tickers") or []))
            if verdict.get("tickers"):
                _record_catalyst([t for t in verdict["tickers"] if t in universe.tickers()], title, score, link)
        lines += ["Market: " + _market_line(tickers), link]
        notify(f"Catalyst: {title}", "\n".join(lines), priority="urgent", tags="rotating_light")
        alerts = load("alerts_today.json", [])
        alerts = [a for a in alerts if a.get("date") == now().strftime("%Y-%m-%d")]
        alerts.append({"date": now().strftime("%Y-%m-%d"), "time": now().strftime("%H:%M"),
                       "title": title, "link": link, "score": score, "tickers": tickers})
        save("alerts_today.json", alerts[-30:])


def first_run():
    return DB.execute("SELECT COUNT(*) FROM seen").fetchone()[0] == 0
