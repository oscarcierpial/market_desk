"""
Shared pieces for news_engine.py (alerts) and market_brief.py (summaries):
feeds, watchlist, keyword scoring, and a fetcher.
"""
import calendar, os, re, time
from concurrent.futures import ThreadPoolExecutor

import feedparser
import requests

UA = os.environ.get("NEWS_UA", "Mozilla/5.0 (personal news-engine; you@example.com)")  # SEC asks for a contact email

WATCHLIST = {  # ticker -> names that count as a mention
    "SPY": ["s&p 500", "stocks"], "QQQ": ["nasdaq"],
    "NVDA": ["nvidia"], "TSLA": ["tesla"], "AAPL": ["apple"], "AMZN": ["amazon"],
    "INTC": ["intel"], "IONQ": ["ionq"], "BB": ["blackberry"], "ONDS": ["ondas"],
    "META": ["meta platforms"], "AMD": ["advanced micro devices"],
}

FEEDS = {  # name: (url, source_bonus)  official sources get a bonus
    # Official sources
    "Federal Reserve":  ("https://www.federalreserve.gov/feeds/press_all.xml", 3),
    "White House":      ("https://www.whitehouse.gov/news/feed/", 2),
    "SEC press":        ("https://www.sec.gov/news/pressreleases.rss", 1),
    "BEA (econ data)":  ("https://apps.bea.gov/rss/rss.xml", 1),
    "EIA (energy)":     ("https://www.eia.gov/rss/todayinenergy.xml", 0),
    # Financial press
    "CNBC Top":         ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", 0),
    "CNBC Economy":     ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258", 0),
    "CNBC World":       ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362", 0),
    "CNBC Tech":        ("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910", 0),
    "MarketWatch":      ("https://feeds.content.dowjones.io/public/rss/mw_topstories", 0),
    "Yahoo Finance":    ("https://finance.yahoo.com/news/rssindex", 0),
    "Seeking Alpha":    ("https://seekingalpha.com/market_currents.xml", 0),
    "Benzinga":         ("https://www.benzinga.com/feed", 0),
    "Investing.com":    ("https://www.investing.com/rss/news_25.rss", 0),
    "BBC Business":     ("https://feeds.bbci.co.uk/news/business/rss.xml", 0),
    # Wire services via Google News (their own RSS feeds were discontinued)
    "Reuters":          ("https://news.google.com/rss/search?q=site:reuters.com+when:2h&hl=en-US&gl=US&ceid=US:en", 0),
    "Bloomberg":        ("https://news.google.com/rss/search?q=site:bloomberg.com+when:2h&hl=en-US&gl=US&ceid=US:en", 0),
    "AP":               ("https://news.google.com/rss/search?q=site:apnews.com+business+when:2h&hl=en-US&gl=US&ceid=US:en", 0),
    # Topic searches and your watchlist
    "Yahoo watchlist":  ("https://feeds.finance.yahoo.com/rss/2.0/headline?s="
                         + ",".join(WATCHLIST) + "&region=US&lang=en-US", 0),
    "Google: geopolitics": ("https://news.google.com/rss/search?q=Hormuz+OR+Iran+OR+tariff+OR+sanctions"
                            "+when:1h&hl=en-US&gl=US&ceid=US:en", 0),
    "Google: Fed/China":   ("https://news.google.com/rss/search?q=%22Federal+Reserve%22+OR+Xi+OR+%22trade+deal%22"
                            "+when:1h&hl=en-US&gl=US&ceid=US:en", 0),
}

# A big catalyst is usually TOPIC + ACTION ("Hormuz" + "reopens"). Topics alone are
# background noise on days like today when every story mentions Xi or Iran.
TOPICS = ["hormuz", "iran", "israel", "xi", "china", "taiwan", "russia", "tariff", "tariffs",
          "sanctions", "federal reserve", "fed", "fomc", "warsh", "williams", "hammack", "opec",
          "oil", "treasury yields", "inflation", "cpi", "pce", "payrolls", "jobs report", "nuclear"]
ACTIONS = ["deal reached", "reach deal", "reached a deal", "agreement", "signed into law",
           "signs", "reopen", "reopens", "reopened", "ceasefire", "collapse", "collapses",
           "talks fail", "breakdown", "strike", "strikes", "airstrike", "missile", "attack",
           "invasion", "blockade", "rate hike", "rate cut", "hikes rates", "cuts rates",
           "raises rates", "emergency", "export ban", "ban", "halted", "trading halt",
           "bankruptcy", "default", "shutdown", "announces", "confirms", "executive order",
           "retaliates", "retaliation", "escalat"]
COMPANY_EVENTS = ["acquisition", "acquire", "acquires", "merger", "buyout", "fda", "guidance",
                  "downgrade", "upgrade", "recall", "investigation", "antitrust", "offering"]
MOVERS = ["beats", "misses", "record", "surge", "surges", "plunge", "plunges", "soars",
          "tumbles", "crash", "rally", "selloff", "jumps", "sinks"]
NOISE = ["photos", "video", "watch", "menu", "pandas", "lip reader", "opinion", "what to know",
         "here's how", "explainer", "podcast", "spouses", "dinner", "protest", "protesters"]
URGENCY = ["breaking", "just in", "urgent", "flash", "alert"]


def norm(title):
    t = re.sub(r"\s+-\s+[^-]+$", "", title.lower())   # drop " - Reuters" style suffixes
    return re.sub(r"[^a-z0-9 ]", " ", t).strip()

def has(term, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text) is not None

def keyword_score(title, summary, source_bonus):
    text = f"{title} {summary}".lower()
    tl = title.lower()
    topics = [t for t in TOPICS if has(t, text)]
    actions = [a for a in ACTIONS if (has(a, tl) if a != "escalat" else "escalat" in tl)]
    score = 2 * min(len(topics), 2) + 4 * min(len(actions), 2)
    if topics and actions:
        score += 4                        # the combo is what matters
    tickers = set()
    for tkr, names in WATCHLIST.items():
        if (len(tkr) > 2 and has(tkr.lower(), text)) or any(has(n, text) for n in names):
            tickers.add(tkr)
    named = tickers - {"SPY", "QQQ"}
    events = [c for c in COMPANY_EVENTS if has(c, tl)]
    if named:
        score += 3 + (4 if events else 0)  # watchlist stock with a real event
    movers = [m for m in MOVERS if has(m, tl)]
    score += min(len(movers), 2)
    if any(has(u, tl) for u in URGENCY):
        score += 3
    if any(has(n, tl) for n in NOISE) or re.match(r"why .* (stock|shares)", tl):
        score -= 8                         # fluff, or an explainer written after the move
    hits = topics[:3] + actions[:3] + events[:2] + movers[:2]
    return score + source_bonus, hits, sorted(tickers)


def fetch_feed(url, timeout=12):
    """Download one feed with a timeout (feedparser alone can hang) and return its entries."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
        return feedparser.parse(r.content).entries
    except Exception:
        return []

def entry_time(e):
    for k in ("published_parsed", "updated_parsed"):
        if e.get(k):
            return calendar.timegm(e[k])
    return time.time()

def fetch_all(max_age_min=180):
    """Pull every feed in parallel; return recent headlines, newest first, with scores."""
    def one(item):
        name, (url, bonus) = item
        out = []
        for e in fetch_feed(url):
            title = (e.get("title") or "").strip()
            ts = entry_time(e)
            if not title or (time.time() - ts) / 60 > max_age_min:
                continue
            summary = re.sub("<[^>]+>", " ", e.get("summary", ""))[:400]
            score, hits, tickers = keyword_score(title, summary, bonus)
            out.append(dict(title=title, summary=summary, source=name, link=e.get("link", ""),
                            ts=ts, score=score, hits=hits, tickers=tickers))
        return out
    with ThreadPoolExecutor(max_workers=12) as pool:
        items = [h for batch in pool.map(one, FEEDS.items()) for h in batch]
    return sorted(items, key=lambda h: h["ts"], reverse=True)
