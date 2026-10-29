"""
The stock universe: S&P 500 + Nasdaq-100 (from Wikipedia, refreshed weekly) plus a list of
popular, optionable momentum names that aren't in either index. Also finds tickers mentioned
in headlines so news catalysts can feed the scanner.
"""
import io, re, time

import pandas as pd
import requests

from common import load, save, log

# Liquid, optionable names outside the two indexes that traders watch. Edit freely.
EXTRAS = {
    "IONQ": "IonQ", "RGTI": "Rigetti", "QBTS": "D-Wave", "RKLB": "Rocket Lab", "NBIS": "Nebius",
    "SOFI": "SoFi", "HOOD": "Robinhood", "ONDS": "Ondas", "BB": "BlackBerry", "CRWV": "CoreWeave",
    "SMCI": "Super Micro", "MSTR": "MicroStrategy", "COIN": "Coinbase", "GME": "GameStop",
    "RIVN": "Rivian", "LCID": "Lucid", "NIO": "NIO", "BABA": "Alibaba", "SNOW": "Snowflake",
    "SHOP": "Shopify", "UBER": "Uber", "OKLO": "Oklo", "HIMS": "Hims & Hers", "RDDT": "Reddit",
    "AFRM": "Affirm", "UPST": "Upstart", "MARA": "MARA Holdings", "RIOT": "Riot Platforms",
    "CVNA": "Carvana", "DKNG": "DraftKings", "ROKU": "Roku", "SNAP": "Snap", "NET": "Cloudflare",
    "U": "Unity", "TSM": "Taiwan Semiconductor", "ASML": "ASML", "ARM": "Arm Holdings",
    "NU": "Nu Holdings", "GRAB": "Grab", "NOK": "Nokia", "AAL": "American Airlines", "F": "Ford",
    "RCL": "Royal Caribbean", "CCL": "Carnival", "NCLH": "Norwegian Cruise", "FSLR": "First Solar",
    "BE": "Bloom Energy", "PLUG": "Plug Power", "VKTX": "Viking Therapeutics", "CRML": "Critical Metals",
    "RDW": "Redwire", "APP": "AppLovin", "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "IWM": "Russell 2000 ETF",
}
# Company names that are also ordinary words; never match these as names (tickers still work).
AMBIGUOUS = {"target", "visa", "match", "public", "block", "general", "progressive", "ball", "fox",
             "gap", "apple hospitality", "best", "global", "american", "united", "first", "ford",
             "news", "nu", "arm", "snap", "u", "f", "nio", "grab", "carnival", "unity", "oracle bone"}
SUFFIX = re.compile(r"\b(inc|incorporated|corp|corporation|company|co|holdings|holding|group|plc|ltd|"
                    r"limited|n\.?v|s\.?a|class [abc]|the)\b\.?", re.I)

_name_re = None
_name_map = {}


def _wiki_table(url, want_cols):
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c) for c in t.columns]
        if all(any(w.lower() in c.lower() for c in cols) for w in want_cols):
            return t
    return None


def build(force=False):
    """Refresh the universe once a week. Falls back to the saved copy or EXTRAS on failure."""
    u = load("universe.json", {})
    if not force and u.get("tickers") and time.time() - u.get("built", 0) < 7 * 86400:
        return u["tickers"]
    tickers = {}
    try:
        sp = _wiki_table("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ["symbol", "security"])
        for _, r in sp.iterrows():
            tickers[str(r["Symbol"]).replace(".", "-")] = str(r["Security"])
        ndx = _wiki_table("https://en.wikipedia.org/wiki/Nasdaq-100", ["ticker", "company"])
        if ndx is not None:
            tcol = [c for c in ndx.columns if "ticker" in str(c).lower()][0]
            ccol = [c for c in ndx.columns if "company" in str(c).lower()][0]
            for _, r in ndx.iterrows():
                tickers.setdefault(str(r[tcol]).replace(".", "-"), str(r[ccol]))
    except Exception as e:
        log(f"[universe] Wikipedia fetch failed ({e}); using saved list")
        tickers = u.get("tickers", {})
    for t, n in EXTRAS.items():
        tickers.setdefault(t, n)
    save("universe.json", {"tickers": tickers, "built": time.time()})
    log(f"[universe] {len(tickers)} stocks")
    return tickers


def tickers():
    return load("universe.json", {}).get("tickers") or build()


def _short(name):
    n = SUFFIX.sub("", name.lower()).replace("&", "and")
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _compile():
    global _name_re, _name_map
    _name_map = {}
    for t, name in tickers().items():
        s = _short(name)
        if len(s) >= 4 and s not in AMBIGUOUS:
            _name_map.setdefault(s, t)
    alts = sorted(_name_map, key=len, reverse=True)
    _name_re = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in alts) + r")(?![a-z0-9])")


def extract_tickers(text):
    """Tickers mentioned via $CASHTAG, (NASDAQ: XYZ), or a known company name."""
    if _name_re is None:
        _compile()
    known = tickers()
    found = set()
    for m in re.finditer(r"\$([A-Z]{1,5})\b", text):
        if m.group(1) in known:
            found.add(m.group(1))
    for m in re.finditer(r"\((?:NYSE|NASDAQ|Nasdaq|NYSE American|AMEX)\s*:\s*([A-Z.]{1,6})\)", text):
        found.add(m.group(1).replace(".", "-"))
    low = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("&", "and"))
    for m in _name_re.finditer(low):
        found.add(_name_map[m.group(1)])
    return sorted(found - {"SPY", "QQQ", "IWM"})
