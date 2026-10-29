"""
Picks the option contract for a setup, using the filter from our 0DTE plan:
delta 0.40-0.60, spread <= $0.03 or 5% of the price, real volume/open interest,
and a price that fits the budget. Returns None if nothing qualifies (then trade shares or skip).
"""
from datetime import datetime, timedelta
from math import erf, log as ln, sqrt

import yfinance as yf

from common import YF_LOCK, ET, now, env, BUDGET

MIN_DELTA, MAX_DELTA = env("MIN_DELTA", 0.40), env("MAX_DELTA", 0.60)
MAX_SPREAD_PCT = env("MAX_SPREAD_PCT", 0.05)
MIN_STRIKE_VOLUME = env("MIN_STRIKE_VOLUME", 300)
MIN_OPEN_INTEREST = env("MIN_OPEN_INTEREST", 500)
MIN_PREMIUM = env("MIN_PREMIUM", 0.20)
ZERO_DTE_CUTOFF = env("ZERO_DTE_CUTOFF", 1430)      # after this (HHMM), skip same-day expiries


def _int(x):
    try:
        return int(x) if x == x else 0      # NaN check
    except (TypeError, ValueError):
        return 0


def _delta(S, K, T, iv, call):
    if T <= 0 or iv <= 0:
        return (1.0 if S > K else 0.0) if call else (-1.0 if S < K else 0.0)
    d1 = (ln(S / K) + 0.5 * iv * iv * T) / (iv * sqrt(T))
    n = 0.5 * (1 + erf(d1 / sqrt(2)))
    return n if call else n - 1


def pick(ticker, direction, spot, stop, target):
    call = direction == "call"
    t = now()
    try:
        with YF_LOCK:
            tk = yf.Ticker(ticker)
            exps = list(tk.options)
    except Exception:
        return None
    today = t.strftime("%Y-%m-%d")
    exps = [e for e in exps if e >= today]
    if exps and exps[0] == today and t.hour * 100 + t.minute >= ZERO_DTE_CUTOFF:
        exps = exps[1:]
    best = None
    for exp in exps[:2]:                                   # nearest two expiries
        try:
            with YF_LOCK:
                chain = tk.option_chain(exp)
        except Exception:
            continue
        df = chain.calls if call else chain.puts
        close = datetime.strptime(exp, "%Y-%m-%d").replace(hour=16, tzinfo=ET)
        T = max((close - t).total_seconds(), 600) / (365 * 86400)
        for _, r in df.iterrows():
            bid, ask = float(r.bid or 0), float(r.ask or 0)
            if bid <= 0 or ask <= 0 or ask < MIN_PREMIUM or ask * 100 > BUDGET:
                continue
            mid, spread = (bid + ask) / 2, ask - bid
            if not (spread <= 0.03 or spread / mid <= MAX_SPREAD_PCT):
                continue
            vol, oi = _int(r.volume), _int(r.openInterest)
            if vol < MIN_STRIKE_VOLUME and oi < MIN_OPEN_INTEREST:
                continue
            iv = float(r.impliedVolatility) if r.impliedVolatility == r.impliedVolatility else 0.5
            d = _delta(spot, float(r.strike), T, iv if iv > 0.05 else 0.5, call)
            if not (MIN_DELTA <= abs(d) <= MAX_DELTA):
                continue
            rank = abs(abs(d) - 0.5) + spread / mid        # closest to 0.50 delta, tightest spread
            if best is None or rank < best["_rank"]:
                n = max(int(BUDGET // (mid * 100)), 1)
                best = {"_rank": rank, "expiry": exp, "strike": float(r.strike), "type": "call" if call else "put",
                        "bid": bid, "ask": ask, "mid": round(mid, 2), "delta": round(d, 2), "iv": round(iv, 2),
                        "volume": vol, "oi": oi, "contracts": n, "zero_dte": exp == today,
                        "stale": bid == ask,
                        "est_loss_each": round(abs(d) * abs(spot - stop) * 100),
                        "est_gain_each": round(abs(d) * abs(target - spot) * 100)}
        if best:
            break                                           # prefer the nearest expiry that qualifies
    if best:
        best.pop("_rank")
    return best


def describe(c):
    if not c:
        return "No contract passes the filter (spread, delta, volume or budget). Trade shares or skip."
    exp = datetime.strptime(c["expiry"], "%Y-%m-%d").strftime("%b %d")
    tag = " (0DTE)" if c["zero_dte"] else ""
    stale = " - quote may be stale, check your broker" if c["stale"] else ""
    return (f"{exp}{tag} ${c['strike']:g} {c['type']} ~${c['mid']:.2f} (bid {c['bid']:.2f}/ask {c['ask']:.2f}, "
            f"delta {c['delta']:+.2f}, vol {c['volume']:,}){stale}\n"
            f"Up to {c['contracts']} contract(s) in budget; per contract roughly "
            f"-${c['est_loss_each']} at stop / +${c['est_gain_each']} at target 1")
