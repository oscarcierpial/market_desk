"""Draws desk.html from the state files. Called after every scan, brief and watchlist refresh."""
import html, json, os
from datetime import datetime

from common import BASE, ET, load, now, MIN_ALERT_SCORE
import journal
import swing

OUT = os.path.join(BASE, "desk.html")
INDEXES = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Small caps", "DIA": "Dow"}
MACRO = {"^VIX": "VIX (fear gauge)", "^TNX": "10-yr yield", "CL=F": "Oil (WTI)",
         "DX-Y.NYB": "US dollar", "GC=F": "Gold", "BTC-USD": "Bitcoin"}
SECTORS = {"XLK": "Tech", "SMH": "Semis", "XLY": "Consumer disc.", "XLC": "Communication",
           "XLF": "Financials", "XLI": "Industrials", "XLE": "Energy", "XLB": "Materials",
           "XLV": "Health care", "XLP": "Staples", "XLU": "Utilities", "XLRE": "Real estate"}
e = html.escape

CSS = """
:root{--ink:#1c2430;--paper:#eef2f5;--card:#fff;--rule:#c9d2da;--mute:#5d6b7a;--bull:#1f7a5a;--bear:#b03a2e;--flat:#8a6d1f}
@media (prefers-color-scheme:dark){:root{--ink:#e3e9ef;--paper:#141a21;--card:#1c242d;--rule:#2e3a46;--mute:#93a1b0;--bull:#4fbf93;--bear:#e0695c;--flat:#d2b25a}}
*{box-sizing:border-box} body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 "Public Sans",system-ui,sans-serif;font-variant-numeric:tabular-nums}
header{padding:24px 20px 20px;border-bottom:6px solid var(--tone)}
header .when{color:var(--mute);font-size:14px} header h1{margin:4px 0;font-size:clamp(40px,9vw,72px);line-height:1;font-weight:800;color:var(--tone)}
header p{margin:6px 0 0;font-size:19px;max-width:62ch} .status{margin-top:10px;color:var(--mute);font-size:14px}
main{max-width:1150px;margin:0 auto;padding:20px;display:grid;gap:28px;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr)}
@media (max-width:860px){main{grid-template-columns:1fr}}
h2{font-size:15px;font-weight:800;margin:26px 0 10px;color:var(--mute)} main>div>h2:first-child,aside>h2:first-child{margin-top:0}
.setup{background:var(--card);border-left:5px solid var(--flat);border-radius:0 8px 8px 0;padding:14px 16px;margin-bottom:12px}
.setup.call{border-color:var(--bull)} .setup.put{border-color:var(--bear)} .setup.quiet{opacity:.72}
.setup h3{margin:0;font-size:19px} .setup .score{float:right;font-weight:800;font-size:19px}
.lv{display:flex;flex-wrap:wrap;gap:6px 18px;margin:8px 0;font-size:15px} .lv b{font-weight:600}
.setup p{margin:4px 0} .meta{color:var(--mute);font-size:13px;display:block}
.pill{display:inline-block;font-size:12px;font-weight:600;padding:1px 8px;border-radius:10px;border:1px solid currentColor;margin-left:6px;vertical-align:2px}
.win{color:var(--bull)} .loss{color:var(--bear)} .timeout{color:var(--flat)}
.lede{font-size:18px;margin:0 0 16px;max-width:65ch}
.drv{background:var(--card);border-left:4px solid var(--flat);padding:10px 14px;margin:0 0 8px;border-radius:0 6px 6px 0}
.drv.bullish{border-color:var(--bull)} .drv.bearish{border-color:var(--bear)} .drv h3{margin:0 0 2px;font-size:16px} .drv p{margin:0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px} @media (max-width:600px){.two{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:6px;overflow:hidden}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;vertical-align:top} th{text-align:left;font-weight:600}
td.l{text-align:left;color:var(--mute);font-size:13px} .tk{color:var(--mute);font-weight:400;font-size:12px;margin-left:6px}
.sub{display:block;color:var(--mute);font-weight:400;font-size:12px} .up{color:var(--bull)} .dn{color:var(--bear)}
.bar{display:grid;grid-template-columns:110px 1fr 60px;align-items:center;gap:8px;font-size:14px;margin:3px 0}
.bar i{display:block;height:10px;border-radius:2px;background:currentColor;opacity:.75} .bar b{text-align:right;font-weight:600} .bar span{color:var(--ink)}
ul{padding-left:18px;margin:0} li{margin:0 0 7px} .plain li{list-style:none;margin-left:-18px}
a{color:var(--ink)} .news a{text-decoration:none;font-weight:600} .news a:hover,.news a:focus{text-decoration:underline}
a:focus-visible{outline:2px solid var(--tone);outline-offset:2px} .empty{background:var(--card);padding:16px;border-radius:8px;color:var(--mute)}
footer{grid-column:1/-1;color:var(--mute);font-size:13px}
.kv{display:grid;grid-template-columns:minmax(130px,max-content) minmax(0,1fr);gap:3px 14px;font-size:14px;margin:10px 0 0}
.kv dt{color:var(--mute)} .kv dd{margin:0;overflow-wrap:anywhere}
"""


def _fmt(v, t):
    return f"{v:.2f}%" if t == "^TNX" else f"{v:,.0f}" if t == "BTC-USD" else f"{v:,.2f}"


def _mkt_row(snap, t, name):
    r = snap.get(t)
    if not r:
        return ""
    extra = f"<span class=sub>VWAP {r['vwap']:.2f} · range {r['lo']:.2f}–{r['hi']:.2f}</span>" if "vwap" in r else ""
    if t == "^TNX":
        bp = 100 * (r["last"] - r["prev"]); cls = "dn" if bp > 0 else "up"; ch = f"{bp:+.0f} bp"
    else:
        cls = "up" if r["chg"] > 0 else "dn"
        if t == "^VIX":
            cls = "dn" if r["chg"] > 0 else "up"
        ch = f"{r['chg']:+.2f}%"
    return f"<tr><th>{e(name)}<span class=tk>{e(t)}</span>{extra}</th><td>{_fmt(r['last'], t)}</td><td class={cls}>{ch}</td></tr>"


def _setup_card(s, prices):
    c = json.loads(s["contract"]) if s.get("contract") else None
    when = datetime.fromisoformat(s["ts"]).strftime("%I:%M %p").lstrip("0")
    status = ""
    if s.get("outcome"):
        status = f"<span class='pill {s['outcome']}'>{s['outcome']} {s['r_multiple']:+.1f}R</span>"
    elif s["ticker"] in prices:
        p = prices[s["ticker"]]
        live = "call" if s["direction"] == "call" else "put"
        hit_stop = p <= s["stop"] if live == "call" else p >= s["stop"]
        hit_t1 = p >= s["t1"] if live == "call" else p <= s["t1"]
        status = (f"<span class='pill loss'>at stop</span>" if hit_stop else
                  f"<span class='pill win'>at target</span>" if hit_t1 else f"<span class=pill>now {p:.2f}</span>")
    contract = ""
    if c:
        exp = datetime.strptime(c["expiry"], "%Y-%m-%d").strftime("%b %d")
        contract = (f"<p>{exp}{' (0DTE)' if c['zero_dte'] else ''} ${c['strike']:g} {c['type']} ~${c['mid']:.2f} "
                    f"· delta {c['delta']:+.2f} · spread {c['ask'] - c['bid']:.2f} · up to {c['contracts']} in budget</p>")
    elif s["alerted"]:
        contract = "<p class=meta>No option passed the filter; shares only.</p>"
    news_line = f"<p class=meta>News: {e(s['catalyst'][:140])}</p>" if s.get("catalyst") else ""
    arrow = "calls" if s["direction"] == "call" else "puts"
    quiet = "" if s["alerted"] else " quiet"
    tag = "alerted" if s["alerted"] else f"logged only (score under {MIN_ALERT_SCORE} or outside trading window)"
    return f"""<article class='setup {s['direction']}{quiet}'><span class=score>{s['score']}/10</span>
<h3>{e(s['ticker'])} · {e(s['setup'])} ({arrow}){status}</h3>
<div class=lv><span>Trigger <b>{s['trigger']:.2f}</b></span><span>Stop <b>{s['stop']:.2f}</b></span>
<span>Targets <b>{s['t1']:.2f}</b> / {s['t2']:.2f}</span><span>R/R <b>{(s['rr'] or 0):.1f}</b></span></div>
<p>{e(s['why'])}</p>{contract}
{news_line}
<span class=meta>{when} · {tag}</span></article>"""


def _score_table(rows, key, label):
    if not rows:
        return ""
    body = "".join(f"<tr><th>{e(str(r[key]))}</th><td>{r['n']}</td><td>{r['win_rate']:.0f}%</td>"
                   f"<td class={'up' if r['avg_r'] > 0 else 'dn'}>{r['avg_r']:+.2f}R</td></tr>" for r in rows)
    return f"<table><tr><th>{label}</th><td>n</td><td>win</td><td>avg</td></tr>{body}</table>"


def render():
    b = load("brief.json", {})
    snap, rules, ai = b.get("snap", {}), b.get("rules", {"label": "Waiting", "score": 0, "reasons": []}), b.get("ai")
    regime = b.get("regime") or "Waiting for data"
    tone = "var(--bull)" if regime.lower().startswith("bull") else "var(--bear)" if regime.lower().startswith("bear") else "var(--flat)"
    st = load("scan_status.json", {})
    prices = st.get("prices", {})
    wl = load("watchlist.json", {}).get("items", [])
    t = now()
    today = t.strftime("%Y-%m-%d")
    sigs = journal.today_signals()
    alerted = [s for s in sigs if s["alerted"]]
    logged = [s for s in sigs if not s["alerted"]]
    sc = journal.scorecard()
    cats = {k: v for k, v in load("catalysts.json", {}).items() if v.get("date") == today}
    alerts = [a for a in load("alerts_today.json", []) if a.get("date") == today]

    one = (ai or {}).get("one_liner") or "; ".join(r for _, r in rules.get("reasons", [])[:3])
    scan_t = datetime.fromisoformat(st["time"]).strftime("%I:%M %p").lstrip("0") if st.get("time") else "not yet"
    status = (f"Last scan {scan_t} · watching {st.get('watching', 0)} stocks · SPY {'above' if st.get('bias') == 'up' else 'below' if st.get('bias') == 'down' else '?'} VWAP · "
              f"{'alerts on' if st.get('in_window') else 'outside alert windows (still logging)'}")

    setups_html = "".join(_setup_card(s, prices) for s in alerted) or \
        "<div class=empty>No alert-quality setups yet today. The scanner checks every minute from 9:45.</div>"
    if logged:
        setups_html += f"<details><summary class=meta style='cursor:pointer;margin:8px 0'>{len(logged)} lower-scoring setup{'s' if len(logged) != 1 else ''} logged for the scorecard</summary>" + \
                       "".join(_setup_card(s, prices) for s in logged[:15]) + "</details>"

    if ai:
        drivers = "".join(f"<article class='drv {e(d.get('direction', 'mixed'))}'><h3>{e(d.get('headline', ''))}</h3>"
                          f"<p>{e(d.get('what_it_means', ''))}</p><span class=meta>{e(d.get('direction', ''))}"
                          f"{'' if d.get('confirmed') else ' · unconfirmed'} · {e(', '.join(d.get('affects', [])))}</span></article>"
                          for d in ai.get("drivers", []))
        lst = lambda k: "".join(f"<li>{e(x)}</li>" for x in ai.get(k, []))
        brief_html = f"""<h2>Market brief</h2><p class=lede>{e(ai.get('summary', ''))}</p>{drivers}
<div class=two><div><h2>For day traders</h2><p>{e(ai.get('for_day_traders', ''))}</p></div>
<div><h2>Getting attention, not moving prices</h2><p>{e(ai.get('not_moving', ''))}</p></div></div>
<div class=two><div><h2>Watch next</h2><ul>{lst('watch_next')}</ul></div><div><h2>What could flip it</h2><ul>{lst('risks')}</ul></div></div>"""
    else:
        brief_html = "<h2>Market brief</h2><p class=meta>Add an AI key in config.env (free Gemini or Claude) for a written explanation of the headlines. The market read above is rule-based.</p>"

    heads = b.get("heads", [])[:20]
    news = "".join(f"<li><a href='{e(h['link'])}' target=_blank rel=noopener>{e(h['title'])}</a><span class=meta>{e(h['source'])}"
                   f"{' +' + str(len(h['outlets']) - 1) + ' more' if len(h.get('outlets', [])) > 1 else ''} · "
                   f"{datetime.fromtimestamp(h['ts'], ET):%H:%M}</span></li>" for h in heads)
    watch = "".join(f"<tr><th>{e(w['ticker'])}</th><td>{prices.get(w['ticker'], '')}</td>"
                    f"<td class=l>{e('; '.join(w['why'])[:110])}</td></tr>" for w in wl)
    secs = sorted((s for s in SECTORS if s in snap), key=lambda s: snap[s]["chg"], reverse=True)
    sec_html = "".join(f"<div class='bar {'up' if snap[s]['chg'] > 0 else 'dn'}'><span>{e(SECTORS[s])}</span>"
                       f"<i style='width:{min(abs(snap[s]['chg']) * 30, 100):.0f}%'></i><b>{snap[s]['chg']:+.2f}%</b></div>" for s in secs)
    reasons = "".join("<li class={}>{:+d} {}</li>".format("up" if p > 0 else "dn", p, e(r)) for p, r in rules.get("reasons", []))
    if sc.get("n"):
        warn = " Small sample: treat as early evidence." if sc["n"] < 30 else ""
        score_html = (f"<p class=meta>{sc['n']} graded setups over {sc['days']} day(s): {sc['win_rate']}% hit target 1 first, "
                      f"average {sc['avg_r']:+.2f}R per setup. Alerted only: {sc['alerted_n']} setups, "
                      f"{(sc['alerted_avg_r'] if sc['alerted_avg_r'] is not None else 0):+.2f}R average.{warn}</p>"
                      + _score_table(sc["by_setup"], "setup", "Setup") + "<div style='height:10px'></div>"
                      + _score_table(sc["by_time"], "bucket", "Time of day"))
    else:
        score_html = "<p class=meta>Setups get graded after each close. Win rates appear here once there's data.</p>"
    try:
        swing_html = swing.render_html(e)
    except Exception as ex:
        swing_html = f"<h2>Swing trades</h2><p class=meta>Swing section failed to draw: {e(repr(ex))}</p>"
    cat_html = "".join(f"<li><b>{e(k)}</b> {e(v['headline'][:110])} <span class=meta>{v['time']}</span></li>" for k, v in cats.items()) \
        or "<li class=meta>None yet today.</li>"
    alert_html = "".join(f"<li><a href='{e(a['link'])}' target=_blank rel=noopener>{e(a['title'])}</a><span class=meta>{a['time']}</span></li>" for a in alerts[::-1]) \
        or "<li class=meta>No catalyst alerts yet today.</li>"

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<meta http-equiv=refresh content=60><title>{e(regime)} · desk {t:%H:%M}</title>
<link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;800&display=swap" rel=stylesheet>
<style>{CSS} :root{{--tone:{tone}}}</style></head><body>
<header><div class=when>{t:%A %b %d, %I:%M %p} ET · page refreshes every minute</div><h1>{e(regime)}</h1>
<p>{e(one)}</p><div class=status>{e(status)}</div></header>
<main><div><h2>Setups today</h2>{setups_html}{swing_html}{brief_html}
<h2>Catalyst alerts today</h2><ul class='plain news'>{alert_html}</ul>
<h2>Top headlines</h2><ul class='plain news'>{news}</ul></div>
<aside><h2>Markets</h2><table>{''.join(_mkt_row(snap, k, v) for k, v in INDEXES.items())}{''.join(_mkt_row(snap, k, v) for k, v in MACRO.items())}</table>
<h2>Scanner watchlist</h2><table>{watch or '<tr><td class=l>Builds at 9:00 a.m. ET.</td></tr>'}</table>
<h2>Scorecard</h2>{score_html}
<h2>Stocks with news today</h2><ul class=plain>{cat_html}</ul>
<h2>Sectors</h2>{sec_html}
<h2>Why the rules say {e(rules['label'].lower())} ({rules['score']:+d})</h2><ul class=plain>{reasons}</ul></aside>
<footer>Free data, about a minute delayed. Setups are screens, not recommendations: confirm on your own chart. Not investment advice.</footer></main></body></html>"""
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(doc)
    os.replace(tmp, OUT)
