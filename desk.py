#!/usr/bin/env python3
"""
Market desk: one app that runs everything.

  every 45 s      news catalysts (all day, every day)
  8:00 a.m.       universe refresh (weekly) + pre-market scan of ~560 stocks
  every 20 min    watchlist refresh from market-wide screens, news and Stocktwits (9:00-3:30)
  every 15 min    market brief: bullish / bearish / stable (8:30-4:15)
  every 60 s      setup scanner on the watchlist (9:45-3:55; alerts only inside the windows)
  10:00 a.m.      swing scan (30-60 DTE options; background thread; weekly backtest first if stale)
  3:50 p.m.       swing tracker marks open picks
  4:10 p.m.       grade today's setups, push the day's scorecard
  after each      redraw desk.html

python desk.py             run it (the RUN launcher does this)
python desk.py --test      run every step once right now, ignoring the clock, then exit
python desk.py --dry-run   print alerts instead of sending them
"""
import argparse, sys, time, traceback

import common
from common import now, hhmm, log, notify, BRIEF_MINUTES


def safe(name, fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as ex:
        tb = traceback.format_exc().strip().splitlines()
        log(f"[{name}] error: {ex!r}")
        for line in tb[-4:]:
            log(f"[{name}]   {line.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run each step once now and exit")
    ap.add_argument("--dry-run", action="store_true", help="print alerts instead of sending")
    ap.add_argument("--no-claude", action="store_true", help="never call any AI provider")
    args = ap.parse_args()
    common.DRY_RUN = args.dry_run or args.test

    import universe, news_engine, market_brief, discovery, scanner, journal, dashboard, swing

    use_claude = not args.no_claude
    log(f"Market desk starting (budget ${common.BUDGET:.0f}, alerts at score {common.MIN_ALERT_SCORE}+"
        f"{', DRY RUN' if common.DRY_RUN else ''})")
    safe("universe", universe.build)
    safe("grader", journal.grade_pending)

    if args.test:
        safe("news", news_engine.run_once, warmup=news_engine.first_run(), verbose=False)
        safe("premarket", discovery.premarket_scan)
        safe("watchlist", discovery.refresh_watchlist)
        safe("brief", market_brief.build, use_claude=use_claude, force_push=True)
        safe("swing", swing.daily, push=False, run_backtest=not common.load("swing_backtest.json", {}))
        found = safe("scanner", scanner.run_scan) or []
        log(f"[test] scanner found {len(found)} new setups")
        safe("dashboard", dashboard.render)
        log(f"[test] done. Open {dashboard.OUT}")
        return

    last = {}
    def due(key, every_s):
        if time.time() - last.get(key, 0) >= every_s:
            last[key] = time.time()
            return True
        return False

    warm = news_engine.first_run()
    if not (830 <= hhmm() <= 1615):
        safe("brief", market_brief.build, use_claude=False)  # something to show outside market hours
    done_today = {}
    while True:
        t = now()
        day, hm, weekday = t.strftime("%Y-%m-%d"), hhmm(t), t.weekday() < 5
        redraw = False

        if due("news", 45):
            safe("news", news_engine.run_once, warmup=warm)
            warm = False

        if weekday and hm >= 800 and hm < 1600 and done_today.get("premarket") != day:
            done_today["premarket"] = day
            if common.load("levels.json", {}).get("date") != day:      # skip if already done today
                safe("universe", universe.build)
                safe("premarket", discovery.premarket_scan)

        if weekday and 900 <= hm <= 1530 and due("watchlist", 20 * 60):
            safe("watchlist", discovery.refresh_watchlist); redraw = True

        if weekday and 830 <= hm <= 1615 and due("brief", BRIEF_MINUTES * 60):
            safe("brief", market_brief.build, use_claude=use_claude); redraw = True

        if weekday and 945 <= hm <= 1555 and due("scan", 60):
            safe("scanner", scanner.run_scan); redraw = True

        if weekday and swing.SCAN_TIME <= hm < 1550 and done_today.get("swing") != day:
            done_today["swing"] = day
            if common.load("swing.json", {}).get("date") != day:          # skip if already scanned today
                swing.start(swing.daily)                                    # runs in the background
        if weekday and 1550 <= hm < 1600 and done_today.get("swing_mark") != day:
            done_today["swing_mark"] = day
            swing.start(swing.update)
            redraw = True

        if weekday and hm >= 1610 and done_today.get("grade") != day:
            done_today["grade"] = day
            safe("grader", journal.grade)
            s = safe("summary", journal.day_summary)
            if s:
                sc = journal.scorecard()
                notify(f"Scorecard {t:%b %d}",
                       f"Today: {s['n']} setups, {s['wins']} wins / {s['losses']} losses / {s['timeouts']} timeouts, "
                       f"{s['total_r']:+.1f}R total. Alerted only: {s['alerted_n']} setups, {s['alerted_r']:+.1f}R.\n"
                       f"All-time: {sc.get('n', 0)} setups, {sc.get('win_rate', 0)}% win rate, "
                       f"{sc.get('avg_r', 0):+.2f}R average.", tags="clipboard")
            redraw = True

        if redraw or due("redraw", 300):
            safe("dashboard", dashboard.render)
        time.sleep(5)


if __name__ == "__main__":
    main()
