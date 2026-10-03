#!/usr/bin/env python3
"""Backtest the Spotify weekly-chart model against the market on settled weeks.

Usage: python3 markets/music_backtest.py [--grid] [--split]

For each settled market in markets/data/music_backfill.json and each point in its week
when 1..6 days of daily charts were public (a day's chart counted from 20:00 UTC the day
after), compare:
  ours:   music_model odds from the daily streams public at that moment
  market: each song's hourly price then (incl. "Other"), scaled to sum to 1
  blend:  50/50
Scores on the winner: log loss, Brier, top-pick hits; paper trades under the live rule
(edge >= 8 points, price 5-97c, $25; at the hourly price, so no spread: optimistic).
Candidates are the songs each market named (a mild look-ahead: markets add songs during
the week). Writes markets/data/music_backtest.json.
"""
import argparse, itertools, json, math, os, sys
from datetime import date, datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import music_model as mm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def price_at(hist, t):
    best = None
    for tt, p in hist or []:
        if tt <= t:
            best = p
        else:
            break
    return best


def rows_for(bf, params):
    rows = []
    tracks = bf["tracks"]
    # kworb keeps each song's last ~31 chart days: the full chart is covered from the
    # median first date among songs still charting
    latest = max((max(t["daily"]) for t in tracks.values() if t["daily"]), default="0")
    firsts = sorted(min(t["daily"]) for t in tracks.values() if t["daily"] and max(t["daily"]) >= latest)
    first = firsts[len(firsts) // 2] if firsts else "9999"
    for m in bf["markets"]:
        if not m.get("winner") or m["weekStart"] < first:
            continue  # need every day of the week in the daily history
        ws = date.fromisoformat(m["weekStart"])
        wi = next((i for i, s in enumerate(m["songs"]) if s["label"] == m["winner"]), None)
        if wi is None:
            continue
        cands = [s["trackId"] for s in m["songs"] if s.get("trackId")]
        for k in range(1, 7):
            known_until = ws + timedelta(days=k - 1)
            t = int(datetime(known_until.year, known_until.month, known_until.day, tzinfo=timezone.utc).timestamp()) + 86400 + 20 * 3600
            mk = [price_at(s.get("history"), t) for s in m["songs"]]
            mk = [0.0 if (x is None and s.get("other")) else x for x, s in zip(mk, m["songs"])]
            if any(x is None for x in mk) or sum(mk) <= 0:
                continue
            res, info = mm.predict(tracks, m["region"], ws, known_until, cands, params)
            if not res or not any(v["known"] > 0 for v in res.values()):
                continue  # no daily chart history for this week (kworb keeps ~31 days)
            ours = mm.market_probs(m, res, info, params)
            if ours[wi] is None:
                continue
            ours = [0.0 if x is None else x for x in ours]
            s = sum(ours) or 1
            ours = [x / s for x in ours]
            tot = sum(mk)
            rows.append({"slug": m["slug"], "region": m["region"], "rank": m["rank"], "k": k, "wi": wi,
                         "ours": ours, "mkt": [x / tot for x in mk], "raw": mk})
    return rows


def score(rows, key):
    if not rows:
        return {}
    ll = br = hit = 0.0
    for r in rows:
        p = r[key] if key != "blend" else [0.5 * a + 0.5 * b for a, b in zip(r["ours"], r["mkt"])]
        ll += -math.log(max(p[r["wi"]], 1e-3))
        br += sum((pi - (1.0 if i == r["wi"] else 0.0)) ** 2 for i, pi in enumerate(p))
        hit += max(range(len(p)), key=lambda i: p[i]) == r["wi"]
    n = len(rows)
    return {"n": n, "logloss": round(ll / n, 3), "brier": round(br / n, 3), "top_hit": round(hit / n, 3)}


def paper(rows, key):
    seen, trades, pnl, wins = set(), 0, 0.0, 0
    for r in rows:
        if r["slug"] in seen:
            continue
        p = r["ours"] if key == "ours" else [0.5 * a + 0.5 * b for a, b in zip(r["ours"], r["mkt"])]
        best = None
        for i, (pi, price) in enumerate(zip(p, r["raw"])):
            if 0.05 <= price <= 0.97 and pi - price >= 0.08 and (best is None or pi - price > best[0]):
                best = (pi - price, i, price)
        if best:
            seen.add(r["slug"])
            trades += 1
            won = best[1] == r["wi"]
            wins += won
            pnl += (25 / best[2] if won else 0) - 25
    return {"trades": trades, "wins": wins, "pnl": round(pnl, 2), "roi": round(pnl / (25 * trades), 3) if trades else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    bf = json.load(open(os.path.join(D, "music_backfill.json")))
    ms = sorted(bf["markets"], key=lambda m: m["weekStart"])
    print(f"{len(ms)} settled markets with a winner")
    tune, test = ms, ms
    if a.split:
        cut = ms[len(ms) // 2]["weekStart"]
        tune = [m for m in ms if m["weekStart"] < cut]
        test = [m for m in ms if m["weekStart"] >= cut]
        print(f"tuning on {len(tune)} markets before {cut}, testing on {len(test)}")
    P = dict(mm.DEFAULT)
    if a.grid:
        best = None
        for sg, dc in itertools.product([0.03, 0.06, 0.1, 0.16, 0.25], [0.88, 0.93, 0.97]):
            Q = dict(P, sigma=sg, decay=dc, sims=1500)
            sc = score(rows_for({"markets": tune, "tracks": bf["tracks"]}, Q), "ours")
            if sc and (best is None or sc["logloss"] < best[0]):
                best = (sc["logloss"], Q)
        if best is None:  # nothing scorable in the tuning set: keep the defaults
            print("grid: no scorable tuning markets; keeping the default parameters")
        else:
            P = dict(best[1], sims=4000)
            print("best by log loss on the tuning set:", best)
    rows = rows_for({"markets": test, "tracks": bf["tracks"]}, P)
    print("params", P)
    for k in ("ours", "mkt", "blend"):
        print(f"  {k:6s}", score(rows, k))
    for k in range(1, 7):
        rk = [r for r in rows if r["k"] == k]
        print(f"  {k} day(s) counted: ours {score(rk, 'ours')}  market {score(rk, 'mkt')}  paper {paper(rk, 'ours')}")
    by = {}
    for reg, rank in itertools.product(("global", "us"), (1, 2)):
        rr = [r for r in rows if r["region"] == reg and r["rank"] == rank]
        by[f"{reg}_{rank}"] = {"ours": score(rr, "ours"), "market": score(rr, "mkt"), "paper": paper(rr, "ours")}
        print(f"  {reg} #{rank}: ours {by[f'{reg}_{rank}']['ours']}  market {by[f'{reg}_{rank}']['market']}  paper {by[f'{reg}_{rank}']['paper']}")
    print("  paper all: ours", paper(rows, "ours"), " blend", paper(rows, "blend"))
    json.dump({"params": {k: v for k, v in P.items() if k != "sims"}, "ours": score(rows, "ours"), "market": score(rows, "mkt"),
               "blend": score(rows, "blend"), "paper_ours": paper(rows, "ours"), "paper_blend": paper(rows, "blend"),
               "by_market": by}, open(os.path.join(D, "music_backtest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
