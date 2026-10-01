#!/usr/bin/env python3
"""Backtest the post-count model against the market on windows that already ended.

Usage: python3 markets/posts_backtest.py [--grid] [--handle elonmusk]

For each ended window with a known winner (markets/data/posts_backfill.json) and at
several points through it (0%, 25%, 50%, 75%, 90% elapsed), compare:
  ours:   posts_model odds from the post history before that moment
  market: the hourly price of each range at that moment, scaled to sum to 1
  blend:  50/50
Scores: log loss and Brier on the winning range (lower is better), how often the
top pick won, and paper trades at the market price under the live rules
(edge >= 8 points, price 5-97c, $25 each; no fees or spread, so optimistic).
--grid tries a few parameter sets and prints the best by log loss.
"""
import argparse, itertools, json, math, os, sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posts_model as pm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FRACS = [0.0, 0.25, 0.5, 0.75, 0.9]


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def price_at(hist, t):
    best = None
    for tt, p in hist or []:
        if tt <= t:
            best = p
        else:
            break
    return best


def load():
    bf = json.load(open(os.path.join(D, "posts_backfill.json")))["windows"]
    hist = {}
    for w in bf:
        if w["handle"] not in hist:
            h = json.load(open(os.path.join(D, "posts_hist", w["handle"] + ".json")))
            hist[w["handle"]] = (sorted(h["posts"].values()), h["from"])
    return bf, hist


def run(bf, hist, params, handle=None, verbose=False):
    rows = []
    for w in bf:
        if not w.get("winner") or (handle and w["handle"] != handle):
            continue
        times, hfrom = hist[w["handle"]]
        s, e = ts(w["start"]), ts(w["end"])
        rng = w["ranges"]
        wi = next(i for i, b in enumerate(rng) if b["label"] == w["winner"])
        for f in FRACS:
            now = int(s + f * (e - s))
            mk = [price_at(b.get("history"), now) for b in rng]
            if any(x is None for x in mk) or sum(mk) <= 0:
                continue
            tot = sum(mk)
            mkn = [x / tot for x in mk]
            _, _, dist, info = pm.predict(times, hfrom, s, e, now, params)
            if info["n"] < 3:
                continue
            ours = pm.range_probs(dist, rng)
            rows.append({"handle": w["handle"], "slug": w["slug"], "f": f, "wi": wi, "ours": ours, "mkt": mkn, "raw": mk})
    return rows


def score(rows, key):
    if not rows:
        return {}
    ll = br = hit = 0.0
    for r in rows:
        p = r[key] if key != "blend" else [0.5 * a + 0.5 * b for a, b in zip(r["ours"], r["mkt"])]
        ll += -math.log(max(p[r["wi"]], 1e-4))
        br += sum((pi - (1.0 if i == r["wi"] else 0.0)) ** 2 for i, pi in enumerate(p))
        hit += max(range(len(p)), key=lambda i: p[i]) == r["wi"]
    n = len(rows)
    return {"n": n, "logloss": round(ll / n, 3), "brier": round(br / n, 3), "top_hit": round(hit / n, 3)}


def paper(rows, key):
    seen, trades, pnl, wins = set(), 0, 0.0, 0
    for r in rows:
        k = (r["slug"], key)
        if k in seen:
            continue
        p = r["ours"] if key == "ours" else [0.5 * a + 0.5 * b for a, b in zip(r["ours"], r["mkt"])]
        best = None
        for i, (pi, price) in enumerate(zip(p, r["raw"])):
            if 0.05 <= price <= 0.97 and pi - price >= 0.08 and (best is None or pi - price > best[0]):
                best = (pi - price, i, price)
        if best:
            seen.add(k)
            trades += 1
            won = best[1] == r["wi"]
            wins += won
            pnl += (25 / best[2] if won else 0) - 25
    return {"trades": trades, "wins": wins, "pnl": round(pnl, 2), "roi": round(pnl / (25 * trades), 3) if trades else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--handle")
    ap.add_argument("--split", action="store_true", help="tune on the older half, report on the newer half")
    a = ap.parse_args()
    bf, hist = load()
    test = bf
    if a.split:
        bf_sorted = sorted(bf, key=lambda w: w["end"])
        cut = bf_sorted[len(bf_sorted) // 2]["end"]
        bf = [w for w in bf_sorted if w["end"] < cut]
        test = [w for w in bf_sorted if w["end"] >= cut]
        print(f"tuning on {len(bf)} windows ending before {cut[:10]}, testing on {len(test)}")
    print(f"{len(bf)} windows, {sum(1 for w in bf if w.get('winner'))} with a winner")
    best = (None, pm.DEFAULT)
    if a.grid:
        for d, pp, inf, wk in itertools.product([0.6, 0.8, 0.95], [0.0, 0.5, 1.0], [1.0, 1.3, 1.8], [4, 8]):
            P = {"decay": d, "pace_power": pp, "inflate": inf, "weeks": wk}
            sc = score(run(bf, hist, P, a.handle), "ours")
            if sc and (best[0] is None or sc["logloss"] < best[0]):
                best = (sc["logloss"], P)
        print("best params by log loss:", best)
    P = best[1]
    rows = run(test, hist, P, a.handle)
    print("params", P)
    for k in ("ours", "mkt", "blend"):
        print(f"  {k:6s}", score(rows, k))
    for f in FRACS:
        rf = [r for r in rows if r["f"] == f]
        print(f"  at {int(f*100):>2d}% elapsed: ours {score(rf,'ours')}  market {score(rf,'mkt')}")
    for hd in sorted({r["handle"] for r in rows}):
        rh = [r for r in rows if r["handle"] == hd]
        print(f"  {hd:16s} ours {score(rh,'ours')}  market {score(rh,'mkt')}  paper {paper(rh,'ours')}")
    print("  paper all: ours", paper(rows, "ours"), " blend", paper(rows, "blend"))
    json.dump({"params": P, "ours": score(rows, "ours"), "market": score(rows, "mkt"), "blend": score(rows, "blend"),
               "paper_ours": paper(rows, "ours"), "paper_blend": paper(rows, "blend"),
               "by_handle": {hd: {"ours": score([r for r in rows if r["handle"] == hd], "ours"),
                                  "market": score([r for r in rows if r["handle"] == hd], "mkt"),
                                  "paper": paper([r for r in rows if r["handle"] == hd], "ours")}
                             for hd in sorted({r["handle"] for r in rows})}},
              open(os.path.join(D, "posts_backtest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
