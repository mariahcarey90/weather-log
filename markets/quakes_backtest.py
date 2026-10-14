#!/usr/bin/env python3
"""Backtest and tune the earthquake-count model on past weeks of the USGS catalogue.

For every Monday-Sunday (ET) week since 2023 and thresholds 5.5 and 6.5, predict the final
count at 0-6 days into the week, then score the probability given to the actual count
(log loss) and the chance the most likely count, or one either side, came in. Grid-searches
the rate blend w and clustering phi on weeks before the last 26, reports the last 26 weeks
out of sample next to a plain Poisson on the yearly rate, and writes
markets/data/quakes_backtest.json (quakes_score.py reads the parameters). Standard library only.
"""
import json, math, os, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quakes_model as qm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ET = ZoneInfo("America/New_York")
THRS = [5.5, 6.5]


def weeks(hist_end):
    d = datetime(2023, 1, 2, tzinfo=ET)  # a Monday
    out = []
    while (d + timedelta(days=7)).timestamp() <= hist_end:
        out.append((d.timestamp(), (d + timedelta(days=7)).timestamp()))
        d += timedelta(days=7)
    return out


def evaluate(hist, wks, params):
    ll, n, top, near = 0.0, 0, 0, 0
    for thr in THRS:
        times = [h[0] for h in hist if h[1] >= thr]
        for s, e in wks:
            actual = qm.count(times, s, e)
            for k in range(7):
                now = s + k * qm.DAY
                pmf, _ = qm.predict(hist, thr, s, e, now, params, times=times)
                pr = pmf.get(actual, 0.0)
                ll += -math.log(max(pr, 1e-6))
                best = max(pmf, key=pmf.get)
                top += best == actual
                near += abs(best - actual) <= 1
                n += 1
    return {"n": n, "logloss": round(ll / n, 4), "top_hit": round(top / n, 3), "within_one": round(near / n, 3)}


def main():
    hist = json.load(open(os.path.join(D, "quakes_hist.json")))
    wks = weeks(hist[-1][0])
    train, test = wks[:-26], wks[-26:]
    grid = [{"w": w, "phi": phi} for w in (0.0, 0.1, 0.2, 0.3, 0.5) for phi in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)]
    scored = sorted(((evaluate(hist, train, g)["logloss"], g) for g in grid), key=lambda x: x[0])
    best = scored[0][1]
    poisson = {"w": 0.0, "phi": 0.0}
    weekly = {}
    for thr in THRS:
        times = [h[0] for h in hist if h[1] >= thr]
        c = [qm.count(times, s, e) for s, e in wks]
        mu = sum(c) / len(c)
        weekly[str(thr)] = {"mean": round(mu, 2), "var": round(sum((x - mu) ** 2 for x in c) / (len(c) - 1), 2),
                            "min": min(c), "max": max(c)}
    out = {"generated": datetime.utcnow().isoformat(timespec="seconds") + "Z", "weeks": len(wks),
           "params": best, "train_top5": [{"params": g, "logloss": l} for l, g in scored[:5]],
           "test_last26": {"model": evaluate(hist, test, best), "poisson_yearly_rate": evaluate(hist, test, poisson)},
           "weekly_counts": weekly}
    json.dump(out, open(os.path.join(D, "quakes_backtest.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
