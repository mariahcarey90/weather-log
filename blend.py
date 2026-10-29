#!/usr/bin/env python3
"""Blend weight and accuracy gate (28 Oct 2026 audit, Fixes 2 and 3).

The question the bet slip has to answer is not "did recent paper trades make money" (a
few dozen long shots are mostly luck) but "do our odds add anything to the market's?".

Rows: one day-ahead run per settled market (the 12-36 h run nearest 24 h ahead), logged
after FIX_TIME (runs before it came from code that applied the city bias twice), not
stale, with every bucket priced. Each row = our odds, the market's odds (mid prices
normalised to sum 1) and the winning bucket.

Blend weight w: the share of our odds in  w*ours + (1-w)*market  that minimises log
loss over the rows (grid 0, 0.05, ..., 1). w = 0 means the market alone is best.

Gate: walk forward through the dates. Each date's rows are scored with the weight
fitted on earlier dates only, so the test never sees its own answer. The gate opens
when, over at least MIN_ROWS such out-of-sample markets on MIN_DATES dates, the blended
odds beat the market's log loss with 95% confidence (bootstrap that resamples whole
dates, because cities on one date share weather and model errors), and the weight fitted
on everything is at least MIN_WEIGHT.

  python3 blend.py --db DIR            print the weight and the gate
Standard library only.
"""
import argparse, glob, json, math, os, random, statistics

FIX_TIME = "2026-10-28T05:05:00Z"   # the bias fix (commit 7145060) was pushed just before this
FLOOR = 0.005                       # log loss floor: a 0% call that comes true costs log(1/0.005)
GRID = tuple(i / 20 for i in range(21))
MIN_ROWS, MIN_DATES, MIN_WEIGHT = 300, 21, 0.2
MIN_TRAIN_DATES = 5                 # dates needed before walk-forward scoring starts
BOOT = 2000


def lead_ok(s):
    lh = s.get("leadHours")
    return lh is not None and 12 <= lh <= 36


def norm(vals):
    if any(v is None for v in vals):
        return None
    t = sum(max(0.0, float(v)) for v in vals)
    return [max(0.0, float(v)) / t for v in vals] if t > 0 else None


def rows_from(snaps, outs, since=FIX_TIME, model_key="model"):
    """[(date, ours, market, winner_index)], one day-ahead run per settled market."""
    best = {}
    for s in snaps:
        if s.get("kind", "high") != "high" or s.get("stale") or not lead_ok(s) or (s.get("runAt") or "") < since:
            continue
        o = outs.get((s.get("city"), s.get("marketDate")))
        if not o or not o.get("resolvedBucket"):
            continue
        k = (s["city"], s["marketDate"])
        if k not in best or abs(s["leadHours"] - 24) < abs(best[k]["leadHours"] - 24):
            best[k] = s
    rows = []
    for (city, date), s in sorted(best.items()):
        bs = s.get("buckets") or []
        labels = [b.get("label") for b in bs]
        win = outs[(city, date)]["resolvedBucket"]
        if win not in labels:
            continue
        pm = norm([b.get(model_key) for b in bs])
        pk = norm([b.get("market") for b in bs])
        if pm and pk:
            rows.append((date, pm, pk, labels.index(win)))
    return rows


def logloss(rows, w):
    return statistics.fmean(-math.log(max(w * pm[i] + (1 - w) * pk[i], FLOOR)) for _, pm, pk, i in rows)


def fit_weight(rows):
    """Weight on our odds that minimises log loss (0 = market only). No rows: 0."""
    if not rows:
        return 0.0
    return min(GRID, key=lambda w: (logloss(rows, w), w))  # ties go to the smaller weight


def walk_forward(rows):
    """{date: [log-loss gain of the blend over the market]} using weights fitted on earlier dates."""
    dates = sorted({r[0] for r in rows})
    gains = {}
    for k, d in enumerate(dates):
        if k < MIN_TRAIN_DATES:
            continue
        w = fit_weight([r for r in rows if r[0] < d])
        gains[d] = [math.log(max(w * pm[i] + (1 - w) * pk[i], FLOOR)) - math.log(max(pk[i], FLOOR))
                    for date, pm, pk, i in rows if date == d]
    return gains


def gate(rows, seed=0):
    """(open, info). info carries the weight, counts, mean gain and its 95% lower bound."""
    w = fit_weight(rows)
    gains = walk_forward(rows)
    n = sum(len(v) for v in gains.values())
    info = {"weight": w, "markets": len(rows), "dates": len({r[0] for r in rows}),
            "oos_markets": n, "oos_dates": len(gains), "gain": None, "gain_low95": None,
            "logloss_market": round(logloss(rows, 0.0), 4) if rows else None,
            "logloss_blend": round(logloss(rows, w), 4) if rows else None,
            "logloss_ours": round(logloss(rows, 1.0), 4) if rows else None}
    if n:
        flat = [g for v in gains.values() for g in v]
        info["gain"] = round(statistics.fmean(flat), 4)
        rnd, ds = random.Random(seed), list(gains)
        boot = sorted(statistics.fmean([g for d in (rnd.choice(ds) for _ in ds) for g in gains[d]])
                      for _ in range(BOOT))
        info["gain_low95"] = round(boot[int(0.05 * BOOT)], 4)  # one-sided 95%
    need = []
    if n < MIN_ROWS:
        need.append(f"{MIN_ROWS - n} more scored markets")
    if len(gains) < MIN_DATES:
        need.append(f"{MIN_DATES - len(gains)} more scored dates")
    if info["gain_low95"] is None or info["gain_low95"] <= 0:
        need.append("blended odds to beat the market's log loss with 95% confidence")
    if w < MIN_WEIGHT:
        need.append(f"a fitted weight of at least {MIN_WEIGHT} (now {w})")
    info["needs"] = need
    return not need, info


def weights_by_city(snaps, outs, min_rows=10):
    """{city: (weight, markets)} for cities with at least min_rows settled markets. For watching only."""
    out = {}
    for city in sorted({s.get("city") for s in snaps if s.get("city")}):
        rows = rows_from([s for s in snaps if s.get("city") == city], outs)
        if len(rows) >= min_rows:
            out[city] = (fit_weight(rows), len(rows))
    return out


def describe(info):
    g = (f"; out-of-sample log-loss gain {info['gain']:+.3f} (95% low {info['gain_low95']:+.3f})"
         if info["gain"] is not None else "")
    return (f"blend weight {info['weight']:.2f} on {info['markets']} day-ahead markets over {info['dates']} dates "
            f"since the 28 Oct fix{g}")


def load_dir(path):
    out = []
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out.append(d.get("data", d))
    return out


def load_db(db):
    snaps = load_dir(os.path.join(db, "snapshots"))
    outs = {(o["city"], o["date"]): o for o in load_dir(os.path.join(db, "outcomes"))}
    return snaps, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--since", default=FIX_TIME)
    a = ap.parse_args()
    snaps, outs = load_db(a.db)
    is_open, info = gate(rows_from(snaps, outs, since=a.since))
    print(describe(info))
    print("gate OPEN" if is_open else "gate closed: needs " + "; ".join(info["needs"]))
    print(json.dumps({"open": is_open, **info}))


if __name__ == "__main__":
    main()
