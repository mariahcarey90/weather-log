#!/usr/bin/env python3
"""Backtest the bucket odds on the backfill (data/backfill_daily.json).

For each city, lead and day: correct each model by its bias measured on the OTHER
days (leave-one-out, so no peeking), weight by 1/sd^2, blend with a Normal kernel,
integrate over market-style buckets (1 C, or 2 F for Atlanta) and score against
the station's observed high. Reports hit rate of the top bucket, average odds given
to the winning bucket, Brier score, and a calibration table. Deterministic models
only (no historical ensembles), so live odds should be a little sharper.
"""
import json, math, statistics, sys

def ncdf(x, m, s):
    return 0.5 * (1 + math.erf((x - m) / (s * math.sqrt(2))))

def bucket_of(v, unit):
    if unit == "F":
        lo = v - (v % 2)  # Polymarket uses even-odd pairs, e.g. 82-83
        return (lo, lo + 1)
    return (v, v)

def run(data, kscale):
    res = {}
    for city, c in data["cities"].items():
        unit, obs = c["unit"], {d: v for d, v in c["obs"].items()}
        for lead, bymodel in c["fc"].items():
            dates = sorted(d for d in obs if all(True for _ in [0]) and sum(d in v for v in bymodel.values()) >= 3)
            rows = []
            for d in dates:
                ov = obs[d]
                ov_c = (ov - 32) * 5 / 9 if unit == "F" else ov
                members, weights, med_errs = [], [], []
                for m, daily in bymodel.items():
                    if d not in daily:
                        continue
                    errs = []
                    for d2, v2 in daily.items():
                        if d2 != d and d2 in obs:
                            o2 = (obs[d2] - 32) * 5 / 9 if unit == "F" else obs[d2]
                            errs.append(v2 - o2)
                    if len(errs) < 10:
                        continue
                    b, sd = statistics.fmean(errs), statistics.stdev(errs)
                    mae = statistics.fmean(abs(e) for e in errs)
                    if mae > 1.8:
                        continue
                    members.append(daily[d] - b)
                    weights.append(1 / max(sd, 0.5) ** 2)
                # blend-level spread (leave-one-out) sets the kernel
                blend_errs = []
                for d2 in dates:
                    if d2 == d:
                        continue
                    vs = [bymodel[m][d2] for m in bymodel if d2 in bymodel[m]]
                    o2 = (obs[d2] - 32) * 5 / 9 if unit == "F" else obs[d2]
                    blend_errs.append(statistics.median(vs) - o2)
                sd_blend = statistics.stdev(blend_errs)
                if not members:
                    continue
                k = max(0.4, kscale * sd_blend) * (9 / 5 if unit == "F" else 1)
                mu = [(x * 9 / 5 + 32) if unit == "F" else x for x in members]
                W = sum(weights)
                cdf = lambda x: sum(w * ncdf(x, m, k) for m, w in zip(mu, weights)) / W
                centre = round(sum(w * m for m, w in zip(mu, weights)) / W)
                step = 2 if unit == "F" else 1
                cands = {}
                for v in range(centre - 12, centre + 13):
                    lo, hi = bucket_of(v, unit)
                    cands[(lo, hi)] = cdf(hi + 0.5) - cdf(lo - 0.5)
                win = bucket_of(ov, unit)
                top = max(cands, key=cands.get)
                brier = sum((p - (1 if b == win else 0)) ** 2 for b, p in cands.items())
                rows.append({"p_win": cands.get(win, 0), "hit": top == win,
                             "near": abs(top[0] - win[0]) <= step, "brier": brier,
                             "probs": list(cands.items()), "win": win})
            res[(city, lead)] = rows
    return res

def summary(rows):
    n = len(rows)
    return {"days": n, "top_hit": sum(r["hit"] for r in rows) / n,
            "within_one": sum(r["near"] for r in rows) / n,
            "avg_p_win": statistics.fmean(r["p_win"] for r in rows),
            "brier": statistics.fmean(r["brier"] for r in rows)}

if __name__ == "__main__":
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/backfill_daily.json"))
    print("kernel scale search (all cities, day1):")
    best = None
    for ks in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1):
        res = run(data, ks)
        rows = [r for (c, l), rs in res.items() if l == "day1" for r in rs]
        s = summary(rows)
        print(f"  k={ks}: brier {s['brier']:.3f}  avg p(win) {s['avg_p_win']:.2f}  top hit {s['top_hit']:.0%}")
        if best is None or s["brier"] < best[1]:
            best = (ks, s["brier"])
    ks = best[0]
    print(f"\nbest kernel scale {ks}\n")
    res = run(data, ks)
    for (city, lead), rows in sorted(res.items()):
        s = summary(rows)
        print(f"{city:12s} {lead}: {s['days']} days | top bucket right {s['top_hit']:.0%} | "
              f"within one bucket {s['within_one']:.0%} | avg odds on winner {s['avg_p_win']:.0%} | brier {s['brier']:.3f}")
    # reliability: when we said X%, how often did it happen?
    bins = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .7), (.7, 1.01)]
    allp = [(p, b == r["win"]) for rows in res.values() for r in rows for b, p in r["probs"] if p > 0.02]
    print("\nreliability (all cities, both leads): predicted -> actual")
    for lo, hi in bins:
        xs = [(p, h) for p, h in allp if lo <= p < hi]
        if xs:
            print(f"  {lo:.0%}-{min(hi,1):.0%}: predicted {statistics.fmean(p for p,_ in xs):.0%}, happened {statistics.fmean(h for _,h in xs):.0%}  (n={len(xs)})")
