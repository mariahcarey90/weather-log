#!/usr/bin/env python3
"""EMOS / non-homogeneous Gaussian regression for each station's daily high.

For every city and lead (day1, day2) in data/backfill_daily.json:
  m = mean of the bias-corrected deterministic models (each model's bias from the
      other days, as in backtest.py; models with MAE > 1.8 C left out)
  s = their spread (sd)
  high ~ Normal(mu = a + b*m, sigma = sqrt(c^2 + d^2 * s^2))
with a, b, c, d chosen to minimise the mean CRPS (Gneiting et al. 2005).

Writes data/emos.json with the parameters fitted on all days, plus a 5-fold
cross-validated comparison with the current kernel method on the same days
(Brier over market-style buckets, top-bucket hit rate, odds on the winner).
score.py uses the parameters for its "emos" challenger. Standard library only.
"""
import json, math, statistics, sys

import backtest

MAE_CUT = 1.8


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def crps(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2 * ncdf(z) - 1) + 2 * npdf(z) - 1 / math.sqrt(math.pi))


def nelder_mead(f, x0, step=0.5, iters=600, tol=1e-7):
    n = len(x0)
    pts = [list(x0)] + [[x0[j] + (step if j == i else 0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        refl = [cen[j] + (cen[j] - pts[-1][j]) for j in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [cen[j] + 2 * (cen[j] - pts[-1][j]) for j in range(n)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                best = pts[0]
                pts = [best] + [[best[j] + 0.5 * (p[j] - best[j]) for j in range(n)] for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = min(range(n + 1), key=lambda k: vals[k])
    return pts[i], vals[i]


def features(city_data, lead):
    """[(date, m, s, y_c)] for one city and lead."""
    unit, obs = city_data["unit"], city_data["obs"]
    bymodel = city_data["fc"].get(lead) or {}
    to_c = (lambda v: (v - 32) * 5 / 9) if unit == "F" else (lambda v: v)
    rows = []
    for d in sorted(obs):
        vals = []
        for m, daily in bymodel.items():
            if d not in daily:
                continue
            errs = [v2 - to_c(obs[d2]) for d2, v2 in daily.items() if d2 != d and d2 in obs]
            if len(errs) < 10 or statistics.fmean(abs(e) for e in errs) > MAE_CUT:
                continue
            vals.append(daily[d] - statistics.fmean(errs))
        if len(vals) >= 3:
            rows.append((d, statistics.fmean(vals), statistics.pstdev(vals), to_c(obs[d])))
    return rows


def fit(rows):
    def loss(p):
        a, b, c, d = p
        return statistics.fmean(crps(a + b * m, math.sqrt(c * c + d * d * s * s) + 1e-6, y) for _, m, s, y in rows)
    best, val = nelder_mead(loss, [0.0, 1.0, 0.8, 0.5])
    return {"a": best[0], "b": best[1], "c": abs(best[2]), "d": abs(best[3]), "crps": val}


def bucket_probs(mu_c, sigma_c, unit, centre=None):
    """{(lo, hi): p} over market-style buckets around mu (market units)."""
    if unit == "F":
        mu, sg = mu_c * 9 / 5 + 32, sigma_c * 9 / 5
    else:
        mu, sg = mu_c, sigma_c
    c = round(mu) if centre is None else centre
    out = {}
    for v in range(c - 12, c + 13):
        lo, hi = backtest.bucket_of(v, unit)
        out[(lo, hi)] = ncdf((hi + 0.5 - mu) / sg) - ncdf((lo - 0.5 - mu) / sg)
    return out


def score_probs(probs, win):
    top = max(probs, key=probs.get)
    return {"brier": sum((p - (1 if b == win else 0)) ** 2 for b, p in probs.items()),
            "hit": top == win, "p_win": probs.get(win, 0.0)}


def main():
    data = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/backfill_daily.json"))
    current = backtest.run(data, 0.5)
    out = {"window": data.get("window"), "cities": {}}
    for city, cd in data["cities"].items():
        unit = cd["unit"]
        for lead in ("day1", "day2"):
            rows = features(cd, lead)
            if len(rows) < 25:
                continue
            # 5-fold cross-validation, days interleaved
            folds = [rows[i::5] for i in range(5)]
            em = []
            for k in range(5):
                train = [r for j, f in enumerate(folds) if j != k for r in f]
                p = fit(train)
                for d, m, s, y in folds[k]:
                    mu, sg = p["a"] + p["b"] * m, math.sqrt(p["c"] ** 2 + p["d"] ** 2 * s * s) + 1e-6
                    obs_u = data["cities"][city]["obs"][d]
                    em.append(score_probs(bucket_probs(mu, sg, unit), backtest.bucket_of(obs_u, unit)))
            cur = [r for r in current.get((city, lead), [])]
            full = fit(rows)
            res = {**{k: round(v, 4) for k, v in full.items()}, "n": len(rows),
                   "cv_brier": round(statistics.fmean(e["brier"] for e in em), 4),
                   "cv_top_hit": round(statistics.fmean(e["hit"] for e in em), 3),
                   "cv_p_win": round(statistics.fmean(e["p_win"] for e in em), 3),
                   "current_brier": round(statistics.fmean(r["brier"] for r in cur), 4) if cur else None,
                   "current_top_hit": round(statistics.fmean(r["hit"] for r in cur), 3) if cur else None,
                   "current_p_win": round(statistics.fmean(r["p_win"] for r in cur), 3) if cur else None}
            out["cities"].setdefault(city, {})[lead] = res
            print(f"{city:12s} {lead}: EMOS cv brier {res['cv_brier']:.3f} top {res['cv_top_hit']:.0%} p(win) {res['cv_p_win']:.0%}"
                  f" | current brier {res['current_brier']:.3f} top {res['current_top_hit']:.0%} p(win) {res['current_p_win']:.0%}"
                  f" | mu = {res['a']:+.2f} + {res['b']:.2f}m, sigma = sqrt({res['c']:.2f}^2 + {res['d']:.2f}^2 s^2)")
    json.dump(out, open("data/emos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
