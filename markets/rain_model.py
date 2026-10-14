"""Odds for monthly rainfall markets: measured so far + forecast + climatology, by simulation.

simulate(rec, clim, params) -> (list of simulated month totals, info)

Each simulated month = rain already measured at the station (complete days before today)
  + one ensemble member's daily rain for today through the forecast horizon (ECMWF/GFS members
    at the station), scaled by log-normal noise for point-vs-grid scatter (sigma_fc)
  + one past year's rain on the same calendar days for anything beyond the horizon (keeps that
    year's wet and dry spells together).
When the "measured" part is a model stand-in rather than the station, it gets its own noise
(sigma_obs). Parameters are deliberately wide until the live record says otherwise.
Standard library only.
"""
import math, random
from datetime import date, timedelta

DEFAULT = {"sims": 6000, "sigma_fc": 0.45, "sigma_obs": 0.3, "seed": 11}


def simulate(rec, clim, params=None):
    p = dict(DEFAULT, **(params or {}))
    rnd = random.Random(p["seed"])
    first, last = date.fromisoformat(rec["first"]), date.fromisoformat(rec["last"])
    today = max(first, date.fromisoformat(rec["localToday"]))
    obs = rec.get("obs") or {}
    measured = sum(v for d, v in obs.items() if first.isoformat() <= d < today.isoformat())
    approx_obs = "model" in (rec.get("obsSource") or "model")
    members = [m for m in (rec.get("ensemble") or {}).values() if m]
    days = [today + timedelta(days=i) for i in range((last - today).days + 1)]
    horizon = max((max(m) for m in members), default=None)
    fc_days = [d.isoformat() for d in days if horizon and d.isoformat() <= horizon]
    cl_days = [d.strftime("%m-%d") for d in days if not horizon or d.isoformat() > horizon]
    years = [y for y in (clim or {}).get("years", {}).values() if y]
    sims = []
    for _ in range(p["sims"]):
        m = measured * (math.exp(rnd.gauss(0, p["sigma_obs"])) if approx_obs and measured > 0 else 1.0)
        f = 0.0
        if fc_days and members:
            mem = rnd.choice(members)
            f = sum(mem.get(d, 0.0) for d in fc_days)
            f *= math.exp(rnd.gauss(-p["sigma_fc"] ** 2 / 2, p["sigma_fc"]))  # mean-preserving
        c = 0.0
        if cl_days and years:
            y = rnd.choice(years)
            c = sum(y.get(d, 0.0) for d in cl_days)
        sims.append(m + f + c)
    sims.sort()
    q = lambda x: sims[min(len(sims) - 1, int(x * len(sims)))]
    info = {"measured": round(measured, 2), "measuredThrough": rec.get("obsThrough"), "obsSource": rec.get("obsSource"),
            "forecastDays": len(fc_days), "climDays": len(cl_days), "members": len(members), "climYears": len(years),
            "median": round(q(0.5), 2), "p10": round(q(0.1), 2), "p90": round(q(0.9), 2)}
    return sims, info


def bucket_probs(sims, buckets):
    """Buckets are [lo, hi) with open ends; ties at an edge go to the higher bucket (NYC rule)."""
    n = len(sims) or 1
    out = []
    for b in buckets:
        lo = -1e9 if b.get("lo") is None else b["lo"]
        hi = 1e9 if b.get("hi") is None else b["hi"]
        out.append(sum(1 for x in sims if lo <= round(x, 2) < hi) / n)
    s = sum(out) or 1
    return [x / s for x in out]
