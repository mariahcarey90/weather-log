"""Odds for earthquake-count markets: count so far + projected rest of the window.

predict(hist, thr, start, end, now, params) -> (pmf over the final count, info)

hist: [unix seconds, magnitude, id] rows from the USGS catalogue (quakes_hist.json).
  so far: events with magnitude >= thr already inside [start, now)
  rest:   negative binomial for the time left, mean = daily rate x days left, where the daily
          rate blends the last 365 days with the last 14 days (weight "w", for aftershock
          sequences and active spells), and variance = mean x (1 + phi x mean) for clustering.
Parameters come from quakes_backtest.py (fitted on past weeks). Standard library only.
"""
import bisect, math

DEFAULT = {"w": 0.2, "phi": 0.05}
DAY = 86400


def nb_pmf(mean, phi, n_max):
    """Negative binomial pmf for 0..n_max with var = mean*(1+phi*mean); Poisson when phi ~ 0."""
    if mean <= 1e-9:
        return [1.0] + [0.0] * n_max
    if phi <= 1e-6:
        out, p = [], math.exp(-mean)
        for k in range(n_max + 1):
            out.append(p)
            p *= mean / (k + 1)
        return out
    r = 1.0 / phi                      # shape
    q = r / (r + mean)                 # success prob
    out = []
    logp = r * math.log(q)
    for k in range(n_max + 1):
        out.append(math.exp(logp))
        logp += math.log((k + r) / (k + 1)) + math.log(1 - q)
    return out


def count(times, a, b):
    return bisect.bisect_left(times, b) - bisect.bisect_left(times, a)


def predict(hist, thr, start, end, now, params=None, n_max=None, times=None):
    p = dict(DEFAULT, **(params or {}))
    if times is None:
        times = [h[0] for h in hist if h[1] >= thr]   # hist is sorted by time
    now_c = min(max(now, start), end)
    so_far = count(times, start, now_c)
    long_rate = count(times, now - 365 * DAY, now) / 365.0
    short_rate = count(times, now - 14 * DAY, now) / 14.0
    rate = (1 - p["w"]) * long_rate + p["w"] * short_rate
    left = max(0.0, (end - now_c) / DAY)
    mean = rate * left
    if n_max is None:
        n_max = int(mean + 10 * math.sqrt(mean * (1 + p["phi"] * mean) + 1) + 10)
    rest = nb_pmf(mean, p["phi"], n_max)
    s = sum(rest)
    pmf = {so_far + k: v / s for k, v in enumerate(rest)}
    info = {"soFar": so_far, "daysLeft": round(left, 2), "ratePerDay": round(rate, 3),
            "longRate": round(long_rate, 3), "shortRate": round(short_rate, 3),
            "expectedFinal": round(so_far + mean, 1),
            "sd": round(math.sqrt(mean * (1 + p["phi"] * mean)), 2)}
    return pmf, info


def bucket_probs(pmf, buckets):
    """buckets: [{"lo": int|None, "hi": int|None}] -> probabilities (renormalised)."""
    ps = []
    for b in buckets:
        lo = -1 if b.get("lo") is None else b["lo"]
        hi = 10 ** 9 if b.get("hi") is None else b["hi"]
        ps.append(sum(v for k, v in pmf.items() if lo <= k <= hi))
    s = sum(ps) or 1
    return [x / s for x in ps]
