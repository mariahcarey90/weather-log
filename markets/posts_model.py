"""Odds for post-count ranges, from an account's own posting history.

predict(times, start, end, now, params) -> (mean final count, var, pmf over final counts)

times: sorted unix seconds of every post the counter captured.
The rest of the window [now, end] is compared with the same hours of the week in
earlier weeks: for each earlier week j (shifted back 7*j days, kept only if fully
inside the history and before now), count the posts in that stretch. Those counts,
weighted toward recent weeks (decay ** (j-1)), give the expected rest-of-window
count and its spread. Two adjustments:
  this week's pace: the posts so far in this window against what the same weeks had
    by this point; the expected rest is scaled by that ratio ** pace_power;
  spread: weighted variance of the samples plus the mean (Poisson part), times
    inflate; counts follow a negative binomial with that mean and variance.
Windows longer than 3 weeks (monthly markets) use daily rates over the last
28 days instead of weekly samples.
Standard library only.
"""
import bisect, math

DAY, WEEK = 86400, 7 * 86400
DEFAULT = {"weeks": 8, "decay": 0.8, "pace_power": 0.0, "inflate": 1.3, "prior": 10.0}  # tuned on older windows (posts_backtest.py --grid --split); pace_power 0 won


def count(times, a, b):
    return bisect.bisect_left(times, b) - bisect.bisect_left(times, a)


def nb_pmf(mean, var, n_max):
    """Negative binomial pmf (Poisson if var <= mean) for 0..n_max."""
    out = []
    if mean <= 0:
        return [1.0] + [0.0] * n_max
    if var <= mean * 1.0001:
        lp = -mean
        for k in range(n_max + 1):
            out.append(math.exp(lp))
            lp += math.log(mean) - math.log(k + 1)
        return out
    p = mean / var
    r = mean * mean / (var - mean)
    lp = r * math.log(p)
    for k in range(n_max + 1):
        out.append(math.exp(lp))
        lp += math.log(k + r) - math.log(k + 1) + math.log(1 - p)
    return out


def samples(times, hist_from, start, end, now, weeks):
    """(elapsed_j, rest_j, weight index j) for earlier weeks."""
    out = []
    span = end - start
    k0 = max(1, math.ceil((end - now) / WEEK))  # the shifted rest-of-window must end before now
    for j in range(k0, k0 + weeks):
        sh = j * WEEK
        if start - sh < hist_from:
            break
        a = max(start, min(now, end))
        out.append((count(times, start - sh, a - sh), count(times, a - sh, end - sh), j - k0 + 1))
    return out


def predict(times, hist_from, start, end, now, params=None):
    P = dict(DEFAULT, **(params or {}))
    now_c = max(start, min(now, end))
    so_far = count(times, start, now_c)
    rest_len = end - now_c
    if rest_len <= 0:
        return so_far, 0.0, {so_far: 1.0}, {"soFar": so_far, "restMean": 0.0, "n": 0}
    sm = samples(times, hist_from, start, end, now_c, P["weeks"]) if end - start <= 21 * DAY else []
    if len(sm) >= 3:
        w = [P["decay"] ** (j - 1) for _, _, j in sm]
        W = sum(w)
        m_rest = sum(wi * r for wi, (_, r, _) in zip(w, sm)) / W
        m_el = sum(wi * e for wi, (e, _, _) in zip(w, sm)) / W
        v_s = sum(wi * (r - m_rest) ** 2 for wi, (_, r, _) in zip(w, sm)) / W
        n = len(sm)
    else:  # long windows or short history: daily rate over the last 28 days
        a = max(hist_from, now_c - 28 * DAY)
        days = max(1.0, (now_c - a) / DAY)
        daily = [count(times, now_c - (i + 1) * DAY, now_c - i * DAY) for i in range(int(days))]
        rate = sum(daily) / len(daily)
        dv = sum((x - rate) ** 2 for x in daily) / max(1, len(daily) - 1)
        L = rest_len / DAY
        m_rest = rate * L
        v_s = max(dv * L, 0.0) + (0.25 * rate * L) ** 2  # day-to-day noise plus drift in the rate
        el_len = (now_c - start) / DAY
        m_el = rate * el_len
        n = len(daily)
    ratio = 1.0
    if now_c > start and m_el > 0:
        ratio = (so_far + P["prior"]) / (m_el + P["prior"])
        m_rest *= ratio ** P["pace_power"]
    var = (v_s * ratio ** (2 * P["pace_power"]) + m_rest) * P["inflate"]
    n_max = int(m_rest + 10 * math.sqrt(var) + 20)
    pmf = nb_pmf(m_rest, var, n_max)
    dist = {so_far + k: p for k, p in enumerate(pmf) if p > 1e-9}
    return so_far + m_rest, var, dist, {"soFar": so_far, "restMean": round(m_rest, 1), "paceRatio": round(ratio, 2), "n": n}


def range_probs(dist, ranges):
    out = []
    for b in ranges:
        lo = -1 if b.get("lo") is None else b["lo"]
        hi = 10 ** 9 if b.get("hi") is None else b["hi"]
        out.append(sum(p for k, p in dist.items() if lo <= k <= hi))
    s = sum(out)
    return [x / s for x in out] if s > 0 else out
