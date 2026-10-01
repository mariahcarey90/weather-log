"""Odds for the weekly Spotify chart markets (#1 / #2 song, global or US).

The weekly chart ranks songs by streams over Friday-Thursday. Spotify's daily charts
(via kworb.net) give each song's streams day by day, about a day and a half late, so
part of the week is already counted when a market is live. For the days still to come:
  a song with a full week of history: same weekday last week x its recent trend
    (its last 3 known days against the same 3 days a week earlier, clipped 0.4-2.5);
  a song released this week: its last known day walked forward on the measured
    new-release curve (day 2 ~41% of day 1, day 3 ~71% of day 2, then ~flat).
Each song's projected rest-of-week gets log-normal noise: sigma 0.07 x sqrt(days left),
at least 0.08, doubled for songs out under a week (measured on 231 song-weeks of
Sep 2026 daily charts: the rest-of-week projection missed by 8-14%), and 4000 simulated weeks give the chance each song
finishes #1 and #2. With no day of the week counted yet the odds are only indicative
(a Friday release can take the top spot), and "early" is set.
Standard library only.
"""
import math, random
from datetime import date, timedelta

# day-over-day ratio by days since a song's first chart day (Sep 2026 releases): the
# release-day spike drops to ~40%, then ~70%, then flattens
NEW_CURVE = {1: 0.41, 2: 0.71, 3: 1.0, 4: 0.9}
DEFAULT = {"sigma": 0.07, "sigma_min": 0.08, "decay": 0.98, "sims": 4000, "new_week_other": 0.15}


def d2s(d):
    return d.isoformat()


def predict(tracks, region, week_start, known_until, cands, params=None, seed=7):
    """tracks: {tid: {'daily': {date: {region: [pos, streams]}}}}
    week_start: date (Friday); known_until: last date whose daily chart is counted (date or None)
    cands: list of track ids to rank.
    Returns ({tid: {'p1','p2','known','proj','mean'}}, info)."""
    P = dict(DEFAULT, **(params or {}))
    days = [week_start + timedelta(days=i) for i in range(7)]
    known_days = [d for d in days if known_until and d <= known_until]
    rest = [d for d in days if d not in known_days]
    rng = random.Random(seed)
    stats = {}
    for tid in cands:
        daily = (tracks.get(tid) or {}).get("daily", {})
        val = lambda d: (daily.get(d2s(d)) or {}).get(region, [None, None])[1]
        known = sum(val(d) or 0 for d in known_days)
        hist = sorted(d for d, v in daily.items() if region in v and (known_until is None or d <= d2s(known_until)))
        if not hist and known == 0:
            continue
        last_d = date.fromisoformat(hist[-1]) if hist else None
        first_d = date.fromisoformat(hist[0]) if hist else None
        new = first_d is not None and (week_start - first_d).days < 7 and first_d >= week_start - timedelta(days=7)
        # trend from the last 3 known days vs a week earlier
        g = None
        if last_d:
            recent = [last_d - timedelta(days=i) for i in range(3)]
            a = sum(val(d) or 0 for d in recent)
            b = sum(val(d - timedelta(days=7)) or 0 for d in recent)
            if a > 0 and b > 0:
                g = min(2.5, max(0.4, a / b))
        proj = 0.0
        for d in rest:
            prev = val(d - timedelta(days=7)) if (d - timedelta(days=7)) <= (known_until or week_start - timedelta(days=1)) else None
            if prev and g and not new:
                proj += prev * g
            elif last_d and val(last_d):
                v = val(last_d)
                for i in range(1, (d - last_d).days + 1):  # walk forward day by day
                    age = (last_d + timedelta(days=i) - first_d).days
                    v *= NEW_CURVE.get(age, P["decay"]) if new else P["decay"]
                proj += v
        sig = max(P["sigma_min"], P["sigma"] * math.sqrt(len(rest))) * (2.0 if new else 1.0) if rest else 0.0
        stats[tid] = {"known": known, "proj": proj, "sig": sig, "new": new}
    if not stats:
        return {}, {"knownDays": len(known_days), "early": True}
    ids = list(stats)
    c1 = dict.fromkeys(ids, 0)
    c2 = dict.fromkeys(ids, 0)
    for _ in range(P["sims"]):
        tot = []
        for t in ids:
            s = stats[t]
            r = s["proj"] * math.exp(rng.gauss(-0.5 * s["sig"] ** 2, s["sig"])) if s["proj"] > 0 else 0.0
            tot.append((s["known"] + r, t))
        tot.sort(reverse=True)
        c1[tot[0][1]] += 1
        if len(tot) > 1:
            c2[tot[1][1]] += 1
    n = P["sims"]
    out = {t: {"p1": c1[t] / n, "p2": c2[t] / n, "known": stats[t]["known"], "proj": round(stats[t]["proj"]),
               "mean": round(stats[t]["known"] + stats[t]["proj"]), "new": stats[t]["new"]} for t in ids}
    return out, {"knownDays": len(known_days), "early": len(known_days) == 0}


def market_probs(market, res, info, params=None):
    """Map simulated odds onto the market's songs; 'Other' gets the rest."""
    P = dict(DEFAULT, **(params or {}))
    key = "p1" if market["rank"] == 1 else "p2"
    named = {s["trackId"] for s in market["songs"] if s.get("trackId")}
    probs = []
    for s in market["songs"]:
        if s.get("other"):
            probs.append(sum(v[key] for t, v in res.items() if t not in named))
        else:
            probs.append(res.get(s.get("trackId"), {}).get(key, 0.0) if s.get("trackId") else None)
    # songs we couldn't link: share nothing; early in the week reserve room for a new release
    reserve = P["new_week_other"] if info.get("early") else 0.0
    tot = sum(p for p in probs if p)
    out = []
    for s, p in zip(market["songs"], probs):
        if p is None:
            out.append(None)
            continue
        q = p / tot * (1 - reserve) if tot > 0 else 0.0
        if s.get("other"):
            q += reserve
        out.append(q)
    return out
