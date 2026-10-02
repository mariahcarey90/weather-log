#!/usr/bin/env python3
"""Scan every city with a Polymarket daily high-temperature market (research only).

For each city: the resolution station and unit from a recent event's rules; average
daily volume over the last 7 events; the order book on tomorrow's event (dollars on
offer within 2c of the best ask on the live buckets); the same 60-day backtest as
screen.py (top-bucket hit rate a day ahead); and a 60-day weather-regime profile using
score.py's own stability rules (model spread + storms, rain, heavy cloud, fronts /
wind shifts, fog), so we can see how often each city would have been Settled, Mixed
or Unsettled, how windy it runs, and how the top pick did on Settled days.
Writes data/scan/scan.json and prints a ranked table. Standard library only.
"""
import inspect, json, os, re, statistics, sys, urllib.parse
import concurrent.futures as cf
from datetime import date, datetime, timedelta

import backtest
import backfill
from backfill import DAYS, observed_highs, past_forecasts
from fetch import get, book_depth, _circ_diff
from score import stability

CITIES = ("hong-kong seoul beijing guangzhou paris qingdao los-angeles chengdu london "
          "san-francisco busan wuhan singapore miami tokyo auckland munich seattle taipei "
          "warsaw chicago dallas shenzhen toronto chongqing wellington nyc "
          "madrid atlanta mexico-city lucknow amsterdam shanghai ankara").split()
CURRENT = {"madrid", "atlanta", "mexico-city", "lucknow", "amsterdam", "shanghai", "ankara"}

# backtest.run with the date kept on each row (so hit rates can be split by regime)
_src = inspect.getsource(backtest.run).replace('rows.append({"p_win"', 'rows.append({"date": d, "p_win"')
_ns = dict(vars(backtest))
exec(_src, _ns)
run_dated = _ns["run"]


def mslug(d):
    return f"{d.strftime('%B').lower()}-{d.day}-{d.year}"


def event(key, d):
    ev = get(f"https://gamma-api.polymarket.com/events?slug=highest-temperature-in-{key}-on-{mslug(d)}")
    return ev[0] if ev else None


def market_info(key):
    info = {"volumes": [], "rules": None, "icao": None, "unit": None, "source": None}
    for back in range(1, 8):
        try:
            ev = event(key, date.today() - timedelta(days=back))
        except Exception:
            continue
        if not ev:
            continue
        info["volumes"].append(float(ev.get("volume") or 0))
        if info["rules"] is None:
            text = ev.get("description") or ((ev.get("markets") or [{}])[0].get("description") or "")
            info["rules"] = text
            labels = " ".join((m.get("groupItemTitle") or "") for m in ev.get("markets", []))
            info["unit"] = "F" if "°F" in labels else "C"
            m = (re.search(r"site=([A-Za-z0-9]{4})\b", text) or re.search(r"wunderground\.com/\S*?/([A-Z]{4})\b", text))
            info["icao"] = m.group(1).upper() if m else None
            src = re.search(r"recorded by (.+?) at", text)
            info["source"] = src.group(1) if src else None
    return info


def tomorrow_book(key):
    ev = event(key, date.today() + timedelta(days=1))
    if not ev:
        return None
    rows = []
    for m in ev.get("markets", []):
        try:
            yes = float(json.loads(m.get("outcomePrices") or "[]")[0])
            tok = json.loads(m.get("clobTokenIds") or "[]")[0]
        except Exception:
            continue
        if 0.05 <= yes <= 0.9 and not m.get("closed"):
            try:
                d = book_depth(tok)
            except Exception:
                continue
            rows.append({"label": m.get("groupItemTitle"), "yes": yes,
                         "ask_2c": d.get("usd_within_2c") or 0, "bid_2c": d.get("bid_usd_within_2c") or 0,
                         "fill25": d.get("avg_fill_25"), "best_ask": d.get("best_ask")})
    if not rows:
        return {"liquidity": float(ev.get("liquidity") or 0), "buckets": []}
    asks = [r["ask_2c"] for r in rows]
    slip = [r["fill25"] - r["best_ask"] for r in rows if r["fill25"] and r["best_ask"]]
    return {"liquidity": round(float(ev.get("liquidity") or 0)),
            "volume_so_far": round(float(ev.get("volume") or 0)),
            "live_buckets": len(rows), "median_ask_2c": round(statistics.median(asks)),
            "min_ask_2c": round(min(asks)), "total_ask_2c": round(sum(asks)),
            "median_slip_25_c": round(100 * statistics.median(slip), 2) if slip else None,
            "buckets": rows}


def coords(icao):
    rows = get(f"https://aviationweather.gov/api/data/stationinfo?ids={icao}&format=json")
    r = rows[0] if isinstance(rows, list) and rows else rows
    return float(r["lat"]), float(r["lon"]), r.get("site") or icao


def conditions_history(lat, lon):
    """Same daily regime fields as fetch.fetch_conditions, over the last DAYS days."""
    vars_ = ["cloud_cover", "precipitation_probability", "precipitation", "cape", "wind_speed_10m",
             "wind_direction_10m", "pressure_msl", "visibility", "temperature_2m"]
    q = {"latitude": lat, "longitude": lon, "hourly": ",".join(vars_), "timezone": "auto",
         "past_days": DAYS, "forecast_days": 1}
    p = get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))
    h, tz = p.get("hourly", {}), p.get("timezone")
    times = h.get("time", [])
    by = {}
    for i, t in enumerate(times):
        by.setdefault(t[:10], []).append((int(t[11:13]), {v: (h.get(v) or [None] * len(times))[i] for v in vars_}))
    out, pmax, p12p = {}, None, None
    for d in sorted(by):
        rows = by[d]
        day = [r for hr, r in rows if 9 <= hr <= 18]
        morn = [r for hr, r in rows if 6 <= hr <= 10]
        vals = lambda rs, k: [r[k] for r in rs if r.get(k) is not None]
        tmax = max(vals([r for _, r in rows], "temperature_2m") or [None]) if rows else None
        wd = lambda H: next((r["wind_direction_10m"] for hr, r in rows if hr == H and r.get("wind_direction_10m") is not None), None)
        p12 = next((r["pressure_msl"] for hr, r in rows if hr == 12 and r.get("pressure_msl") is not None), None)
        rain_p = max(vals(day, "precipitation_probability")) if vals(day, "precipitation_probability") else None
        rain_mm = sum(vals(day, "precipitation")) if vals(day, "precipitation") else None
        if rain_p is None and rain_mm is not None:  # past days often lack the probability: use what fell
            rain_p = 60 if rain_mm >= 1 else (30 if rain_mm > 0.1 else 0)
        out[d] = {
            "cloud_mean": statistics.fmean(vals(day, "cloud_cover")) if vals(day, "cloud_cover") else None,
            "rain_chance_max": rain_p,
            "cape_max": max(vals(day, "cape")) if vals(day, "cape") else None,
            "wind_mean_kmh": statistics.fmean(vals(day, "wind_speed_10m")) if vals(day, "wind_speed_10m") else None,
            "wind_shift_deg": _circ_diff(wd(9), wd(17)) if wd(9) is not None and wd(17) is not None else None,
            "pressure_change_hpa": (p12 - p12p) if p12 is not None and p12p is not None else None,
            "morning_vis_min_m": min(vals(morn, "visibility")) if vals(morn, "visibility") else None,
            "tmax_change_c": (tmax - pmax) if tmax is not None and pmax is not None else None,
        }
        pmax, p12p = tmax, p12
    return out, tz


def city(key):
    rec = {"key": key, "current": key in CURRENT, "notes": []}
    mi = market_info(key)
    if not mi["volumes"]:
        rec["error"] = "no recent events under this slug"
        return rec
    rec["avg_daily_volume"] = round(statistics.fmean(mi["volumes"]))
    rec["unit"], rec["icao"], rec["resolution_source"] = mi["unit"], mi["icao"], mi["source"]
    rec["rules_excerpt"] = (mi["rules"] or "")[:300]
    try:
        rec["book_tomorrow"] = tomorrow_book(key)
    except Exception as e:
        rec["notes"].append(f"book: {e}")
    if not mi["icao"]:
        rec["error"] = "no airport station in rules (not testable with METARs)"
        return rec
    try:
        lat, lon, name = coords(mi["icao"])
    except Exception as e:
        rec["error"] = f"station lookup {mi['icao']}: {e}"
        return rec
    rec.update({"station": name, "lat": lat, "lon": lon})
    try:
        cond, tz = conditions_history(lat, lon)
    except Exception as e:
        rec["error"] = f"conditions: {e}"
        return rec
    rec["tz"] = tz
    st = {"key": key, "icao": mi["icao"], "lat": lat, "lon": lon, "tz": tz, "unit": mi["unit"] or "C"}
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=DAYS)
    try:
        obs = observed_highs(dict(st, icao=st["icao"][1:] if st["icao"].startswith("K") else st["icao"]),
                             start, end + timedelta(days=1))
    except Exception as e:
        obs = {}
        rec["notes"].append(f"observations: {e}")
    fc = past_forecasts(st)
    # regime per day: bias-corrected spread of the day-ahead models + conditions flags
    day1 = fc.get("day1") or {}
    unit = st["unit"]
    toc = lambda v: (v - 32) * 5 / 9 if unit == "F" else v
    bias = {}
    for m, daily in day1.items():
        errs = [v - toc(obs[d]) for d, v in daily.items() if d in obs]
        if len(errs) >= 10:
            bias[m] = statistics.fmean(errs)
    labels = {}
    for d, c in cond.items():
        if d > end.isoformat() or d < start.isoformat():
            continue
        corr = [day1[m][d] - b for m, b in bias.items() if d in day1[m]]
        spread = statistics.pstdev(corr) if len(corr) >= 4 else None
        labels[d] = stability(spread, c)[0]
    n = len(labels) or 1
    rec["regime"] = {k: round(sum(1 for v in labels.values() if v == k) / n, 2) for k in ("Settled", "Mixed", "Unsettled")}
    days_c = [c for d, c in cond.items() if d in labels]
    f = lambda k: [c[k] for c in days_c if c.get(k) is not None]
    rec["climate"] = {
        "wind_mean_kmh": round(statistics.fmean(f("wind_mean_kmh")), 1) if f("wind_mean_kmh") else None,
        "windy_days_pct": round(100 * sum(1 for w in f("wind_mean_kmh") if w >= 20) / max(1, len(f("wind_mean_kmh")))),
        "wind_shift_front_pct": round(100 * sum(1 for c in days_c if (c.get("wind_shift_deg") or 0) >= 90 and (c.get("wind_mean_kmh") or 0) >= 12) / max(1, len(days_c))),
        "heavy_cloud_pct": round(100 * sum(1 for x in f("cloud_mean") if x >= 75) / max(1, len(f("cloud_mean")))),
        "rain_pct": round(100 * sum(1 for x in f("rain_chance_max") if x >= 50) / max(1, len(f("rain_chance_max")))),
        "storm_pct": round(100 * sum(1 for c in days_c if (c.get("cape_max") or 0) >= 1000 and (c.get("rain_chance_max") or 0) >= 40) / max(1, len(days_c))),
        "pressure_front_pct": round(100 * sum(1 for x in f("pressure_change_hpa") if abs(x) >= 6) / max(1, len(f("pressure_change_hpa")))),
        "fog_pct": round(100 * sum(1 for x in f("morning_vis_min_m") if x < 1000) / max(1, len(f("morning_vis_min_m")))),
    }
    data = {"cities": {key: {"unit": unit, "obs": obs, "fc": fc}}}
    try:
        res = run_dated(data, 0.5)
        for lead in ("day1", "day2"):
            rows = res.get((key, lead)) or []
            if rows:
                s = backtest.summary(rows)
                rec[f"backtest_{lead}"] = {k: round(v, 3) for k, v in s.items()}
        rows = res.get((key, "day1")) or []
        for lab in ("Settled", "Mixed", "Unsettled"):
            rs = [r for r in rows if labels.get(r["date"]) == lab]
            if rs:
                rec.setdefault("hit_by_regime", {})[lab] = {"days": len(rs), "top_hit": round(sum(r["hit"] for r in rs) / len(rs), 2)}
    except Exception as e:
        rec["notes"].append(f"backtest: {e}")
    rec["obs_days"] = len(obs)
    return rec


def main():
    only = sys.argv[1:] or CITIES
    with cf.ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(city, only))
    os.makedirs("data/scan", exist_ok=True)
    for r in recs:
        for b in ((r.get("book_tomorrow") or {}).get("buckets") or []):
            b.pop("best_ask", None)
    json.dump({"generated": datetime.utcnow().isoformat(timespec="seconds") + "Z", "cities": recs},
              open("data/scan/scan.json", "w"), indent=1)
    print(f"{'city':14s} {'vol/day':>8s} {'book2c':>7s} {'slip25':>6s} {'hit1':>5s} {'hitSet':>6s} {'Set':>4s} {'Mix':>4s} {'Uns':>4s} {'wind':>5s} station")
    for r in sorted(recs, key=lambda r: -(r.get("avg_daily_volume") or 0)):
        b = r.get("book_tomorrow") or {}
        g = r.get("regime") or {}
        hs = (r.get("hit_by_regime") or {}).get("Settled", {})
        print(f"{r['key']:14s} {r.get('avg_daily_volume') or 0:8d} {b.get('median_ask_2c') or 0:7d} "
              f"{str(b.get('median_slip_25_c')):>6s} {(r.get('backtest_day1') or {}).get('top_hit', 0):5.0%} "
              f"{hs.get('top_hit', 0):6.0%} {g.get('Settled', 0):4.0%} {g.get('Mixed', 0):4.0%} {g.get('Unsettled', 0):4.0%} "
              f"{(r.get('climate') or {}).get('wind_mean_kmh') or 0:5.1f} {r.get('icao')} {r.get('error') or ''}")


if __name__ == "__main__":
    main()
