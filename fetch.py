#!/usr/bin/env python3
"""Fetch forecast, observation and Polymarket data for four airport stations.

Runs on GitHub Actions (which has open internet). Writes:
  data/latest.json              full snapshot of the most recent fetch
  data/history/YYYY-MM.jsonl    one compact line per fetch, for later analysis

Standard library only. All temperatures stored in degrees Celsius.

Weather data by Open-Meteo.com (CC BY 4.0), NOAA/NWS, aviationweather.gov.
Market data from Polymarket's public Gamma API.
"""
import json, os, re, statistics, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

UA = "weather-log/1.0 (personal research; github.com/mariahcarey90/weather-log)"

STATIONS = [
    {"key": "madrid", "name": "Madrid", "icao": "LEMD", "lat": 40.4719, "lon": -3.5626,
     "tz": "Europe/Madrid", "unit": "C", "slug": "madrid"},
    {"key": "atlanta", "name": "Atlanta", "icao": "KATL", "lat": 33.6367, "lon": -84.4281,
     "tz": "America/New_York", "unit": "F", "slug": "atlanta"},
    {"key": "mexico-city", "name": "Mexico City", "icao": "MMMX", "lat": 19.4363, "lon": -99.0721,
     "tz": "America/Mexico_City", "unit": "C", "slug": "mexico-city"},
    {"key": "lucknow", "name": "Lucknow", "icao": "VILK", "lat": 26.7606, "lon": 80.8893,
     "tz": "Asia/Kolkata", "unit": "C", "slug": "lucknow"},
]

DET_MODELS = ["ecmwf_ifs025", "ecmwf_aifs025_single", "gfs_seamless", "icon_seamless",
              "gem_seamless", "meteofrance_seamless", "ukmo_seamless", "jma_seamless",
              "cma_grapes_global", "bom_access_global"]
ENS_MODELS = ["ecmwf_ifs025", "gfs025", "icon_seamless"]

ERRORS = []


def get(url, tries=3, timeout=40):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last = f"HTTP {e.code}: {body}"
            if 400 <= e.code < 500 and e.code != 429:
                break
        except Exception as e:  # network hiccup
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} -> {last}")


def daily_max_by_local_date(times, values):
    """times are local ISO strings (Open-Meteo with timezone=...)."""
    out = {}
    for t, v in zip(times, values):
        if v is None:
            continue
        d = t[:10]
        out[d] = max(out.get(d, -999), v)
    return {d: round(v, 2) for d, v in out.items()}


def fetch_open_meteo(base, st, models):
    """Try all models in one call; on failure fall back to one call per model.
    Returns list of (requested_models, payload)."""
    def call(ms):
        q = {"latitude": st["lat"], "longitude": st["lon"], "hourly": "temperature_2m",
             "models": ",".join(ms), "timezone": st["tz"], "forecast_days": 3, "past_days": 1}
        return get(base + "?" + urllib.parse.urlencode(q))
    try:
        return [(models, call(models))]
    except RuntimeError as e:
        ERRORS.append(f"{st['key']} {base} batch failed, retrying per model: {e}")
        res = []
        for m in models:
            try:
                res.append(([m], call([m])))
            except RuntimeError as e2:
                ERRORS.append(f"{st['key']} {base} model {m} failed: {e2}")
        return res


def parse_hourly(results):
    """Return {series_name: {date: max}}. Unsuffixed keys get the single requested model's name."""
    series = {}
    for ms, p in results:
        h = p.get("hourly", {})
        times = h.get("time", [])
        for k in h:
            if not k.startswith("temperature_2m"):
                continue
            suffix = k[len("temperature_2m"):].lstrip("_")
            if len(ms) == 1:
                # single model: keys are "temperature_2m" or "temperature_2m_memberNN"
                name = (suffix + "_" if suffix else "") + ms[0]
            else:
                name = suffix
            series[name] = daily_max_by_local_date(times, h[k])
    return series


def fetch_models(st):
    return parse_hourly(fetch_open_meteo("https://api.open-meteo.com/v1/forecast", st, DET_MODELS))


def fetch_ensemble(st):
    s = parse_hourly(fetch_open_meteo("https://ensemble-api.open-meteo.com/v1/ensemble", st, ENS_MODELS))
    # keys look like "member01_ecmwf_ifs025_ensemble" or "ecmwf_ifs025_ensemble" (control run)
    groups = {}
    for name, bydate in s.items():
        m = re.sub(r"^member\d+_", "", name)
        m = re.sub(r"_ensemble$", "", m)
        for d, v in bydate.items():
            groups.setdefault(m, {}).setdefault(d, []).append(v)
    return groups


def fetch_nws_katl():
    st = next(s for s in STATIONS if s["key"] == "atlanta")
    pts = get(f"https://api.weather.gov/points/{st['lat']},{st['lon']}")
    grid = get(pts["properties"]["forecastGridData"])
    tz = ZoneInfo(st["tz"])
    out = {}
    for v in grid["properties"]["maxTemperature"]["values"]:
        start = datetime.fromisoformat(v["validTime"].split("/")[0])
        d = start.astimezone(tz).date().isoformat()
        if v["value"] is not None:
            out[d] = round(v["value"], 2)  # deg C
    return out


def fetch_metar():
    ids = ",".join(s["icao"] for s in STATIONS)
    rows = get(f"https://aviationweather.gov/api/data/metar?ids={ids}&format=json&hours=40")
    obs = {}
    for r in rows:
        icao = r.get("icaoId")
        t = r.get("obsTime") or r.get("reportTime")
        temp = r.get("temp")
        if icao is None or temp is None or t is None:
            continue
        if isinstance(t, (int, float)):
            ts = datetime.fromtimestamp(t, tz=timezone.utc)
        else:
            ts = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        obs.setdefault(icao, []).append({"t": ts.isoformat(), "temp": temp,
                                         "maxT": r.get("maxT"), "raw": r.get("rawOb")})
    return obs


def month_name_slug(d):
    dt = datetime.fromisoformat(d)
    return f"{dt.strftime('%B').lower()}-{dt.day}-{dt.year}"


def fetch_market(st, d):
    slug = f"highest-temperature-in-{st['slug']}-on-{month_name_slug(d)}"
    ev = get(f"https://gamma-api.polymarket.com/events?slug={slug}")
    if not ev:
        return None
    ev = ev[0]
    buckets = []
    for m in ev.get("markets", []):
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
        except Exception:
            prices = []
        yes = float(prices[0]) if prices else None
        buckets.append({
            "label": m.get("groupItemTitle") or m.get("question"),
            "yes": yes,
            "bestBid": m.get("bestBid"), "bestAsk": m.get("bestAsk"),
            "lastTrade": m.get("lastTradePrice"),
            "volume": float(m.get("volume") or 0),
            "closed": m.get("closed"),
            "threshold": m.get("groupItemThreshold"),
        })
    return {"slug": slug, "title": ev.get("title"), "closed": ev.get("closed"),
            "volume": float(ev.get("volume") or 0), "liquidity": float(ev.get("liquidity") or 0),
            "endDate": ev.get("endDate"), "buckets": buckets}


def main():
    now = datetime.now(timezone.utc)
    out = {"fetched_at": now.isoformat(timespec="seconds"), "stations": {}, "errors": ERRORS,
           "attribution": "Weather data by Open-Meteo.com (CC BY 4.0), NOAA/NWS, aviationweather.gov"}
    try:
        metar = fetch_metar()
    except RuntimeError as e:
        metar = {}
        ERRORS.append(f"metar: {e}")
    try:
        nws = fetch_nws_katl()
    except Exception as e:
        nws = {}
        ERRORS.append(f"nws: {e}")

    for st in STATIONS:
        tz = ZoneInfo(st["tz"])
        local_now = now.astimezone(tz)
        dates = [(local_now.date() + timedelta(days=i)).isoformat() for i in (-1, 0, 1)]
        rec = {k: st[k] for k in ("name", "icao", "lat", "lon", "tz", "unit")}
        rec["local_now"] = local_now.isoformat(timespec="minutes")
        try:
            models = fetch_models(st)
        except Exception as e:
            models = {}
            ERRORS.append(f"{st['key']} models: {e}")
        try:
            ens = fetch_ensemble(st)
        except Exception as e:
            ens = {}
            ERRORS.append(f"{st['key']} ensemble: {e}")
        obs = metar.get(st["icao"], [])
        days = {}
        for d in dates:
            day = {"models": {m: v[d] for m, v in models.items() if d in v},
                   "ensemble": {m: sorted(v[d]) for m, v in ens.items() if d in v}}
            if st["key"] == "atlanta" and d in nws:
                day["nws_grid_max"] = nws[d]
            day_obs = [o for o in obs if datetime.fromisoformat(o["t"]).astimezone(tz).date().isoformat() == d]
            if day_obs:
                day["obs_max_so_far"] = max(o["temp"] for o in day_obs)
                day["obs_count"] = len(day_obs)
                day["obs_last"] = max(o["t"] for o in day_obs)
            if d >= local_now.date().isoformat():
                try:
                    day["market"] = fetch_market(st, d)
                except Exception as e:
                    ERRORS.append(f"{st['key']} market {d}: {e}")
            days[d] = day
        rec["days"] = days
        rec["metar"] = obs
        out["stations"][st["key"]] = rec

    os.makedirs("data/history", exist_ok=True)
    with open("data/latest.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)

    # compact history line: model maxima, ensemble quantiles, obs, market yes prices
    compact = {"t": out["fetched_at"], "s": {}}
    for k, rec in out["stations"].items():
        cd = {}
        for d, day in rec["days"].items():
            e = {}
            for m, vals in day.get("ensemble", {}).items():
                if len(vals) >= 5:
                    q = statistics.quantiles(vals, n=10)
                    e[m] = {"n": len(vals), "mean": round(statistics.fmean(vals), 2),
                            "p10": round(q[0], 2), "p50": round(statistics.median(vals), 2), "p90": round(q[-1], 2)}
            mk = day.get("market") or {}
            cd[d] = {"models": day.get("models"), "ens": e, "nws": day.get("nws_grid_max"),
                     "obs": day.get("obs_max_so_far"),
                     "mkt": {b["label"]: b["yes"] for b in mk.get("buckets", [])} if mk else None}
        compact["s"][k] = cd
    with open(f"data/history/{now.strftime('%Y-%m')}.jsonl", "a") as f:
        f.write(json.dumps(compact, separators=(",", ":")) + "\n")

    print(json.dumps({"fetched_at": out["fetched_at"], "errors": ERRORS}, indent=1))
    n_models = {k: {d: len(v["models"]) for d, v in r["days"].items()} for k, r in out["stations"].items()}
    n_ens = {k: {d: {m: len(x) for m, x in v["ensemble"].items()} for d, v in r["days"].items()} for k, r in out["stations"].items()}
    print("models per day:", json.dumps(n_models))
    print("ensemble members per day:", json.dumps(n_ens))


if __name__ == "__main__":
    main()
