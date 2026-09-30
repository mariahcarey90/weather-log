#!/usr/bin/env python3
"""Fetch forecast, observation and Polymarket data for four airport stations.

Runs on GitHub Actions (which has open internet). Writes:
  data/latest.json              full snapshot of the most recent fetch
  data/history/YYYY-MM.jsonl    one compact line per fetch, for later analysis

Standard library only. All temperatures stored in degrees Celsius.

Weather data by Open-Meteo.com (CC BY 4.0), NOAA/NWS, aviationweather.gov
(METARs and TAFs). Market data from Polymarket's public Gamma API.
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
    # added after the 60-day city screen (data/screen): all settle on these airports per the market rules
    {"key": "amsterdam", "name": "Amsterdam", "icao": "EHAM", "lat": 52.315, "lon": 4.79,
     "tz": "Europe/Amsterdam", "unit": "C", "slug": "amsterdam"},
    {"key": "shanghai", "name": "Shanghai", "icao": "ZSPD", "lat": 31.146, "lon": 121.8,
     "tz": "Asia/Shanghai", "unit": "C", "slug": "shanghai"},
    {"key": "ankara", "name": "Ankara", "icao": "LTAC", "lat": 40.128, "lon": 32.995,
     "tz": "Europe/Istanbul", "unit": "C", "slug": "ankara"},
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


REST_KEY = "rest"  # pseudo-date key: max over the remaining hours of the station's local today
NOW_LOCAL = {}     # station key -> "YYYY-MM-DDTHH:00" local, set in main()


def daily_max_by_local_date(times, values, now_local=None):
    """times are local ISO strings (Open-Meteo with timezone=...). Adds REST_KEY:
    the max over hours from now_local to the end of that local day (None if none left)."""
    out, rest = {}, None
    for t, v in zip(times, values):
        if v is None:
            continue
        d = t[:10]
        out[d] = max(out.get(d, -999), v)
        if now_local and d == now_local[:10] and t[:13] >= now_local[:13]:
            rest = v if rest is None else max(rest, v)
    res = {d: round(v, 2) for d, v in out.items()}
    if rest is not None:
        res[REST_KEY] = round(rest, 2)
    return res


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


HOURLY_TODAY = {}  # station key -> {model: {"YYYY-MM-DDTHH:00": temp}} for the local today


def parse_hourly(results, now_local=None, keep_today=None):
    """Return {series_name: {date: max}}. Unsuffixed keys get the single requested model's name.
    If keep_today is a dict, today's hourly values per series are stored in it."""
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
            series[name] = daily_max_by_local_date(times, h[k], now_local)
            if keep_today is not None and now_local:
                keep_today[name] = {t: round(v, 2) for t, v in zip(times, h[k])
                                    if v is not None and t[:10] == now_local[:10]}
    return series


COND_VARS = ["cloud_cover", "precipitation_probability", "cape", "wind_speed_10m",
             "wind_direction_10m", "pressure_msl", "visibility", "temperature_2m"]


def _circ_diff(a, b):
    d = abs(a - b) % 360
    return 360 - d if d > 180 else d


def fetch_conditions(st):
    """Weather regime per local date from Open-Meteo's best-match forecast: daytime
    (09-18 local) cloud, rain chance, storm energy (CAPE), wind; pressure change over
    the day; morning (06-10) visibility for fog; change in the day's max vs the day before."""
    q = {"latitude": st["lat"], "longitude": st["lon"], "hourly": ",".join(COND_VARS),
         "timezone": st["tz"], "forecast_days": 3, "past_days": 1}
    h = get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q)).get("hourly", {})
    times = h.get("time", [])
    by_date = {}
    for i, t in enumerate(times):
        by_date.setdefault(t[:10], []).append((int(t[11:13]), {v: (h.get(v) or [None] * len(times))[i] for v in COND_VARS}))
    out, prev_max, prev_p12 = {}, None, None
    for d in sorted(by_date):
        rows = by_date[d]
        day = [r for hr, r in rows if 9 <= hr <= 18]
        morn = [r for hr, r in rows if 6 <= hr <= 10]
        vals = lambda rs, k: [r[k] for r in rs if r.get(k) is not None]
        tmax = max(vals([r for _, r in rows], "temperature_2m") or [None]) if rows else None
        wd9 = next((r["wind_direction_10m"] for hr, r in rows if hr == 9 and r.get("wind_direction_10m") is not None), None)
        wd17 = next((r["wind_direction_10m"] for hr, r in rows if hr == 17 and r.get("wind_direction_10m") is not None), None)
        # same hour vs the day before, so the normal daily pressure cycle cancels out
        p12 = next((r["pressure_msl"] for hr, r in rows if hr == 12 and r.get("pressure_msl") is not None), None)
        c = {
            "cloud_mean": round(statistics.fmean(vals(day, "cloud_cover")), 0) if vals(day, "cloud_cover") else None,
            "rain_chance_max": max(vals(day, "precipitation_probability") or [None]) if vals(day, "precipitation_probability") else None,
            "cape_max": max(vals(day, "cape")) if vals(day, "cape") else None,
            "wind_mean_kmh": round(statistics.fmean(vals(day, "wind_speed_10m")), 1) if vals(day, "wind_speed_10m") else None,
            "wind_shift_deg": round(_circ_diff(wd9, wd17)) if wd9 is not None and wd17 is not None else None,
            "pressure_change_hpa": round(p12 - prev_p12, 1) if p12 is not None and prev_p12 is not None else None,
            "morning_vis_min_m": min(vals(morn, "visibility")) if vals(morn, "visibility") else None,
            "tmax_change_c": round(tmax - prev_max, 1) if tmax is not None and prev_max is not None else None,
        }
        prev_max, prev_p12 = tmax, p12
        out[d] = c
    return out


def fetch_models(st):
    keep = HOURLY_TODAY.setdefault(st["key"], {})
    return parse_hourly(fetch_open_meteo("https://api.open-meteo.com/v1/forecast", st, DET_MODELS),
                        NOW_LOCAL.get(st["key"]), keep)


def fetch_ensemble(st):
    s = parse_hourly(fetch_open_meteo("https://ensemble-api.open-meteo.com/v1/ensemble", st, ENS_MODELS), NOW_LOCAL.get(st["key"]))
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


def fetch_nws_hourly_katl():
    """NWS hourly forecast for the KATL grid point, in deg C: {local_iso_hour: temp}."""
    st = next(s for s in STATIONS if s["key"] == "atlanta")
    pts = get(f"https://api.weather.gov/points/{st['lat']},{st['lon']}")
    # default US units: whole deg F is finer than whole deg C for 2 F market buckets
    hourly = get(pts["properties"]["forecastHourly"])
    tz = ZoneInfo(st["tz"])
    out = {}
    for p in hourly["properties"]["periods"]:
        t = p.get("temperature")
        if isinstance(t, dict):  # newer schema: {"unitCode": ..., "value": ...}
            unit, t = t.get("unitCode", ""), t.get("value")
        else:
            unit = p.get("temperatureUnit", "C")
        if t is None:
            continue
        if unit.endswith("degF") or unit == "F":
            t = (t - 32) * 5 / 9
        start = datetime.fromisoformat(p["startTime"]).astimezone(tz)
        out[start.strftime("%Y-%m-%dT%H:00")] = round(t, 2)
    return out


TAF_TX = re.compile(r"\bTX(M?)(\d{2})/(\d{2})(\d{2})Z")
TAF_HEAD = re.compile(r"\b([A-Z]{4})\s+(\d{2})(\d{2})(\d{2})Z")


def resolve_ddhh(dd, hh, near):
    """TAF day-of-month + hour (UTC) -> the datetime closest to `near`."""
    best = None
    for dm in (-1, 0, 1):
        y, m = near.year, near.month + dm
        if m == 0:
            y, m = y - 1, 12
        elif m == 13:
            y, m = y + 1, 1
        try:
            cand = datetime(y, m, dd, tzinfo=timezone.utc) + timedelta(hours=hh)  # hh may be 24
        except ValueError:
            continue
        if best is None or abs(cand - near) < abs(best - near):
            best = cand
    return best


def parse_taf_tx(raw, now):
    """Forecast maximum temperatures (TX groups) in a raw TAF: [{"t": utc iso, "value": deg C}]."""
    head = TAF_HEAD.search(raw)
    issued = resolve_ddhh(int(head.group(2)), int(head.group(3)), now) if head else now
    out = []
    for neg, temp, dd, hh in TAF_TX.findall(raw):
        t = resolve_ddhh(int(dd), int(hh), issued)
        if t is not None:
            out.append({"t": t.isoformat(), "value": -int(temp) if neg else int(temp)})
    return out


def fetch_taf(now):
    """Latest TAF per station: {icao: {"raw", "issued", "tx": [...]}}. Not every office issues TX groups."""
    ids = ",".join(s["icao"] for s in STATIONS)
    rows = get(f"https://aviationweather.gov/api/data/taf?ids={ids}&format=json")
    out = {}
    for r in rows:
        icao, raw = r.get("icaoId"), r.get("rawTAF")
        if not icao or not raw:
            continue
        issued = r.get("issueTime")
        prev = out.get(icao)
        if prev and str(prev.get("issued") or "") >= str(issued or ""):
            continue  # keep the newest TAF per station
        out[icao] = {"raw": raw, "issued": issued, "tx": parse_taf_tx(raw, now)}
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


BOOK_MIN_YES, BOOK_MAX_YES = 0.02, 0.97  # only read books for buckets that are still live


def book_depth(token_id):
    """What buying the YES side would cost now, from the public CLOB order book:
    best ask, dollars on offer within 1c and 2c of it, and the average price a
    $10 and a $25 market order would pay (None if the book can't fill it)."""
    book = get(f"https://clob.polymarket.com/book?token_id={token_id}")
    asks = sorted(((float(a["price"]), float(a["size"])) for a in book.get("asks", [])), key=lambda x: x[0])
    bids = sorted(((float(b["price"]), float(b["size"])) for b in book.get("bids", [])), key=lambda x: -x[0])
    if not asks:
        bb = bids[0][0] if bids else None
        return {"best_ask": None, "best_bid": bb,
                "bid_usd_within_2c": round(sum(p * q for p, q in bids if p >= bb - 0.02 - 1e-9), 2) if bids else 0.0}
    best = asks[0][0]
    within = lambda c: round(sum(p * q for p, q in asks if p <= best + c + 1e-9), 2)

    def avg_fill(dollars):
        left, shares = dollars, 0.0
        for p, q in asks:
            take = min(left, p * q)
            shares += take / p
            left -= take
            if left <= 1e-9:
                return round(dollars / shares, 4)
        return None
    bb = bids[0][0] if bids else None
    bid_usd_2c = round(sum(p * q for p, q in bids if bb is not None and p >= bb - 0.02 - 1e-9), 2) if bids else 0.0
    return {"best_ask": best, "best_bid": bb,
            "usd_at_best": within(0), "usd_within_1c": within(0.01), "usd_within_2c": within(0.02),
            "avg_fill_10": avg_fill(10), "avg_fill_25": avg_fill(25),
            "bid_usd_within_2c": bid_usd_2c}


def fetch_market(st, d, with_books=False):
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
        depth = None
        if with_books and yes is not None and BOOK_MIN_YES <= yes <= BOOK_MAX_YES and not m.get("closed"):
            try:
                tokens = json.loads(m.get("clobTokenIds") or "[]")
                if tokens:
                    depth = book_depth(tokens[0])
            except Exception as e:
                ERRORS.append(f"{st['key']} book {d} {m.get('groupItemTitle')}: {e}")
        buckets.append({
            "depth": depth,
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
    try:
        nws_hourly = fetch_nws_hourly_katl()
    except Exception as e:
        nws_hourly = {}
        ERRORS.append(f"nws hourly: {e}")
    try:
        tafs = fetch_taf(now)
    except Exception as e:
        tafs = {}
        ERRORS.append(f"taf: {e}")

    for st in STATIONS:
        tz = ZoneInfo(st["tz"])
        local_now = now.astimezone(tz)
        dates = [(local_now.date() + timedelta(days=i)).isoformat() for i in (-1, 0, 1)]
        rec = {k: st[k] for k in ("name", "icao", "lat", "lon", "tz", "unit")}
        rec["local_now"] = local_now.isoformat(timespec="minutes")
        NOW_LOCAL[st["key"]] = local_now.strftime("%Y-%m-%dT%H:00")
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
        try:
            conds = fetch_conditions(st)
        except Exception as e:
            conds = {}
            ERRORS.append(f"{st['key']} conditions: {e}")
        obs = metar.get(st["icao"], [])
        days = {}
        for d in dates:
            day = {"models": {m: v[d] for m, v in models.items() if d in v},
                   "ensemble": {m: sorted(v[d]) for m, v in ens.items() if d in v}}
            if d == local_now.date().isoformat():
                # max over the rest of today (from the current local hour), for same-day markets
                day["rest_of_day_from"] = NOW_LOCAL[st["key"]]
                day["models_rest"] = {m: v[REST_KEY] for m, v in models.items() if REST_KEY in v}
                day["ensemble_rest"] = {m: sorted(v[REST_KEY]) for m, v in ens.items() if REST_KEY in v}
                day["models_hourly"] = {m: v for m, v in HOURLY_TODAY.get(st["key"], {}).items() if v}
            if d in conds:
                day["conditions"] = conds[d]
            if st["key"] == "atlanta" and d in nws:
                day["nws_grid_max"] = nws[d]
            if st["key"] == "atlanta":
                hrs = {h: v for h, v in nws_hourly.items() if h[:10] == d}
                if hrs:
                    day["nws_hourly_max"] = max(hrs.values())
                    rest = [v for h, v in hrs.items() if h >= NOW_LOCAL[st["key"]]]
                    if d == local_now.date().isoformat() and rest:
                        day["nws_hourly_rest"] = max(rest)
            taf = tafs.get(st["icao"])
            if taf:
                tx = [x for x in taf["tx"]
                      if datetime.fromisoformat(x["t"]).astimezone(tz).date().isoformat() == d]
                if tx:
                    day["taf_max"] = max(x["value"] for x in tx)
                    day["taf_issued"] = taf["issued"]
            day_obs = [o for o in obs if datetime.fromisoformat(o["t"]).astimezone(tz).date().isoformat() == d]
            if day_obs:
                day["obs_max_so_far"] = max(o["temp"] for o in day_obs)
                day["obs_count"] = len(day_obs)
                day["obs_last"] = max(o["t"] for o in day_obs)
            if d >= local_now.date().isoformat():
                try:
                    day["market"] = fetch_market(st, d, with_books=True)
                except Exception as e:
                    ERRORS.append(f"{st['key']} market {d}: {e}")
            days[d] = day
        rec["days"] = days
        # settled (or settling) markets for the last two local days, for scoring outcomes
        resolved = {}
        for back in (2, 1):
            d = (local_now.date() - timedelta(days=back)).isoformat()
            try:
                m = fetch_market(st, d)
            except Exception as e:
                ERRORS.append(f"{st['key']} resolved market {d}: {e}")
                continue
            if m:
                win = [b["label"] for b in m["buckets"] if b.get("yes") is not None and b["yes"] >= 0.99]
                m["winner"] = win[0] if len(win) == 1 and all(
                    (b.get("yes") or 0) <= 0.01 for b in m["buckets"] if b["label"] != win[0]) else None
                m["settled"] = bool(m["buckets"]) and all(b.get("closed") for b in m["buckets"])
                resolved[d] = m
        rec["resolved"] = resolved
        rec["metar"] = obs
        if st["icao"] in tafs:
            rec["taf"] = {k: tafs[st["icao"]][k] for k in ("raw", "issued")}
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
                     "nws_h": day.get("nws_hourly_max"), "taf": day.get("taf_max"),
                     "cond": day.get("conditions"),
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
