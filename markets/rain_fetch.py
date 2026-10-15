#!/usr/bin/env python3
"""Fetch data for Polymarket's monthly rainfall markets ("Precipitation in NYC in October?").

Runs on GitHub Actions. Writes markets/data/rain_latest.json with, per open market:
  the month window, buckets (prices + order books), the resolution station's rainfall so far
  this month (daily values), the ensemble forecast of daily rainfall for the next ~15 days
  (ECMWF 51 + GFS 31 members, at the station), and a 20-year daily climatology for the month
  (station history where ACIS has it, else ERA5 at the station's coordinates).
Plus recently resolved markets (winner) for scoring. Standard library only.

Observed rainfall, best source first:
  1. NOAA ACIS (data.rcc-acis.org) station data: US stations and GHCN-Daily international
     stations (Heathrow, Seoul) - the same station the market names
  2. Hong Kong Observatory open data (Hong Kong)
  3. Meteostat's free station archive (bulk.meteostat.net, WMO station id; usually 1-3 days behind)
  4. Open-Meteo's model analysis at the station's coordinates ("model-approx": a stand-in,
     flagged in the output, used only when the station hasn't reported)
"""
import csv, gzip, io, json, os, re, sys, time, urllib.parse, urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get, book_depth, UA  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LATEST = os.path.join(D, "rain_latest.json")
CLIM_DIR = os.path.join(D, "rain_clim")
ERRORS = []
DEBUG = {}
BOOK_MIN_YES, BOOK_MAX_YES = 0.02, 0.98
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]

# city name as in the market title -> resolution station (from each market's rules)
CITIES = {
    "nyc": {"names": ["nyc", "new york", "new york city"], "station": "Central Park, NY (NOAA)", "lat": 40.779,
            "lon": -73.969, "tz": "America/New_York", "unit": "in", "acis": "USW00094728"},
    "london": {"names": ["london"], "station": "Heathrow (Met Office)", "lat": 51.479, "lon": -0.449,
               "tz": "Europe/London", "unit": "mm", "acis": "UKM00003772"},
    "seoul": {"names": ["seoul"], "station": "Seoul (KMA)", "lat": 37.571, "lon": 126.966,
              "tz": "Asia/Seoul", "unit": "mm", "acis": "KSM00047108"},
    "hong-kong": {"names": ["hong kong"], "station": "Hong Kong Observatory", "lat": 22.302, "lon": 114.174,
                  "tz": "Asia/Hong_Kong", "unit": "mm", "hko": "HKO", "acis": "CHM00045005"},
    "seattle": {"names": ["seattle"], "station": "Seattle-Tacoma Intl (NOAA; check market rules)", "lat": 47.4447,
                "lon": -122.3144, "tz": "America/Los_Angeles", "unit": "in", "acis": "USW00024233"},
    "tokyo": {"names": ["tokyo"], "station": "Tokyo (JMA)", "lat": 35.692, "lon": 139.750,
              "tz": "Asia/Tokyo", "unit": "mm", "acis": "JA000047662"},
}


def city_of(title):
    t = title.lower()
    for k, c in CITIES.items():
        if any(re.search(rf"\bin {re.escape(n)}\b", t) for n in c["names"]):
            return k
    return None


def to_unit(mm, unit):
    return mm / 25.4 if unit == "in" else mm


def acis_daily(sid, sdate, edate, unit):
    """{date: value in market unit} from ACIS (pcpn in inches; 'T' trace -> 0; 'M' missing skipped)."""
    q = {"sid": sid, "sdate": sdate, "edate": edate, "elems": "pcpn", "output": "json"}
    r = get("https://data.rcc-acis.org/StnData?" + urllib.parse.urlencode(q))
    out = {}
    for d, v in (r or {}).get("data", []):
        if v in ("M", "S", "") or v is None:
            continue
        v = 0.0 if v == "T" else float(str(v).rstrip("AS"))
        out[d] = v if unit == "in" else v * 25.4
    return out


def get_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def hko_daily(station, y, m):
    q = {"dataType": "CLMRAIN", "rformat": "csv", "station": station, "year": y, "month": m}
    txt = get_bytes("https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?" + urllib.parse.urlencode(q)).decode("utf-8-sig", "replace")
    DEBUG[f"hko_{station}"] = txt[:400]
    out = {}
    for row in csv.reader(io.StringIO(txt)):
        try:
            yy, mm, dd, val = int(row[0]), int(row[1]), int(row[2]), row[3].strip()
            out[f"{yy:04d}-{mm:02d}-{dd:02d}"] = 0.0 if str(val).lower().startswith("trace") else float(val)
        except Exception:
            continue
    return out


def meteostat_daily(wmo, first, last, unit):
    """{date: value} from Meteostat's bulk daily file (prcp in mm); skips days with no value."""
    raw = gzip.decompress(get_bytes(f"https://bulk.meteostat.net/v2/daily/{wmo}.csv.gz")).decode("utf-8", "replace")
    lines = raw.strip().splitlines()
    DEBUG[f"meteostat_{wmo}"] = {"rows": len(lines), "first": lines[:1], "last": lines[-3:]}
    out = {}
    for row in csv.reader(io.StringIO(raw)):
        if len(row) > 4 and first <= row[0] <= last and row[4] not in ("", None):
            out[row[0]] = to_unit(float(row[4]), unit)
    return out


def openmeteo_past(c, days):
    q = {"latitude": c["lat"], "longitude": c["lon"], "daily": "precipitation_sum", "timezone": c["tz"],
         "past_days": days, "forecast_days": 1}
    r = get("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))
    dd = r.get("daily", {})
    return {d: to_unit(v, c["unit"]) for d, v in zip(dd.get("time", []), dd.get("precipitation_sum", [])) if v is not None}


def ensemble(c):
    """{member: {date: value}} daily sums of hourly precipitation, local days."""
    q = {"latitude": c["lat"], "longitude": c["lon"], "hourly": "precipitation", "timezone": c["tz"],
         "models": "ecmwf_ifs025,gfs025", "forecast_days": 16}
    r = get("https://ensemble-api.open-meteo.com/v1/ensemble?" + urllib.parse.urlencode(q))
    h = r.get("hourly", {})
    times = h.get("time", [])
    out = {}
    for k, vals in h.items():
        if not k.startswith("precipitation"):
            continue
        day = {}
        for t, v in zip(times, vals):
            if v is None:
                continue
            day[t[:10]] = day.get(t[:10], 0.0) + v
        if day:
            out[k] = {d: round(to_unit(v, c["unit"]), 4) for d, v in day.items()}
    return out


def climatology(key, c, month):
    """{year: {MM-DD: value}} for the month over the last 20 years; cached per city+month."""
    os.makedirs(CLIM_DIR, exist_ok=True)
    fp = os.path.join(CLIM_DIR, f"{key}_{month:02d}.json")
    if os.path.exists(fp):
        cached = json.load(open(fp))
        if cached.get("built", "")[:7] >= (date.today() - timedelta(days=40)).isoformat()[:7]:
            return cached
    this = date.today().year
    years, source = {}, None
    if c.get("acis"):
        try:
            for y in range(this - 20, this):
                last = (date(y + (month == 12), month % 12 + 1, 1) - timedelta(days=1)).day
                v = acis_daily(c["acis"], f"{y}-{month:02d}-01", f"{y}-{month:02d}-{last}", c["unit"])
                if len(v) >= last - 2:
                    years[str(y)] = {d[5:]: x for d, x in v.items()}
            source = "station (ACIS)" if len(years) >= 10 else None
        except Exception as e:
            ERRORS.append(f"{key} ACIS climatology: {e}")
    if source is None:
        years = {}
        q = {"latitude": c["lat"], "longitude": c["lon"], "daily": "precipitation_sum", "timezone": c["tz"],
             "start_date": f"{this - 20}-01-01", "end_date": f"{this - 1}-12-31"}
        r = get("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
        dd = r.get("daily", {})
        for d, v in zip(dd.get("time", []), dd.get("precipitation_sum", [])):
            if v is not None and int(d[5:7]) == month:
                years.setdefault(d[:4], {})[d[5:]] = round(to_unit(v, c["unit"]), 4)
        source = "ERA5 reanalysis at the station"
    out = {"built": date.today().isoformat(), "source": source, "years": years}
    json.dump(out, open(fp, "w"), separators=(",", ":"))
    return out


def parse_bucket(label):
    s = (label or "").replace("″", '"').replace("”", '"').replace("≤", "<").replace("≥", ">")
    n = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not n:
        return None, None
    if re.search(r"<|under|less than|below", s, re.I):
        return None, n[0]
    if re.search(r">|or more|\+|over|above", s, re.I):
        return n[0], None
    if len(n) >= 2:
        return n[0], n[1]
    return n[0], n[0]


def find_events():
    seen = {}
    for status in ("active", "resolved"):
        for q in ("precipitation", "precipitation in", "monthly precipitation"):
            try:
                r = get("https://gamma-api.polymarket.com/public-search?" + urllib.parse.urlencode(
                    {"q": q, "limit_per_type": 50, "events_status": status}))
            except Exception as e:
                ERRORS.append(f"search {q}: {e}")
                continue
            for ev in (r or {}).get("events") or []:
                if re.match(r"precipitation in .+ in [a-z]+\??$", (ev.get("title") or "").strip(), re.I):
                    seen[ev["slug"]] = ev
    return list(seen.values())


def month_window(ev):
    t = ev["title"]
    mon = MONTHS.index(re.search(r"in ([A-Za-z]+)\??$", t.strip()).group(1).lower()) + 1
    rules = ev.get("description") or ""
    m = re.search(rf"{MONTHS[mon - 1].title()}\s+(?:1,\s*)?(\d{{4}})", rules)
    year = int(m.group(1)) if m else int((ev.get("startDate") or ev.get("creationDate") or "2026")[:4])
    first = date(year, mon, 1)
    last = date(year + (mon == 12), mon % 12 + 1, 1) - timedelta(days=1)
    return first, last


def main():
    now = datetime.now(timezone.utc)
    open_m, resolved = [], []
    for ev in find_events():
        try:
            full = get(f"https://gamma-api.polymarket.com/events?slug={ev['slug']}")
            ev = full[0] if full else ev
            key = city_of(ev["title"])
            if not key:
                ERRORS.append(f"no station configured for: {ev['title']}")
                continue
            c = CITIES[key]
            first, last = month_window(ev)
            local_today = now.astimezone(ZoneInfo(c["tz"])).date()
            buckets = []
            for m in ev.get("markets", []):
                try:
                    prices = json.loads(m.get("outcomePrices") or "[]")
                except Exception:
                    prices = []
                yes = float(prices[0]) if prices else None
                lo, hi = parse_bucket(m.get("groupItemTitle") or m.get("question"))
                depth = None
                if not ev.get("closed") and not m.get("closed") and yes is not None and BOOK_MIN_YES <= yes <= BOOK_MAX_YES \
                        and local_today <= last:
                    try:
                        tokens = json.loads(m.get("clobTokenIds") or "[]")
                        if tokens:
                            depth = book_depth(tokens[0])
                    except Exception as e:
                        ERRORS.append(f"book {ev['slug']}: {e}")
                buckets.append({"label": m.get("groupItemTitle") or m.get("question"), "lo": lo, "hi": hi, "yes": yes,
                                "closed": m.get("closed"), "volume": float(m.get("volume") or 0), "depth": depth})
            rec = {"slug": ev["slug"], "title": ev["title"], "city": key, "station": c["station"], "unit": c["unit"],
                   "first": first.isoformat(), "last": last.isoformat(), "tz": c["tz"],
                   "localToday": local_today.isoformat(), "volume": float(ev.get("volume") or 0),
                   "liquidity": float(ev.get("liquidity") or 0), "rules": (ev.get("description") or "")[:1500],
                   "buckets": buckets}
            win = [b for b in buckets if (b["yes"] or 0) >= 0.99]
            if local_today > last:
                rec["winner"] = win[0]["label"] if len(win) == 1 else None
                if now.date() - last < timedelta(days=60):
                    resolved.append(rec)
                continue
            # observed so far (complete local days before today)
            obs, src = {}, None
            end_obs = min(last, local_today - timedelta(days=1))
            if end_obs >= first:
                if c.get("hko"):
                    try:
                        obs = {d: v for d, v in hko_daily(c["hko"], first.year, first.month).items()
                               if first.isoformat() <= d <= end_obs.isoformat()}
                        src = "Hong Kong Observatory" if obs else None
                    except Exception as e:
                        ERRORS.append(f"{key} HKO: {e}")
                if src is None and c.get("acis"):
                    try:
                        obs = acis_daily(c["acis"], first.isoformat(), end_obs.isoformat(), c["unit"])
                        DEBUG[f"acis_{c['acis']}"] = len(obs)
                        src = f"station via NOAA ACIS ({c['acis']})" if obs else None
                    except Exception as e:
                        ERRORS.append(f"{key} ACIS obs: {e}")
                if src is None and c.get("wmo"):
                    try:
                        obs = meteostat_daily(c["wmo"], first.isoformat(), end_obs.isoformat(), c["unit"])
                        src = f"station via Meteostat (WMO {c['wmo']})" if obs else None
                    except Exception as e:
                        ERRORS.append(f"{key} Meteostat obs: {e}")
                approx = {}
                try:
                    approx = {d: v for d, v in openmeteo_past(c, (local_today - first).days + 1).items()
                              if first.isoformat() <= d <= end_obs.isoformat()}
                except Exception as e:
                    ERRORS.append(f"{key} Open-Meteo past: {e}")
                missing = [d for d in (first + timedelta(days=i) for i in range((end_obs - first).days + 1))
                           if d.isoformat() not in obs]
                if missing:  # fill days the station hasn't reported yet with the model analysis
                    for d in missing:
                        if d.isoformat() in approx:
                            obs[d.isoformat()] = approx[d.isoformat()]
                    src = (src + f"; {len(missing)} day(s) filled from model analysis") if src else "model-approx"
                rec["daysFilled"] = len(missing)
                rec["obsApprox"] = approx
            rec["obs"] = obs
            rec["obsSource"] = src
            rec["obsThrough"] = end_obs.isoformat() if end_obs >= first else None
            try:
                rec["ensemble"] = ensemble(c)
            except Exception as e:
                ERRORS.append(f"{key} ensemble: {e}")
                rec["ensemble"] = {}
            try:
                cl = climatology(key, c, first.month)
                rec["climSource"], rec["climYears"] = cl["source"], len(cl["years"])
            except Exception as e:
                ERRORS.append(f"{key} climatology: {e}")
            open_m.append(rec)
        except Exception as e:
            ERRORS.append(f"event {ev.get('slug')}: {e}")
    out = {"fetched_at": now.isoformat(timespec="seconds"), "open": open_m, "resolved": resolved, "errors": ERRORS, "sourceDebug": DEBUG}
    os.makedirs(D, exist_ok=True)
    json.dump(out, open(LATEST, "w"), indent=1)
    print(json.dumps({"open": [(o["title"], o["obsSource"], round(sum(o["obs"].values()), 2), len(o["ensemble"]),
                                o.get("climSource"), o.get("climYears")) for o in open_m],
                      "resolved": [(r["title"], r.get("winner")) for r in resolved], "errors": ERRORS}, indent=1))


if __name__ == "__main__":
    main()
