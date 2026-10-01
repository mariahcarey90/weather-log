#!/usr/bin/env python3
"""Measure how past forecasts did at each airport, to calibrate the odds.

For the last ~60 days:
  forecasts: Open-Meteo Previous Runs API, the forecast made 1 and 2 days ahead
             (temperature_2m_previous_day1/2), per model, daily max from hourly data
  observed:  hourly METARs from the Iowa Environmental Mesonet archive, max per
             local calendar day, rounded to whole degrees in the market's unit

Writes data/calibration.json: per city and lead (day1 = ~12-36 h ahead,
day2 = ~36-60 h), per model and for the model median: bias (forecast - observed,
deg C), sd of the error, mean absolute error, n. Standard library only.
"""
import csv, io, json, statistics, urllib.parse
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from fetch import STATIONS, get, UA
import urllib.request

DAYS = 60
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless",
          "meteofrance_seamless", "ukmo_seamless", "jma_seamless"]


def get_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", "replace")


OBS_LOWS = {}   # station key -> {date: observed daily low, market unit}; filled by observed_highs
FC_LOWS = {}    # station key -> {lead: {model: {date: forecast daily min}}}; filled by past_forecasts


def observed_highs(st, start, end):
    q = [("station", st["icao"] if st["icao"] != "KATL" else "ATL"), ("data", "tmpf"),
         ("year1", start.year), ("month1", start.month), ("day1", start.day),
         ("year2", end.year), ("month2", end.month), ("day2", end.day),
         ("tz", st["tz"]), ("format", "onlycomma"), ("latlon", "no"), ("missing", "empty"),
         ("trace", "empty"), ("direct", "no"), ("report_type", "3"), ("report_type", "4")]
    txt = get_text("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urllib.parse.urlencode(q))
    highs, lows = {}, {}
    for row in csv.DictReader(io.StringIO(txt)):
        t, v = row.get("valid"), row.get("tmpf")
        if not t or not v:
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        d = t[:10]
        highs[d] = max(highs.get(d, -999), f)
        lows[d] = min(lows.get(d, 999), f)
    conv = lambda f: round(f) if st["unit"] == "F" else round((f - 32) * 5 / 9)
    out = {d: conv(f) for d, f in highs.items()}
    OBS_LOWS[st["key"]] = {d: conv(f) for d, f in lows.items()}
    return out  # whole degrees in market unit


def past_forecasts(st):
    fields = ["temperature_2m_previous_day1", "temperature_2m_previous_day2"]
    res = {}
    for m in MODELS:
        q = {"latitude": st["lat"], "longitude": st["lon"], "hourly": ",".join(fields),
             "models": m, "timezone": st["tz"], "past_days": DAYS, "forecast_days": 1}
        try:
            p = get("https://previous-runs-api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(q))
        except Exception as e:
            print(f"  {st['key']} {m}: {e}")
            continue
        h = p.get("hourly", {})
        for fld, lead in zip(fields, ("day1", "day2")):
            vals = h.get(fld) or h.get(f"{fld}_{m}")
            if not vals:
                continue
            daily, dmin = {}, {}
            for t, v in zip(h["time"], vals):
                if v is not None:
                    daily[t[:10]] = max(daily.get(t[:10], -999), v)
                    dmin[t[:10]] = min(dmin.get(t[:10], 999), v)
            res.setdefault(lead, {})[m] = daily
            FC_LOWS.setdefault(st["key"], {}).setdefault(lead, {})[m] = dmin
    return res


def summarize(errs):
    if len(errs) < 3:
        return None
    return {"n": len(errs), "bias": round(statistics.fmean(errs), 2),
            "sd": round(statistics.stdev(errs), 2),
            "mae": round(statistics.fmean(abs(e) for e in errs), 2)}


def build(obs, fc, unit, start, end):
    """Per-lead calibration (per model and the model median) for one station."""
    city = {"obs_days": len(obs)}
    for lead, bymodel in fc.items():
        per_model, med_errs = {}, []
        for m, daily in bymodel.items():
            errs = []
            for d, v in daily.items():
                if d in obs and start.isoformat() <= d <= end.isoformat():
                    ov_c = (obs[d] - 32) * 5 / 9 if unit == "F" else obs[d]
                    errs.append(v - ov_c)
            s = summarize(errs)
            if s:
                per_model[m] = s
        dates = sorted(set().union(*[set(x) for x in bymodel.values()])) if bymodel else []
        for d in dates:
            if d in obs and start.isoformat() <= d <= end.isoformat():
                vs = [bymodel[m][d] for m in bymodel if d in bymodel[m]]
                if len(vs) >= 3:
                    ov_c = (obs[d] - 32) * 5 / 9 if unit == "F" else obs[d]
                    med_errs.append(statistics.median(vs) - ov_c)
        city[lead] = {"median": summarize(med_errs), "models": per_model}
    return city


def main():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=DAYS)
    cal = {"generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
           "window": [start.isoformat(), end.isoformat()], "cities": {}}
    daily_rows = {}
    for st in STATIONS:
        print(st["key"])
        obs = observed_highs(st, start, end + timedelta(days=1))
        fc = past_forecasts(st)
        city = {"obs_days": len(obs)}
        daily_rows[st["key"]] = {"unit": st["unit"], "obs": obs,
                                 "fc": {lead: bymodel for lead, bymodel in fc.items()}}
        for lead, bymodel in fc.items():
            per_model, med_errs = {}, []
            for m, daily in bymodel.items():
                errs = []
                for d, v in daily.items():
                    if d in obs and start.isoformat() <= d <= end.isoformat():
                        ov_c = (obs[d] - 32) * 5 / 9 if st["unit"] == "F" else obs[d]
                        errs.append(v - ov_c)
                s = summarize(errs)
                if s:
                    per_model[m] = s
            dates = sorted(set().union(*[set(x) for x in bymodel.values()])) if bymodel else []
            for d in dates:
                if d in obs and start.isoformat() <= d <= end.isoformat():
                    vs = [bymodel[m][d] for m in bymodel if d in bymodel[m]]
                    if len(vs) >= 3:
                        ov_c = (obs[d] - 32) * 5 / 9 if st["unit"] == "F" else obs[d]
                        med_errs.append(statistics.median(vs) - ov_c)
            city[lead] = {"median": summarize(med_errs), "models": per_model}
        cal["cities"][st["key"]] = city
        print(json.dumps(city, indent=1)[:1500])
    with open("data/calibration.json", "w") as f:
        json.dump(cal, f, indent=1)
    with open("data/backfill_daily.json", "w") as f:
        json.dump({"window": cal["window"], "cities": daily_rows}, f, separators=(",", ":"))

    # the same for daily lows (lowest-temperature markets)
    cal_low = {"generated": cal["generated"], "window": cal["window"], "cities": {}}
    rows_low = {}
    for st in STATIONS:
        obs, fc = OBS_LOWS.get(st["key"], {}), FC_LOWS.get(st["key"], {})
        if not obs or not fc:
            continue
        cal_low["cities"][st["key"]] = build(obs, fc, st["unit"], start, end)
        rows_low[st["key"]] = {"unit": st["unit"], "obs": obs, "fc": fc}
    with open("data/calibration_low.json", "w") as f:
        json.dump(cal_low, f, indent=1)
    with open("data/backfill_daily_low.json", "w") as f:
        json.dump({"window": cal["window"], "cities": rows_low}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
