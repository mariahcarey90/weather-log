#!/usr/bin/env python3
"""Screen candidate cities for the daily high-temperature markets.

For each candidate: read a recent Polymarket event's rules to find the resolution
station (the ICAO code in its Weather Underground link) and the unit, look the
station up on aviationweather.gov, then build the same 60-day table as
backfill.py (Open-Meteo previous-run forecasts 1 and 2 days ahead, per model,
vs the station's observed daily high from the IEM METAR archive) and score it
with backtest.py's method. Writes data/screen/backfill_screen.json and
data/screen/summary.json. Standard library only.
"""
import json, os, re, statistics, urllib.parse
from datetime import date, datetime, timedelta

import backtest
from backfill import DAYS, observed_highs, past_forecasts
from fetch import get

CANDIDATES = [
    # key (Polymarket slug), fallback ICAO if the rules can't be parsed, IANA tz
    # first screen (1 Oct 2026)
    ("ankara", "LTAC", "Europe/Istanbul"),
    ("paris", "LFPB", "Europe/Paris"),          # rules: Paris-Le Bourget Airport
    ("london", "EGLC", "Europe/London"),        # rules: London City Airport
    ("amsterdam", "EHAM", "Europe/Amsterdam"),
    ("munich", "EDDM", "Europe/Berlin"),
    ("shanghai", "ZSPD", "Asia/Shanghai"),
    ("miami", "KMIA", "America/New_York"),
    # wider scan (5 Oct 2026): every city seen with a daily high market, plus the current ones
    # for a like-for-like comparison. A key with no recent market still gets a forecast-skill score.
    ("madrid", "LEMD", "Europe/Madrid"),
    ("atlanta", "KATL", "America/New_York"),
    ("mexico-city", "MMMX", "America/Mexico_City"),
    ("lucknow", "VILK", "Asia/Kolkata"),
    ("dallas", "KDAL", "America/Chicago"),
    ("seattle", "KSEA", "America/Los_Angeles"),
    ("toronto", "CYYZ", "America/Toronto"),
    ("nyc", "KLGA", "America/New_York"),
    ("chicago", "KORD", "America/Chicago"),
    ("los-angeles", "KLAX", "America/Los_Angeles"),
    ("denver", "KBKF", "America/Denver"),
    ("houston", "KHOU", "America/Chicago"),
    ("austin", "KAUS", "America/Chicago"),
    ("phoenix", "KPHX", "America/Phoenix"),
    ("warsaw", "EPWA", "Europe/Warsaw"),
    ("milan", "LIMC", "Europe/Rome"),
    ("tel-aviv", "LLBG", "Asia/Jerusalem"),
    ("istanbul", "LTFM", "Europe/Istanbul"),
    ("seoul", "RKSI", "Asia/Seoul"),
    ("busan", "RKPK", "Asia/Seoul"),
    ("tokyo", "RJTT", "Asia/Tokyo"),
    ("beijing", "ZBAA", "Asia/Shanghai"),
    ("guangzhou", "ZGGG", "Asia/Shanghai"),
    ("shenzhen", "ZGSZ", "Asia/Shanghai"),
    ("chengdu", "ZUUU", "Asia/Shanghai"),
    ("wuhan", "ZHHH", "Asia/Shanghai"),
    ("singapore", "WSSS", "Asia/Singapore"),
    ("wellington", "NZWN", "Pacific/Auckland"),
    ("buenos-aires", "SAEZ", "America/Argentina/Buenos_Aires"),
    ("sao-paulo", "SBGR", "America/Sao_Paulo"),
    # hong-kong settles on the Hong Kong Observatory, not an airport METAR: not testable this way
]


def month_slug(d):
    return f"{d.strftime('%B').lower()}-{d.day}-{d.year}"


def market_info(key):
    """Rules text, unit, ICAO and a typical day's volume from recent events."""
    info = {"rules": None, "unit": None, "icao": None, "volumes": [], "slugs": []}
    for back in range(1, 8):
        d = date.today() - timedelta(days=back)
        slug = f"highest-temperature-in-{key}-on-{month_slug(d)}"
        try:
            ev = get(f"https://gamma-api.polymarket.com/events?slug={slug}")
        except Exception as e:
            continue
        if not ev:
            continue
        ev = ev[0]
        info["slugs"].append(slug)
        info["volumes"].append(float(ev.get("volume") or 0))
        if info["rules"] is None:
            text = ev.get("description") or ""
            if not text and ev.get("markets"):
                text = ev["markets"][0].get("description") or ""
            info["rules"] = text
            labels = " ".join((m.get("groupItemTitle") or "") for m in ev.get("markets", []))
            info["unit"] = "F" if "°F" in labels else "C"
            m = re.search(r"wunderground\.com/\S*?/([A-Z]{4})\b", text) or \
                re.search(r"timeseries\?site=([A-Za-z]{4})\b", text)
            if m:
                info["icao"] = m.group(1).upper()
            else:
                m = re.search(r"\b([A-Z]{4})\b(?=[^.]*(?:airport|station|Airport|Station))", text)
                info["icao"] = m.group(1) if m and m.group(1) not in ("NOAA", "NWS", "METAR") else None
    return info


def station_coords(icao):
    rows = get(f"https://aviationweather.gov/api/data/stationinfo?ids={icao}&format=json")
    r = rows[0] if isinstance(rows, list) and rows else rows
    return float(r["lat"]), float(r["lon"]), r.get("site") or r.get("name") or icao


def main():
    import sys
    only = set(sys.argv[1:]) or set(os.environ.get("SCREEN_ONLY", "").split()) - {""}
    os.makedirs("data/screen", exist_ok=True)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=DAYS)
    cities, meta = {}, {}
    if only and os.path.exists("data/screen/summary.json"):  # rerun a few: keep the others
        cities = json.load(open("data/screen/backfill_screen.json")).get("cities", {})
        meta = json.load(open("data/screen/summary.json")).get("meta", {})
    for key, fallback, tz in CANDIDATES:
        if only and key not in only:
            continue
        print("==", key)
        mi = market_info(key)
        icao, note = mi["icao"] or fallback, []
        if not mi["icao"]:
            note.append(f"station not found in rules; used {fallback}")
        try:
            lat, lon, name = station_coords(icao)
        except Exception as e:
            if icao == fallback:
                meta[key] = {"error": f"station lookup {icao}: {e}", "unit": mi["unit"], "slugs": mi["slugs"]}
                continue
            note.append(f"rules gave {icao} but it isn't a known station; used {fallback}")
            icao = fallback
            try:
                lat, lon, name = station_coords(icao)
            except Exception as e2:
                meta[key] = {"error": f"station lookup {icao}: {e2}", "unit": mi["unit"], "slugs": mi["slugs"]}
                continue
        unit = mi["unit"] or "C"
        st = {"key": key, "icao": icao,
              "lat": lat, "lon": lon, "tz": tz, "unit": unit}
        iem = dict(st, icao=icao[1:] if icao.startswith("K") else icao)
        try:
            obs = observed_highs(iem, start, end + timedelta(days=1))
        except Exception as e:
            obs = {}
            note.append(f"observations failed: {e}")
        fc = past_forecasts(st)
        cities[key] = {"unit": unit, "obs": obs, "fc": fc}
        meta[key] = {"icao": icao, "station": name, "lat": lat, "lon": lon, "unit": unit,
                     "obs_days": len(obs), "models": sorted((fc.get("day1") or {}).keys()),
                     "avg_daily_volume": round(statistics.fmean(mi["volumes"])) if mi["volumes"] else None,
                     "rules_excerpt": (mi["rules"] or "")[:700], "notes": note}
        print(json.dumps({k: v for k, v in meta[key].items() if k != "rules_excerpt"}))

    data = {"window": [start.isoformat(), end.isoformat()], "cities": cities}
    with open("data/screen/backfill_screen.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    # score with the same method and kernel as the live backtest
    res = backtest.run(data, 0.5)
    summary = {}
    for (city, lead), rows in sorted(res.items()):
        if rows:
            s = backtest.summary(rows)
            summary.setdefault(city, {})[lead] = {k: round(v, 3) for k, v in s.items()}
            print(f"{city:10s} {lead}: {s['days']} days | top {s['top_hit']:.0%} | within one {s['within_one']:.0%} "
                  f"| p(win) {s['avg_p_win']:.0%} | brier {s['brier']:.3f}")
    with open("data/screen/summary.json", "w") as f:
        json.dump({"generated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                   "window": data["window"], "meta": meta, "scores": summary}, f, indent=1)


if __name__ == "__main__":
    main()
