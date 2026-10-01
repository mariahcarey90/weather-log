#!/usr/bin/env python3
"""One-off: sample chart data sources for the Spotify weekly-chart markets.
Writes markets/data/music_probe.json with status, size and trimmed text of each URL."""
import json, os, re, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import UA  # noqa: E402

URLS = {
    "kworb_global_daily": "https://kworb.net/spotify/country/global_daily.html",
    "kworb_us_daily": "https://kworb.net/spotify/country/us_daily.html",
    "kworb_global_weekly": "https://kworb.net/spotify/country/global_weekly.html",
    "kworb_us_weekly": "https://kworb.net/spotify/country/us_weekly.html",
    "kworb_global_weekly_totals": "https://kworb.net/spotify/country/global_weekly_totals.html",
    "kworb_us_daily_totals": "https://kworb.net/spotify/country/us_daily_totals.html",
    "spotify_public_charts": "https://charts-spotify-com-service.spotify.com/public/v0/charts",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.status, r.read().decode("utf-8", "replace")


out = {}
for k, u in URLS.items():
    try:
        st, body = get(u)
        out[k] = {"url": u, "status": st, "bytes": len(body), "head": body[:4000],
                  "links": sorted(set(re.findall(r'href="([^"]+)"', body)))[:80]}
    except Exception as e:
        out[k] = {"url": u, "error": str(e)[:200]}

# a track page from the global daily chart
try:
    body = out["kworb_global_daily"]["head"]
    st, full = get(URLS["kworb_global_daily"])
    tl = re.findall(r'href="(\.\./track/[^"]+)"', full)[:2]
    for i, t in enumerate(tl):
        u = "https://kworb.net/spotify/" + t.replace("../", "")
        st, b = get(u)
        out[f"kworb_track_{i}"] = {"url": u, "status": st, "bytes": len(b), "head": b[:6000],
                                   "tail": b[-3000:], "n_rows": b.count("<tr")}
except Exception as e:
    out["kworb_track_error"] = str(e)[:200]

# the public charts API: which charts and fields it carries
try:
    d = json.loads(get(URLS["spotify_public_charts"])[1])
    charts = []
    for c in d.get("chartEntryViewResponses", []):
        md = c.get("displayChart", {}).get("chartMetadata", {})
        e0 = (c.get("entries") or [{}])[0]
        charts.append({"alias": md.get("alias"), "date": c.get("displayChart", {}).get("date"),
                       "dims": md.get("dimensions"), "n": len(c.get("entries") or []), "entry0": e0})
    out["spotify_public_charts_summary"] = charts
except Exception as e:
    out["spotify_public_charts_summary"] = str(e)[:200]

os.makedirs("markets/data", exist_ok=True)
json.dump(out, open("markets/data/music_probe.json", "w"), indent=1, ensure_ascii=False)
print({k: (v.get("status"), v.get("bytes"), v.get("error")) if isinstance(v, dict) else "list" for k, v in out.items()})
