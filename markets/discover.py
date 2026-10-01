#!/usr/bin/env python3
"""One-off research for new market tracks (post counts, music charts).

Runs on GitHub Actions. Writes markets/data/discovery.json with:
  events:  recent and active Polymarket events whose titles look like post-count or
           music-chart markets (title, slug, dates, volume, liquidity, rules text,
           outcome labels/prices, resolution source)
  sources: whether candidate data sources respond, with a short sample of each
Standard library only.
"""
import json, os, re, sys, time, urllib.request, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get, UA  # noqa: E402

PATTERNS = {
    "posts": re.compile(r"(#|number of|how many)\s*(tweets|posts)|tweets?\b.*\b(week|month|\d)", re.I),
    "music": re.compile(r"spotify|billboard|hot 100|top song|#1 song|top artist|album|streams|apple music|youtube music", re.I),
}


def get_text(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def scan_events():
    found, seen = [], set()
    for closed in ("false", "true"):
        for offset in range(0, 4000, 500):
            q = {"limit": 500, "offset": offset, "closed": closed, "order": "startDate", "ascending": "false"}
            try:
                evs = get("https://gamma-api.polymarket.com/events?" + urllib.parse.urlencode(q))
            except Exception as e:
                print("scan error", closed, offset, e)
                break
            if not evs:
                break
            for ev in evs:
                title = ev.get("title") or ""
                kind = next((k for k, p in PATTERNS.items() if p.search(title)), None)
                if not kind or ev.get("slug") in seen:
                    continue
                seen.add(ev.get("slug"))
                mk = ev.get("markets") or []
                outs = []
                for m in mk:
                    try:
                        pr = json.loads(m.get("outcomePrices") or "[]")
                    except Exception:
                        pr = []
                    outs.append({"label": m.get("groupItemTitle") or m.get("question"),
                                 "yes": float(pr[0]) if pr else None, "volume": float(m.get("volume") or 0)})
                found.append({"kind": kind, "title": title, "slug": ev.get("slug"),
                              "start": ev.get("startDate"), "end": ev.get("endDate"), "closed": ev.get("closed"),
                              "volume": float(ev.get("volume") or 0), "liquidity": float(ev.get("liquidity") or 0),
                              "resolutionSource": ev.get("resolutionSource") or (mk[0].get("resolutionSource") if mk else None),
                              "rules": (ev.get("description") or (mk[0].get("description") if mk else "") or "")[:1500],
                              "n_outcomes": len(mk), "outcomes": outs[:40]})
            time.sleep(0.5)
    return found


SOURCES = {
    "xtracker_io": "https://www.xtracker.io/",
    "xtracker_polymarket": "https://xtracker.polymarket.com/",
    "xtracker_api_users": "https://xtracker.polymarket.com/api/users",
    "kworb_global_daily": "https://kworb.net/spotify/country/global_daily.html",
    "kworb_us_daily": "https://kworb.net/spotify/country/us_daily.html",
    "kworb_global_weekly": "https://kworb.net/spotify/country/global_weekly.html",
    "kworb_us_weekly": "https://kworb.net/spotify/country/us_weekly.html",
    "spotify_charts_api": "https://charts-spotify-com-service.spotify.com/public/v0/charts",
    "nitter_check": "https://nitter.net/elonmusk",
}


def probe_sources():
    out = {}
    for k, url in SOURCES.items():
        try:
            status, body = get_text(url)
            text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", body, flags=re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            out[k] = {"url": url, "ok": True, "status": status, "bytes": len(body), "sample": text[:1500]}
        except Exception as e:
            out[k] = {"url": url, "ok": False, "error": repr(e)[:300]}
    return out


def main():
    os.makedirs("markets/data", exist_ok=True)
    events = scan_events()
    sources = probe_sources()
    json.dump({"events": events, "sources": sources}, open("markets/data/discovery.json", "w"), indent=1, ensure_ascii=False)
    print(len(events), "events;", {k: v.get("ok") for k, v in sources.items()})


if __name__ == "__main__":
    main()
