#!/usr/bin/env python3
"""One-off: sample the post-counter API (the resolution source for post-count markets).
Writes markets/data/xtracker_probe.json with trimmed responses from each endpoint."""
import json, os, sys, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get  # noqa: E402

B = "https://xtracker.polymarket.com/api"


def trim(x, n=4):
    if isinstance(x, list):
        return [trim(v, n) for v in x[:n]] + ([f"... {len(x)} items"] if len(x) > n else [])
    if isinstance(x, dict):
        return {k: trim(v, n) for k, v in x.items()}
    return x


def call(path, **q):
    url = B + path + ("?" + urllib.parse.urlencode(q) if q else "")
    try:
        return {"url": url, "body": get(url)}
    except Exception as e:
        return {"url": url, "error": str(e)[:300]}


out = {}
out["trackings_active"] = call("/trackings", activeOnly="true")
out["elon_trackings_all"] = call("/users/elonmusk/trackings", activeOnly="false")
out["elon_posts_day"] = call("/users/elonmusk/posts", startDate="2026-09-29T16:00:00.000Z", endDate="2026-09-30T16:00:00.000Z")
out["elon_posts_old"] = call("/users/elonmusk/posts", startDate="2026-03-01T00:00:00.000Z", endDate="2026-03-02T00:00:00.000Z")
tid = None
try:
    tr = out["elon_trackings_all"]["body"]["data"]
    tid = next(t["id"] for t in tr if t.get("isActive"))
except Exception:
    pass
if tid:
    out["tracking_stats"] = call(f"/trackings/{tid}", includeStats="true")
summary = {}
for k, v in out.items():
    b = v.get("body")
    d = b.get("data") if isinstance(b, dict) else b
    summary[k] = {"url": v["url"], "error": v.get("error"),
                  "n": len(d) if isinstance(d, list) else None, "sample": trim(b)}
os.makedirs("markets/data", exist_ok=True)
json.dump(summary, open("markets/data/xtracker_probe.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: (v["n"], v["error"]) for k, v in summary.items()}))
