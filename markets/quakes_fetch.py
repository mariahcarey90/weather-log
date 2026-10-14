#!/usr/bin/env python3
"""Fetch data for Polymarket's earthquake-count markets ("How many 5.5 or above
earthquakes worldwide, Oct 12-18?").

Runs on GitHub Actions. Writes:
  markets/data/quakes_hist.json    USGS catalogue: [unix seconds, magnitude, id] for every M5.0+
                                   event since HIST_START (rebuilt with --backfill, else the last
                                   45 days are re-read and merged so magnitude revisions land)
  markets/data/quakes_latest.json  open count markets (window, threshold, buckets, prices, order
                                   books) and recently resolved ones (winner), plus the rules text
These markets resolve on the USGS Earthquake Hazards Program catalogue. Standard library only.
"""
import csv, io, json, os, re, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get, UA, book_depth  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HIST = os.path.join(D, "quakes_hist.json")
LATEST = os.path.join(D, "quakes_latest.json")
HIST_START = "2021-01-01"
MIN_MAG = 5.0
ET = ZoneInfo("America/New_York")
BOOK_MIN_YES, BOOK_MAX_YES = 0.02, 0.98
ERRORS = []
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
MON3 = {k[:3]: v for k, v in MONTHS.items()}


def get_text(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def usgs(start, end):
    """[ts, mag, id] for M>=MIN_MAG between two ISO dates, in yearly chunks (the API caps at 20000 rows)."""
    out, s = [], datetime.fromisoformat(start)
    e_all = datetime.fromisoformat(end)
    while s < e_all:
        e = min(e_all, s + timedelta(days=366))
        q = {"format": "csv", "starttime": s.strftime("%Y-%m-%dT%H:%M:%S"), "endtime": e.strftime("%Y-%m-%dT%H:%M:%S"),
             "minmagnitude": MIN_MAG, "orderby": "time-asc"}
        for i in range(3):
            try:
                txt = get_text("https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(q))
                break
            except Exception as ex:
                if i == 2:
                    raise
                time.sleep(5 * (i + 1))
        for row in csv.DictReader(io.StringIO(txt)):
            try:
                t = datetime.fromisoformat(row["time"].replace("Z", "+00:00")).timestamp()
                out.append([int(t), float(row["mag"]), row["id"]])
            except Exception:
                continue
        s = e
    return out


def parse_date(txt, year_hint):
    """'October 12, 2026' / 'Oct 12' -> date."""
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2})(?:,?\s*(\d{4}))?", txt.strip())
    if not m:
        return None
    mon = MONTHS.get(m.group(1).lower()) or MON3.get(m.group(1).lower()[:3])
    if not mon:
        return None
    return datetime(int(m.group(3) or year_hint), mon, int(m.group(2)))


def window(ev):
    """(start, end) as aware datetimes (end exclusive) from the rules text, else the title."""
    rules = ev.get("description") or ""
    year = int((ev.get("endDate") or "2026")[:4])
    m = re.search(r"([A-Z][a-z]+\.? \d{1,2}(?:, \d{4})?)[^A-Za-z0-9]{0,4}(?:\(?\s*12:00 ?AM ET\)?)?\s*(?:to|until|through|-|–)\s*"
                  r"([A-Z][a-z]+\.? \d{1,2}(?:, \d{4})?)", rules)
    a = b = None
    if m:
        a, b = parse_date(m.group(1), year), parse_date(m.group(2), year)
    if not (a and b):
        t = ev.get("title") or ""
        m = re.search(r"([A-Z][a-z]+\.? \d{1,2})\s*[-–]\s*(?:([A-Z][a-z]+\.?) )?(\d{1,2})\b", t)
        if m:
            a = parse_date(m.group(1), year)
            b = parse_date(f"{m.group(2) or m.group(1).split()[0]} {m.group(3)}", year)
        else:
            m = re.search(r"\bin (\d{4})\b", t)  # yearly markets: "in 2026"
            if m:
                a, b = datetime(int(m.group(1)), 1, 1), datetime(int(m.group(1)), 12, 31)
            m2 = re.search(r"\bin ([A-Z][a-z]+)\b(?! \d)", t)  # monthly: "in October"
            if not a and m2 and m2.group(1).lower() in MONTHS:
                mon = MONTHS[m2.group(1).lower()]
                a = datetime(year, mon, 1)
                b = (datetime(year + (mon == 12), mon % 12 + 1, 1) - timedelta(days=1))
    if not (a and b):
        return None
    if b < a:
        b = b.replace(year=b.year + 1)
    start = a.replace(tzinfo=ET)
    end = (b + timedelta(days=1)).replace(tzinfo=ET)  # through 11:59 PM ET on the last day
    return start, end


def parse_bucket(label):
    s = (label or "").replace("≤", "<=").replace("≥", ">=").strip()
    n = [int(x) for x in re.findall(r"\d+", s)]
    if not n:
        return None, None
    if re.search(r"more than|over|\+|>", s, re.I):
        return (n[0] + 1 if re.search(r"more than|over|>(?!=)", s, re.I) else n[0]), None
    if re.search(r"<=|or (fewer|less)|at most", s, re.I):
        return None, n[0]
    if re.search(r"fewer than|less than|<", s, re.I):
        return None, n[0] - 1
    if len(n) >= 2:
        return n[0], n[1]
    return n[0], n[0]


def events():
    seen = {}
    for status in ("active", "resolved"):
        for q in ("earthquakes", "earthquakes worldwide", "or above earthquakes"):
            try:
                r = get("https://gamma-api.polymarket.com/public-search?" + urllib.parse.urlencode(
                    {"q": q, "limit_per_type": 50, "events_status": status}))
            except Exception as e:
                ERRORS.append(f"search {q} {status}: {e}")
                continue
            for ev in (r or {}).get("events") or []:
                t = ev.get("title") or ""
                if re.search(r"how many .*?(\d(?:\.\d)?) or (above|higher|greater)\b.*earthquake", t, re.I):
                    seen[ev.get("slug")] = ev
    return list(seen.values())


def market_detail(ev):
    full = get(f"https://gamma-api.polymarket.com/events?slug={ev['slug']}")
    return full[0] if full else ev


def main():
    backfill = "--backfill" in sys.argv
    now = datetime.now(timezone.utc)
    os.makedirs(D, exist_ok=True)
    hist = json.load(open(HIST)) if os.path.exists(HIST) and not backfill else []
    since = HIST_START if not hist else (now - timedelta(days=45)).strftime("%Y-%m-%d")
    fresh = usgs(since, (now + timedelta(days=1)).strftime("%Y-%m-%d"))
    cut = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())
    keep = [h for h in hist if h[0] < cut]
    ids = {h[2] for h in fresh}
    hist = sorted(keep + [h for h in fresh if h[2] not in {k[2] for k in keep}], key=lambda h: h[0])
    json.dump(hist, open(HIST, "w"), separators=(",", ":"))

    open_m, resolved = [], []
    for ev in events():
        try:
            ev = market_detail(ev)
        except Exception as e:
            ERRORS.append(f"event {ev.get('slug')}: {e}")
            continue
        t = ev.get("title") or ""
        thr = float(re.search(r"(\d(?:\.\d)?) or (above|higher|greater)", t, re.I).group(1))
        w = window(ev)
        if not w:
            ERRORS.append(f"no window parsed for {ev.get('slug')}")
            continue
        start, end = w
        buckets = []
        for m in ev.get("markets", []):
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
            except Exception:
                prices = []
            yes = float(prices[0]) if prices else None
            lo, hi = parse_bucket(m.get("groupItemTitle") or m.get("question"))
            depth = None
            if (not ev.get("closed") and yes is not None and BOOK_MIN_YES <= yes <= BOOK_MAX_YES
                    and not m.get("closed") and end > now):
                try:
                    tokens = json.loads(m.get("clobTokenIds") or "[]")
                    if tokens:
                        depth = book_depth(tokens[0])
                except Exception as e:
                    ERRORS.append(f"book {ev['slug']} {m.get('groupItemTitle')}: {e}")
            buckets.append({"label": m.get("groupItemTitle") or m.get("question"), "lo": lo, "hi": hi, "yes": yes,
                            "closed": m.get("closed"), "volume": float(m.get("volume") or 0), "depth": depth})
        rec = {"slug": ev["slug"], "title": t, "threshold": thr, "start": start.isoformat(), "end": end.isoformat(),
               "volume": float(ev.get("volume") or 0), "liquidity": float(ev.get("liquidity") or 0),
               "closed": bool(ev.get("closed")), "rules": (ev.get("description") or "")[:1500], "buckets": buckets}
        win = [b for b in buckets if (b["yes"] or 0) >= 0.99]
        if end <= now:
            rec["winner"] = win[0]["label"] if len(win) == 1 else None
            rec["countUSGS"] = sum(1 for h in hist if h[1] >= thr and start.timestamp() <= h[0] < end.timestamp())
            if now - end < timedelta(days=21):
                resolved.append(rec)
        else:
            open_m.append(rec)
    out = {"fetched_at": now.isoformat(timespec="seconds"), "histFrom": hist[0][0] if hist else None,
           "histEvents": len(hist), "open": open_m, "resolved": resolved, "errors": ERRORS}
    json.dump(out, open(LATEST, "w"), indent=1)
    print(json.dumps({"hist": len(hist), "open": [(o["title"], o["start"], o["end"]) for o in open_m],
                      "resolved": [(r["title"], r.get("winner"), r.get("countUSGS")) for r in resolved],
                      "errors": ERRORS}, indent=1))


if __name__ == "__main__":
    main()
