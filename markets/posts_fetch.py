#!/usr/bin/env python3
"""Collect data for post-count markets ("Elon Musk # tweets September 29 - October 6, 2026?").

Runs on GitHub Actions (the counter and Polymarket APIs aren't reachable elsewhere).

Live mode (default) writes markets/data/posts_latest.json:
  trackings: every active counter window that has a Polymarket event: handle, window
             start/end, the counter's live total, the event's ranges with prices,
             token ids and order-book depth for buying each range
  resolved:  windows that ended in the last 10 days: final count from the post
             history, and Polymarket's winning range (priced 99%+)
and keeps markets/data/posts_hist/<handle>.json up to date: the time of every post
the counter has captured (unix seconds; no post text is stored), back to HIST_DAYS.

--backfill also writes markets/data/posts_backfill.json: windows that ended in the
last BACKFILL_DAYS, each with its ranges, the winner and every range's hourly price
history, for backtesting the model against the market.

The resolution source for these markets is the counter's "Post Counter" figure
(xtracker.polymarket.com); its API is public. Standard library only.
"""
import argparse, json, os, re, sys, time, urllib.parse
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get, book_depth  # noqa: E402

API = "https://xtracker.polymarket.com/api"
GAMMA = "https://gamma-api.polymarket.com"
OUT = "markets/data"
HIST_DAYS = 140
BACKFILL_DAYS = 70
RESOLVED_DAYS = 10
BOOK_MIN, BOOK_MAX = 0.005, 0.97


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def ts_of(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def slug_of(link):
    m = re.search(r"/event/([^/?#]+)", link or "")
    return m.group(1) if m else None


def parse_range(label):
    """'<20' -> (None, 19); '20-39' -> (20, 39); '500+' -> (500, None)."""
    s = (label or "").replace(",", "").strip()
    m = re.match(r"^(?:<|less than |under )\s*(\d+)$", s, re.I)
    if m:
        return None, int(m.group(1)) - 1
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d+)\s*(?:\+|or more)$", s, re.I)
    if m:
        return int(m.group(1)), None
    return None


def event_ranges(ev, books):
    out = []
    for m in ev.get("markets") or []:
        label = m.get("groupItemTitle") or m.get("question")
        rng = parse_range(label)
        if rng is None:
            continue
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            toks = json.loads(m.get("clobTokenIds") or "[]")
        except Exception:
            prices, toks = [], []
        yes = float(prices[0]) if prices else None
        depth = None
        if books and toks and yes is not None and BOOK_MIN <= yes <= BOOK_MAX and not m.get("closed"):
            try:
                depth = book_depth(toks[0])
            except Exception as e:
                depth = {"error": str(e)[:120]}
            time.sleep(0.1)
        out.append({"label": label, "lo": rng[0], "hi": rng[1], "market": yes,
                    "token": toks[0] if toks else None, "closed": bool(m.get("closed")), "depth": depth})
    out.sort(key=lambda b: -1 if b["lo"] is None else b["lo"])
    return out


def winner(ranges):
    w = [b for b in ranges if b["market"] is not None and b["market"] >= 0.99]
    return w[0]["label"] if len(w) == 1 else None


def update_history(handle, now):
    path = os.path.join(OUT, "posts_hist", f"{handle}.json")
    h = json.load(open(path)) if os.path.exists(path) else {"handle": handle, "from": None, "posts": {}}
    posts = h["posts"]  # platformId -> unix seconds
    start = now - HIST_DAYS * 86400
    if h.get("synced_to"):
        start = max(start, h["synced_to"] - 6 * 3600)  # small overlap catches late imports
    t = start
    while t < now:
        e = min(t + 86400, now)
        try:
            r = get(f"{API}/users/{handle}/posts?" + urllib.parse.urlencode({"startDate": iso(t), "endDate": iso(e)}))
            for p in (r or {}).get("data") or []:
                posts[str(p.get("platformId") or p.get("id"))] = ts_of(p["createdAt"])
        except Exception as ex:
            print("history error", handle, iso(t), ex)
            break
        t = e
        time.sleep(0.15)
    else:
        h["synced_to"] = now
    cutoff = now - HIST_DAYS * 86400
    h["posts"] = {k: v for k, v in posts.items() if v >= cutoff}
    h["from"] = max(cutoff, min(h["posts"].values())) if h["posts"] else cutoff
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(h, open(path, "w"), separators=(",", ":"))
    return sorted(h["posts"].values())


def count_between(times, a, b):
    import bisect
    return bisect.bisect_left(times, b + 1) - bisect.bisect_left(times, a)


def all_trackings(handles):
    out = []
    for hd in handles:
        try:
            r = get(f"{API}/users/{hd}/trackings?activeOnly=false")
            for t in (r or {}).get("data") or []:
                t["handle"] = hd
                out.append(t)
        except Exception as e:
            print("trackings error", hd, e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    now = int(time.time())
    os.makedirs(OUT, exist_ok=True)

    active = (get(f"{API}/trackings?activeOnly=true") or {}).get("data") or []
    handles = sorted({t["user"]["handle"] for t in active if t.get("user") and slug_of(t.get("marketLink"))})
    users = {t["user"]["handle"]: {"name": t["user"].get("name"), "lastSync": t["user"].get("lastSync"),
                                   "lastSyncError": t["user"].get("lastSyncError")} for t in active if t.get("user")}
    hist = {hd: update_history(hd, now) for hd in handles}

    live = []
    for t in active:
        slug = slug_of(t.get("marketLink"))
        hd = (t.get("user") or {}).get("handle")
        if not slug or hd not in hist:
            continue
        try:
            ev = get(f"{GAMMA}/events?slug={slug}")
        except Exception as e:
            print("event error", slug, e)
            continue
        if not ev:
            continue
        ev = ev[0]
        if ev.get("closed"):
            continue
        try:
            stats = (get(f"{API}/trackings/{t['id']}?includeStats=true") or {}).get("data", {}).get("stats") or {}
        except Exception as e:
            stats = {"error": str(e)[:120]}
        s, e = ts_of(t["startDate"]), ts_of(t["endDate"])
        live.append({"id": t["id"], "handle": hd, "name": users.get(hd, {}).get("name"), "title": t["title"],
                     "slug": slug, "start": t["startDate"], "end": t["endDate"],
                     "counterTotal": stats.get("total"), "counterStats": {k: stats.get(k) for k in ("pace", "percentComplete", "daysRemaining")},
                     "histCount": count_between(hist[hd], s, min(e, now)) if s <= now else 0,
                     "volume": float(ev.get("volume") or 0), "liquidity": float(ev.get("liquidity") or 0),
                     "ranges": event_ranges(ev, books=s - 86400 <= now)})
        time.sleep(0.2)

    trk = all_trackings(handles)
    resolved = []
    for t in trk:
        slug = slug_of(t.get("marketLink"))
        e = ts_of(t["endDate"])
        if not slug or not (now - RESOLVED_DAYS * 86400 <= e <= now):
            continue
        try:
            ev = get(f"{GAMMA}/events?slug={slug}")
        except Exception:
            continue
        if not ev:
            continue
        rng = event_ranges(ev[0], books=False)
        s = ts_of(t["startDate"])
        resolved.append({"handle": t["handle"], "title": t["title"], "slug": slug, "start": t["startDate"], "end": t["endDate"],
                         "histCount": count_between(hist[t["handle"]], s, e), "winner": winner(rng),
                         "settled": bool(ev[0].get("closed"))})
        time.sleep(0.2)

    json.dump({"fetched_at": iso(now), "users": users, "trackings": live, "resolved": resolved},
              open(os.path.join(OUT, "posts_latest.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(live)} live windows, {len(resolved)} recently ended, handles {handles}")
    for l in live:
        print(f"  {l['title']}: counter {l['counterTotal']} vs history {l['histCount']}")

    if a.backfill:
        bf = []
        for t in trk:
            slug = slug_of(t.get("marketLink"))
            s, e = ts_of(t["startDate"]), ts_of(t["endDate"])
            if not slug or not (now - BACKFILL_DAYS * 86400 <= e <= now - 3600):
                continue
            try:
                ev = get(f"{GAMMA}/events?slug={slug}")
            except Exception:
                continue
            if not ev:
                continue
            rng = event_ranges(ev[0], books=False)
            for b in rng:
                if not b["token"]:
                    continue
                try:
                    ph = get("https://clob.polymarket.com/prices-history?" + urllib.parse.urlencode(
                        {"market": b["token"], "startTs": s - 3 * 86400, "endTs": e, "fidelity": 60}))
                    b["history"] = [[p["t"], p["p"]] for p in (ph or {}).get("history") or []]
                except Exception as ex:
                    b["history"] = []
                    print("price history error", slug, b["label"], ex)
                time.sleep(0.1)
            bf.append({"handle": t["handle"], "title": t["title"], "slug": slug, "start": t["startDate"], "end": t["endDate"],
                       "histCount": count_between(hist[t["handle"]], s, e), "winner": winner(rng),
                       "volume": float(ev[0].get("volume") or 0), "ranges": rng})
        json.dump({"fetched_at": iso(now), "windows": bf}, open(os.path.join(OUT, "posts_backfill.json"), "w"),
                  separators=(",", ":"))
        ok = [w for w in bf if w["winner"]]
        agree = sum(1 for w in ok for b in w["ranges"] if b["label"] == w["winner"]
                    and (b["lo"] is None or b["lo"] <= w["histCount"]) and (b["hi"] is None or w["histCount"] <= b["hi"]))
        print(f"backfill: {len(bf)} windows, {len(ok)} with a winner; history count inside the winning range for {agree}")


if __name__ == "__main__":
    main()
