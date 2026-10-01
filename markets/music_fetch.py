#!/usr/bin/env python3
"""Collect data for the weekly Spotify chart markets ("#1 Spotify song this week? (October 2)",
"#2 Spotify song in the US this week? (October 2)").

Runs on GitHub Actions. The market settles on Spotify's weekly Top Songs chart (global or
USA) for the Friday-to-Thursday week; the chart "labeled October 2" covers Sep 25 - Oct 1.

Live mode (default) writes markets/data/music_latest.json:
  markets:  every open weekly #1/#2 market: region, rank, chart week, each named song with
            price, token and order-book depth
  tracks:   daily global and US streams (and chart position) for every candidate song over
            the last ~5 weeks, from kworb.net's per-track Spotify chart history; candidates are
            the top 15 on the latest global and US daily charts plus every song a market names
  resolved: markets that ended in the last 10 days with Polymarket's winner (99%+)
Also keeps markets/data/music_hist.json, our own growing copy of every candidate's daily
  streams (kworb shows only each song's last ~31 chart days), for later backtests.
--backfill also writes markets/data/music_backfill.json: settled weekly markets since
  BACKFILL_FROM with their songs, the winner and each song's hourly price history, plus
  the daily streams of every song they name.
Standard library only.
"""
import argparse, html, json, os, re, sys, time, unicodedata, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import get, book_depth, UA  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
KW = "https://kworb.net/spotify"
OUT = "markets/data"
BACKFILL_FROM = "2026-03-01"
TRACK_DAYS = 40
TITLE_RE = re.compile(r"^#([12])\s+(?:Spotify song|song on (US )?Spotify)( in the US)?\s+this week\?\s*\((\w+ \d+)\)", re.I)
PLACEHOLDER_RE = re.compile(r"^(Song \w{1,2}|Other)$", re.I)
BOOK_MIN, BOOK_MAX = 0.005, 0.97


def get_text(url, tries=3):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def norm(s):
    s = unicodedata.normalize("NFKD", html.unescape(s or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\((feat|with|w/|ft)\.?[^)]*\)", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def key_of(title, artist):
    first = re.split(r",|&| ft\.? | feat\.? | featuring | x | and | with ", artist, flags=re.I)[0]
    return norm(title) + "|" + norm(first)


# ---------------- kworb ----------------
ROW_LINK = re.compile(r'<a href="\.\./artist/[^"]+">([^<]+)</a>\s*-\s*<a href="\.\./track/([A-Za-z0-9]+)\.html">([^<]+)</a>')


def chart_rows(page):
    """(artist, title, track_id) in chart order from a kworb chart or totals page."""
    out = []
    for tr in page.split("<tr")[1:]:
        m = ROW_LINK.search(tr)
        if m:
            out.append((html.unescape(m.group(1)), html.unescape(m.group(3)), m.group(2)))
    return out


def cell_val(td):
    t = re.sub(r"<[^>]+>", "", td)
    m = re.match(r"\s*(\d+)\s*\(\s*([\d,]+)\s*\)", t)
    if m:
        return [int(m.group(1)), int(m.group(2).replace(",", ""))]
    return None


def parse_track(page, since):
    """{'daily': {date: {'global': [pos, streams], 'us': [...]}}, 'weekly': {...}} for dates >= since."""
    res = {}
    for kind in ("daily", "weekly"):
        m = re.search(r'<div class="%s">\s*<table>(.*?)</table>' % kind, page, re.S)
        rows = {}
        if m:
            parts = m.group(1).split("<tr")
            head = re.findall(r"<th>([^<]*)</th>", parts[1]) if len(parts) > 1 else []
            idx = {h: i for i, h in enumerate(head)}
            for tr in parts[2:]:
                tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
                if not tds:
                    continue
                d = re.sub(r"<[^>]+>", "", tds[0]).strip().replace("/", "-")
                if not re.match(r"\d{4}-\d{2}-\d{2}$", d) or d < since:
                    continue
                rec = {}
                for reg, col in (("global", "Global"), ("us", "US")):
                    if col in idx and idx[col] < len(tds):
                        v = cell_val(tds[idx[col]])
                        if v:
                            rec[reg] = v
                if rec:
                    rows[d] = rec
        res[kind] = rows
    return res


# ---------------- markets ----------------
def parse_event(ev, books):
    m = TITLE_RE.match(ev.get("title") or "")
    if not m:
        return None
    rank = int(m.group(1))
    region = "us" if (m.group(2) or m.group(3)) else "global"
    end = ev.get("endDate") or ""
    year = int(end[:4]) if end[:4].isdigit() else datetime.now(timezone.utc).year
    label = datetime.strptime(f"{m.group(4)} {year}", "%B %d %Y").date()
    week_start = label - timedelta(days=7)  # Friday the week began
    songs = []
    for mk in ev.get("markets") or []:
        lab = (mk.get("groupItemTitle") or mk.get("question") or "").strip()
        try:
            prices = json.loads(mk.get("outcomePrices") or "[]")
            toks = json.loads(mk.get("clobTokenIds") or "[]")
        except Exception:
            prices, toks = [], []
        yes = float(prices[0]) if prices else None
        placeholder = bool(PLACEHOLDER_RE.match(lab)) and lab.lower() != "other"
        if placeholder:
            continue
        title, artist = (lab.rsplit(" - ", 1) + [""])[:2] if " - " in lab else (lab, "")
        depth = None
        if books and toks and yes is not None and BOOK_MIN <= yes <= BOOK_MAX and not mk.get("closed"):
            try:
                depth = book_depth(toks[0])
            except Exception as e:
                depth = {"error": str(e)[:100]}
            time.sleep(0.1)
        songs.append({"label": lab, "title": title, "artist": artist, "key": key_of(title, artist) if artist else None,
                      "other": lab.lower() == "other", "market": yes, "token": toks[0] if toks else None,
                      "closed": bool(mk.get("closed")), "depth": depth})
    return {"slug": ev.get("slug"), "title": ev.get("title"), "rank": rank, "region": region,
            "weekStart": week_start.isoformat(), "weekEnd": (week_start + timedelta(days=6)).isoformat(),
            "chartLabel": label.isoformat(), "start": ev.get("startDate"), "end": end, "closed": bool(ev.get("closed")),
            "volume": float(ev.get("volume") or 0), "liquidity": float(ev.get("liquidity") or 0), "songs": songs}


def find_events(closed):
    seen, out = set(), []
    qs = ["Spotify song this week", "song on Spotify this week", "Spotify song in the US this week", "song on US Spotify this week"]
    for q in qs:
        try:
            r = get(f"{GAMMA}/public-search?" + urllib.parse.urlencode(
                {"q": q, "limit_per_type": 50, "events_status": "resolved" if closed else "active"}))
            evs = (r or {}).get("events") or []
        except Exception as e:
            print("search error", q, e)
            evs = []
        for ev in evs:
            if ev.get("slug") not in seen and TITLE_RE.match(ev.get("title") or ""):
                seen.add(ev["slug"])
                out.append(ev["slug"])
        time.sleep(0.3)
    for off in range(0, 600, 100):
        try:
            evs = get(f"{GAMMA}/events?" + urllib.parse.urlencode(
                {"tag_slug": "spotify", "closed": "true" if closed else "false", "limit": 100, "offset": off,
                 "order": "endDate", "ascending": "false"})) or []
        except Exception as e:
            print("tag error", e)
            break
        if not evs:
            break
        for ev in evs:
            if ev.get("slug") not in seen and TITLE_RE.match(ev.get("title") or ""):
                seen.add(ev["slug"])
                out.append(ev["slug"])
        time.sleep(0.3)
    return out


def load_event(slug, books):
    ev = get(f"{GAMMA}/events?slug={urllib.parse.quote(slug)}")
    return parse_event(ev[0], books) if ev else None


def winner(m):
    w = [s for s in m["songs"] if s["market"] is not None and s["market"] >= 0.99]
    return w[0]["label"] if len(w) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    os.makedirs(OUT, exist_ok=True)

    # kworb index: key -> track id, from current charts (and all-time totals for backfill)
    index, cands = {}, set()
    for reg in ("global", "us"):
        for kind in ("daily", "weekly"):
            try:
                rows = chart_rows(get_text(f"{KW}/country/{reg}_{kind}.html"))
            except Exception as e:
                print("kworb chart error", reg, kind, e)
                rows = []
            for i, (ar, ti, tid) in enumerate(rows):
                index.setdefault(key_of(ti, ar), tid)
                if kind == "daily" and i < 15:
                    cands.add(tid)
            time.sleep(1)
    if a.backfill:
        for page in ("global_weekly_totals", "us_daily_totals"):
            try:
                for ar, ti, tid in chart_rows(get_text(f"{KW}/country/{page}.html")):
                    index.setdefault(key_of(ti, ar), tid)
            except Exception as e:
                print("kworb totals error", page, e)
            time.sleep(1)

    # open markets
    live = []
    for slug in find_events(closed=False):
        try:
            m = load_event(slug, books=True)
        except Exception as e:
            print("event error", slug, e)
            continue
        if m and not m["closed"]:
            live.append(m)
        time.sleep(0.2)
    # recently ended
    resolved, bf_markets = [], []
    old = find_events(closed=True)
    for slug in old:
        try:
            m = load_event(slug, books=False)
        except Exception as e:
            print("event error", slug, e)
            continue
        if not m:
            continue
        m["winner"] = winner(m)
        if m["weekEnd"] >= (now - timedelta(days=12)).date().isoformat():
            resolved.append({k: m[k] for k in ("slug", "title", "rank", "region", "weekStart", "weekEnd", "chartLabel", "winner", "closed")})
        if a.backfill and m["weekStart"] >= BACKFILL_FROM and m["winner"]:
            bf_markets.append(m)
        time.sleep(0.2)

    def link(m, use):
        for s in m["songs"]:
            s["trackId"] = index.get(s["key"]) if s.get("key") else None
            if s["trackId"] and use:
                cands.add(s["trackId"])
    for m in live:
        link(m, True)
    for m in bf_markets:
        link(m, True)

    since = (now - timedelta(days=TRACK_DAYS)).date().isoformat()
    if a.backfill and bf_markets:
        since = min(since, (datetime.fromisoformat(min(m["weekStart"] for m in bf_markets)) - timedelta(days=14)).date().isoformat())
    tracks = {}
    for tid in sorted(cands):
        try:
            page = get_text(f"{KW}/track/{tid}.html")
            t = re.search(r"<title>(.*?) - Spotify Chart History</title>", page)
            name = html.unescape(t.group(1)) if t else tid
            tracks[tid] = {"name": name, **parse_track(page, since)}
        except Exception as e:
            print("track error", tid, e)
        time.sleep(0.7)

    # keep our own growing copy of the daily history (kworb shows only ~31 chart days per song)
    hpath = os.path.join(OUT, "music_hist.json")
    hist = json.load(open(hpath)) if os.path.exists(hpath) else {}
    keep = (now - timedelta(days=200)).date().isoformat()
    for tid, t in tracks.items():
        h = hist.setdefault(tid, {"name": t["name"], "daily": {}})
        h["name"] = t["name"]
        h["daily"].update(t["daily"])
        h["daily"] = {d: v for d, v in sorted(h["daily"].items()) if d >= keep}
    json.dump(hist, open(hpath, "w"), ensure_ascii=False, separators=(",", ":"))

    recent_since = (now - timedelta(days=TRACK_DAYS)).date().isoformat()
    live_tracks = {tid: {"name": t["name"], "daily": {d: v for d, v in t["daily"].items() if d >= recent_since},
                         "weekly": {d: v for d, v in t["weekly"].items() if d >= recent_since}} for tid, t in tracks.items()}
    json.dump({"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": live, "resolved": resolved,
               "tracks": live_tracks}, open(os.path.join(OUT, "music_latest.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"{len(live)} open markets, {len(resolved)} recently ended, {len(live_tracks)} tracks")
    for m in live:
        miss = [s["label"] for s in m["songs"] if not s["other"] and not s.get("trackId")]
        print(f"  {m['title']}: {len(m['songs'])} songs" + (f"; no chart history found for {miss}" if miss else ""))

    if a.backfill:
        for m in bf_markets:
            for s in m["songs"]:
                if s["token"]:
                    try:
                        st = int(datetime.fromisoformat(m["weekStart"]).replace(tzinfo=timezone.utc).timestamp()) - 3 * 86400
                        en = int(datetime.fromisoformat(m["chartLabel"]).replace(tzinfo=timezone.utc).timestamp()) + 2 * 86400
                        ph = get("https://clob.polymarket.com/prices-history?" + urllib.parse.urlencode(
                            {"market": s["token"], "startTs": st, "endTs": en, "fidelity": 60}))
                        s["history"] = [[p["t"], p["p"]] for p in (ph or {}).get("history") or []]
                    except Exception as e:
                        s["history"] = []
                        print("price history error", m["slug"], s["label"], e)
                    time.sleep(0.1)
        json.dump({"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "markets": bf_markets, "tracks": tracks},
                  open(os.path.join(OUT, "music_backfill.json"), "w"), ensure_ascii=False, separators=(",", ":"))
        linked = sum(1 for m in bf_markets for s in m["songs"] if s.get("trackId"))
        named = sum(1 for m in bf_markets for s in m["songs"] if not s["other"])
        print(f"backfill: {len(bf_markets)} settled markets; {linked}/{named} named songs linked to chart history")


if __name__ == "__main__":
    main()
