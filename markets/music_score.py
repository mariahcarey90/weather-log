#!/usr/bin/env python3
"""Score open Spotify weekly-chart markets and run their paper trading.

Usage:
  python3 markets/music_score.py --db DIR --out DIR

Reads markets/data/music_latest.json (and music_backtest.json for the track record) plus
exported tracker collections in DIR (paper, paper_results, music_outcomes; any may be
missing). Writes to --out:
  music_batch.json  ArtifactData batch entries with file_path, all create-only:
                    music_snapshots, new music_outcomes, paper trades and results
and prints a short report.

Versions (paper only): music_model (music_model.py) and music_blend (50/50 with the
market). Paper rule as weather: one $25 per version per market on the biggest edge
(our odds minus a $25 order's average price >= 8 points, best ask >= 5c, market 3-97%),
not before a day of the chart week is counted, and not while the market's prices are
unformed (adding up to more than 125%). Standard library only.
"""
import argparse, glob, json, os, re, sys, time
from datetime import date, datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import music_model as mm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EDGE, MIN_ASK, STAKE = 0.08, 0.05, 25.0


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def slug_label(label):
    return re.sub(r"[^A-Za-z0-9]+", "", label)[:40] or "x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%MZ")
    run_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    L = json.load(open(os.path.join(D, "music_latest.json")))
    bt_path = os.path.join(D, "music_backtest.json")
    bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
    params = dict(mm.DEFAULT, **(bt.get("params") or {}))
    tracks = L["tracks"]
    latest_day = {reg: max((d for t in tracks.values() for d, v in t["daily"].items() if reg in v), default=None)
                  for reg in ("global", "us")}

    snaps, lines = [], []
    for m in L["markets"]:
        reg = m["region"]
        ws = date.fromisoformat(m["weekStart"])
        ku = date.fromisoformat(latest_day[reg]) if latest_day[reg] else None
        if ku and ku > date.fromisoformat(m["weekEnd"]):
            ku = date.fromisoformat(m["weekEnd"])
        # candidates: every song the market names plus anything else charting high
        top = sorted(((sum((t["daily"].get(d) or {}).get(reg, [0, 0])[1] for d in t["daily"] if d >= m["weekStart"]), tid)
                      for tid, t in tracks.items()), reverse=True)[:12]
        cands = list({s["trackId"] for s in m["songs"] if s.get("trackId")} | {tid for _, tid in top})
        res, info = mm.predict(tracks, reg, ws, ku if ku and ku >= ws else None, cands, params)
        ours = mm.market_probs(m, res, info, params)
        # an "Other" outcome left at 50c with no book is a placeholder, not a price
        for s in m["songs"]:
            if s.get("other") and s.get("market") == 0.5 and not (s.get("depth") or {}).get("best_ask"):
                s["market"] = None
        unformed = sum(s["market"] or 0 for s in m["songs"]) > 1.25
        key = "p1" if m["rank"] == 1 else "p2"
        songs = []
        for s, p in zip(m["songs"], ours):
            r = res.get(s.get("trackId")) or {}
            mk = None if unformed else s.get("market")
            songs.append({"label": s["label"], "trackId": s.get("trackId"), "other": s.get("other", False),
                          "market": mk, "marketRaw": s.get("market"), "model": None if p is None else round(p, 4),
                          "streamsSoFar": r.get("known"), "expectedWeek": r.get("mean"), "new": r.get("new"),
                          "depth": s.get("depth"),
                          "variants": {} if p is None else {"music_model": round(p, 4),
                                                            "music_blend": round(0.5 * p + 0.5 * mk, 4) if mk is not None else round(p, 4)}})
        # standings among all candidates (named or not)
        standings = sorted(({"trackId": t, "name": tracks.get(t, {}).get("name", t), "soFar": v["known"], "expected": v["mean"],
                             "pTop": round(v[key], 4)} for t, v in res.items()), key=lambda x: -x["expected"])[:6]
        priced = [x for x in songs if x["model"] is not None]
        top_ours = max(priced, key=lambda x: x["model"]) if priced else None
        top_mkt = None if unformed else max(songs, key=lambda x: x["market"] or 0)
        hours_left = round((datetime.fromisoformat(m["end"].replace("Z", "+00:00")) - now).total_seconds() / 3600, 1) if m.get("end") else None
        doc = {"kind": "music", "slug": m["slug"], "title": m["title"], "region": reg, "rank": m["rank"],
               "city": f"spotify_{reg}_{m['rank']}", "marketDate": m["chartLabel"], "weekStart": m["weekStart"],
               "weekEnd": m["weekEnd"], "runAt": run_at, "hoursLeft": hours_left, "daysCounted": info.get("knownDays", 0),
               "countedThrough": ku.isoformat() if ku else None, "early": info.get("early", True),
               "pricesUnformed": unformed, "volume": m.get("volume"), "liquidity": m.get("liquidity"),
               "songs": songs, "standings": standings, "backtest": (bt.get("by_market") or {}).get(f"{reg}_{m['rank']}"),
               "topPick": top_ours["label"] if top_ours else None, "marketTop": top_mkt["label"] if top_mkt else None}
        snaps.append({"doc_id": f"{m['slug'][:120]}_{stamp}", "data": doc})
        gaps = [((x["model"] or 0) - x["market"], x) for x in songs if x["market"] is not None and x["model"] is not None]
        g, gb = max(gaps, key=lambda z: z[0]) if gaps else (0, None)
        lines.append(f"{m['title']}: {doc['daysCounted']}/7 days counted (through {doc['countedThrough']}); "
                     + (f"ours {top_ours['label']} {top_ours['model']*100:.0f}%" if top_ours else "no odds")
                     + (f" vs market's {top_mkt['label']} {(top_mkt['market'] or 0)*100:.0f}%" if top_mkt else "; market prices not formed")
                     + (f"; biggest gap {gb['label']} +{g*100:.0f} pts" if gb and g >= 0.10 else "")
                     + "; leaders so far: " + ", ".join(f"{x['name'].split(' - ')[-1]} {x['soFar']/1e6:.1f}M" for x in standings[:3]))

    # outcomes
    have = load_dir(os.path.join(a.db, "music_outcomes"))
    outcomes = []
    for r in L.get("resolved", []):
        if r.get("winner") and r["slug"][:200] not in have:
            outcomes.append({"doc_id": r["slug"][:200], "data": dict(r, kind="music", city=f"spotify_{r['region']}_{r['rank']}",
                                                                      date=r["chartLabel"], resolvedBucket=r["winner"])})
    outs = {o["slug"]: o for o in have.values()}
    outs.update({o["data"]["slug"]: o["data"] for o in outcomes})

    # paper trading
    paper = load_dir(os.path.join(a.db, "paper"))
    results = load_dir(os.path.join(a.db, "paper_results"))
    writes, opened, settled = [], [], []
    taken = {(p["variant"], p.get("slug")) for p in paper.values()}
    for sn in snaps:
        s = sn["data"]
        if s["early"] or s["pricesUnformed"] or (s["hoursLeft"] is not None and s["hoursLeft"] < 1):
            continue
        best = {}
        for b in s["songs"]:
            d = b.get("depth") or {}
            fill, ask, mkt = d.get("avg_fill_25"), d.get("best_ask"), b.get("market")
            if fill is None or ask is None or mkt is None or ask < MIN_ASK or not (0.03 <= mkt <= 0.97):
                continue
            for v, p in b["variants"].items():
                edge = p - fill
                if edge >= EDGE and (v not in best or edge > best[v][0]):
                    best[v] = (edge, b, fill, ask, p)
        for v, (edge, b, fill, ask, p) in best.items():
            if (v, s["slug"]) in taken:
                continue
            taken.add((v, s["slug"]))
            writes.append({"op": "set", "collection": "paper", "doc_id": f"{v}_{s['slug'][:110]}_{slug_label(b['label'])}", "data": {
                "kind": "music", "variant": v, "city": s["city"], "marketDate": s["marketDate"], "slug": s["slug"], "title": s["title"],
                "bucket": b["label"], "stake": STAKE, "price": round(fill, 4), "bestAsk": ask, "entryMid": b["market"],
                "shares": round(STAKE / fill, 2), "ourOdds": round(p, 4), "edge": round(edge, 4), "daysCounted": s["daysCounted"],
                "runAt": run_at, "placedAt": run_at}})
            opened.append(f"{v} {s['title']} {b['label']}: ours {p*100:.0f}% vs {fill*100:.1f}c (+{edge*100:.0f})")
    for doc_id, p in paper.items():
        if p.get("kind") != "music" or doc_id in results:
            continue
        o = outs.get(p.get("slug"))
        if not o:
            continue
        won = o["resolvedBucket"] == p["bucket"]
        payout = p["shares"] if won else 0.0
        writes.append({"op": "set", "collection": "paper_results", "doc_id": doc_id, "data": {
            "kind": "music", "variant": p["variant"], "city": p["city"], "marketDate": p["marketDate"], "slug": p["slug"],
            "bucket": p["bucket"], "status": "won" if won else "lost", "stake": p["stake"], "payout": round(payout, 2),
            "pnl": round(payout - p["stake"], 2), "resolvedBucket": o["resolvedBucket"], "settledAt": run_at}})
        settled.append(f"{p['variant']} {p.get('title', p['slug'])} {p['bucket']}: {'won' if won else 'lost'}")

    batch = []
    for kind, items in (("snap", [(x["doc_id"], "music_snapshots", x["data"]) for x in snaps]),
                        ("out", [(x["doc_id"], "music_outcomes", x["data"]) for x in outcomes]),
                        ("paper", [(w["doc_id"], w["collection"], w["data"]) for w in writes])):
        for i, (doc_id, coll, data) in enumerate(items):
            fp = os.path.abspath(os.path.join(a.out, f"music_{kind}_{i}.json"))
            json.dump(data, open(fp, "w"), ensure_ascii=False)
            batch.append({"op": "set", "collection": coll, "doc_id": doc_id, "file_path": fp})
    json.dump(batch, open(os.path.join(a.out, "music_batch.json"), "w"), indent=1)
    print(f"music data fetched {L['fetched_at']}; daily charts counted through global {latest_day['global']}, US {latest_day['us']}")
    for l in lines:
        print("  ", l)
    print(f"paper trades opened: {len(opened)}")
    for l in opened:
        print("  +", l)
    print(f"paper trades settled: {len(settled)}")
    for l in settled:
        print("  =", l)
    print(f"outcomes: {len(outcomes)} new; {len(snaps)} snapshots; all {len(batch)} database writes listed in "
          f"{os.path.join(a.out, 'music_batch.json')} (create-only sets)")


if __name__ == "__main__":
    main()
