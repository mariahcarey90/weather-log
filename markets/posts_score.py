#!/usr/bin/env python3
"""Score open post-count markets and run their paper trading.

Usage:
  python3 markets/posts_score.py --db DIR --out DIR [--now ISO]

Reads markets/data/posts_latest.json and posts_hist/, plus exported tracker
collections in DIR (paper, paper_results, posts_outcomes; any may be missing).
Writes to --out:
  posts_snapshots.json  list of {"doc_id", "data"} for collection posts_snapshots
  posts_outcomes.json   list of {"doc_id", "data"} for windows that ended with a winner
  posts_paper.json      create-only batch writes for paper / paper_results
  posts_batch.json      every write above as ArtifactData batch entries with file_path
and prints a short report.

Model versions (paper-trade side by side, like weather):
  posts_pace   the posting-history model (posts_model.py, parameters from the backtest)
  posts_blend  50/50 with the market's price
Paper rule, as weather: one $25 trade per version per market, on the range with the
biggest edge where our odds minus the average price of a $25 order >= 8 points,
best ask >= 5c and the market 3-97%. Paper only; never trades.
Focus (5 Oct 2026): only accounts in PAPER_FOCUS open new paper trades; REST_SCALE trims an
account's rest-of-window forecast; SKIP_HANDLES are not scored at all.
Paused accounts: no new paper trades for an account whose backtest paper trading
(markets/data/posts_backtest.json, re-tuned weekly by retune.yml) lost money over at
least PAUSE_MIN_TRADES trades, so the live record only covers accounts we would bet on.
Their markets are still scored and logged, and open paper trades still settle.
Standard library only.
"""
import argparse, glob, json, os, re, sys, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import posts_model as pm  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EDGE, MIN_ASK, STAKE = 0.08, 0.05, 25.0
PAUSE_MIN_TRADES = 8
# 5 Oct 2026 tuning, from the first live fortnight (63 scored snapshots):
# - our odds beat the market's overall almost entirely on Trump (log loss 0.43 vs 0.77; top pick 7/7 vs 3/7),
#   so only these accounts open new paper trades; the rest are still scored and logged.
PAPER_FOCUS = {"realDonaldTrump"}
# - Ted Cruz's final counts came in ~9% under our expected final: scale his rest-of-window forecast.
REST_SCALE = {"tedcruz": 0.91}
# - Khamenei posts ~0-1 a week: nothing to learn or trade, so his markets are no longer scored or logged.
SKIP_HANDLES = {"khamenei_ir"}


def paused_handles(by_handle):
    """Accounts whose backtest paper trades lost money over enough trades."""
    out = set()
    for hd, v in (by_handle or {}).items():
        pp = v.get("paper") or {}
        if (pp.get("trades") or 0) >= PAUSE_MIN_TRADES and (pp.get("roi") is not None and pp["roi"] < 0):
            out.add(hd)
    return out


def ts(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def slug_label(label):
    return re.sub(r"[^A-Za-z0-9]+", "", label.replace("<", "lt").replace("+", "plus")) or "x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    now = ts(a.now) if a.now else int(time.time())
    stamp = datetime.fromtimestamp(now, timezone.utc).strftime("%Y%m%dT%H%MZ")
    run_at = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    latest = json.load(open(os.path.join(D, "posts_latest.json")))
    bt_path = os.path.join(D, "posts_backtest.json")
    bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
    params = bt.get("params") or pm.DEFAULT
    by_handle = bt.get("by_handle") or {}
    paused = paused_handles(by_handle)
    hist = {}

    def H(hd):
        if hd not in hist:
            h = json.load(open(os.path.join(D, "posts_hist", hd + ".json")))
            hist[hd] = (sorted(h["posts"].values()), h["from"])
        return hist[hd]

    snaps, lines = [], []
    for t in latest["trackings"]:
        s, e = ts(t["start"]), ts(t["end"])
        if now >= e or t["handle"] in SKIP_HANDLES:
            continue
        times, hfrom = H(t["handle"])
        mean, var, dist, info = pm.predict(times, hfrom, s, e, now, params)
        k = REST_SCALE.get(t["handle"])
        if k:  # scale the posts still to come (not the ones already counted)
            base = info["soFar"]
            scaled = {}
            for c, p in dist.items():
                c2 = base + round((c - base) * k)
                scaled[c2] = scaled.get(c2, 0.0) + p
            dist = scaled
            mean = base + (mean - base) * k
            var = var * k * k
            info["restScale"] = k
        # the counter's own live total is what settles the market: shift our count to it
        if t.get("counterTotal") is not None and s <= now:
            shift = t["counterTotal"] - info["soFar"]
            if shift:
                dist = {k + shift: p for k, p in dist.items()}
                mean += shift
            info["counterShift"] = shift
        rng = t["ranges"]
        ours = pm.range_probs(dist, rng)
        # a brand-new market has quotes on one side only and prices that add up to far more
        # than 100%: treat those prices as not formed yet (no comparison, no paper trade)
        unformed = sum(b.get("market") or 0 for b in rng) > 1.25
        if unformed:
            rng = [dict(b, marketRaw=b.get("market"), market=None) for b in rng]
        buckets = []
        for b, p in zip(rng, ours):
            mk = b.get("market")
            buckets.append({"label": b["label"], "lo": b["lo"], "hi": b["hi"], "market": mk, "model": round(p, 4),
                            "depth": b.get("depth"),
                            "variants": {"posts_pace": round(p, 4),
                                         "posts_blend": round(0.5 * p + 0.5 * mk, 4) if mk is not None else round(p, 4)}})
        top = max(buckets, key=lambda b: b["model"])
        if unformed:
            for x, r in zip(buckets, rng):
                x["marketRaw"] = r.get("marketRaw")
        mtop = max(buckets, key=lambda b: b["market"] or 0)
        hours_left = round((e - now) / 3600, 1)
        sd = var ** 0.5
        doc = {"kind": "posts", "handle": t["handle"], "name": t.get("name"), "title": t["title"], "slug": t["slug"],
               "city": t["handle"], "marketDate": t["end"][:10], "start": t["start"], "end": t["end"],
               "runAt": run_at, "hoursLeft": hours_left, "started": s <= now,
               "count": t.get("counterTotal") if s <= now else 0, "expectedFinal": round(mean, 1),
               "range80": [max(t.get("counterTotal") or 0, round(mean - 1.28 * sd)) if s <= now else max(0, round(mean - 1.28 * sd)), round(mean + 1.28 * sd)],
               "paceRatio": info.get("paceRatio"), "samples": info.get("n"), "restScale": info.get("restScale"),
               "volume": t.get("volume"), "liquidity": t.get("liquidity"),
               "backtest": by_handle.get(t["handle"]), "paused": t["handle"] in paused, "buckets": buckets,
               "topPick": top["label"], "marketTop": None if unformed else mtop["label"],
               "pricesUnformed": unformed}
        doc_id = f"{t['slug'][:120]}_{stamp}"
        snaps.append({"doc_id": doc_id, "data": doc})
        gaps = [(b["model"] - (b["market"] or 0), b) for b in buckets if b["market"] is not None]
        g, gb = max(gaps, key=lambda x: x[0]) if gaps else (0, None)
        lines.append(f"{t['title']}: count {doc['count']} with {hours_left}h left, expect ~{mean:.0f} "
                     f"(80%: {doc['range80'][0]}-{doc['range80'][1]}); ours {top['label']} {top['model']*100:.0f}% "
                     + (f"vs market's {mtop['label']} {(mtop['market'] or 0)*100:.0f}%" if not unformed else "; market prices not formed yet")
                     + (f"; biggest gap {gb['label']} +{g*100:.0f} pts" if gb and g >= 0.10 else ""))

    # outcomes for windows that ended
    outcomes = []
    for r in latest.get("resolved", []):
        if r.get("winner"):
            outcomes.append({"doc_id": r["slug"][:200], "data": {
                "kind": "posts", "handle": r["handle"], "slug": r["slug"], "title": r["title"], "city": r["handle"],
                "date": r["end"][:10], "end": r["end"], "finalCountHistory": r["histCount"],
                "resolvedBucket": r["winner"], "settled": r.get("settled")}})

    # paper trading
    paper = load_dir(os.path.join(a.db, "paper"))
    results = load_dir(os.path.join(a.db, "paper_results"))
    have = load_dir(os.path.join(a.db, "posts_outcomes"))
    outs = {o["slug"]: o for o in have.values()}
    outs.update({o["data"]["slug"]: o["data"] for o in outcomes})
    outcomes = [o for o in outcomes if o["doc_id"] not in have]  # outcomes are written once
    writes, opened, settled = [], [], []
    taken = {(p["variant"], p.get("slug") or p["city"] + p["marketDate"]) for p in paper.values()}
    for sn in snaps:
        s = sn["data"]
        if s["hoursLeft"] < 1 or s.get("paused") or s["handle"] not in PAPER_FOCUS:
            continue
        best = {}
        for b in s["buckets"]:
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
            doc_id = f"{v}_{s['slug'][:120]}_{slug_label(b['label'])}"
            writes.append({"op": "set", "collection": "paper", "doc_id": doc_id, "data": {
                "kind": "posts", "variant": v, "city": s["handle"], "marketDate": s["marketDate"], "slug": s["slug"],
                "title": s["title"], "bucket": b["label"], "lo": b["lo"], "hi": b["hi"], "stake": STAKE,
                "price": round(fill, 4), "bestAsk": ask, "entryMid": b["market"], "shares": round(STAKE / fill, 2),
                "ourOdds": round(p, 4), "edge": round(edge, 4), "countAtEntry": s["count"], "hoursLeft": s["hoursLeft"],
                "runAt": run_at, "placedAt": run_at}})
            opened.append(f"{v} {s['title']} {b['label']}: ours {p*100:.0f}% vs {fill*100:.1f}c (+{edge*100:.0f})")
    for doc_id, p in paper.items():
        if p.get("kind") != "posts" or doc_id in results:
            continue
        o = outs.get(p.get("slug"))
        if not o:
            continue
        won = o["resolvedBucket"] == p["bucket"]
        payout = p["shares"] if won else 0.0
        writes.append({"op": "set", "collection": "paper_results", "doc_id": doc_id, "data": {
            "kind": "posts", "variant": p["variant"], "city": p["city"], "marketDate": p["marketDate"], "slug": p["slug"],
            "bucket": p["bucket"], "status": "won" if won else "lost", "stake": p["stake"], "payout": round(payout, 2),
            "pnl": round(payout - p["stake"], 2), "resolvedBucket": o["resolvedBucket"], "settledAt": run_at}})
        settled.append(f"{p['variant']} {p.get('title', p['slug'])} {p['bucket']}: {'won' if won else 'lost'}")

    json.dump(snaps, open(os.path.join(a.out, "posts_snapshots.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(outcomes, open(os.path.join(a.out, "posts_outcomes.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(writes, open(os.path.join(a.out, "posts_paper.json"), "w"), ensure_ascii=False, indent=1)
    batch = []
    for kind, items in (("snap", [(x["doc_id"], "posts_snapshots", x["data"]) for x in snaps]),
                        ("out", [(x["doc_id"], "posts_outcomes", x["data"]) for x in outcomes]),
                        ("paper", [(w["doc_id"], w["collection"], w["data"]) for w in writes])):
        for i, (doc_id, coll, data) in enumerate(items):
            fp = os.path.abspath(os.path.join(a.out, f"posts_{kind}_{i}.json"))
            json.dump(data, open(fp, "w"), ensure_ascii=False)
            batch.append({"op": "set", "collection": coll, "doc_id": doc_id, "file_path": fp})
    json.dump(batch, open(os.path.join(a.out, "posts_batch.json"), "w"), indent=1)
    print(f"posts data fetched {latest['fetched_at']}; params {params}")
    for l in lines:
        print("  ", l)
    print("paper trading focus: " + ", ".join(sorted(PAPER_FOCUS)) + "; not scored: " + ", ".join(sorted(SKIP_HANDLES)))
    if paused:
        print("paused for paper trading (backtest paper losses): " + ", ".join(sorted(paused)))
    print(f"paper trades opened: {len(opened)}")
    for l in opened:
        print("  +", l)
    print(f"paper trades settled: {len(settled)}")
    for l in settled:
        print("  =", l)
    print(f"outcomes: {len(outcomes)} new; {len(snaps)} snapshots, {len(writes)} paper writes; "
          f"all {len(batch)} database writes listed in {os.path.join(a.out, 'posts_batch.json')} (create-only sets)")


if __name__ == "__main__":
    main()
