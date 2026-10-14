"""Shared paper-trading and output helpers for the bucketed market tracks (quakes, rain).

Same paper rule as weather, posts and music: one $25 paper trade per version per market, on the
bucket with the biggest edge where our odds minus the average price of a $25 order >= 8 points,
best ask >= 5c and the market 3-97%. Paper only; never trades. Standard library only.
"""
import glob, json, os, re

EDGE, MIN_ASK, STAKE = 0.08, 0.05, 25.0


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def slug_label(label):
    return re.sub(r"[^A-Za-z0-9]+", "", (label or "").replace("<", "lt").replace(">", "gt").replace("+", "plus")
                  .replace("≤", "le").replace("≥", "ge")) or "x"


def open_trades(snaps, paper, kind, run_at, extra=lambda s: {}):
    """snaps: list of snapshot docs (data) with slug, title, city, marketDate and buckets carrying
    variants/market/depth. Returns (writes, lines)."""
    writes, lines = [], []
    taken = {(p["variant"], p.get("slug")) for p in paper.values()}
    for s in snaps:
        if s.get("noTrade"):
            continue
        best = {}
        for b in s["buckets"]:
            d = b.get("depth") or {}
            fill, ask, mkt = d.get("avg_fill_25"), d.get("best_ask"), b.get("market")
            if fill is None or ask is None or mkt is None or ask < MIN_ASK or not (0.03 <= mkt <= 0.97):
                continue
            for v, p in (b.get("variants") or {}).items():
                edge = p - fill
                if edge >= EDGE and (v not in best or edge > best[v][0]):
                    best[v] = (edge, b, fill, ask, p)
        for v, (edge, b, fill, ask, p) in best.items():
            if (v, s["slug"]) in taken:
                continue
            taken.add((v, s["slug"]))
            doc_id = f"{v}_{s['slug'][:110]}_{slug_label(b['label'])}"
            data = {"kind": kind, "variant": v, "city": s["city"], "marketDate": s["marketDate"], "slug": s["slug"],
                    "title": s["title"], "bucket": b["label"], "lo": b.get("lo"), "hi": b.get("hi"), "stake": STAKE,
                    "price": round(fill, 4), "bestAsk": ask, "entryMid": b["market"], "shares": round(STAKE / fill, 2),
                    "ourOdds": round(p, 4), "edge": round(edge, 4), "runAt": run_at, "placedAt": run_at}
            data.update(extra(s))
            writes.append({"op": "set", "collection": "paper", "doc_id": doc_id, "data": data})
            lines.append(f"{v} {s['title']} {b['label']}: ours {p*100:.0f}% vs {fill*100:.1f}c (+{edge*100:.0f})")
    return writes, lines


def settle(paper, results, outcomes_by_slug, kind, run_at, closes=None):
    writes, lines = [], []
    for doc_id, p in paper.items():
        if p.get("kind") != kind or doc_id in results:
            continue
        o = outcomes_by_slug.get(p.get("slug"))
        if not o or not o.get("resolvedBucket"):
            continue
        won = o["resolvedBucket"] == p["bucket"]
        payout = p["shares"] if won else 0.0
        close = (closes or {}).get((p["slug"], p["bucket"]))
        writes.append({"op": "set", "collection": "paper_results", "doc_id": doc_id, "data": {
            "kind": kind, "variant": p["variant"], "city": p["city"], "marketDate": p["marketDate"], "slug": p["slug"],
            "bucket": p["bucket"], "status": "won" if won else "lost", "stake": p["stake"], "payout": round(payout, 2),
            "pnl": round(payout - p["stake"], 2), "resolvedBucket": o["resolvedBucket"],
            "closeMid": close, "priceMove": round(close - p["entryMid"], 4) if close is not None and p.get("entryMid") is not None else None,
            "settledAt": run_at}})
        lines.append(f"{p['variant']} {p.get('title', p['slug'])} {p['bucket']}: {'won' if won else 'lost'} {payout - p['stake']:+.2f}")
    return writes, lines


def write_batch(out_dir, prefix, groups):
    """groups: [(collection, [(doc_id, data)])] -> writes data files + <prefix>_batch.json; returns entries."""
    batch = []
    for coll, items in groups:
        for i, (doc_id, data) in enumerate(items):
            fp = os.path.abspath(os.path.join(out_dir, f"{prefix}_{coll}_{i}.json"))
            json.dump(data, open(fp, "w"), ensure_ascii=False)
            batch.append({"op": "set", "collection": coll, "doc_id": doc_id, "file_path": fp})
    json.dump(batch, open(os.path.join(out_dir, f"{prefix}_batch.json"), "w"), indent=1)
    return batch
