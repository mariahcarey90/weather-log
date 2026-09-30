#!/usr/bin/env python3
"""Paper-trade every model version's signals, and settle them when results come in.

Usage:
  python3 paper.py --snaps FILE --db DIR [--data data/latest.json] --out FILE

--snaps  score.py's output for this run (JSON list of {"doc_id", "data"}); each
         bucket carries "variants" (current, blend, emos) and order-book "depth".
--db     directory of exported tracker collections: DIR/paper, DIR/paper_results,
         DIR/outcomes, DIR/snapshots (any may be missing).
--out    JSON list of ArtifactData batch writes to apply (all create-only "set"s).

New paper trade, per version and market (city + date), at most one: the bucket with
the biggest edge where
  our odds - the average price a $25 order would pay >= 8 points,
  best ask >= 5c, the market still open (3-97%), the weather label not Unsettled,
  and the book can fill $25.
Stake $25 at that average fill price. A version gets one entry per market, ever,
so repeated runs don't pile in.

Settlement: once `outcomes` has the city+date, a result doc (same id) in
`paper_results`: won/lost, payout, pnl, and the closing price move (last logged
market price for the bucket minus its price at entry: positive means the market
moved toward us).
Standard library only.
"""
import argparse, glob, json, os, re
from datetime import datetime, timezone

EDGE = 0.08
MIN_ASK = 0.05
STAKE = 25.0


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def slug(label):
    return re.sub(r"[^A-Za-z0-9]+", "", label.replace("°", "")) or "x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", default="data/latest.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    paper = load_dir(os.path.join(a.db, "paper"))
    results = load_dir(os.path.join(a.db, "paper_results"))
    outcomes = {(o["city"], o["date"]): o for o in load_dir(os.path.join(a.db, "outcomes")).values()}
    snapdb = list(load_dir(os.path.join(a.db, "snapshots")).values())
    run = [e["data"] for e in json.load(open(a.snaps))]
    writes, opened, settled = [], [], []

    # --- new entries
    taken = {(p["variant"], p["city"], p["marketDate"]) for p in paper.values()}
    for s in run:
        if s.get("stability") == "Unsettled":
            continue
        best = {}
        for b in s.get("buckets", []):
            d = b.get("depth") or {}
            fill, ask, mkt = d.get("avg_fill_25"), d.get("best_ask"), b.get("market")
            if fill is None or ask is None or mkt is None or ask < MIN_ASK or not (0.03 <= mkt <= 0.97):
                continue
            for v, p in (b.get("variants") or {}).items():
                edge = p - fill
                if edge >= EDGE and (v not in best or edge > best[v][0]):
                    best[v] = (edge, b, fill, ask, p)
        for v, (edge, b, fill, ask, p) in best.items():
            key = (v, s["city"], s["marketDate"])
            if key in taken:
                continue
            taken.add(key)
            doc_id = f"{v}_{s['city']}_{s['marketDate']}_{slug(b['label'])}"
            data = {"variant": v, "city": s["city"], "marketDate": s["marketDate"], "bucket": b["label"],
                    "lo": b.get("lo"), "hi": b.get("hi"), "stake": STAKE, "price": round(fill, 4),
                    "bestAsk": ask, "entryMid": b.get("market"), "shares": round(STAKE / fill, 2),
                    "ourOdds": round(p, 4), "edge": round(edge, 4), "stability": s.get("stability"),
                    "leadHours": s.get("leadHours"), "runAt": s.get("runAt"), "placedAt": now}
            writes.append({"op": "set", "collection": "paper", "doc_id": doc_id, "data": data})
            opened.append(f"{v:7s} {s['city']} {s['marketDate']} {b['label']}: ours {p*100:.0f}% vs {fill*100:.1f}c "
                          f"(+{edge*100:.0f}) [{s.get('stability')}]")

    # --- settle
    for doc_id, p in paper.items():
        if doc_id in results:
            continue
        o = outcomes.get((p["city"], p["marketDate"]))
        if not o:
            continue
        if o.get("resolvedBucket"):
            won = o["resolvedBucket"] == p["bucket"]
        else:
            lo = -1e9 if p.get("lo") is None else p["lo"]
            hi = 1e9 if p.get("hi") is None else p["hi"]
            won = lo <= o["observedHigh"] <= hi
        close = None
        same = [x for x in snapdb if x.get("city") == p["city"] and x.get("marketDate") == p["marketDate"]]
        if same:
            last = max(same, key=lambda x: x.get("runAt", ""))
            close = next((b.get("market") for b in last.get("buckets", []) if b["label"] == p["bucket"]), None)
        payout = p["shares"] if won else 0.0
        data = {"variant": p["variant"], "city": p["city"], "marketDate": p["marketDate"], "bucket": p["bucket"],
                "status": "won" if won else "lost", "stake": p["stake"], "payout": round(payout, 2),
                "pnl": round(payout - p["stake"], 2), "observedHigh": o.get("observedHigh"),
                "closeMid": close, "priceMove": round(close - p["entryMid"], 4) if close is not None and p.get("entryMid") is not None else None,
                "settledAt": now}
        writes.append({"op": "set", "collection": "paper_results", "doc_id": doc_id, "data": data})
        settled.append(f"{p['variant']:7s} {p['city']} {p['marketDate']} {p['bucket']}: {data['status']} {data['pnl']:+.2f}")

    json.dump(writes, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"paper trades opened: {len(opened)}")
    for l in opened:
        print("  +", l)
    print(f"paper trades settled: {len(settled)}")
    for l in settled:
        print("  =", l)
    print(f"wrote {len(writes)} writes to {a.out}")


if __name__ == "__main__":
    main()
