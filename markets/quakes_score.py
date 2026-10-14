#!/usr/bin/env python3
"""Score open earthquake-count markets and run their paper trading (paper only).

Usage:  python3 markets/quakes_score.py --db DIR --out DIR [--now ISO]

Reads markets/data/quakes_latest.json, quakes_hist.json and quakes_backtest.json (parameters),
plus exported tracker collections in DIR (paper, paper_results, quakes_outcomes; any may be
missing). Writes to --out: quakes_batch.json (ArtifactData batch entries with file_path:
quakes_snapshots, new quakes_outcomes, paper and paper_results, all create-only) and prints
one line per open market. Version: quakes_model (count so far + negative-binomial rest).
"""
import argparse, json, os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quakes_model as qm  # noqa: E402
import paper_common as pc  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    latest = json.load(open(os.path.join(D, "quakes_latest.json")))
    hist = json.load(open(os.path.join(D, "quakes_hist.json")))
    bt_path = os.path.join(D, "quakes_backtest.json")
    params = json.load(open(bt_path))["params"] if os.path.exists(bt_path) else dict(qm.DEFAULT)
    now_dt = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    now, run_at, stamp = now_dt.timestamp(), now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), now_dt.strftime("%Y%m%dT%H%MZ")
    age_h = (now_dt - datetime.fromisoformat(latest["fetched_at"])).total_seconds() / 3600

    snaps, lines = [], []
    for m in latest.get("open", []):
        s, e = datetime.fromisoformat(m["start"]).timestamp(), datetime.fromisoformat(m["end"]).timestamp()
        if e <= now:
            continue
        pmf, info = qm.predict(hist, m["threshold"], s, e, now, params)
        bks = [b for b in m["buckets"] if b.get("lo") is not None or b.get("hi") is not None]
        probs = qm.bucket_probs(pmf, bks)
        buckets = []
        for b, p in zip(bks, probs):
            buckets.append({"label": b["label"], "lo": b["lo"], "hi": b["hi"], "market": b.get("yes"),
                            "model": round(p, 4), "depth": b.get("depth"), "variants": {"quakes_model": round(p, 4)}})
        top = max(buckets, key=lambda b: b["model"])
        mtop = max(buckets, key=lambda b: b["market"] or 0)
        doc = {"kind": "quakes", "slug": m["slug"], "title": m["title"], "city": f"M{m['threshold']}+",
               "marketDate": m["end"][:10], "start": m["start"], "end": m["end"], "threshold": m["threshold"],
               "runAt": run_at, "dataAgeHours": round(age_h, 1), "started": s <= now, **info,
               "params": params, "volume": m.get("volume"), "liquidity": m.get("liquidity"),
               "buckets": buckets, "topPick": top["label"], "marketTop": mtop["label"],
               "noTrade": age_h > 3}  # stale catalogue: score but don't paper-trade
        snaps.append({"doc_id": f"{m['slug'][:120]}_{stamp}", "data": doc})
        gaps = [(b["model"] - (b["market"] or 0), b) for b in buckets if b["market"] is not None]
        g, gb = max(gaps, key=lambda x: x[0]) if gaps else (0, None)
        lines.append(f"{m['title']}: {info['soFar']} so far, {info['daysLeft']} days left, expect ~{info['expectedFinal']} "
                     f"(sd {info['sd']}); ours {top['label']} {top['model']*100:.0f}% vs market's {mtop['label']} "
                     f"{(mtop['market'] or 0)*100:.0f}%" + (f"; biggest gap {gb['label']} +{g*100:.0f} pts" if gb and g >= 0.10 else ""))

    have = pc.load_dir(os.path.join(a.db, "quakes_outcomes"))
    new_out = []
    for r in latest.get("resolved", []):
        if r.get("winner") and r["slug"][:200] not in have:
            new_out.append((r["slug"][:200], {"kind": "quakes", "slug": r["slug"], "title": r["title"],
                                              "city": f"M{r['threshold']}+", "date": r["end"][:10],
                                              "resolvedBucket": r["winner"], "countUSGS": r.get("countUSGS")}))
    outs = {o["slug"]: o for o in have.values()}
    outs.update({d["slug"]: d for _, d in new_out})

    paper = pc.load_dir(os.path.join(a.db, "paper"))
    results = pc.load_dir(os.path.join(a.db, "paper_results"))
    opened_w, opened = pc.open_trades([x["data"] for x in snaps], paper, "quakes", run_at,
                                      extra=lambda s: {"countAtEntry": s["soFar"], "daysLeft": s["daysLeft"]})
    snapdb = pc.load_dir(os.path.join(a.db, "quakes_snapshots"))
    closes = {}
    for sd in sorted(snapdb.values(), key=lambda x: x.get("runAt", "")):
        for b in sd.get("buckets", []):
            closes[(sd["slug"], b["label"])] = b.get("market")
    settled_w, settled = pc.settle(paper, results, outs, "quakes", run_at, closes)
    writes = opened_w + settled_w
    batch = pc.write_batch(a.out, "quakes", [
        ("quakes_snapshots", [(x["doc_id"], x["data"]) for x in snaps]),
        ("quakes_outcomes", new_out),
        ("paper", [(w["doc_id"], w["data"]) for w in writes if w["collection"] == "paper"]),
        ("paper_results", [(w["doc_id"], w["data"]) for w in writes if w["collection"] == "paper_results"])])
    print(f"quakes data fetched {latest['fetched_at']} ({age_h:.1f} h old); {latest.get('histEvents')} catalogue events; params {params}")
    for e in latest.get("errors", []):
        print("  fetch error:", e[:200])
    for l in lines:
        print("  ", l)
    print(f"paper trades opened: {len(opened)}")
    for l in opened:
        print("  +", l)
    print(f"paper trades settled: {len(settled)}")
    for l in settled:
        print("  =", l)
    print(f"outcomes: {len(new_out)} new; {len(snaps)} snapshots; {len(batch)} database writes in "
          f"{os.path.join(a.out, 'quakes_batch.json')} (create-only sets)")


if __name__ == "__main__":
    main()
