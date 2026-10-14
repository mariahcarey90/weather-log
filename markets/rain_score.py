#!/usr/bin/env python3
"""Score open monthly rainfall markets and run their paper trading (paper only).

Usage:  python3 markets/rain_score.py --db DIR --out DIR [--now ISO]

Reads markets/data/rain_latest.json and rain_clim/, plus exported tracker collections in DIR
(paper, paper_results, rain_outcomes, rain_snapshots; any may be missing). Writes to --out:
rain_batch.json (ArtifactData batch entries with file_path: rain_snapshots, new rain_outcomes,
paper and paper_results, all create-only) and prints one line per open market.
Version: rain_model (measured + ensemble forecast + climatology simulation).
"""
import argparse, json, os, sys
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rain_model as rm  # noqa: E402
import paper_common as pc  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--now")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    latest = json.load(open(os.path.join(D, "rain_latest.json")))
    now_dt = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    run_at, stamp = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), now_dt.strftime("%Y%m%dT%H%MZ")
    age_h = (now_dt - datetime.fromisoformat(latest["fetched_at"])).total_seconds() / 3600

    snaps, lines = [], []
    for m in latest.get("open", []):
        fp = os.path.join(D, "rain_clim", f"{m['city']}_{int(m['first'][5:7]):02d}.json")
        clim = json.load(open(fp)) if os.path.exists(fp) else {}
        sims, info = rm.simulate(m, clim)
        bks = [b for b in m["buckets"] if b.get("lo") is not None or b.get("hi") is not None]
        probs = rm.bucket_probs(sims, bks)
        buckets = [{"label": b["label"], "lo": b["lo"], "hi": b["hi"], "market": b.get("yes"), "model": round(p, 4),
                    "depth": b.get("depth"), "variants": {"rain_model": round(p, 4)}} for b, p in zip(bks, probs)]
        top = max(buckets, key=lambda b: b["model"])
        mtop = max(buckets, key=lambda b: b["market"] or 0)
        u = '"' if m["unit"] == "in" else "mm"
        doc = {"kind": "rain", "slug": m["slug"], "title": m["title"], "city": m["city"], "station": m["station"],
               "unit": m["unit"], "marketDate": m["last"], "first": m["first"], "runAt": run_at,
               "dataAgeHours": round(age_h, 1), **info, "climSource": m.get("climSource"),
               "volume": m.get("volume"), "liquidity": m.get("liquidity"), "buckets": buckets,
               "topPick": top["label"], "marketTop": mtop["label"],
               # no paper trades when the rain so far is only a model estimate, or the station is 4+ days behind:
               # mid-month the measured part drives the answer, and a stand-in can be badly off
               "noTrade": age_h > 6 or not info["members"] or (info["obsSource"] or "model").startswith("model")
                          or (m.get("daysFilled") or 0) >= 4}
        snaps.append({"doc_id": f"{m['slug'][:120]}_{stamp}", "data": doc})
        gaps = [(b["model"] - (b["market"] or 0), b) for b in buckets if b["market"] is not None]
        g, gb = max(gaps, key=lambda x: x[0]) if gaps else (0, None)
        lines.append(f"{m['title']}: {info['measured']}{u} measured to {info['measuredThrough']} ({info['obsSource']}), "
                     f"month likely {info['p10']}-{info['p90']}{u} (median {info['median']}); ours {top['label']} "
                     f"{top['model']*100:.0f}% vs market's {mtop['label']} {(mtop['market'] or 0)*100:.0f}%"
                     + (f"; biggest gap {gb['label']} +{g*100:.0f} pts" if gb and g >= 0.10 else ""))

    have = pc.load_dir(os.path.join(a.db, "rain_outcomes"))
    new_out = [(r["slug"][:200], {"kind": "rain", "slug": r["slug"], "title": r["title"], "city": r["city"],
                                  "date": r["last"], "resolvedBucket": r["winner"]})
               for r in latest.get("resolved", []) if r.get("winner") and r["slug"][:200] not in have]
    outs = {o["slug"]: o for o in have.values()}
    outs.update({d["slug"]: d for _, d in new_out})
    paper = pc.load_dir(os.path.join(a.db, "paper"))
    results = pc.load_dir(os.path.join(a.db, "paper_results"))
    opened_w, opened = pc.open_trades([x["data"] for x in snaps], paper, "rain", run_at,
                                      extra=lambda s: {"measuredAtEntry": s["measured"], "obsSource": s["obsSource"]})
    closes = {}
    for sd in sorted(pc.load_dir(os.path.join(a.db, "rain_snapshots")).values(), key=lambda x: x.get("runAt", "")):
        for b in sd.get("buckets", []):
            closes[(sd["slug"], b["label"])] = b.get("market")
    settled_w, settled = pc.settle(paper, results, outs, "rain", run_at, closes)
    writes = opened_w + settled_w
    batch = pc.write_batch(a.out, "rain", [
        ("rain_snapshots", [(x["doc_id"], x["data"]) for x in snaps]),
        ("rain_outcomes", new_out),
        ("paper", [(w["doc_id"], w["data"]) for w in writes if w["collection"] == "paper"]),
        ("paper_results", [(w["doc_id"], w["data"]) for w in writes if w["collection"] == "paper_results"])])
    print(f"rain data fetched {latest['fetched_at']} ({age_h:.1f} h old)")
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
          f"{os.path.join(a.out, 'rain_batch.json')} (create-only sets)")


if __name__ == "__main__":
    main()
