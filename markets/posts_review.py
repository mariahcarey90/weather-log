#!/usr/bin/env python3
"""Weekly readiness check for the Posts track: is it ready for real money?

Usage: python3 markets/posts_review.py --db DIR [--min-trades 30] [--min-windows 20]

Reads exported tracker collections from DIR: paper, paper_results, posts_snapshots,
posts_outcomes; and markets/data/posts_backtest.json (re-tuned weekly) for which
accounts are active (paused accounts are the ones posts_score.py stops paper trading).

Per version (posts_pace, posts_blend), on active accounts only:
  live paper trading: trades, settled, wins, profit/loss, return on stake;
  live accuracy vs the market: Brier on every ended window with a known winner,
  averaged over all logged snapshots of that window, next to the market's own Brier.
Verdict READY only when a version has at least --min-trades settled live paper trades
with a positive return AND its Brier is no worse than the market's over at least
--min-windows settled windows; otherwise NOT YET with what is missing. READY means
real stakes can be proposed to the owner (a code change they approve), never placed.
Also lists each account's live record. Prints a report and a JSON summary last.
Standard library only.
"""
import argparse, glob, json, os, statistics, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posts_score import paused_handles  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VERSIONS = ("posts_pace", "posts_blend")


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def brier(buckets, key, winner):
    ps = [(b["label"], (b.get("variants") or {}).get(key) if key != "market" else b.get("market")) for b in buckets]
    ps = [(l, p) for l, p in ps if p is not None]
    tot = sum(p for _, p in ps)
    if not ps or tot <= 0 or winner not in {l for l, _ in ps}:
        return None
    return sum((p / tot - (1 if l == winner else 0)) ** 2 for l, p in ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-windows", type=int, default=20)
    a = ap.parse_args()
    bt_path = os.path.join(D, "posts_backtest.json")
    bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
    paused = paused_handles(bt.get("by_handle"))
    paper = {k: p for k, p in load_dir(os.path.join(a.db, "paper")).items() if p.get("kind") == "posts"}
    res = load_dir(os.path.join(a.db, "paper_results"))
    outs = {o["slug"]: o for o in load_dir(os.path.join(a.db, "posts_outcomes")).values() if o.get("slug")}
    snaps = [s for s in load_dir(os.path.join(a.db, "posts_snapshots")).values() if s.get("kind") == "posts"]

    pt = defaultdict(lambda: {"trades": 0, "settled": 0, "wins": 0, "staked": 0.0, "pnl": 0.0})
    acct = defaultdict(lambda: {"settled": 0, "wins": 0, "pnl": 0.0})
    for k, p in paper.items():
        if p.get("city") in paused:
            continue
        v = p["variant"]
        pt[v]["trades"] += 1
        r = res.get(k)
        if r:
            pt[v]["settled"] += 1
            pt[v]["wins"] += r.get("status") == "won"
            pt[v]["staked"] += r.get("stake") or 25
            pt[v]["pnl"] += r.get("pnl") or 0
            acct[(v, p["city"])]["settled"] += 1
            acct[(v, p["city"])]["wins"] += r.get("status") == "won"
            acct[(v, p["city"])]["pnl"] += r.get("pnl") or 0

    # accuracy: per settled window, average Brier over its snapshots (active accounts only)
    per_win = defaultdict(lambda: defaultdict(list))
    for s in snaps:
        o = outs.get(s.get("slug"))
        if not o or s.get("handle") in paused or not s.get("started", True):
            continue
        for key in VERSIONS + ("market",):
            b = brier(s.get("buckets") or [], key, o.get("resolvedBucket"))
            if b is not None:
                per_win[s["slug"]][key].append(b)
    wins_scored = {slug: {k: statistics.fmean(v) for k, v in d.items()} for slug, d in per_win.items()
                   if all(d.get(k) for k in VERSIONS + ("market",))}

    lines = [f"POSTS READINESS (active accounts: all except {', '.join(sorted(paused)) or 'none'})"]
    summary, ready = {"paused": sorted(paused), "versions": {}}, []
    mb = statistics.fmean(w["market"] for w in wins_scored.values()) if wins_scored else None
    for v in VERSIONS:
        p = pt[v]
        roi = p["pnl"] / p["staked"] if p["staked"] else None
        b = statistics.fmean(w[v] for w in wins_scored.values()) if wins_scored else None
        need = []
        if p["settled"] < a.min_trades:
            need.append(f"{a.min_trades - p['settled']} more settled paper trades")
        elif roi is None or roi <= 0:
            need.append("a positive paper return")
        if len(wins_scored) < a.min_windows:
            need.append(f"{a.min_windows - len(wins_scored)} more settled windows")
        elif b is None or b > mb:
            need.append("accuracy at least level with the market")
        verdict = "READY: propose real stakes on active accounts" if not need else "NOT YET: needs " + " and ".join(need)
        if not need:
            ready.append(v)
        summary["versions"][v] = {"trades": p["trades"], "settled": p["settled"], "wins": p["wins"],
                                  "pnl": round(p["pnl"], 2), "roi": round(roi, 3) if roi is not None else None,
                                  "brier": round(b, 4) if b is not None else None,
                                  "market_brier": round(mb, 4) if mb is not None else None,
                                  "windows": len(wins_scored), "verdict": verdict}
        lines.append(f"  {v:11s} {p['trades']} trades, {p['settled']} settled, {p['wins']} won, P/L {p['pnl']:+.2f}"
                     + (f", return {roi*100:+.0f}%" if roi is not None else "")
                     + (f" | Brier {b:.3f} vs market {mb:.3f} over {len(wins_scored)} windows" if b is not None else " | no settled windows yet")
                     + f"\n    -> {verdict}")
    lines.append("BY ACCOUNT (live paper P/L / settled)")
    for hd in sorted({h for (_, h) in acct}):
        lines.append(f"  {hd:16s} " + " | ".join(f"{v} {acct[(v, hd)]['pnl']:+.0f}/{acct[(v, hd)]['settled']}" for v in VERSIONS))
    summary["ready"] = ready
    print("\n".join(lines))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
