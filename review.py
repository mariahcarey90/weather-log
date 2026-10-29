#!/usr/bin/env python3
"""Weekly model review: how is each version doing, and should one be promoted?

Usage:  python3 review.py --db DIR [--min-trades 30] [--min-days 30]

Reads exported tracker collections from DIR: paper, paper_results, snapshots,
outcomes. For each model version ("current", "blend", "emos"):
  paper trading: trades, settled, win rate, profit/loss, return on stake, and the
    average closing price move (positive = the market moved toward our price after
    the signal, the earliest sign of a real edge);
  accuracy vs the market: Brier score on every scored market-day, using one run per
    day (the 12-36 h run closest to 24 h ahead), next to the market's own Brier on
    the same days; lower is better;
  per city: the same, so weak cities show up.
Then a verdict per challenger: promote only when it beats "current" on both paper
profit/loss and Brier with at least --min-trades settled trades and --min-days
scored market-days; otherwise say what evidence is still missing.
(29 Oct 2026) First comes the accuracy gate from blend.py: the blend weight, log loss of
the market, our odds and the blend, and whether the gate is open. The gate decides the
bet slip; the version verdicts are for watching only.
Prints a report and a JSON summary on the last line. Standard library only.
"""
import argparse, glob, json, os, statistics
from collections import defaultdict

import blend


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def hit(b, obs, resolved):
    if resolved:
        return b["label"] == resolved
    lo = -1e9 if b.get("lo") is None else b["lo"]
    hi = 1e9 if b.get("hi") is None else b["hi"]
    return lo <= obs <= hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-days", type=int, default=30)
    a = ap.parse_args()
    paper = load_dir(os.path.join(a.db, "paper"))
    res = load_dir(os.path.join(a.db, "paper_results"))
    outs = {(o["city"], o["date"]): o for o in load_dir(os.path.join(a.db, "outcomes")).values()}
    snaps = list(load_dir(os.path.join(a.db, "snapshots")).values())

    # paper trading
    pt = defaultdict(lambda: {"trades": 0, "settled": 0, "wins": 0, "staked": 0.0, "pnl": 0.0, "moves": []})
    ptc = defaultdict(lambda: {"settled": 0, "pnl": 0.0})
    for k, p in paper.items():
        v = p["variant"]
        pt[v]["trades"] += 1
        r = res.get(k)
        if r:
            pt[v]["settled"] += 1
            pt[v]["wins"] += r["status"] == "won"
            pt[v]["staked"] += r["stake"]
            pt[v]["pnl"] += r["pnl"]
            if r.get("priceMove") is not None:
                pt[v]["moves"].append(r["priceMove"])
            ptc[(v, p["city"])]["settled"] += 1
            ptc[(v, p["city"])]["pnl"] += r["pnl"]

    # accuracy vs market, one day-ahead run per scored market-day
    best = {}
    for s in snaps:
        k = (s.get("city"), s.get("marketDate"))
        lh = s.get("leadHours")
        if k not in outs or lh is None or not (12 <= lh <= 36):
            continue
        if k not in best or abs(lh - 24) < abs(best[k]["leadHours"] - 24):
            best[k] = s
    br = defaultdict(list)
    brc = defaultdict(list)
    brd = defaultdict(dict)   # version -> {(city, date): brier}, for same-day comparisons
    for (city, date), s in best.items():
        o = outs[(city, date)]
        bs = s.get("buckets", [])
        if not bs:
            continue
        msum = sum(b.get("market") or 0 for b in bs) or 1
        br["market"].append(sum(((b.get("market") or 0) / msum - hit(b, o["observedHigh"], o.get("resolvedBucket"))) ** 2 for b in bs))
        brc[("market", city)].append(br["market"][-1])
        vs = set().union(*[set((b.get("variants") or {"current": 0}).keys()) for b in bs])
        for v in vs:
            if any(v not in (b.get("variants") or {"current": b["model"]}) for b in bs):
                continue
            val = sum(((b.get("variants") or {"current": b["model"]})[v] - hit(b, o["observedHigh"], o.get("resolvedBucket"))) ** 2 for b in bs)
            br[v].append(val)
            brc[(v, city)].append(val)
            brd[v][(city, date)] = val

    # accuracy gate (29 Oct 2026): this, not the paper verdicts below, decides the bet slip
    rows = blend.rows_from(snaps, outs)
    gate_open, ginfo = blend.gate(rows)
    lines = ["ACCURACY GATE (decides the bet slip)", "  " + blend.describe(ginfo),
             "  log loss on those markets: market alone {m}, ours alone {o}, blend {b} (lower is better)".format(
                 m=ginfo["logloss_market"], o=ginfo["logloss_ours"], b=ginfo["logloss_blend"]),
             "  " + ("OPEN" if gate_open else "closed: needs " + "; ".join(ginfo["needs"]))]
    per_city = blend.weights_by_city(snaps, outs)
    if per_city:
        lines.append("  weight by city, for watching only (too few days each to act on): "
                     + ", ".join(f"{c} {w:.2f} ({n})" for c, (w, n) in sorted(per_city.items(), key=lambda kv: -kv[1][0])))
    lines.append("MODEL VERSIONS")
    versions = sorted(set(pt) | {v for v in br if v != "market"} | {"current"})
    summary = {}
    for v in versions:
        p = pt[v]
        roi = p["pnl"] / p["staked"] if p["staked"] else None
        mv = statistics.fmean(p["moves"]) if p["moves"] else None
        b = statistics.fmean(br[v]) if br[v] else None
        bm = statistics.fmean(br["market"]) if br["market"] else None
        summary[v] = {"trades": p["trades"], "settled": p["settled"], "wins": p["wins"], "pnl": round(p["pnl"], 2),
                      "roi": round(roi, 3) if roi is not None else None, "avg_price_move": round(mv, 4) if mv is not None else None,
                      "brier": round(b, 4) if b is not None else None, "market_brier": round(bm, 4) if bm is not None else None,
                      "scored_days": len(br[v])}
        lines.append(f"  {v:8s} paper: {p['trades']} trades, {p['settled']} settled, {p['wins']} won, P/L {p['pnl']:+.2f}"
                     + (f", return {roi*100:+.0f}%" if roi is not None else "")
                     + (f", avg price move {mv*100:+.1f} pts" if mv is not None else "")
                     + f" | Brier {b:.3f} vs market {bm:.3f} over {len(br[v])} days" if b is not None else
                     f"  {v:8s} paper: {p['trades']} trades, {p['settled']} settled, P/L {p['pnl']:+.2f} | no scored days yet")
    lines.append("BY CITY (paper P/L; Brier ours vs market)")
    for city in sorted({c for (_, c) in brc} | {c for (_, c) in ptc}):
        parts = []
        for v in versions:
            q = ptc[(v, city)]
            bb = statistics.fmean(brc[(v, city)]) if brc[(v, city)] else None
            parts.append(f"{v} {q['pnl']:+.0f}/{q['settled']}" + (f" B{bb:.2f}" if bb is not None else ""))
        mb = statistics.fmean(brc[("market", city)]) if brc[("market", city)] else None
        lines.append(f"  {city:12s} " + " | ".join(parts) + (f" | market B{mb:.2f}" if mb is not None else ""))
    lines.append("VERDICTS")
    cur = summary.get("current", {})
    for v in versions:
        if v == "current":
            continue
        s = summary[v]
        common = sorted(set(brd[v]) & set(brd["current"]))
        s["paired_days"] = len(common)
        if common:
            s["paired_brier"] = round(statistics.fmean(brd[v][k] for k in common), 4)
            s["paired_brier_current"] = round(statistics.fmean(brd["current"][k] for k in common), 4)
            lines.append(f"  {v} vs current on the same {len(common)} days: Brier {s['paired_brier']:.3f} vs {s['paired_brier_current']:.3f}")
        need = []
        if s["settled"] < a.min_trades:
            need.append(f"{a.min_trades - s['settled']} more settled paper trades")
        if s["paired_days"] < a.min_days:
            need.append(f"{a.min_days - s['paired_days']} more scored market-days")
        if need:
            verdict = "keep testing: needs " + " and ".join(need)
        else:
            better_pnl = s["pnl"] > (cur.get("pnl") or 0)
            better_brier = s["paired_brier"] < s["paired_brier_current"]
            verdict = ("PROMOTE: beats current on paper P/L and Brier" if better_pnl and better_brier else
                       "keep current: challenger does not beat it on both paper P/L and Brier")
        s["verdict"] = verdict
        lines.append(f"  {v}: {verdict}")
    summary["gate"] = {"open": gate_open, **ginfo}
    print("\n".join(lines))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
