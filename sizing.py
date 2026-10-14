#!/usr/bin/env python3
"""Stake sizing and risk rules: turn this run's signals into a bet slip.

Usage:
  python3 sizing.py --snaps FILE --db DIR [--settings FILE] [--out FILE]

--snaps     score.py's output for this run.
--db        exported tracker collections: DIR/bets, DIR/paper, DIR/paper_results.
--settings  JSON with the owner's settings (defaults below; the tracker keeps a copy
            in `settings/sizing`).
--out       optional: write the slip as JSON.

How a stake is sized, for each market's best edge for the live ("current") model:
 0. (5 Oct 2026) Edges are measured on our odds shrunk toward the market's price
    (shrink_to_market, 40%): entry needs shrunk odds - price >= EDGE_MIN. Cities in
    slip_excluded_cities stay off the slip. Side rule: where past picks keep missing on
    one side (e.g. the actual high keeps coming in warmer), no bet on the other side of
    the market's top pick.
 1. Sizing odds: half our odds, half the market's price (the market is a strong
    prior until our record says otherwise). No bet unless our own odds beat the
    price by EDGE_MIN and the sizing odds still beat it.
 2. Quarter Kelly on the sizing odds: stake = 0.25 * (p - price) / (1 - price) * bankroll,
    where bankroll = starting bankroll + settled profit/loss on real bets.
 3. Risk cuts: half on Mixed weather; no bet on Unsettled; half for long shots
    (price under 15c); half for cities under 45% day-ahead backtest accuracy.
 4. Caps: the unlocked tier cap (below), the dollars on offer within 2c of the best
    price (so the order doesn't walk the book), MAX_STAKE, and the daily risk budget
    (open stakes at most DAILY_RISK of the bankroll). Under MIN_STAKE: skip.
 5. Tiers unlock on the live model's settled paper trades:
      $50 to start; $100 after 30 settled with positive return and a non-negative
      average price move; $250 after 75 settled with return above +5%.
    Drawdown brake: a real-money bankroll 20%+ below its peak drops one tier.
Scaled mode (settings "mode": "scaled"), the owner's guide points: long shots
(under 15c) $10-15; everything else $15-50, about $25 for a typical signal and up
to $50 for the strongest (Settled weather, edge 20+ points), from a confidence
score built from the edge and the weather (x0.6 in a weaker city). Whole dollars.
The same skips and caps apply (no Unsettled days, tier cap, order book, daily budget).
Prints the slip and a JSON summary on the last line. Standard library only.
"""
import argparse, glob, json, os, re

DEFAULTS = {
    "bankroll_start": 2500.0, "kelly_fraction": 0.25, "max_stake": 250.0, "min_stake": 5.0,
    "edge_min": 0.08, "market_weight": 0.5, "daily_risk": 0.10,
    "tiers": [
        {"cap": 50.0, "min_settled": 0, "min_roi": None, "min_move": None},
        {"cap": 100.0, "min_settled": 30, "min_roi": 0.0, "min_move": 0.0},
        {"cap": 250.0, "min_settled": 75, "min_roi": 0.05, "min_move": 0.0},
    ],
    "drawdown_brake": 0.20,
    "weak_cities": {"mexico-city": 0.38, "lucknow": 0.37},  # day-ahead backtest top-pick hit rate
    # trial cities: paper trades only, never on the slip, and not counted toward tier unlocks
    "paper_only_cities": ["dallas", "seattle", "toronto", "tel-aviv", "singapore", "austin", "warsaw",
                          "paris", "istanbul", "wellington", "munich", "milan"],
    # "flat": the owner's fixed stake levels; "kelly": quarter Kelly as described above
    "mode": "kelly",
    "flat": {"confident": 50.0, "standard": 25.0, "longshot": 10.0, "longshot_settled": 15.0},
    "confident_edge": 0.12,
    # 5 Oct 2026 tuning, from the first two weeks of settled trades:
    # our odds ran overconfident (paper picks averaged 32% odds, won 22%), so every edge is
    # measured on odds shrunk this far toward the market's price before anything else.
    "shrink_to_market": 0.4,
    # cities left off the slip (still scored and paper-traded): the market's day-ahead top pick
    # in Mexico City was right 97% of the time vs our 62%, so there is no edge to find there.
    "slip_excluded_cities": ["mexico-city"],
    # skip a bet on the side of the market's pick where a city keeps missing: if at least
    # side_skip_min_misses past picks (real and paper, one per market and bucket) missed, and at
    # least side_skip_share of them landed on the same side (actual warmer, or actual colder),
    # don't back a bucket on the other side of the market's top pick there.
    "side_skip_min_misses": 2,  # low on purpose: it only ever removes bets, and one opposite miss lifts it
    "side_skip_share": 0.75,
}


def load_dir(path):
    out = {}
    for f in glob.glob(os.path.join(path, "*.json")):
        d = json.load(open(f))
        out[os.path.splitext(os.path.basename(f))[0]] = d.get("data", d)
    return out


def bucket_key(label):
    """A sortable temperature for a bucket label ('18°C', '82-83°F', '25°C or higher')."""
    m = re.search(r"-?\d+", label or "")
    return int(m.group()) if m else None


def miss_sides(cfg, paper, results, bets, outcomes):
    """Per city: 'warmer' or 'colder' when past picks keep missing on one side, else absent.
    Uses settled paper trades (every version) and settled real bets, one per market and bucket."""
    picks = {}
    for k, r in results.items():
        p = paper.get(k, {})
        if p.get("kind", "high") != "high" or k.startswith(("posts_", "music_", "low_")):
            continue
        picks[(r["city"], r["marketDate"], r["bucket"])] = 1
    for b in bets.values():
        if b.get("status") in ("won", "lost"):
            picks[(b["city"], b["marketDate"], b["bucket"])] = 1
    sides, detail = {}, {}
    for (city, date, bucket) in picks:
        o = outcomes.get((city, date))
        if not o or not o.get("resolvedBucket") or o["resolvedBucket"] == bucket:
            continue
        a, w = bucket_key(bucket), bucket_key(o["resolvedBucket"])
        if a is None or w is None or a == w:
            continue
        detail.setdefault(city, [0, 0])[0 if w > a else 1] += 1
    for city, (warm, cold) in detail.items():
        n = warm + cold
        if n >= cfg["side_skip_min_misses"]:
            if warm / n >= cfg["side_skip_share"]:
                sides[city] = ("warmer", warm, n)
            elif cold / n >= cfg["side_skip_share"]:
                sides[city] = ("colder", cold, n)
    return sides


def tier_state(cfg, paper, results):
    trial = set(cfg.get("paper_only_cities") or [])
    settled = [results[k] for k, p in paper.items()
               if p.get("variant") == "current" and k in results and p.get("city") not in trial]
    n = len(settled)
    staked = sum(r["stake"] for r in settled)
    roi = sum(r["pnl"] for r in settled) / staked if staked else None
    moves = [r["priceMove"] for r in settled if r.get("priceMove") is not None]
    move = sum(moves) / len(moves) if moves else None
    level = 0
    for i, t in enumerate(cfg["tiers"]):
        ok = n >= t["min_settled"]
        if t["min_roi"] is not None:
            ok = ok and roi is not None and roi > t["min_roi"]
        if t["min_move"] is not None:
            ok = ok and move is not None and move >= t["min_move"]
        if ok:
            level = i
    return level, {"settled": n, "roi": roi, "avg_move": move}


def bankroll_state(cfg, bets):
    done = sorted((b for b in bets.values() if b.get("status") in ("won", "lost", "sold")),
                  key=lambda b: b.get("settledAt") or b.get("placedAt") or "")
    bank, peak = cfg["bankroll_start"], cfg["bankroll_start"]
    for b in done:
        bank += float(b.get("pnl") or 0)
        peak = max(peak, bank)
    open_stake = sum(float(b.get("stake") or 0) for b in bets.values() if b.get("status", "open") == "open")
    return bank, peak, open_stake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--settings")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = dict(DEFAULTS)
    if a.settings and os.path.exists(a.settings):
        s = json.load(open(a.settings))
        cfg.update({k: v for k, v in s.get("data", s).items() if k in DEFAULTS})

    bets = load_dir(os.path.join(a.db, "bets"))
    paper = load_dir(os.path.join(a.db, "paper"))
    results = load_dir(os.path.join(a.db, "paper_results"))
    outcomes = {(o["city"], o["date"]): o for o in load_dir(os.path.join(a.db, "outcomes")).values()}
    sides = miss_sides(cfg, paper, results, bets, outcomes)
    level, evidence = tier_state(cfg, paper, results)
    bank, peak, open_stake = bankroll_state(cfg, bets)
    braked = peak > 0 and (peak - bank) / peak >= cfg["drawdown_brake"]
    if braked:
        level = max(0, level - 1)
    tier_cap = cfg["tiers"][level]["cap"]
    budget_left = max(0.0, cfg["daily_risk"] * bank - open_stake)

    slip, lines = [], []
    for e in json.load(open(a.snaps)):
        s = e["data"]
        if s.get("stability") == "Unsettled" or (s.get("leadHours") or 0) < -2:
            continue
        if s.get("paperOnly") or s["city"] in cfg.get("paper_only_cities", []):
            continue  # trial city: paper trades only until promoted
        if s["city"] in cfg.get("slip_excluded_cities", []):
            continue  # scored and paper-traded, but kept off the slip
        msum = sum(b.get("market") or 0 for b in s.get("buckets", [])) or 1
        mtop = max(s.get("buckets", []), key=lambda b: b.get("market") or 0, default=None)
        side = sides.get(s["city"])
        skipped_side = None
        best = None
        for b in s.get("buckets", []):
            d = b.get("depth") or {}
            ask, mkt, p = d.get("best_ask"), b.get("market"), b.get("model")
            if ask is None or mkt is None or ask < 0.05 or not (0.03 <= mkt <= 0.97):
                continue
            q = (1 - cfg["shrink_to_market"]) * p + cfg["shrink_to_market"] * mkt / msum  # shrunk odds
            if q - ask < cfg["edge_min"]:
                continue
            ps = (1 - cfg["market_weight"]) * p + cfg["market_weight"] * mkt / msum
            if ps <= ask:
                continue
            if side and mtop is not None and b is not mtop:
                kb, km = bucket_key(b["label"]), bucket_key(mtop["label"])
                if kb is not None and km is not None and (
                        (side[0] == "warmer" and kb < km) or (side[0] == "colder" and kb > km)):
                    skipped_side = skipped_side or (b["label"], mtop["label"])
                    continue
            if best is None or q - ask > best[0]:
                best = (q - ask, b, ask, p, ps, d, q)
        if not best:
            if skipped_side:
                lines.append(f"{s['city']} {s['marketDate']} {skipped_side[0]}: SKIP (side rule: our pick is "
                             f"{'colder' if side[0] == 'warmer' else 'warmer'} than the market's {skipped_side[1]}, and "
                             f"{side[1]} of the last {side[2]} missed picks here came in {side[0]})")
            continue
        edge, b, ask, p, ps, d, q = best
        cuts = []
        if cfg["mode"] in ("flat", "scaled"):
            # The owner's guide: Confident ~$50, Standard ~$25 (the median), Long shot $10-15.
            # A confidence score from 0 to 1 slides the stake between those points:
            #   edge part: 8 points over the price -> 0, 20+ points -> 1
            #   weather part: Settled 1, Mixed 0.3
            #   score = average of the two, x0.6 in a weaker city
            # Not a long shot: stake = 15 + 35 x score  ($15 weak ... ~$25 typical ... $50 strongest)
            # Long shot (price under 15c): stake = 10 + 5 x score  ($10-15)
            f = cfg["flat"]
            edge_part = min(1.0, max(0.0, (edge - cfg["edge_min"]) / 0.12))
            wx_part = 1.0 if s.get("stability") == "Settled" else 0.3
            score = (edge_part + wx_part) / 2
            if s["city"] in cfg["weak_cities"]:
                score *= 0.6
                cuts.append("weaker city")
            if s.get("stability") == "Mixed":
                cuts.append("Mixed weather")
            cuts.append(f"edge {edge*100:.0f} pts, confidence {score:.2f}")
            if ask < 0.15:
                blevel = "Long shot"
                raw = f["longshot"] + (f["longshot_settled"] - f["longshot"]) * score
            else:
                raw = f["longshot_settled"] + (f["confident"] - f["longshot_settled"]) * score
                blevel = "Confident" if raw >= 40 else ("Standard" if raw >= 20 else "Small")
        else:
            kelly = (ps - ask) / (1 - ask)
            raw = cfg["kelly_fraction"] * kelly * bank
            blevel = "Kelly"
            if s.get("stability") == "Mixed":
                raw *= 0.5; cuts.append("Mixed weather: half")
            if ask < 0.15:
                raw *= 0.5; cuts.append("long shot: half")
            if s["city"] in cfg["weak_cities"]:
                raw *= 0.5; cuts.append("weaker city: half")
        depth_cap = float(d.get("usd_within_2c") or 0)
        # bets the owner already holds in this market count against the per-market cap
        held = [x for x in bets.values() if x.get("status", "open") == "open"
                and x.get("city") == s["city"] and x.get("marketDate") == s["marketDate"]]
        held_usd = sum(float(x.get("stake") or 0) for x in held)
        room = max(0.0, tier_cap - held_usd)
        if held:
            same = [x for x in held if x.get("bucket") == b["label"]]
            cuts.append(f"you already hold ${held_usd:.0f} in this market" + (" on this bucket" if same else ""))
        stake = min(raw, room, cfg["max_stake"], depth_cap, budget_left)
        limiter = min(((blevel, raw), ("tier cap" if not held else "tier cap less what you hold", room), ("max", cfg["max_stake"]),
                       ("order book", depth_cap), ("daily risk budget", budget_left)), key=lambda x: x[1])[0]
        stake = round(stake) if cfg["mode"] in ("flat", "scaled") else (round(stake / 5) * 5 if stake >= 10 else round(stake, 0))
        rec = {"level": blevel, "city": s["city"], "marketDate": s["marketDate"], "bucket": b["label"], "stability": s.get("stability"),
               "ourOdds": round(p, 3), "shrunkOdds": round(q, 3), "sizingOdds": round(ps, 3), "market": b.get("market"), "limitPrice": ask,
               "stake": stake if stake >= cfg["min_stake"] else 0, "limitedBy": limiter, "cuts": cuts,
               "payoutIfWin": round(stake / ask, 2) if stake >= cfg["min_stake"] else 0,
               "expectedValue": round(stake * (ps / ask - 1), 2) if stake >= cfg["min_stake"] else 0,
               "maxLoss": stake if stake >= cfg["min_stake"] else 0, "leadHours": s.get("leadHours"),
               "alreadyHeld": round(held_usd, 2),
               # walk-away price: the most worth paying when the price runs while you buy. Above it the
               # bet no longer clears the entry rule (our odds 8+ points over the price) or the sizing odds.
               "walkAway": round(max(ask, min(q - cfg["edge_min"], ps - 0.02)), 3)}
        if rec["stake"]:
            budget_left -= rec["stake"]
        slip.append(rec)
        lines.append(
            f"{s['city']} {s['marketDate']} {b['label']} [{s.get('stability')}]: "
            + (f"{blevel.upper()} ${rec['stake']:.0f} as a limit order at {ask*100:.0f}c, walk away above {rec['walkAway']*100:.0f}c (pays ${rec['payoutIfWin']:.0f}, "
               f"expected {rec['expectedValue']:+.2f}, max loss ${rec['maxLoss']:.0f}); "
               if rec["stake"] else ("SKIP (already held at the cap); " if held and room < cfg["min_stake"] else "SKIP (sized under the minimum); "))
            + f"our odds {p*100:.0f}% (shrunk {q*100:.0f}%), sizing odds {ps*100:.0f}% vs {ask*100:.0f}c; limited by {limiter}"
            + (f"; {', '.join(cuts)}" if cuts else ""))

    head = (f"Bankroll ${bank:,.0f} (start ${cfg['bankroll_start']:,.0f}, peak ${peak:,.0f}); open stakes ${open_stake:,.0f}; "
            f"stake cap now ${tier_cap:.0f} (tier {level + 1} of {len(cfg['tiers'])}"
            + (", drawdown brake on" if braked else "") + "); "
            f"live-model paper record: {evidence['settled']} settled"
            + (f", return {evidence['roi']*100:+.0f}%" if evidence["roi"] is not None else "")
            + (f", avg price move {evidence['avg_move']*100:+.1f} pts" if evidence["avg_move"] is not None else ""))
    nxt = cfg["tiers"][level + 1] if level + 1 < len(cfg["tiers"]) else None
    unlock = (f"Next cap ${nxt['cap']:.0f}: needs {max(0, nxt['min_settled'] - evidence['settled'])} more settled paper trades"
              f" with return above {nxt['min_roi']*100:+.0f}% and a non-negative average price move" if nxt else "Top cap unlocked")
    if sides:
        head += "; side rule on: " + ", ".join(f"{c} (misses ran {v[0]}, {v[1]} of {v[2]})" for c, v in sorted(sides.items()))
    print(head)
    print(unlock)
    print("\n".join(lines) if lines else "No bets pass the rules this run.")
    summary = {"bankroll": round(bank, 2), "peak": round(peak, 2), "open_stake": open_stake, "tier": level + 1,
               "tier_cap": tier_cap, "braked": braked, "sideRule": {c: list(v) for c, v in sides.items()}, "evidence": evidence, "unlock": unlock, "slip": slip}
    if a.out:
        json.dump(summary, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
