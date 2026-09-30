#!/usr/bin/env python3
"""Check open test bets against the latest odds and prices: hold or sell?

Usage:
  python3 positions.py --bets DIR --snaps FILE [--data data/latest.json] [--window]

--bets   directory of exported `bets` documents (DIR/*.json), as written by the
         ArtifactData tool with out_dir (DIR is usually /home/claude/db/bets).
--snaps  score.py's output for this run (a JSON list of {"doc_id", "data"}).
--window only report bets whose city is between 10:00 and 13:30 local time on
         the bet's market day (the late-morning check).

The rule compares what a buyer will pay now (best bid) with our current odds for
the bet's bucket. Past profit or loss plays no part.
  SELL         bid is 5+ points above our odds: the market pays more than we think it's worth
  TAKE PROFIT  a long shot (bought at 15c or less) now bids at 2x+ the entry and not
               clearly below our odds: consider locking some or all of it in
  HOLD         our odds are 5+ points above the bid
  FAIR         within 5 points either way: holding or selling are both reasonable
  DECIDED      the market already prices the bucket at 97%+ or 3%-: waiting for settlement
Prints one line per bet and a JSON summary on the last line.
"""
import argparse, glob, json, os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

EDGE = 0.05
LONG_SHOT = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bets", required=True)
    ap.add_argument("--snaps", required=True)
    ap.add_argument("--data", default="data/latest.json")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--now")
    a = ap.parse_args()
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)

    latest = json.load(open(a.data))
    snaps = {}
    for e in json.load(open(a.snaps)):
        d = e["data"]
        snaps[(d["city"], d["marketDate"])] = d
    bets = []
    for f in glob.glob(os.path.join(a.bets, "*.json")):
        b = json.load(open(f))
        b = b.get("data", b)
        if b.get("status", "open") == "open":
            bets.append(b)

    lines, summary = [], []
    for b in sorted(bets, key=lambda x: (x.get("marketDate", ""), x.get("city", ""))):
        city, date, label = b.get("city"), b.get("marketDate"), b.get("bucket")
        st = latest["stations"].get(city)
        if not st:
            continue
        local = now.astimezone(ZoneInfo(st["tz"]))
        in_window = local.date().isoformat() == date and (10, 0) <= (local.hour, local.minute) <= (13, 30)
        if a.window and not in_window:
            continue
        stake = float(b.get("stake") or 0)
        entry = float(b.get("price") or 0) or None
        shares = float(b.get("shares") or (stake / entry if entry else 0))
        day = st["days"].get(date, {})
        mk = {x["label"]: x for x in (day.get("market") or {}).get("buckets", [])}
        bk = mk.get(label)
        snap = snaps.get((city, date))
        ours = next((x["model"] for x in (snap or {}).get("buckets", []) if x["label"] == label), None)
        if bk is None:
            rec = {"city": city, "date": date, "bucket": label, "action": "AWAIT",
                   "why": "market no longer trading in this run's data; wait for the result"}
            summary.append(rec)
            lines.append(f"{city} {date} {label}: AWAIT - {rec['why']}")
            continue
        depth = bk.get("depth") or {}
        bid = depth.get("best_bid", bk.get("bestBid"))
        bid = float(bid) if bid is not None else None
        bid_usd = depth.get("bid_usd_within_2c")
        obs = day.get("obs_max_so_far")
        value = shares * bid if bid is not None else None
        yes = bk.get("yes")
        if yes is not None and (yes >= 0.97 or yes <= 0.03):
            action = "DECIDED"
            why = (f"market prices this bucket at {yes*100:.1f}%: effectively won, waiting for settlement"
                   if yes >= 0.97 else f"market prices this bucket at {yes*100:.1f}%: effectively lost, waiting for settlement")
        elif bid is None or bid <= 0:
            action, why = "HOLD", "no buyers bidding right now, so there is nothing to sell into"
        elif ours is None:
            action, why = "FAIR", "no current odds for this bucket in this run"
        elif bid >= ours + EDGE:
            action, why = "SELL", f"bid {bid*100:.0f}c is above our odds {ours*100:.0f}%"
        elif entry and entry <= LONG_SHOT and bid >= 2 * entry and bid >= ours - 0.02:
            action, why = "TAKE PROFIT", f"long shot now bids {bid*100:.0f}c vs {entry*100:.1f}c paid; our odds {ours*100:.0f}%"
        elif ours >= bid + EDGE:
            action, why = "HOLD", f"our odds {ours*100:.0f}% are above the bid {bid*100:.0f}c"
        else:
            action, why = "FAIR", f"our odds {ours*100:.0f}% vs bid {bid*100:.0f}c: holding or selling are both reasonable"
        extra = []
        if value is not None:
            extra.append(f"selling now returns ~${value:.2f} ({value - stake:+.2f})")
        if bid_usd is not None and value is not None and bid_usd < value:
            extra.append(f"only ${bid_usd:.0f} bid within 2c, so sell part or use a limit order")
        if obs is not None and local.date().isoformat() == date:
            shown = f"{obs * 9 / 5 + 32:.0f}°F" if st["unit"] == "F" else f"{obs}°C"
            extra.append(f"high so far {shown} at {local.strftime('%H:%M')} local")
        if snap and snap.get("stability"):
            extra.append(f"weather {snap['stability']}")
        rec = {"city": city, "date": date, "bucket": label, "action": action, "why": why,
               "bid": bid, "ours": ours, "entry": entry, "shares": round(shares, 2),
               "value_now": round(value, 2) if value is not None else None,
               "bid_usd_within_2c": bid_usd, "obs_so_far": obs, "local_time": local.strftime("%H:%M"),
               "in_late_morning_window": in_window}
        summary.append(rec)
        lines.append(f"{city} {date} {label}: {action} - {why}" + (f"; {'; '.join(extra)}" if extra else ""))

    print("\n".join(lines) if lines else ("no open bets in their late-morning window" if a.window else "no open bets"))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
