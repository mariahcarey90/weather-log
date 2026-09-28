#!/usr/bin/env python3
"""Turn data/latest.json into bucket odds and tracker documents.

Usage:
  python3 score.py [--db DIR] [--out FILE] [--run-at ISO]

--db   directory holding the tracker's exported collections, as written by the
       ArtifactData tool with out_dir: DIR/snapshots/*.json and DIR/outcomes/*.json.
       Used to learn each city's forecast bias and typical miss. Optional.
--out  where to write the snapshot docs (default: out/snapshots.json), a JSON list
       of {"doc_id": ..., "data": {...}} ready for an ArtifactData batch.

Method
  1. Predictive members = every ensemble member's daily max (ECMWF 51, GFS 31,
     ICON ~40) plus each deterministic model, all from hourly data at the
     station's coordinates.
  2. Shift every member by the learned bias for that city and lead band
     (members run cool at airport stations: gridded hourly max < METAR max).
  3. Smooth each member with a Normal kernel (learned from past misses,
     floor 0.6 C) and average -> continuous distribution of the day's high.
  4. For today, the high can't finish below what the station already reported:
     truncate the distribution at the observed max so far and renormalise.
  5. Integrate over each market bucket (whole-degree settlement: [lo-0.5, hi+0.5]).
"""
import argparse, glob, json, math, os, re, statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

DEFAULT_KERNEL_C = {"same": 0.9, "day": 1.0, "far": 1.2}
MIN_KERNEL_C = 0.6
MIN_RESOLVED = 5


def c2f(c):
    return c * 9 / 5 + 32


def ncdf(x, m, s):
    return 0.5 * (1 + math.erf((x - m) / (s * math.sqrt(2))))


def lead_band(h):
    return "same" if h < 12 else ("day" if h <= 36 else "far")


def parse_bucket(label):
    """'25°C' -> (25,25); '84-85°F' -> (84,85); '≤20°C' / '20°C or below' -> (None,20);
    '≥30°C' / '30°C or higher' -> (30,None)."""
    s = label.replace("–", "-").replace("—", "-").replace("º", "°")
    rng = re.search(r"(-?\d+)\s*-\s*(-?\d+)", s)
    if rng:
        return int(rng.group(1)), int(rng.group(2))
    nums = [int(n) for n in re.findall(r"-?\d+", s)]
    low = re.search(r"(or below|or less|or lower|≤|<=|under)", s, re.I)
    high = re.search(r"(or higher|or above|or more|≥|>=|\+|over)", s, re.I)
    if low and nums:
        return None, nums[0]
    if high and nums:
        return nums[0], None
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    return None, None


def load_db(dirpath):
    snaps, outs = [], {}
    if not dirpath:
        return snaps, outs
    for f in glob.glob(os.path.join(dirpath, "snapshots", "*.json")):
        d = json.load(open(f))
        snaps.append(d.get("data", d))
    for f in glob.glob(os.path.join(dirpath, "outcomes", "*.json")):
        d = json.load(open(f))
        d = d.get("data", d)
        outs[(d["city"], d["date"])] = d
    return snaps, outs


def calibration(snaps, outs):
    """Per city and lead band: bias (forecastRaw - observed, in market units) and
    sd of misses, from resolved days. Returned in deg C."""
    errs = {}
    for s in snaps:
        o = outs.get((s["city"], s["marketDate"]))
        if not o or o.get("observedHigh") is None:
            continue
        raw = s.get("forecastRaw", s.get("forecastHigh"))
        if raw is None or s.get("leadHours") is None:
            continue
        e = raw - o["observedHigh"]
        if s.get("unit") == "°F" or s["city"] == "atlanta":
            e = e * 5 / 9
        errs.setdefault((s["city"], lead_band(s["leadHours"])), []).append(e)
    cal = {}
    for k, es in errs.items():
        if len(es) >= MIN_RESOLVED:
            cal[k] = {"bias": statistics.fmean(es), "sd": statistics.stdev(es), "n": len(es)}
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latest.json")
    ap.add_argument("--db")
    ap.add_argument("--out", default="out/snapshots.json")
    ap.add_argument("--run-at")
    a = ap.parse_args()

    latest = json.load(open(a.data))
    snaps, outs = load_db(a.db)
    cal = calibration(snaps, outs)
    run_at = datetime.fromisoformat(a.run_at.replace("Z", "+00:00")) if a.run_at else datetime.now(timezone.utc)
    fetched = datetime.fromisoformat(latest["fetched_at"])
    age_h = (run_at - fetched).total_seconds() / 3600

    docs, report = [], []
    for city, st in latest["stations"].items():
        tz = ZoneInfo(st["tz"])
        fahrenheit = st["unit"] == "F"
        unit = "°F" if fahrenheit else "°C"
        now_local = run_at.astimezone(tz)
        for d, day in sorted(st["days"].items()):
            mk = day.get("market")
            if not mk or not mk.get("buckets") or mk.get("closed"):
                continue
            if d < now_local.date().isoformat():
                continue
            peak = datetime.fromisoformat(d + "T15:00").replace(tzinfo=tz)
            lead = (peak - run_at).total_seconds() / 3600
            if lead < -2:  # past ~17:00 local: the high is in, skip
                continue
            band = lead_band(max(lead, 0))
            c = cal.get((city, band))
            bias_c = c["bias"] if c else 0.0
            kern_c = max(c["sd"], MIN_KERNEL_C) if c else DEFAULT_KERNEL_C[band]

            is_today = d == now_local.date().isoformat()
            obs = day.get("obs_max_so_far") if is_today else None
            # Same day with obs: final high = max(observed so far, max over the remaining hours).
            use_rest = is_today and obs is not None and "models_rest" in day
            det = day.get("models_rest" if use_rest else "models", {})
            ens = day.get("ensemble_rest" if use_rest else "ensemble", {})
            members = []
            for vals in ens.values():
                members.extend(vals)
            det_vals = list(det.values())
            if not use_rest and day.get("nws_grid_max") is not None:
                det_vals.append(day["nws_grid_max"])
            # weight deterministic runs so together they count as much as the ensembles
            weights = [1.0] * len(members)
            if det_vals:
                w = (len(members) / len(det_vals)) if members else 1.0
                members += det_vals
                weights += [w] * len(det_vals)
            members = [m - bias_c for m in members]
            full_det = list(day.get("models", {}).values())
            raw_c = statistics.median(full_det) if full_det else (statistics.median(members) if members else obs)
            if raw_c is None:
                continue

            to_u = c2f if fahrenheit else (lambda x: x)
            ku = kern_c * (9 / 5 if fahrenheit else 1)
            mem_u = [to_u(m) for m in members]
            obs_u = to_u(obs) if obs is not None else None

            def cdf(x):
                """P(settled high <= x) with x on the continuous scale."""
                if obs_u is not None and x < obs_u:
                    return 0.0
                if not mem_u:  # nothing left of the day: high is what was observed
                    return 1.0
                return sum(wi * ncdf(x, m, ku) for m, wi in zip(mem_u, weights)) / sum(weights)

            buckets, sum_m = [], 0.0
            for bk in mk["buckets"]:
                lo, hi = parse_bucket(bk["label"] or "")
                if lo is None and hi is None:
                    continue
                a_ = -1e9 if lo is None else lo - 0.5
                b_ = 1e9 if hi is None else hi + 0.5
                p = max(0.0, cdf(b_) - cdf(a_))
                sum_m += p
                buckets.append({"label": bk["label"], "lo": lo, "hi": hi,
                                "market": round(bk["yes"], 4) if bk.get("yes") is not None else None,
                                "model": p})
            for bk in buckets:
                bk["model"] = round(bk["model"] / sum_m, 4) if sum_m else 0
            buckets.sort(key=lambda b: (-1e9 if b["lo"] is None else b["lo"]))
            gaps = [b["label"] for b in buckets if b["market"] is not None and abs(b["model"] - b["market"]) >= 0.10]

            sources = {k: round(to_u(v), 1) for k, v in det.items()}
            if day.get("nws_grid_max") is not None:
                sources["nws_grid"] = round(to_u(day["nws_grid_max"]), 1)
            for m, vals in day.get("ensemble", {}).items():
                sources[f"{m} ens mean ({len(vals)})"] = round(to_u(statistics.fmean(vals)), 1)
            if use_rest and mem_u:
                sources["rest of today (median)"] = round(statistics.median(mem_u), 1)
            if obs is not None:
                sources["observed so far"] = round(to_u(obs), 1)

            note = (f"GitHub data fetched {latest['fetched_at']} ({age_h:.1f} h before run). "
                    f"{len(det)} models, {sum(len(v) for v in day.get('ensemble', {}).values())} ensemble members. "
                    + (f"Calibrated from {c['n']} resolved runs: bias {bias_c:+.2f}°C, kernel {kern_c:.2f}°C."
                       if c else f"No calibration yet (bias 0, kernel {kern_c}°C).")
                    + (f" Same-day: high so far {obs}°C, combined with forecast for the hours after {day.get('rest_of_day_from')}." if use_rest else ""))
            doc = {
                "city": city, "marketDate": d, "runAt": run_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "leadHours": round(lead, 1), "unit": unit,
                "forecastRaw": round(to_u(raw_c), 1),
                "forecastHigh": round(sum(b["model"] * ((b["lo"] if b["lo"] is not None else b["hi"]) + (b["hi"] if b["hi"] is not None else b["lo"])) / 2 for b in buckets), 1),
                "bias": round(bias_c * (9 / 5 if fahrenheit else 1), 2),
                "sigma": round(ku, 2), "sources": sources, "buckets": buckets, "gaps": gaps,
                "volume": mk.get("volume"), "liquidity": mk.get("liquidity"), "note": note,
            }
            doc_id = f"{city}_{d}_{run_at.strftime('%Y%m%dT%H%MZ')}"
            docs.append({"doc_id": doc_id, "data": doc})
            top_m = max(buckets, key=lambda b: b["model"])
            top_k = max(buckets, key=lambda b: b["market"] or 0)
            report.append(f"{st['name']} {d} (lead {lead:.0f}h): forecast {doc['forecastRaw']}{unit}; "
                          f"ours {top_m['label']} {top_m['model']:.0%} vs market {top_k['label']} {top_k['market']:.0%}; "
                          f"gaps: {', '.join(gaps) or 'none'}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(docs, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"data age {age_h:.1f} h; errors in fetch: {len(latest.get('errors', []))}")
    for e in latest.get("errors", []):
        print("  fetch error:", e[:200])
    print("calibration:", {f"{k[0]}/{k[1]}": {kk: round(vv, 2) for kk, vv in v.items()} for k, v in cal.items()} or "none yet")
    print("\n".join(report))
    print(f"wrote {len(docs)} docs to {a.out}")


if __name__ == "__main__":
    main()
