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

Station forecasts made for the airport itself (a TAF's TX maximum; on market day in
Atlanta, the NWS hourly forecast for the rest of the day) join as extra members with
no bias shift (they already target the station) and the weight of the best
calibrated model there. They have no track record yet: the weight is a starting
assumption to revisit once the tracker has scored days.

Ensembles run on coarser grids than their parent models and can sit several
degrees apart at an airport (Mexico City's GEFS runs ~3.8 C below GFS). Each
ensemble is re-centred on its parent model for that day, so it adds spread,
not a location error, before the parent's bias correction is applied.

Learning from results: every logged run records each source's forecast. Once
days are scored, each source's bias and typical miss per city (one run per day,
12-36 h ahead) are blended with its prior: the 60-day backfill for the models,
and "no bias, as accurate as the best model" for the airport TAF and NWS
forecasts, worth PRIOR_N_STATION days. Weights follow the blended accuracy.

Stability label (Settled / Mixed / Unsettled): how far the bias-corrected models
spread (in the 60-day history, days with a tight spread missed by 0.53 C on
average vs 0.81 C when they disagreed) plus weather flags from the conditions
forecast: storms, rain, heavy cloud, a front (pressure 6+ hPa off the same hour
the day before, a 3 C jump
in the daily max, or a wind shift), and morning fog. The thresholds are starting
assumptions, logged with every run so they can be checked against results.

Challenger versions (scored alongside, never shown as our pick): each bucket also
carries "variants": "current" (the method above), "blend" (half ours, half the
market's normalised price) and "emos" (non-homogeneous Gaussian regression fitted
by emos.py on the 60-day history: mu = a + b*m, sigma = sqrt(c^2 + d^2 s^2) from
the bias-corrected models' mean m and spread s; day-ahead markets only).
paper.py paper-trades every variant's signals and review.py compares them.

Same-day nowcast: from 10:00 local, each model's error over the last 3 hours
of airport readings (minus its usual bias) shifts its forecast for the rest of
the day by NOWCAST_ALPHA of that error; ensembles follow their parent model.
"""
import argparse, glob, json, math, os, re, statistics
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_KERNEL_C = {"same": 0.7, "day": 0.8, "far": 1.0}
MIN_KERNEL_C = 0.5
MIN_RESOLVED = 5
MIN_LIVE = 30
PRIOR_N_STATION = 8    # how many scored days the TAF/NWS starting assumption is worth
STATION_SOURCES = {"taf_max": "airport forecast (TAF) max", "nws_grid": "nws_grid",
                   "nws_hourly_rest": "nws hourly max"}
BLEND_W_OURS = 0.5      # "blend" challenger: share of our odds vs the market's
SPREAD_SETTLED_C = 1.0
SPREAD_UNSETTLED_C = 1.4
NOWCAST_ALPHA = 0.5    # share of this morning's forecast error assumed to persist to the peak
NOWCAST_FROM_HOUR = 10  # earlier readings (fog, inversions) say little about the afternoon
NOWCAST_WINDOW_H = 3  # live-log calibration overrides the backfill only once it has this many runs
ENS_PARENT = {"ecmwf_ifs025": "ecmwf_ifs025", "ncep_gefs025": "gfs_seamless", "gfs025": "gfs_seamless",
              "icon_seamless_eps": "icon_seamless", "icon_seamless": "icon_seamless"}


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


def source_stats(snaps, outs):
    """Per (city, source key): errors (forecast - observed, deg C) from scored days,
    using one run per day: the 12-36 h run closest to 24 h ahead."""
    best = {}
    for s in snaps:
        o = outs.get((s["city"], s["marketDate"]))
        lh = s.get("leadHours")
        if not o or o.get("observedHigh") is None or lh is None or not (12 <= lh <= 36):
            continue
        k = (s["city"], s["marketDate"])
        if k not in best or abs(lh - 24) < abs(best[k]["leadHours"] - 24):
            best[k] = s
    names = {v: k for k, v in STATION_SOURCES.items()}
    errs = {}
    for (city, _), s in best.items():
        f = s.get("unit") == "°F"
        obs = outs[(city, s["marketDate"])]["observedHigh"]
        for name, v in (s.get("sources") or {}).items():
            if not isinstance(v, (int, float)) or "ens mean" in name or name.startswith(("observed", "rest of", "nowcast")):
                continue
            e = (v - obs) * (5 / 9 if f else 1)
            errs.setdefault((city, names.get(name, name)), []).append(e)
    return errs


def blend_stats(prior_bias, prior_sd, prior_n, errs):
    """Shrink live errors toward the prior: returns (bias, sd, live_n)."""
    if not errs:
        return prior_bias, prior_sd, 0
    n = len(errs)
    b = statistics.fmean(errs)
    live_var = statistics.fmean([(e - b) ** 2 for e in errs]) if n > 1 else prior_sd ** 2
    bias = (prior_n * prior_bias + n * b) / (prior_n + n)
    sd = math.sqrt((prior_n * prior_sd ** 2 + n * live_var) / (prior_n + n))
    return bias, sd, n


def stability(spread_c, cond):
    """(label, reasons) from model spread (deg C) and the day's conditions forecast."""
    cond = cond or {}
    g = lambda k: cond.get(k)
    flags = []
    if (g("cape_max") or 0) >= 1000 and (g("rain_chance_max") or 0) >= 40:
        flags.append("storms likely")
    elif (g("rain_chance_max") or 0) >= 50:
        flags.append("rain likely")
    if (g("cloud_mean") or 0) >= 75:
        flags.append("heavy cloud")
    front = (abs(g("pressure_change_hpa") or 0) >= 6 or abs(g("tmax_change_c") or 0) >= 3
             or ((g("wind_shift_deg") or 0) >= 90 and (g("wind_mean_kmh") or 0) >= 12))
    if front:
        flags.append("front / air-mass change")
    if g("morning_vis_min_m") is not None and g("morning_vis_min_m") < 1000:
        flags.append("morning fog")
    reasons = list(flags)
    if spread_c is not None:
        reasons.insert(0, f"models spread {spread_c:.1f}°C")
    if (spread_c is not None and spread_c >= SPREAD_UNSETTLED_C) or "storms likely" in flags or front or len(flags) >= 2:
        return "Unsettled", reasons
    if (spread_c is None or spread_c <= SPREAD_SETTLED_C) and not flags:
        return "Settled", reasons
    return "Mixed", reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latest.json")
    ap.add_argument("--db")
    ap.add_argument("--out", default="out/snapshots.json")
    ap.add_argument("--run-at")
    ap.add_argument("--calib", default="data/calibration.json")
    a = ap.parse_args()

    latest = json.load(open(a.data))
    emos = json.load(open("data/emos.json")).get("cities", {}) if os.path.exists("data/emos.json") else {}
    snaps, outs = load_db(a.db)
    cal = calibration(snaps, outs)
    src_errs = source_stats(snaps, outs)
    learned = {}  # (city, source) -> (bias, sd, live_n), for the report
    prior = {}
    if os.path.exists(a.calib):
        prior = json.load(open(a.calib)).get("cities", {})
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
            lead_key = "day2" if band == "far" else "day1"
            pcity = (prior.get(city) or {}).get(lead_key) or {}
            pc = pcity.get("median")
            pmodels = pcity.get("models", {})
            if c and c["n"] >= MIN_LIVE:  # live log has plenty of resolved runs: single bias on the blend
                bias_c, kern_c, per_model = c["bias"], max(0.7 * c["sd"], MIN_KERNEL_C), {}
                cal_note = f"Calibrated from {c['n']} logged runs: bias {bias_c:+.2f}°C, kernel {kern_c:.2f}°C."
            elif pc:  # 60-day backfill: correct each model by its own bias, weight by its accuracy
                bias_c = pc["bias"]
                kern_c = max(0.5 * pc["sd"] * (0.8 if band == "same" else 1.0), MIN_KERNEL_C)
                per_model = pmodels
                cal_note = (f"Per-model bias correction from {pc['n']}-day backfill (blend ran {bias_c:+.2f}°C "
                            f"vs station, miss sd {pc['sd']:.2f}°C); kernel {kern_c:.2f}°C.")
            else:
                bias_c, kern_c, per_model = 0.0, DEFAULT_KERNEL_C[band], {}
                cal_note = f"No calibration yet (bias 0, kernel {kern_c}°C)."

            best_sd = min((pm["sd"] for pm in per_model.values()), default=1.0)

            def corr(model):
                """(bias, weight) for a model name, ensembles mapped to their parent model."""
                parent = ENS_PARENT.get(model, model)
                pm = per_model.get(parent)
                if pm:
                    b, sd, n = blend_stats(pm["bias"], pm["sd"], pm.get("n", 60), src_errs.get((city, parent)))
                    if n:
                        learned[(city, parent)] = (b, sd, n)
                    return b, 1.0 / max(sd, 0.5) ** 2
                if per_model:  # uncalibrated model: blend bias, low weight
                    return bias_c, 0.5 / max(pc["sd"] if pc else 1.0, 0.5) ** 2
                return bias_c, 1.0

            is_today = d == now_local.date().isoformat()
            obs = day.get("obs_max_so_far") if is_today else None
            # Same day with obs: final high = max(observed so far, max over the remaining hours).
            use_rest = is_today and obs is not None and "models_rest" in day
            det = day.get("models_rest" if use_rest else "models", {})
            ens = day.get("ensemble_rest" if use_rest else "ensemble", {})
            # same-day nowcast: each model's anomaly vs this morning's readings
            anomaly = {}
            if use_rest and now_local.hour >= NOWCAST_FROM_HOUR:
                recent = []
                for o in st.get("metar", []):
                    t = datetime.fromisoformat(o["t"])
                    obs_age_h = (run_at - t).total_seconds() / 3600
                    if o.get("temp") is None or not (0 <= obs_age_h <= NOWCAST_WINDOW_H):
                        continue
                    lt = (t.astimezone(tz) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:00")
                    recent.append((lt, o["temp"]))
                for m, hours in (day.get("models_hourly") or {}).items():
                    errs = [ob - hours[h] for h, ob in recent if h in hours]
                    if len(errs) >= 2:
                        bm, _ = corr(m)
                        anomaly[m] = statistics.fmean(errs) + bm  # beyond the model's usual bias
            mean_anom = statistics.fmean(anomaly.values()) if anomaly else 0.0

            members, weights = [], []
            ens_total = 0
            for m, vals in ens.items():
                bm, wm = corr(m)
                parent = ENS_PARENT.get(m)
                if parent in det and vals:  # re-centre on the parent model
                    shift = det[parent] - statistics.fmean(vals)
                    vals = [v + shift for v in vals]
                if anomaly:
                    an = anomaly.get(parent, mean_anom)
                    vals = [v + NOWCAST_ALPHA * an for v in vals]
                members += [v - bm for v in vals]
                weights += [wm / len(vals)] * len(vals)   # each ensemble counts once in total
                ens_total += 1
            dets = dict(det)
            for m, v in dets.items():
                if per_model and m in per_model and per_model[m]["mae"] > 1.8:
                    continue  # historically poor at this station
                bm, wm = corr(m)
                if anomaly:
                    v = v + NOWCAST_ALPHA * anomaly.get(m, mean_anom)
                members.append(v - bm)
                weights.append(wm)
            if not per_model:
                members = [x - bias_c for x in members]
            # station forecasts (made for the airport): prior of no bias and the best model's
            # accuracy, updated from scored days
            station_fc = {}
            if day.get("taf_max") is not None:
                station_fc["taf_max"] = day["taf_max"]
            if not use_rest and day.get("nws_grid_max") is not None:
                station_fc["nws_grid"] = day["nws_grid_max"]
            if use_rest and day.get("nws_hourly_rest") is not None:
                station_fc["nws_hourly_rest"] = day["nws_hourly_rest"]
            for k, v in station_fc.items():
                b, sd, n = blend_stats(0.0, best_sd, PRIOR_N_STATION, src_errs.get((city, k)))
                if n:
                    learned[(city, k)] = (b, sd, n)
                members.append(v - b)
                weights.append(1.0 / max(sd, 0.5) ** 2)
            # stability: spread of the bias-corrected models (full day), plus the conditions flags
            # (only models with a track record here, as in the 60-day analysis; skip poor ones)
            corr_full = [v - corr(m)[0] for m, v in day.get("models", {}).items()
                         if (not per_model or m in per_model) and not (per_model and m in per_model and per_model[m]["mae"] > 1.8)]
            spread_c = statistics.pstdev(corr_full) if len(corr_full) >= 4 else None
            stab, stab_why = stability(spread_c, day.get("conditions"))
            full_det = list(day.get("models", {}).values())
            raw_c = statistics.median(full_det) if full_det else obs
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
                                "model": p, "depth": bk.get("depth")})
            for bk in buckets:
                bk["model"] = round(bk["model"] / sum_m, 4) if sum_m else 0
            # challengers
            mk_sum = sum(b["market"] or 0 for b in buckets)
            ep = None
            ec = (emos.get(city) or {}).get(lead_key)
            if ec and not is_today and spread_c is not None and len(corr_full) >= 3:
                mu_c = ec["a"] + ec["b"] * statistics.fmean(corr_full)
                sg_c = math.sqrt(ec["c"] ** 2 + ec["d"] ** 2 * spread_c ** 2) + 1e-6
                mu_u, sg_u = to_u(mu_c), sg_c * (9 / 5 if fahrenheit else 1)
                ep = []
                for b in buckets:
                    lo_ = -1e9 if b["lo"] is None else b["lo"] - 0.5
                    hi_ = 1e9 if b["hi"] is None else b["hi"] + 0.5
                    ep.append(max(0.0, ncdf(hi_, mu_u, sg_u) - ncdf(lo_, mu_u, sg_u)))
                es = sum(ep) or 1
                ep = [x / es for x in ep]
            for i, b in enumerate(buckets):
                v = {"current": b["model"]}
                if mk_sum > 0 and b["market"] is not None:
                    v["blend"] = round(BLEND_W_OURS * b["model"] + (1 - BLEND_W_OURS) * b["market"] / mk_sum, 4)
                if ep is not None:
                    v["emos"] = round(ep[i], 4)
                b["variants"] = v
            buckets.sort(key=lambda b: (-1e9 if b["lo"] is None else b["lo"]))
            gaps = [b["label"] for b in buckets if b["market"] is not None and abs(b["model"] - b["market"]) >= 0.10]

            sources = {k: round(to_u(v), 1) for k, v in det.items()}
            if day.get("nws_grid_max") is not None:
                sources["nws_grid"] = round(to_u(day["nws_grid_max"]), 1)
            for m, vals in day.get("ensemble", {}).items():
                sources[f"{m} ens mean ({len(vals)})"] = round(to_u(statistics.fmean(vals)), 1)
            if day.get("taf_max") is not None:
                sources["airport forecast (TAF) max"] = round(to_u(day["taf_max"]), 1)
            if day.get("nws_hourly_max") is not None:
                sources["nws hourly max"] = round(to_u(day["nws_hourly_max"]), 1)
            if anomaly:
                sources["nowcast shift"] = round(NOWCAST_ALPHA * mean_anom * (9 / 5 if fahrenheit else 1), 1)
            if use_rest and mem_u:
                sources["rest of today (median)"] = round(statistics.median(mem_u), 1)
            if obs is not None:
                sources["observed so far"] = round(to_u(obs), 1)

            note = (f"GitHub data fetched {latest['fetched_at']} ({age_h:.1f} h before run). "
                    f"{len(det)} models, {sum(len(v) for v in day.get('ensemble', {}).values())} ensemble members. "
                    + cal_note
                    + (f" Station forecasts added: {', '.join(station_fc)}." if station_fc else "")
                    + (f" Nowcast: models ran {mean_anom:+.1f}°C vs the last {NOWCAST_WINDOW_H} h of readings"
                       f" beyond their usual bias; {NOWCAST_ALPHA:.0%} carried to the rest of the day." if anomaly else "")
                    + (f" Same-day: high so far {obs}°C, combined with forecast for the hours after {day.get('rest_of_day_from')}." if use_rest else ""))
            doc = {
                "city": city, "marketDate": d, "runAt": run_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "leadHours": round(lead, 1), "unit": unit,
                "forecastRaw": round(to_u(raw_c), 1),
                "forecastHigh": round(sum(b["model"] * ((b["lo"] if b["lo"] is not None else b["hi"]) + (b["hi"] if b["hi"] is not None else b["lo"])) / 2 for b in buckets), 1),
                "bias": round(bias_c * (9 / 5 if fahrenheit else 1), 2),
                "sigma": round(ku, 2), "sources": sources, "buckets": buckets, "gaps": gaps,
                "volume": mk.get("volume"), "liquidity": mk.get("liquidity"), "note": note,
                "stability": stab, "stabilityReasons": stab_why,
                "modelSpread": round(spread_c * (9 / 5 if fahrenheit else 1), 2) if spread_c is not None else None,
                "conditions": day.get("conditions"),
            }
            doc_id = f"{city}_{d}_{run_at.strftime('%Y%m%dT%H%MZ')}"
            docs.append({"doc_id": doc_id, "data": doc})
            top_m = max(buckets, key=lambda b: b["model"])
            top_k = max(buckets, key=lambda b: b["market"] or 0)
            dp = top_m.get("depth") or {}
            depth_txt = (f"; our pick's book: best {dp['best_ask']*100:.0f}c, ${dp.get('usd_within_2c', 0):.0f} within 2c, "
                         f"$25 fills at {dp['avg_fill_25']*100:.1f}c" if dp.get("best_ask") and dp.get("avg_fill_25")
                         else (f"; our pick's book: best {dp['best_ask']*100:.0f}c, ${dp.get('usd_within_2c', 0):.0f} within 2c, "
                               f"too thin to fill $25" if dp.get("best_ask") else ""))
            report.append(f"{st['name']} {d} [{stab}: {', '.join(stab_why)}] (lead {lead:.0f}h): forecast {doc['forecastRaw']}{unit}; "
                          f"ours {top_m['label']} {top_m['model']:.0%} vs market {top_k['label']} {top_k['market']:.0%}; "
                          f"gaps: {', '.join(gaps) or 'none'}{depth_txt}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(docs, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"data age {age_h:.1f} h; errors in fetch: {len(latest.get('errors', []))}")
    for e in latest.get("errors", []):
        print("  fetch error:", e[:200])
    print("calibration:", {f"{k[0]}/{k[1]}": {kk: round(vv, 2) for kk, vv in v.items()} for k, v in cal.items()} or "none yet")
    if learned:
        print("learned from scored days (bias C, typical miss C, days):",
              {f"{c}/{k}": (round(b, 2), round(sd, 2), n) for (c, k), (b, sd, n) in sorted(learned.items())})
    else:
        print("learned from scored days: none yet")
    print("\n".join(report))
    print(f"wrote {len(docs)} docs to {a.out}")


if __name__ == "__main__":
    main()
