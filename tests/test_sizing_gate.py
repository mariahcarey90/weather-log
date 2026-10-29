"""Fix 3 (29 Oct 2026): the slip opens only on the accuracy gate, never on recent paper profit."""
import json, pathlib, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def run_sizing(tmp_path, paper_results, history=None, now=None):
    db = tmp_path / "db"
    for c in ("bets", "paper", "paper_results", "outcomes", "snapshots"):
        (db / c).mkdir(parents=True, exist_ok=True)
    for k, r in paper_results.items():
        (db / "paper" / f"{k}.json").write_text(json.dumps({"variant": r["variant"], "city": r["city"],
                                                            "marketDate": r["marketDate"]}))
        (db / "paper_results" / f"{k}.json").write_text(json.dumps(r))
    for i, (snap, outcome) in enumerate(history or []):
        (db / "snapshots" / f"h{i}.json").write_text(json.dumps(snap))
        (db / "outcomes" / f"o{i}.json").write_text(json.dumps(outcome))
    bucket = {"label": "21°C", "lo": 21, "hi": 21, "market": 0.10, "model": 0.60,
              "depth": {"best_ask": 0.10, "usd_within_2c": 500}}
    snap = {"city": "madrid", "marketDate": "2026-11-02", "leadHours": 24, "stability": "Settled", "buckets": [bucket]}
    (tmp_path / "snaps.json").write_text(json.dumps([{"doc_id": "s", "data": snap}]))
    subprocess.run([sys.executable, str(REPO / "sizing.py"), "--snaps", "snaps.json", "--db", "db", "--out", "slip.json"]
                   + (["--now", now] if now else []), cwd=tmp_path, check=True, capture_output=True)
    return json.loads((tmp_path / "slip.json").read_text())


def sharp_history(n_dates=45, cities=15):
    """Day-ahead runs where our odds were sharp (80% on the winner) and the market's flat."""
    out = []
    for d in range(n_dates):
        date = f"2026-11-{d + 1:02d}" if d < 30 else f"2026-12-{d - 29:02d}"
        for c in range(cities):
            win = (d + c) % 4
            buckets = [{"label": f"{20 + j}°C", "model": 0.8 if j == win else 0.2 / 3, "market": 0.25} for j in range(4)]
            out.append(({"city": f"c{c}", "marketDate": date, "runAt": "2026-10-29T00:00:00Z", "leadHours": 24,
                         "buckets": buckets}, {"city": f"c{c}", "date": date, "resolvedBucket": f"{20 + win}°C"}))
    return out


def test_profitable_paper_trades_do_not_reopen_the_slip(tmp_path):
    wins = {f"t{i}": {"variant": "current", "city": "madrid", "marketDate": "2026-11-01", "bucket": "21°C",
                      "stake": 25, "pnl": 100, "status": "won", "settledAt": f"2026-11-01T{i % 24:02d}:00Z"}
            for i in range(50)}
    slip = run_sizing(tmp_path, wins)
    assert slip["paused"] is True
    assert slip["slip"] == []
    assert "accuracy gate is closed" in slip["unlock"]


def test_open_gate_still_waits_for_blended_paper_trades(tmp_path):
    slip = run_sizing(tmp_path, {}, history=sharp_history(), now="2026-12-31T00:00:00Z")
    assert slip["paused"] is True and "more settled blended paper trades" in slip["unlock"]


def test_open_gate_with_paper_record_puts_blended_odds_on_the_slip(tmp_path):
    blended = {f"b{i}": {"variant": "blended", "city": "madrid", "marketDate": "2026-11-20", "bucket": "21°C",
                         "stake": 25, "pnl": -25, "status": "lost"} for i in range(20)}
    slip = run_sizing(tmp_path, blended, history=sharp_history(), now="2026-12-31T00:00:00Z")
    assert slip["paused"] is False, slip["unlock"]
    rec = slip["slip"][0]
    w = slip["gate"]["weight"]
    assert w >= 0.2
    assert abs(rec["blendedOdds"] - round(w * 0.60 + (1 - w) * 1.0, 3)) < 1e-9  # one bucket: market share is 1
    assert rec["priceWithFee"] > rec["limitPrice"]  # no schedule reported: the weather rate is assumed
