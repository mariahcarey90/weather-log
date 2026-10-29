"""Finding 11 (29 Oct 2026): paper trades and the slip pay the taker fee."""
import json, pathlib, subprocess, sys

from fees import effective_price, net_shares

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_zero_fee_market():
    zero = {"rate": 0.0, "exponent": 1.0}
    assert net_shares(25, 0.10, zero) == 250
    assert effective_price(0.10, zero) == 0.10


def test_weather_rate_at_ten_cents():
    rate = {"rate": 0.05, "exponent": 1.0}
    # fee in shares = shares * 0.05 * (1 - p): 250 shares lose 11.25
    assert abs(net_shares(25, 0.10, rate) - 238.75) < 1e-9
    assert abs(effective_price(0.10, rate) - 0.10 / 0.955) < 1e-12


def test_unknown_schedule_assumes_the_weather_rate():
    assert abs(net_shares(25, 0.10) - 238.75) < 1e-9


def test_paper_trade_pays_the_fee(tmp_path):
    bucket = {"label": "21°C", "lo": 21, "hi": 21, "market": 0.10, "model": 0.30,
              "variants": {"current": 0.30}, "fee": {"rate": 0.05, "exponent": 1.0},
              "depth": {"best_ask": 0.10, "avg_fill_25": 0.10}}
    snap = {"city": "x", "marketDate": "2026-11-02", "runAt": "2026-11-01T06:00:00Z", "leadHours": 24,
            "stability": "Settled", "buckets": [bucket]}
    (tmp_path / "snaps.json").write_text(json.dumps([{"doc_id": "s", "data": snap}]))
    (tmp_path / "db").mkdir()
    subprocess.run([sys.executable, str(REPO / "paper.py"), "--snaps", "snaps.json", "--db", "db", "--out", "w.json"],
                   cwd=tmp_path, check=True, capture_output=True)
    writes = json.loads((tmp_path / "w.json").read_text())
    assert len(writes) == 1
    p = writes[0]["data"]
    assert p["shares"] == 238.75 and p["effectivePrice"] == round(0.10 / 0.955, 4)
