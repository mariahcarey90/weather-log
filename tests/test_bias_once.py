"""Fix 1 (28 Oct 2026): in live-calibration mode the city bias must be applied once, not twice."""
import json, pathlib, subprocess, sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_city_bias_applied_once(tmp_path):
    day = "2026-11-02"
    station = {"name": "Test", "icao": "TEST", "tz": "UTC", "unit": "C", "metar": [],
               "days": {day: {"market": {"buckets": [{"label": f"{t}°C", "yes": 0.1} for t in range(17, 26)]},
                              "models": {m: 20.0 for m in ("ecmwf_ifs025", "gfs_seamless",
                                                           "icon_seamless", "gem_seamless")}}}}
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "latest.json").write_text(json.dumps(
        {"fetched_at": "2026-11-01T11:50:00+00:00", "errors": [], "stations": {"testcity": station}}))
    for sub in ("snapshots", "outcomes"):
        (tmp_path / "db" / sub).mkdir(parents=True)
    for i in range(35):  # day-ahead runs: raw median 20.0 C, observed 21 C, so bias -1.0 C
        d = f"2026-10-{i % 28 + 1:02d}"
        (tmp_path / "db" / "snapshots" / f"s{i}.json").write_text(json.dumps(
            {"city": "testcity", "marketDate": d, "forecastRaw": 20.0, "leadHours": 24.0, "unit": "°C"}))
        (tmp_path / "db" / "outcomes" / f"testcity_{d}.json").write_text(json.dumps(
            {"city": "testcity", "date": d, "observedHigh": 21}))
    subprocess.run([sys.executable, str(REPO / "score.py"), "--db", "db", "--out", "out.json",
                    "--run-at", "2026-11-01T12:00:00Z", "--calib", "none.json"],
                   cwd=tmp_path, check=True, capture_output=True)
    doc = json.loads((tmp_path / "out.json").read_text())[0]["data"]
    assert "scored days" in doc["note"], doc["note"]  # the live-calibration branch ran
    top = max(doc["buckets"], key=lambda b: b["model"])
    assert top["label"] == "21°C", top  # with the double correction the peak sat on 22°C
