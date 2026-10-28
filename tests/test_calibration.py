"""Fix 4 (28 Oct 2026): live calibration counts market-days, not logged runs, over a 30-day window."""
from datetime import datetime, timezone

from score import calibration


def test_runs_on_one_day_count_once():
    snaps = [{"city": "x", "marketDate": f"2026-10-0{d}", "forecastRaw": 20.0, "leadHours": h, "unit": "°C"}
             for d in range(1, 6) for h in (14, 20, 26, 32)]
    outs = {("x", f"2026-10-0{d}"): {"observedHigh": 21} for d in range(1, 6)}
    cal = calibration(snaps, outs)
    assert cal[("x", "day")]["n"] == 5  # 20 runs, 5 market-days
    assert abs(cal[("x", "day")]["bias"] + 1.0) < 1e-9


def test_old_days_drop_out_of_the_window():
    snaps = [{"city": "x", "marketDate": f"2026-09-{d:02d}", "forecastRaw": 20.0, "leadHours": 24, "unit": "°C"}
             for d in range(1, 11)]
    outs = {("x", f"2026-09-{d:02d}"): {"observedHigh": 21} for d in range(1, 11)}
    run_at = datetime(2026, 10, 28, tzinfo=timezone.utc)
    assert calibration(snaps, outs, run_at=run_at) == {}  # all older than 30 days


def test_fahrenheit_errors_convert_to_celsius():
    snaps = [{"city": "y", "marketDate": f"2026-10-0{d}", "forecastRaw": 70.0, "leadHours": 24, "unit": "°F"}
             for d in range(1, 6)]
    outs = {("y", f"2026-10-0{d}"): {"observedHigh": 79} for d in range(1, 6)}
    assert abs(calibration(snaps, outs)[("y", "day")]["bias"] + 5.0) < 1e-9  # -9 F = -5 C
