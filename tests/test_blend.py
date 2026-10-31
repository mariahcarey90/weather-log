"""Fixes 2 and 3 (29 Oct 2026): the blend weight and the accuracy gate."""
import blend
from blend import fit_weight, gate, rows_from, walk_forward


def flat_rows(n_dates=25, per_date=15):
    """Our odds are flat (no information); the market's are sharp."""
    return [(f"2026-11-{d:02d}", [0.2] * 5, [0.6 if j == i % 5 else 0.1 for j in range(5)], i % 5)
            for d in range(1, n_dates + 1) for i in range(per_date)]


def sharp_rows(n_dates=25, per_date=15):
    """Our odds are sharp; the market's are flat."""
    return [(f"2026-11-{d:02d}", [0.9 if j == i % 5 else 0.025 for j in range(5)], [0.2] * 5, i % 5)
            for d in range(1, n_dates + 1) for i in range(per_date)]


def test_flat_model_gets_no_weight():
    assert fit_weight(flat_rows()) == 0.0


def test_sharp_model_gets_full_weight():
    assert fit_weight(sharp_rows()) == 1.0


def test_no_rows_means_market_only():
    assert fit_weight([]) == 0.0


def test_gate_stays_shut_without_an_edge():
    is_open, info = gate(flat_rows())
    assert is_open is False and info["weight"] == 0.0


def test_gate_stays_shut_when_model_equals_market():
    rows = [(f"2026-11-{d:02d}", [0.2, 0.5, 0.3], [0.2, 0.5, 0.3], 1) for d in range(1, 29) for _ in range(12)]
    assert gate(rows)[0] is False


def test_gate_opens_for_a_clear_edge_with_enough_evidence():
    is_open, info = gate(sharp_rows(n_dates=27))
    assert is_open, info["needs"]
    assert info["oos_markets"] == (27 - blend.MIN_TRAIN_DATES) * 15


def test_gate_needs_enough_dates_even_with_an_edge():
    is_open, info = gate(sharp_rows(n_dates=12))
    assert is_open is False and any("more scored dates" in n for n in info["needs"])


def test_walk_forward_never_uses_the_scored_date():
    # The model is useless on the first 10 dates and perfect after: the first scored dates must
    # be judged with a weight fitted on the useless period, so their gain is not positive.
    rows = flat_rows(n_dates=10) + [r for r in sharp_rows(n_dates=20) if r[0] > "2026-11-10"]
    gains = walk_forward(sorted(rows))
    assert sum(gains["2026-11-11"]) <= 0


def test_equally_close_runs_pick_the_later_one_whatever_the_order():
    early = [{"label": "20°C", "model": 0.9, "market": 0.5}, {"label": "21°C", "model": 0.1, "market": 0.5}]
    late = [{"label": "20°C", "model": 0.1, "market": 0.5}, {"label": "21°C", "model": 0.9, "market": 0.5}]
    a = {"city": "x", "marketDate": "2026-11-02", "runAt": "2026-10-31T06:00:00Z", "leadHours": 27, "buckets": early}
    b = {"city": "x", "marketDate": "2026-11-02", "runAt": "2026-10-31T12:00:00Z", "leadHours": 21, "buckets": late}
    outs = {("x", "2026-11-02"): {"resolvedBucket": "21°C"}}
    assert rows_from([a, b], outs)[0][1] == rows_from([b, a], outs)[0][1] == [0.1, 0.9]


def test_rows_skip_old_stale_and_unsettled_runs():
    b = [{"label": "20°C", "model": 0.5, "market": 0.5}, {"label": "21°C", "model": 0.5, "market": 0.5}]
    snaps = [
        {"city": "x", "marketDate": "2026-11-01", "runAt": "2026-10-31T00:10:00Z", "leadHours": 30, "buckets": b},
        {"city": "x", "marketDate": "2026-11-01", "runAt": "2026-10-31T06:10:00Z", "leadHours": 24, "buckets": b},
        {"city": "x", "marketDate": "2026-11-02", "runAt": "2026-10-27T06:10:00Z", "leadHours": 24, "buckets": b},  # before the fix
        {"city": "x", "marketDate": "2026-11-03", "runAt": "2026-11-02T06:10:00Z", "leadHours": 24, "buckets": b, "stale": True},
        {"city": "x", "marketDate": "2026-11-04", "runAt": "2026-11-03T06:10:00Z", "leadHours": 24, "buckets": b},  # no outcome
    ]
    outs = {("x", d): {"resolvedBucket": "21°C"} for d in ("2026-11-01", "2026-11-02", "2026-11-03")}
    rows = rows_from(snaps, outs)
    assert len(rows) == 1 and rows[0][0] == "2026-11-01" and rows[0][3] == 1
