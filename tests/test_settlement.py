"""Fix 5 (28 Oct 2026): read observations the way the markets settle them."""
import fetch
from fetch import settled_value


def test_settled_value_rounds_half_up():
    assert settled_value(23.9, "F") == 75   # 75.02 F
    assert settled_value(22.5, "F") == 73   # exactly 72.5 F: the rules round up, round() gives 72
    assert settled_value(24.5, "C") == 25   # round() gives 24
    assert settled_value(23.4, "C") == 23
    assert settled_value(-2.5, "C") == -2   # halves go up, toward warmer


def test_fetch_metar_one_request_per_station_and_no_speci(monkeypatch):
    urls = []

    def fake_get(url, *a, **k):
        urls.append(url)
        icao = url.split("ids=")[1].split("&")[0]
        return [
            {"icaoId": icao, "obsTime": 1793170200, "temp": 21.0, "rawOb": f"{icao} 281230Z 21/10 RMK"},
            {"icaoId": icao, "obsTime": 1793171400, "temp": 23.0, "rawOb": f"SPECI {icao} 281250Z 23/10"},
        ]

    monkeypatch.setattr(fetch, "get", fake_get)
    obs = fetch.fetch_metar(pause_s=0)
    assert len(urls) == len(fetch.STATIONS)          # one request per station, under the 400-entry cap
    assert all("," not in u.split("ids=")[1].split("&")[0] for u in urls)
    for st in fetch.STATIONS:
        temps = [o["temp"] for o in obs[st["icao"]]]
        assert temps == [21.0]                         # the SPECI reading never counts
