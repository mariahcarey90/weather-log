# weather-log

Personal, non-commercial research: compares weather forecasts with Polymarket daily
high-temperature markets for four airport stations (Madrid LEMD, Atlanta KATL,
Mexico City MMMX, Lucknow VILK).

- `fetch.py` runs on GitHub Actions every 2 hours and saves forecasts, ensembles,
  airport observations and market prices to `data/latest.json` (plus a compact
  line per run in `data/history/`).
- `score.py` turns that data into odds per market bucket; a scheduled Claude task
  logs them to the tracker page and scores them against the official highs.

Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0), NOAA/NWS and
aviationweather.gov. Market data from Polymarket's public API. Not financial advice.
