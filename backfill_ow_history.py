"""
Backfill past AQI from OpenWeather's own Air Pollution *history* endpoint,
with hourly weather from the Open-Meteo archive.

- Same provider + same PM2.5 -> AQI formula as feature_pipeline.py, so the
  history matches your live data (no synthetic values anywhere).
- Writes feature_store/ow_history.csv with data_source = "ow_history".
  Nothing in Supabase is touched or overwritten.

Usage:
    python backfill_ow_history.py                      # 2021-01-01 -> 2026-06-06
    python backfill_ow_history.py --start 2023-11-01
    python backfill_ow_history.py --end 2026-06-06     # live data starts here

Needs OPENWEATHER_KEY in .env (same key you already use).
"""
import os
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("OPENWEATHER_KEY")
LAT, LON, CITY = 33.6844, 73.0479, "Islamabad"
OUT = os.path.join(os.getenv("FEATURE_STORE_PATH", "feature_store"), "ow_history.csv")
CHUNK_DAYS = 30

# Classic US EPA PM2.5 breakpoints (these match your live rows best).
BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]


def pm25_to_aqi(pm25):
    if pm25 is None or np.isnan(pm25):
        return np.nan
    c = np.floor(pm25 * 10) / 10          # EPA truncates to 0.1
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS:
        if c <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (max(c, c_lo) - c_lo) + i_lo, 1)
    return 500.0


def _get(url, params, tries=5):
    for k in range(tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (k + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if k == tries - 1:
                raise
            print(f"   retry after error: {e}")
            time.sleep(5 * (k + 1))


def fetch_pollution(start, end):
    """Hourly pollutants from OpenWeather history, in CHUNK_DAYS windows."""
    rows, t = [], start
    while t < end:
        t2 = min(t + pd.Timedelta(days=CHUNK_DAYS), end)
        js = _get("http://api.openweathermap.org/data/2.5/air_pollution/history", {
            "lat": LAT, "lon": LON, "appid": KEY,
            "start": int(t.timestamp()), "end": int(t2.timestamp())})
        for item in js.get("list", []):
            c = item["components"]
            rows.append({"timestamp": pd.Timestamp(item["dt"], unit="s"),
                         "pm25": c.get("pm2_5"), "pm10": c.get("pm10"),
                         "no2": c.get("no2"), "o3": c.get("o3"),
                         "co": c.get("co"), "so2": c.get("so2")})
        print(f"   pollution {t.date()} -> {t2.date()}: {len(js.get('list', []))} rows")
        t = t2
        time.sleep(1.1)                    # free plan: 60 calls/min
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = df["timestamp"].dt.floor("h")
    return df.drop_duplicates("timestamp").sort_values("timestamp")


def fetch_weather(start, end):
    """Hourly weather from Open-Meteo archive (free, no key), one year at a time."""
    names = {"temperature_2m": "temp", "apparent_temperature": "feels_like",
             "relative_humidity_2m": "humidity", "surface_pressure": "pressure",
             "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_direction",
             "precipitation": "precipitation"}
    frames, y = [], start.year
    while y <= end.year:
        s = max(start, pd.Timestamp(f"{y}-01-01"))
        e = min(end, pd.Timestamp(f"{y}-12-31"))
        js = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": LAT, "longitude": LON,
            "start_date": s.strftime("%Y-%m-%d"), "end_date": e.strftime("%Y-%m-%d"),
            "hourly": ",".join(names), "timezone": "UTC", "wind_speed_unit": "ms"})
        h = pd.DataFrame(js["hourly"])
        h["timestamp"] = pd.to_datetime(h.pop("time"))
        frames.append(h.rename(columns=names))
        print(f"   weather {y}: {len(h)} rows")
        y += 1
        time.sleep(1)
    return pd.concat(frames).drop_duplicates("timestamp")


def build(start, end):
    poll = fetch_pollution(start, end)
    if poll.empty:
        raise SystemExit("No pollution data returned - check key / dates.")
    wx = fetch_weather(poll.timestamp.min().normalize(), poll.timestamp.max().normalize())
    df = poll.merge(wx, on="timestamp", how="left")
    df = df[(df.timestamp >= start) & (df.timestamp < end)].copy()

    df["aqi"] = df["pm25"].apply(pm25_to_aqi)
    df = df.dropna(subset=["aqi"]).reset_index(drop=True)
    df["city"] = CITY
    df["hour"] = df.timestamp.dt.hour
    df["day"] = df.timestamp.dt.day
    df["month"] = df.timestamp.dt.month
    df["dayofweek"] = df.timestamp.dt.weekday
    df["is_weekend"] = (df.dayofweek >= 5).astype(int)
    df["is_rush_hour"] = df.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["aqi_change_rate"] = df["aqi"].diff().round(1)
    df["data_source"] = "ow_history"
    df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S+00")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-06-06",
                    help="exclusive; day your live OpenWeather rows begin")
    a = ap.parse_args()
    if not KEY:
        raise SystemExit("OPENWEATHER_KEY missing from .env")

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    out = build(start, end)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} rows -> {OUT}")
    print(f"AQI mean by month:\n{out.groupby('month').aqi.mean().round(0).to_string()}")
