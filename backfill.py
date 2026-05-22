import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

AQICN_TOKEN  = os.getenv("AQICN_TOKEN")
CITY         = os.getenv("CITY", "Islamabad")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Islamabad coordinates
LAT, LON = 33.6844, 73.0479

# Local CSV backup
FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
os.makedirs(FEATURE_STORE, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────
# 1. FETCH REAL HISTORICAL WEATHER (Open-Meteo)
# ─────────────────────────────────────────────
def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Fetching real hourly weather: {start_date} → {end_date}")
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature", "precipitation",
            "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "visibility", "weather_code",
        ],
        "timezone": "Asia/Karachi",
    }
    r  = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Open-Meteo error: {r.text}")

    data   = r.json()
    df     = pd.DataFrame(data["hourly"])
    df.rename(columns={
        "time":                 "timestamp",
        "temperature_2m":       "temp",
        "relative_humidity_2m": "humidity",
        "apparent_temperature": "feels_like",
        "precipitation":        "precipitation",
        "surface_pressure":     "pressure",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_direction",
        "visibility":           "visibility",
        "weather_code":         "weather_code",
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"   Got {len(df)} hourly weather rows")
    return df


# ─────────────────────────────────────────────
# 2. FETCH CURRENT AQI (AQICN)
# ─────────────────────────────────────────────
def fetch_current_aqi() -> dict:
    print("📡 Fetching current AQI from AQICN...")
    url  = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    r    = requests.get(url, timeout=10).json()
    if r["status"] != "ok":
        raise Exception(f"AQICN error: {r}")
    data = r["data"]
    iaqi = data.get("iaqi", {})
    aqi  = {
        "aqi_now": data["aqi"],
        "pm25":    iaqi.get("pm25", {}).get("v", None),
        "pm10":    iaqi.get("pm10", {}).get("v", None),
        "no2":     iaqi.get("no2",  {}).get("v", None),
        "o3":      iaqi.get("o3",   {}).get("v", None),
        "co":      iaqi.get("co",   {}).get("v", None),
        "so2":     iaqi.get("so2",  {}).get("v", None),
    }
    print(f"   Current AQI: {aqi['aqi_now']} | PM2.5: {aqi['pm25']}")
    return aqi


# ─────────────────────────────────────────────
# 3. ESTIMATE AQI FROM WEATHER
# ─────────────────────────────────────────────
def estimate_aqi(row, base_aqi: float) -> float:
    month         = row["timestamp"].month
    hour          = row["timestamp"].hour
    humidity      = row["humidity"]      if not pd.isna(row["humidity"])      else 50
    wind_speed    = row["wind_speed"]    if not pd.isna(row["wind_speed"])    else 3
    temp          = row["temp"]          if not pd.isna(row["temp"])          else 25
    precipitation = row["precipitation"] if not pd.isna(row["precipitation"]) else 0

    seasonal = {1:1.6,2:1.4,3:1.1,4:0.95,5:1.0,6:0.9,
                7:0.8,8:0.82,9:0.9,10:1.1,11:1.4,12:1.6}
    hourly   = {0:1.1,1:1.05,2:1.0,3:0.95,4:0.95,5:1.0,
                6:1.1,7:1.25,8:1.3,9:1.2,10:1.1,11:1.05,
                12:1.1,13:1.05,14:1.0,15:1.0,16:1.1,17:1.25,
                18:1.3,19:1.2,20:1.15,21:1.1,22:1.1,23:1.1}

    humidity_factor = 1 + (humidity - 50) * 0.004
    wind_factor     = max(0.5, 1 - wind_speed * 0.06)
    rain_factor     = max(0.4, 1 - precipitation * 0.3)
    temp_factor     = 1 + max(0, temp - 35) * 0.01

    aqi_est = (base_aqi
               * seasonal.get(month, 1.0)
               * hourly.get(hour, 1.0)
               * humidity_factor
               * wind_factor
               * rain_factor
               * temp_factor)

    noise   = np.random.normal(0, base_aqi * 0.03)
    return max(10, round(aqi_est + noise, 1))


# ─────────────────────────────────────────────
# 4. BUILD FEATURE ROWS
# ─────────────────────────────────────────────
def build_features(weather_df: pd.DataFrame, aqi_ref: dict) -> list:
    print(f"⚙️  Building features for {len(weather_df)} hourly rows...")
    base_aqi = aqi_ref["aqi_now"]
    rows     = []

    for _, row in weather_df.iterrows():
        ts      = row["timestamp"]
        aqi_est = estimate_aqi(row, base_aqi)
        scale   = aqi_est / base_aqi if base_aqi > 0 else 1.0

        prev_aqi        = rows[-1]["aqi"] if rows else aqi_est
        aqi_change_rate = round(aqi_est - prev_aqi, 2)

        feature_row = {
            "aqi":              aqi_est,
            "pm25":             round(aqi_ref["pm25"] * scale, 1) if aqi_ref["pm25"] else None,
            "pm10":             round(aqi_ref["pm10"] * scale, 1) if aqi_ref["pm10"] else None,
            "no2":              round(aqi_ref["no2"]  * scale, 1) if aqi_ref["no2"]  else None,
            "o3":               round(aqi_ref["o3"]   * scale, 1) if aqi_ref["o3"]   else None,
            "co":               round(aqi_ref["co"]   * scale, 1) if aqi_ref["co"]   else None,
            "so2":              round(aqi_ref["so2"]  * scale, 1) if aqi_ref["so2"]  else None,
            "temp":             row["temp"],
            "feels_like":       row["feels_like"],
            "humidity":         row["humidity"],
            "pressure":         row["pressure"],
            "wind_speed":       row["wind_speed"],
            "wind_direction":   row["wind_direction"],
            "visibility":       row["visibility"],
            "precipitation":    row["precipitation"],
            "weather_code":     row["weather_code"],
            "hour":             int(ts.hour),
            "day":              int(ts.day),
            "month":            int(ts.month),
            "dayofweek":        int(ts.weekday()),
            "is_weekend":       int(ts.weekday() >= 5),
            "is_rush_hour":     int(ts.hour in [7, 8, 9, 17, 18, 19]),
            "aqi_change_rate":  aqi_change_rate,
            "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
            "city":             CITY,
        }
        # Replace NaN with None for Supabase
        feature_row = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in feature_row.items()}
        rows.append(feature_row)

    print(f"   Built {len(rows)} feature rows")
    return rows


# ─────────────────────────────────────────────
# 5. STORE TO SUPABASE (batch insert)
# ─────────────────────────────────────────────
def store_to_supabase(rows: list):
    print(f"\nUploading {len(rows)} rows to Supabase...")
    batch_size = 100
    success    = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            (supabase.table("aqi_features")
             .upsert(batch, on_conflict="timestamp")
             .execute())
            success += len(batch)
            print(f"   Uploaded batch {i//batch_size + 1} ({success}/{len(rows)} rows)")
        except Exception as e:
            print(f"   Batch {i//batch_size + 1} failed: {e}")

    print(f"   Supabase upload complete! {success}/{len(rows)} rows stored")


# ─────────────────────────────────────────────
# 6. STORE CSV BACKUP
# ─────────────────────────────────────────────
def store_csv_backup(rows: list):
    csv_path    = os.path.join(FEATURE_STORE, "aqi_features.csv")
    df_new      = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"   CSV backup saved ({len(df_combined)} rows)")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AQI PREDICTOR — HISTORICAL BACKFILL (Supabase)")
    print("="*55)

    end_date   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"\nBackfill period: {start_date} → {end_date}")
    print(f"   Expected rows:  ~{90 * 24} hourly readings\n")

    weather_df  = fetch_historical_weather(start_date, end_date)
    aqi_ref     = fetch_current_aqi()
    rows        = build_features(weather_df, aqi_ref)

    store_to_supabase(rows)
    store_csv_backup(rows)

    # Verify count in Supabase
    result = supabase.table("aqi_features").select("id", count="exact").execute()
    print(f"\nBackfill complete!")
    print(f"   Rows in Supabase: {result.count}")
    print(f"\n   Run next: python training_pipeline.py")