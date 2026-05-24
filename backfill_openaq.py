"""
Real Historical Backfill using OpenAQ + Open-Meteo
- PM2.5 from OpenAQ (real measured data)
- Weather from Open-Meteo archive (real data)
- AQI calculated from PM2.5 using US EPA formula
"""
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

OPENAQ_KEY   = os.getenv("OPENAQ_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CITY         = os.getenv("CITY", "Islamabad")

# Islamabad coordinates
LAT, LON = 33.6844, 73.0479

# OpenAQ location IDs for Islamabad
LOCATION_IDS = [
    233470,   # Islamabad — 2021 to Feb 2026
    8634,     # US Diplomatic Post — 2019 to Mar 2025
]

FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
os.makedirs(FEATURE_STORE, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

OPENAQ_HEADERS = {"X-API-Key": OPENAQ_KEY}


# ─────────────────────────────────────────────
# 1. PM2.5 → AQI CONVERSION (US EPA formula)
# ─────────────────────────────────────────────
def pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 concentration (µg/m³) to AQI using US EPA formula"""
    if pm25 is None or np.isnan(pm25) or pm25 < 0:
        return None

    # EPA breakpoints: (PM2.5_low, PM2.5_high, AQI_low, AQI_high)
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
            return round(aqi, 1)
    return 500.0  # Beyond scale


# ─────────────────────────────────────────────
# 2. FETCH PM2.5 FROM OPENAQ
# ─────────────────────────────────────────────
def fetch_openaq_measurements(location_id: int, date_from: str, date_to: str) -> pd.DataFrame:
    """Fetch hourly PM2.5 measurements from OpenAQ for a location"""
    print(f"   Fetching OpenAQ location {location_id}: {date_from} → {date_to}")

    all_results = []
    page = 1
    limit = 1000

    while True:
        try:
            url = f"https://api.openaq.org/v3/locations/{location_id}/measurements"
            params = {
                "datetime_from": f"{date_from}T00:00:00Z",
                "datetime_to":   f"{date_to}T23:59:59Z",
                "limit":         limit,
                "page":          page,
                "parameters_id": 2,  # PM2.5 parameter ID
            }
            r = requests.get(url, params=params, headers=OPENAQ_HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"   API error {r.status_code}: {r.text[:200]}")
                break

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            print(f"   Page {page}: {len(results)} measurements (total: {len(all_results)})")

            if len(results) < limit:
                break
            page += 1

        except Exception as e:
            print(f"   Error on page {page}: {e}")
            break

    if not all_results:
        return pd.DataFrame()

    # Parse results
    rows = []
    for m in all_results:
        try:
            ts  = pd.to_datetime(m["period"]["datetimeFrom"]["utc"])
            val = m["value"]
            if val is not None and val >= 0:
                rows.append({"timestamp": ts, "pm25_raw": float(val)})
        except:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.drop_duplicates(subset=["timestamp"], keep="mean", inplace=True)
    df.sort_values("timestamp", inplace=True)
    print(f"   Got {len(df)} valid PM2.5 readings from location {location_id}")
    return df


# ─────────────────────────────────────────────
# 3. FETCH REAL WEATHER FROM OPEN-METEO
# ─────────────────────────────────────────────
def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch real hourly weather from Open-Meteo archive"""
    print(f"\n🌤️  Fetching Open-Meteo weather: {start_date} → {end_date}")
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature", "precipitation",
            "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "weather_code",
        ],
        "timezone": "Asia/Karachi",
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise Exception(f"Open-Meteo error: {r.text}")

    data = r.json()["hourly"]
    df   = pd.DataFrame(data)
    df.rename(columns={
        "time":                 "timestamp",
        "temperature_2m":       "temp",
        "relative_humidity_2m": "humidity",
        "apparent_temperature": "feels_like",
        "surface_pressure":     "pressure",
        "wind_speed_10m":       "wind_speed",
        "wind_direction_10m":   "wind_direction",
        "weather_code":         "weather_code",
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print(f"   Got {len(df)} hourly weather rows")
    return df


# ─────────────────────────────────────────────
# 4. COMBINE PM2.5 + WEATHER INTO FEATURES
# ─────────────────────────────────────────────
def build_feature_rows(pm25_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Merge PM2.5 and weather into feature rows"""
    print(f"\n⚙️  Building feature rows...")

    # Round timestamps to nearest hour for merging
    pm25_df["timestamp"] = pm25_df["timestamp"].dt.round("h")
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"]).dt.round("h")

    # Merge on timestamp
    df = pd.merge(weather_df, pm25_df, on="timestamp", how="inner")
    print(f"   Merged: {len(df)} rows with both PM2.5 and weather")

    if df.empty:
        return df

    # Convert PM2.5 → AQI
    df["aqi"]  = df["pm25_raw"].apply(pm25_to_aqi)
    df["pm25"] = df["pm25_raw"].round(2)
    df.drop(columns=["pm25_raw"], inplace=True)

    # Add null pollutant columns
    for col in ["pm10", "no2", "o3", "co", "so2"]:
        df[col] = None

    # Add time features
    df["hour"]            = df["timestamp"].dt.hour
    df["day"]             = df["timestamp"].dt.day
    df["month"]           = df["timestamp"].dt.month
    df["dayofweek"]       = df["timestamp"].dt.weekday
    df["is_weekend"]      = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"]    = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0).round(2)
    df["city"]            = CITY
    df["visibility"]      = None
    df["weather"]         = None

    # Format timestamp
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove rows with null AQI
    before = len(df)
    df.dropna(subset=["aqi"], inplace=True)
    print(f"   Removed {before - len(df)} rows with null AQI")
    print(f"   Final: {len(df)} rows")
    print(f"   AQI range: {df['aqi'].min():.0f} – {df['aqi'].max():.0f}")
    print(f"   PM2.5 range: {df['pm25'].min():.1f} – {df['pm25'].max():.1f} µg/m³")

    return df


# ─────────────────────────────────────────────
# 5. CLEAN EXISTING SYNTHETIC DATA
# ─────────────────────────────────────────────
def delete_synthetic_rows():
    """Delete backfill synthetic rows — keep only real pipeline data"""
    print("\n🗑️  Deleting synthetic backfill rows from Supabase...")

    # Real data starts from May 15 2026 (when feature pipeline started)
    # Everything before that from backfill is synthetic
    cutoff = "2026-05-15 00:00:00"

    try:
        result = (supabase.table("aqi_features")
                  .delete()
                  .lt("timestamp", cutoff)
                  .execute())
        print(f"   Deleted synthetic rows before {cutoff}")

        # Check remaining
        count = supabase.table("aqi_features").select("id", count="exact").execute()
        print(f"   Remaining real rows: {count.count}")
    except Exception as e:
        print(f"   Delete failed: {e}")


# ─────────────────────────────────────────────
# 6. STORE TO SUPABASE
# ─────────────────────────────────────────────
def store_to_supabase(df: pd.DataFrame):
    """Upload feature rows to Supabase"""
    print(f"\n💾 Uploading {len(df)} rows to Supabase...")

    # Fix data types
    int_cols = ["hour", "day", "month", "dayofweek", "is_weekend", "is_rush_hour"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    def clean_row(row):
        cleaned = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                cleaned[k] = None
            elif isinstance(v, np.integer):
                cleaned[k] = int(v)
            elif isinstance(v, np.floating):
                cleaned[k] = round(float(v), 4)
            elif isinstance(v, float):
                cleaned[k] = round(v, 4)
            else:
                cleaned[k] = v
        return cleaned

    rows    = [clean_row(r) for r in df.to_dict(orient="records")]
    success = 0
    failed  = 0

    for i in range(0, len(rows), 100):
        batch = rows[i:i+100]
        try:
            supabase.table("aqi_features").upsert(batch, on_conflict="timestamp").execute()
            success += len(batch)
            print(f"   Uploaded {success}/{len(rows)} rows...")
        except Exception as e:
            failed += len(batch)
            print(f"   Batch {i//100+1} failed: {e}")

    print(f"   ✅ Done! {success} uploaded | {failed} failed")
    return success


# ─────────────────────────────────────────────
# 7. SAVE CSV BACKUP
# ─────────────────────────────────────────────
def save_csv_backup(df: pd.DataFrame):
    csv_path    = os.path.join(FEATURE_STORE, "aqi_features.csv")
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df
    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"   ✅ CSV backup: {len(df_combined)} rows")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  REAL HISTORICAL BACKFILL (OpenAQ + Open-Meteo)")
    print("="*60)

    # Date range: last 1 year
    end_date   = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
    print(f"\n📅 Date range: {start_date} → {end_date}")

    # Step 1: Delete synthetic rows
    delete_synthetic_rows()

    # Step 2: Fetch real weather
    try:
        weather_df = fetch_historical_weather(start_date, end_date)
    except Exception as e:
        print(f"❌ Weather fetch failed: {e}")
        print("Try running with a VPN if Open-Meteo is blocked")
        exit(1)

    # Step 3: Fetch PM2.5 from all locations
    print(f"\n📡 Fetching PM2.5 from OpenAQ...")
    all_pm25 = []
    for loc_id in LOCATION_IDS:
        df_loc = fetch_openaq_measurements(loc_id, start_date, end_date)
        if not df_loc.empty:
            all_pm25.append(df_loc)

    if not all_pm25:
        print("❌ No PM2.5 data found! Check your OpenAQ API key and location IDs.")
        exit(1)

    # Combine all locations — average if overlap
    pm25_df = pd.concat(all_pm25, ignore_index=True)
    pm25_df["timestamp"] = pm25_df["timestamp"].dt.round("h")
    pm25_df = pm25_df.groupby("timestamp")["pm25_raw"].mean().reset_index()
    pm25_df.sort_values("timestamp", inplace=True)
    print(f"\n   Combined PM2.5: {len(pm25_df)} hourly readings")

    # Step 4: Build features
    features_df = build_feature_rows(pm25_df, weather_df)

    if features_df.empty:
        print("❌ No features built — check date ranges overlap")
        exit(1)

    # Step 5: Upload to Supabase
    store_to_supabase(features_df)

    # Step 6: Save CSV backup
    save_csv_backup(features_df)

    # Final count
    count = supabase.table("aqi_features").select("id", count="exact").execute()
    print(f"\n🎉 Backfill complete!")
    print(f"   Total rows in Supabase: {count.count}")
    print(f"   Real PM2.5 → AQI data: {len(features_df)} rows")
    print(f"\n   Run next: python training_pipeline.py")
    
### What this code is doing:
# 1. Deletes synthetic rows from Supabase (before May 15)
# 2. Fetches real hourly weather from Open-Meteo (1 year)
# 3. Fetches real PM2.5 from OpenAQ locations 233470 + 8634
# 4. Converts PM2.5 → AQI using US EPA formula (legitimate method)
# 5. Merges weather + AQI into feature rows
# 6. Uploads to Supabase + saves CSV backup