import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

AQICN_TOKEN   = os.getenv("AQICN_TOKEN")
OW_KEY        = os.getenv("OPENWEATHER_KEY")
CITY          = os.getenv("CITY", "Islamabad")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")

# Also keep local CSV as backup
FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
os.makedirs(FEATURE_STORE, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────
# 1. FETCH DATA
# ─────────────────────────────────────────────
def fetch_aqi() -> dict:
    """Fetch live AQI and pollutant data from AQICN (US Embassy station — only valid Islamabad feed)"""
    url  = f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}"
    r    = requests.get(url, timeout=10).json()
    if r.get("status") != "ok":
        raise Exception(f"AQICN API error: {r}")
    data = r["data"]
    iaqi = data.get("iaqi", {})
    return {
        "aqi":  data["aqi"],
        "pm25": iaqi.get("pm25", {}).get("v", None),
        "pm10": iaqi.get("pm10", {}).get("v", None),
        "no2":  iaqi.get("no2",  {}).get("v", None),
        "o3":   iaqi.get("o3",   {}).get("v", None),
        "co":   iaqi.get("co",   {}).get("v", None),
        "so2":  iaqi.get("so2",  {}).get("v", None),
    }


def fetch_weather() -> dict:
    """Fetch live weather from OpenWeatherMap"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OW_KEY}&units=metric"
    r   = requests.get(url, timeout=10).json()
    if r.get("cod") != 200:
        raise Exception(f"OpenWeather API error: {r}")
    return {
        "temp":           r["main"]["temp"],
        "feels_like":     r["main"]["feels_like"],
        "humidity":       r["main"]["humidity"],
        "pressure":       r["main"]["pressure"],
        "wind_speed":     r["wind"]["speed"],
        "wind_direction": r["wind"].get("deg", 0),
        "visibility":     r.get("visibility", 10000),
        "weather_code":   r["weather"][0]["id"],
        "precipitation":  r.get("rain", {}).get("1h", 0.0),
    }


# ─────────────────────────────────────────────
# 2. ENGINEER FEATURES
# ─────────────────────────────────────────────
def engineer_features(aqi_data: dict, weather_data: dict) -> dict:
    """Combine API data and add time-based engineered features"""
    from datetime import timezone, timedelta
    PKT = timezone(timedelta(hours=5))
    now      = datetime.now(PKT).replace(tzinfo=None)  # PKT, tz-naive
    features = {**aqi_data, **weather_data}

    # Time-based features
    features["hour"]            = now.hour
    features["day"]             = now.day
    features["month"]           = now.month
    features["dayofweek"]       = now.weekday()
    features["is_weekend"]      = int(now.weekday() >= 5)
    features["is_rush_hour"]    = int(now.hour in [7, 8, 9, 17, 18, 19])
    features["aqi_change_rate"] = 0.0
    features["timestamp"]       = now.strftime("%Y-%m-%d %H:%M:%S")
    features["city"]            = CITY
    return features


def compute_aqi_change_rate(features: dict) -> dict:
    """Calculate AQI change vs last stored row in Supabase"""
    try:
        result = (supabase.table("aqi_features")
                  .select("aqi")
                  .order("timestamp", desc=True)
                  .limit(1)
                  .execute())
        if result.data:
            last_aqi = result.data[0]["aqi"]
            features["aqi_change_rate"] = round(features["aqi"] - last_aqi, 2)
    except Exception as e:
        print(f"   Could not compute change rate: {e}")
    return features


# ─────────────────────────────────────────────
# 3. STORE TO SUPABASE + CSV BACKUP
# ─────────────────────────────────────────────
def store_to_supabase(features: dict) -> bool:
    """Insert feature row into Supabase table"""
    try:
        # Convert None values and ensure JSON serializable
        row = {k: (float(v) if isinstance(v, float) else v)
               for k, v in features.items()}

        result = (supabase.table("aqi_features")
                  .upsert(row, on_conflict="timestamp")
                  .execute())

        print(f"   Stored to Supabase!")
        return True
    except Exception as e:
        print(f"   Supabase insert failed: {e}")
        return False


def store_to_csv_backup(features: dict):
    """Keep local CSV as backup"""
    csv_path    = os.path.join(FEATURE_STORE, "aqi_features.csv")
    df_new      = pd.DataFrame([features])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"   CSV backup updated ({len(df_combined)} rows)")


# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    print(f"\n{'='*55}")
    print(f"  Feature Pipeline — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*55}")

    print("Fetching AQI data...")
    aqi_data = fetch_aqi()
    print(f"   AQI: {aqi_data['aqi']} | PM2.5: {aqi_data['pm25']}")

    print("Fetching weather data...")
    weather_data = fetch_weather()
    print(f"   Temp: {weather_data['temp']}°C | Humidity: {weather_data['humidity']}%")

    print("Engineering features...")
    features = engineer_features(aqi_data, weather_data)
    features = compute_aqi_change_rate(features)

    print("Storing features...")
    store_to_supabase(features)
    store_to_csv_backup(features)

    print(f"\nStored row summary:")
    print(f"   Timestamp:      {features['timestamp']}")
    print(f"   AQI:            {features['aqi']}")
    print(f"   Temp:           {features['temp']}°C")
    print(f"   Humidity:       {features['humidity']}%")
    print(f"   Wind Speed:     {features['wind_speed']} m/s")
    print(f"   Hour:           {features['hour']}")
    print(f"   Is Rush Hour:   {features['is_rush_hour']}")
    print(f"   AQI Change:     {features['aqi_change_rate']}")
    return features


if __name__ == "__main__":
    run_pipeline()