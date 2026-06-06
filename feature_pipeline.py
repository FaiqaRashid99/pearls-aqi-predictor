import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

OW_KEY        = os.getenv("OPENWEATHER_KEY")
CITY          = os.getenv("CITY", "Islamabad")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")

# Islamabad coordinates — used for OpenWeatherMap Air Pollution API
LAT, LON = 33.6844, 73.0479

FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
os.makedirs(FEATURE_STORE, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────
# PM2.5 → AQI  (US EPA formula)
# ─────────────────────────────────────────────
def pm25_to_aqi(pm25: float) -> int:
    breakpoints = [
        (0.0,   12.0,   0,   50),
        (12.1,  35.4,  51,  100),
        (35.5,  55.4, 101,  150),
        (55.5, 150.4, 151,  200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    return 500


# ─────────────────────────────────────────────
# 1. FETCH AQI — OpenWeatherMap Air Pollution
#    (replaced dead AQICN US Embassy station)
# ─────────────────────────────────────────────
def fetch_aqi() -> dict:
    """
    Fetch real-time PM2.5 and pollutants from OpenWeatherMap Air Pollution API.
    Converts PM2.5 → AQI using US EPA formula.
    This replaced AQICN which stopped reporting in Feb 2026.
    """
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    r   = requests.get(url, params={"lat": LAT, "lon": LON, "appid": OW_KEY}, timeout=10)

    if r.status_code != 200:
        raise Exception(f"OpenWeatherMap Air Pollution API error {r.status_code}: {r.text}")

    data       = r.json()
    components = data["list"][0]["components"]

    pm25 = components.get("pm2_5", 0.0)
    pm10 = components.get("pm10",  0.0)
    no2  = components.get("no2",   0.0)
    o3   = components.get("o3",    0.0)
    co   = components.get("co",    0.0)
    so2  = components.get("so2",   0.0)

    aqi = pm25_to_aqi(pm25)

    print(f"   PM2.5: {pm25} µg/m³  →  AQI: {aqi}")
    return {
        "aqi":  aqi,
        "pm25": round(pm25, 2),
        "pm10": round(pm10, 2),
        "no2":  round(no2,  2),
        "o3":   round(o3,   2),
        "co":   round(co,   2),
        "so2":  round(so2,  2),
    }


# ─────────────────────────────────────────────
# 2. FETCH WEATHER — OpenWeatherMap Current
# ─────────────────────────────────────────────
def fetch_weather() -> dict:
    """Fetch live weather from OpenWeatherMap"""
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={CITY}&appid={OW_KEY}&units=metric")
    r   = requests.get(url, timeout=10).json()
    if r.get("cod") != 200:
        raise Exception(f"OpenWeather weather API error: {r}")
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
# 3. ENGINEER FEATURES
# ─────────────────────────────────────────────
def engineer_features(aqi_data: dict, weather_data: dict) -> dict:
    """Combine API data and add time-based engineered features"""
    PKT = timezone(timedelta(hours=5))
    now = datetime.now(PKT).replace(tzinfo=None)  # PKT, tz-naive

    features = {**aqi_data, **weather_data}
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
# 4. STORE TO SUPABASE + CSV BACKUP
# ─────────────────────────────────────────────
def store_to_supabase(features: dict) -> bool:
    """Insert feature row into Supabase table"""
    try:
        row = {k: (float(v) if isinstance(v, float) else v)
               for k, v in features.items()}
        supabase.table("aqi_features").upsert(row, on_conflict="timestamp").execute()
        print(f"   Stored to Supabase!")
        return True
    except Exception as e:
        print(f"   Supabase insert failed: {e}")
        return False


def store_to_csv_backup(features: dict):
    """Keep local CSV as backup"""
    csv_path = os.path.join(FEATURE_STORE, "aqi_features.csv")
    df_new   = pd.DataFrame([features])
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
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    print(f"\n{'='*55}")
    print(f"  Feature Pipeline — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*55}")

    print("Fetching AQI data (OpenWeatherMap Air Pollution)...")
    aqi_data = fetch_aqi()
    print(f"   AQI: {aqi_data['aqi']} | PM2.5: {aqi_data['pm25']} µg/m³")

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
    print(f"   Timestamp:    {features['timestamp']}")
    print(f"   AQI:          {features['aqi']}")
    print(f"   PM2.5:        {features['pm25']} µg/m³")
    print(f"   Temp:         {features['temp']}°C")
    print(f"   Humidity:     {features['humidity']}%")
    print(f"   Wind Speed:   {features['wind_speed']} m/s")
    print(f"   Hour:         {features['hour']}")
    print(f"   Is Rush Hour: {features['is_rush_hour']}")
    print(f"   AQI Change:   {features['aqi_change_rate']}")
    return features


if __name__ == "__main__":
    run_pipeline()