import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN   = os.getenv("AQICN_TOKEN")
OW_KEY        = os.getenv("OPENWEATHER_KEY")
CITY          = os.getenv("CITY", "Islamabad")
FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")

os.makedirs(FEATURE_STORE, exist_ok=True)


def fetch_aqi() -> dict:
    """Fetch live AQI and pollutant data from AQICN"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    r   = requests.get(url, timeout=10).json()

    if r["status"] != "ok":
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


def engineer_features(aqi_data: dict, weather_data: dict) -> dict:
    """Combine API data and add time-based engineered features"""
    now      = datetime.utcnow()
    features = {**aqi_data, **weather_data}

    features["hour"]             = now.hour
    features["day"]              = now.day
    features["month"]            = now.month
    features["dayofweek"]        = now.weekday()
    features["is_weekend"]       = int(now.weekday() >= 5)
    features["is_rush_hour"]     = int(now.hour in [7, 8, 9, 17, 18, 19])
    features["aqi_change_rate"]  = 0.0   # updated below if history exists
    features["timestamp"]        = now.strftime("%Y-%m-%d %H:%M:%S")
    features["city"]             = CITY

    return features


def compute_aqi_change_rate(features: dict, csv_path: str) -> dict:
    """Calculate AQI change rate compared to previous row"""
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        if not df_existing.empty:
            last_aqi = df_existing["aqi"].iloc[-1]
            features["aqi_change_rate"] = round(features["aqi"] - last_aqi, 2)
    return features


def store_features(features: dict) -> pd.DataFrame:
    """Append new feature row to CSV feature store"""
    csv_path   = os.path.join(FEATURE_STORE, "aqi_features.csv")
    features   = compute_aqi_change_rate(features, csv_path)
    df_new     = pd.DataFrame([features])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.sort_values("timestamp", inplace=True)
    df_combined.to_csv(csv_path, index=False)

    print(f"Feature stored! Total rows: {len(df_combined)}")
    return df_combined


def run_pipeline():
    print(f"\n{'='*50}")
    print(f"Feature Pipeline — {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'='*50}")

    print("Fetching AQI data...")
    aqi_data = fetch_aqi()
    print(f"   AQI: {aqi_data['aqi']} | PM2.5: {aqi_data['pm25']}")

    print("Fetching weather data...")
    weather_data = fetch_weather()
    print(f"   Temp: {weather_data['temp']}°C | Humidity: {weather_data['humidity']}%")

    print("Engineering features...")
    features = engineer_features(aqi_data, weather_data)

    print("Storing to feature store...")
    df = store_features(features)

    print(f"\nLatest entry:")
    cols = ["timestamp", "aqi", "pm25", "temp", "humidity",
            "wind_speed", "hour", "is_rush_hour", "aqi_change_rate"]
    available = [c for c in cols if c in df.columns]
    print(df[available].tail(1).to_string(index=False))
    return df


if __name__ == "__main__":
    run_pipeline()