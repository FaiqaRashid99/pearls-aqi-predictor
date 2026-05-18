import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OW_KEY = os.getenv("OPENWEATHER_KEY")
CITY = os.getenv("CITY", "Islamabad")
FEATURE_STORE_PATH = os.getenv("FEATURE_STORE_PATH", "feature_store")

# Make sure folder exists
os.makedirs(FEATURE_STORE_PATH, exist_ok=True)


def fetch_aqi():
    """Fetch current AQI and pollutant data from AQICN"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    r = requests.get(url, timeout=10).json()

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


def fetch_weather():
    """Fetch current weather from OpenWeatherMap"""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OW_KEY}&units=metric"
    r = requests.get(url, timeout=10).json()

    if r.get("cod") != 200:
        raise Exception(f"OpenWeather API error: {r}")

    return {
        "temp":       r["main"]["temp"],
        "feels_like": r["main"]["feels_like"],
        "humidity":   r["main"]["humidity"],
        "pressure":   r["main"]["pressure"],
        "wind_speed": r["wind"]["speed"],
        "visibility": r.get("visibility", None),
        "weather":    r["weather"][0]["description"],
    }


def engineer_features(aqi_data, weather_data):
    """Combine and add time-based features"""
    now = datetime.utcnow()

    features = {**aqi_data, **weather_data}

    # Time-based features
    features["hour"]       = now.hour
    features["day"]        = now.day
    features["month"]      = now.month
    features["dayofweek"]  = now.weekday()   # 0=Monday, 6=Sunday
    features["is_weekend"] = int(now.weekday() >= 5)
    features["timestamp"]  = now.strftime("%Y-%m-%d %H:%M:%S")
    features["city"]       = CITY

    return features


def store_features(features: dict):
    """Append features to local CSV feature store"""
    csv_path = os.path.join(FEATURE_STORE_PATH, "aqi_features.csv")
    df_new = pd.DataFrame([features])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Remove duplicate timestamps
    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ Features stored! Total rows: {len(df_combined)}")
    return df_combined


def run_pipeline():
    print(f"\n{'='*45}")
    print(f"Feature Pipeline — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*45}")

    print("📡 Fetching AQI data...")
    aqi_data = fetch_aqi()
    print(f"   AQI: {aqi_data['aqi']} | PM2.5: {aqi_data['pm25']} | PM10: {aqi_data['pm10']}")

    print("🌤️  Fetching weather data...")
    weather_data = fetch_weather()
    print(f"   Temp: {weather_data['temp']}°C | Humidity: {weather_data['humidity']}% | Wind: {weather_data['wind_speed']} m/s")

    print("⚙️  Engineering features...")
    features = engineer_features(aqi_data, weather_data)

    print("💾 Storing features...")
    df = store_features(features)

    print(f"\n📊 Latest row saved:")
    print(pd.DataFrame([features])[["timestamp","aqi","pm25","pm10","temp","humidity","hour","dayofweek"]].to_string(index=False))
    return df


if __name__ == "__main__":
    run_pipeline()