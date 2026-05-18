import os
import requests
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
OW_KEY = os.getenv("OPENWEATHER_KEY")
CITY = os.getenv("CITY", "Rawalpindi")

print("=" * 40)
print("Testing AQICN API...")
try:
    r = requests.get(f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}")
    data = r.json()
    if data["status"] == "ok":
        print(f"✅ AQICN works! Current AQI in {CITY}: {data['data']['aqi']}")
    else:
        print(f"❌ AQICN failed: {data}")
except Exception as e:
    print(f"❌ AQICN error: {e}")

print("=" * 40)
print("Testing OpenWeather API...")
try:
    r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OW_KEY}&units=metric")
    data = r.json()
    if r.status_code == 200:
        print(f"✅ OpenWeather works! Temp: {data['main']['temp']}°C, Humidity: {data['main']['humidity']}%")
    else:
        print(f"❌ OpenWeather failed: {data['message']}")
except Exception as e:
    print(f"❌ OpenWeather error: {e}")

print("=" * 40)
print("Testing Hopsworks...")
try:
    import hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    print(f"✅ Hopsworks works! Connected to project: {project.name}")
except Exception as e:
    print(f"❌ Hopsworks error: {e}")

print("=" * 40)
