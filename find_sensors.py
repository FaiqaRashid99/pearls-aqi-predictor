import requests
import os
from dotenv import load_dotenv

load_dotenv()

headers = {"X-API-Key": os.getenv("OPENAQ_KEY")}
url = "https://api.openaq.org/v3/locations"
params = {
    "coordinates": "33.6844,73.0479",
    "radius": 25000,
    "limit": 10
}

r = requests.get(url, params=params, headers=headers).json()

if "results" not in r:
    print("Error:", r)
else:
    print(f"Found {len(r['results'])} locations near Islamabad:\n")
    for loc in r["results"]:
        sensors = [s["parameter"]["name"] for s in loc.get("sensors", [])]
        first = loc.get("datetimeFirst", {}).get("utc", "?")
        last  = loc.get("datetimeLast",  {}).get("utc", "?")
        print(f"ID: {loc['id']} | Name: {loc['name']}")
        print(f"   Sensors: {sensors}")
        print(f"   Data from: {first} to {last}")
        print()
        