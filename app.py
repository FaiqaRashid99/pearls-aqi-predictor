import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ Pearls AQI Predictor")
st.subheader("Real-time Air Quality Monitoring - Islamabad")

TOKEN = os.getenv("AQICN_TOKEN")

if not TOKEN:
    st.error(":() AQICN_TOKEN not found in .env file!")
    st.stop()

# List of stations to combine
stations_list = {
    "Islamabad US Embassy": "islamabad",
    "Islamabad General": "pakistan/islamabad",
    "E-11 Sector": "e11/4-sector-islamabad",
    "F-7 Street 40": "pakistan-islamabad-street-40",
    "Rawalpindi": "rawalpindi"
}

def get_aqi_data(station_name, station_key):
    url = f"https://api.waqi.info/feed/{station_key}/?token={TOKEN}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                return {
                    'name': station_name,
                    'aqi': data['data']['aqi'],
                    'city': data['data']['city']['name'],
                    'time': data['data']['time']['s']
                }
    except:
        pass
    return None

# Fetch data from all stations
st.sidebar.header("Settings")
selected_view = st.sidebar.radio("View Mode", ["City Average", "Individual Stations"])

if st.button("Refresh Data"):
    st.rerun()

# Main Dashboard
if selected_view == "City Average":
    st.subheader("Islamabad City Average AQI")
    
    all_data = []
    for name, key in stations_list.items():
        result = get_aqi_data(name, key)
        if result:
            all_data.append(result)
    
    if all_data:
        avg_aqi = round(sum(d['aqi'] for d in all_data) / len(all_data))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average AQI", avg_aqi)
        with col2:
            if avg_aqi <= 50:
                st.success("🟢 Good")
            elif avg_aqi <= 100:
                st.warning("🟡 Moderate")
            elif avg_aqi <= 150:
                st.error("🟠 Unhealthy for Sensitive Groups")
            else:
                st.error("🔴 Unhealthy / Hazardous")
        with col3:
            st.metric("Stations Used", len(all_data))
        
        # Show individual stations
        st.subheader("Data from Stations")
        for d in all_data:
            st.write(f"**{d['name']}**: {d['aqi']} AQI")
    else:
        st.error("Failed to fetch data from any station")

else:
    # Individual Stations View
    st.subheader("Individual Stations")
    cols = st.columns(2)
    
    for idx, (name, key) in enumerate(stations_list.items()):
        data = get_aqi_data(name, key)
        if data:
            with cols[idx % 2]:
                st.metric(
                    label=f"{data['name']}",
                    value=data['aqi'],
                    delta=None
                )
                if data['aqi'] <= 50:
                    st.success("Good")
                elif data['aqi'] <= 100:
                    st.warning("Moderate")
                elif data['aqi'] <= 150:
                    st.error("Unhealthy for Sensitive")
                else:
                    st.error("Unhealthy / Hazardous")

# Footer
st.caption("Data Source: AQICN.org | Updated in real-time")
st.caption("Project: Pearls AQI Predictor")
