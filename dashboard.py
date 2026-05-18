import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
AQICN_TOKEN   = os.getenv("AQICN_TOKEN")
OW_KEY        = os.getenv("OPENWEATHER_KEY")
CITY          = os.getenv("CITY", "Islamabad")
FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
MODEL_DIR     = "models"

LAT, LON = 33.6844, 73.0479

FEATURE_COLS = [
    "temp", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_direction",
    "precipitation", "weather_code",
    "hour", "day", "month", "dayofweek",
    "is_weekend", "is_rush_hour", "aqi_change_rate",
]

# AQI categories
def aqi_category(aqi):
    if aqi <= 50:   return "Good",                    "#00e400", "😊"
    if aqi <= 100:  return "Moderate",                "#ffff00", "😐"
    if aqi <= 150:  return "Unhealthy for Sensitive", "#ff7e00", "😷"
    if aqi <= 200:  return "Unhealthy",               "#ff0000", "🤢"
    if aqi <= 300:  return "Very Unhealthy",          "#8f3f97", "🚨"
    return             "Hazardous",                   "#7e0023", "☠️"

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"AQI Predictor — {CITY}",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .aqi-badge {
        font-size: 3rem;
        font-weight: bold;
    }
    .alert-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(MODEL_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_resource
def load_metadata():
    path = os.path.join(MODEL_DIR, "model_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data(ttl=3600)
def load_feature_store():
    path = os.path.join(FEATURE_STORE, "aqi_features.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df

@st.cache_data(ttl=1800)
def fetch_current_aqi():
    try:
        url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
        r   = requests.get(url, timeout=10).json()
        if r["status"] == "ok":
            return r["data"]["aqi"], r["data"].get("iaqi", {})
    except:
        pass
    return None, {}

@st.cache_data(ttl=1800)
def fetch_current_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={OW_KEY}&units=metric"
        r   = requests.get(url, timeout=10).json()
        if r.get("cod") == 200:
            return r
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def fetch_forecast_weather():
    """Fetch 3-day hourly forecast from Open-Meteo (free, no key needed)"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT, "longitude": LON,
            "hourly": [
                "temperature_2m", "relative_humidity_2m",
                "apparent_temperature", "precipitation",
                "surface_pressure", "wind_speed_10m",
                "wind_direction_10m", "weather_code"
            ],
            "forecast_days": 4,
            "timezone": "Asia/Karachi"
        }
        r = requests.get(url, params=params, timeout=15).json()
        hourly = r["hourly"]
        df = pd.DataFrame(hourly)
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
        # Filter to next 3 days only
        now = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None)
        df  = df[df["timestamp"] > now].head(72)
        return df
    except Exception as e:
        st.warning(f"Forecast weather fetch failed: {e}")
        return pd.DataFrame()

def load_shap_importance():
    path = os.path.join(MODEL_DIR, "shap_importance.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def make_forecast(model, forecast_df: pd.DataFrame, last_aqi: float):
    if forecast_df.empty or model is None:
        return pd.DataFrame()

    rows = []
    prev_aqi = last_aqi

    for _, row in forecast_df.iterrows():
        ts = row["timestamp"]
        features = {
            "temp":             row.get("temp", 30),
            "feels_like":       row.get("feels_like", 30),
            "humidity":         row.get("humidity", 50),
            "pressure":         row.get("pressure", 1010),
            "wind_speed":       row.get("wind_speed", 3),
            "wind_direction":   row.get("wind_direction", 180),
            "precipitation":    row.get("precipitation", 0),
            "weather_code":     row.get("weather_code", 0),
            "hour":             ts.hour,
            "day":              ts.day,
            "month":            ts.month,
            "dayofweek":        ts.weekday(),
            "is_weekend":       int(ts.weekday() >= 5),
            "is_rush_hour":     int(ts.hour in [7, 8, 9, 17, 18, 19]),
            "aqi_change_rate":  0.0,
        }
        X = pd.DataFrame([features])[FEATURE_COLS]
        predicted_aqi = float(model.predict(X)[0])
        features["aqi_change_rate"] = predicted_aqi - prev_aqi
        prev_aqi = predicted_aqi

        rows.append({
            "timestamp":     ts,
            "predicted_aqi": round(predicted_aqi, 1),
            "temp":          row.get("temp", 30),
            "humidity":      row.get("humidity", 50),
            "wind_speed":    row.get("wind_speed", 3),
            "precipitation": row.get("precipitation", 0),
            "date":          ts.date(),
            "hour":          ts.hour,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
def main():
    # Load everything
    model    = load_model()
    metadata = load_metadata()
    hist_df  = load_feature_store()
    shap_imp = load_shap_importance()

    current_aqi, iaqi = fetch_current_aqi()
    weather           = fetch_current_weather()
    forecast_weather  = fetch_forecast_weather()

    # ── HEADER ──────────────────────────────
    st.title(f"AQI Predictor — {CITY}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Model: {metadata.get('best_model','N/A')} | R²: {metadata.get('r2','N/A')}")

    # ── HAZARD ALERT ────────────────────────
    if current_aqi and current_aqi > 150:
        cat, color, icon = aqi_category(current_aqi)
        st.markdown(f"""
        <div class="alert-box" style="background:{color}22; border-left: 5px solid {color}; color:{color}">
            {icon} ALERT: Current AQI is {current_aqi} — {cat}!
            Sensitive groups should avoid outdoor activities.
        </div>
        """, unsafe_allow_html=True)

    # ── CURRENT CONDITIONS ───────────────────
    st.subheader("Current Conditions")
    col1, col2, col3, col4, col5 = st.columns(5)

    if current_aqi:
        cat, color, icon = aqi_category(current_aqi)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background:linear-gradient(135deg,{color}88,{color}44)">
                <div class="aqi-badge">{current_aqi}</div>
                <div>{icon} {cat}</div>
                <div style="font-size:0.8rem;margin-top:5px">Current AQI</div>
            </div>""", unsafe_allow_html=True)

    if weather:
        with col2:
            st.metric("Temperature", f"{weather['main']['temp']:.1f}°C",
                      f"Feels {weather['main']['feels_like']:.1f}°C")
        with col3:
            st.metric("Humidity", f"{weather['main']['humidity']}%")
        with col4:
            st.metric("Wind Speed", f"{weather['wind']['speed']} m/s")
        with col5:
            st.metric("Pressure", f"{weather['main']['pressure']} hPa")

    st.divider()

    # ── 3-DAY FORECAST ───────────────────────
    st.subheader("📅 3-Day AQI Forecast")

    if model is None:
        st.error("Model not found. Run training_pipeline.py first.")
        return

    forecast_df = make_forecast(model, forecast_weather, current_aqi or 150)

    if not forecast_df.empty:
        # Daily summary cards
        days     = sorted(forecast_df["date"].unique())[:3]
        day_cols = st.columns(3)

        for col, day in zip(day_cols, days):
            day_data  = forecast_df[forecast_df["date"] == day]
            avg_aqi   = day_data["predicted_aqi"].mean()
            max_aqi   = day_data["predicted_aqi"].max()
            min_aqi   = day_data["predicted_aqi"].min()
            cat, color, icon = aqi_category(avg_aqi)
            label     = pd.Timestamp(day).strftime("%A, %b %d")

            with col:
                st.markdown(f"""
                <div class="metric-card" style="background:linear-gradient(135deg,{color}88,{color}22)">
                    <div style="font-size:1.1rem;font-weight:bold">{label}</div>
                    <div class="aqi-badge" style="font-size:2.5rem">{avg_aqi:.0f}</div>
                    <div>{icon} {cat}</div>
                    <div style="font-size:0.8rem;margin-top:8px">
                        Min: {min_aqi:.0f} | Max: {max_aqi:.0f}
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Hourly forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"],
            y=forecast_df["predicted_aqi"],
            mode="lines+markers",
            name="Predicted AQI",
            line=dict(color="#4CAF50", width=2),
            marker=dict(size=4),
            fill="tozeroy",
            fillcolor="rgba(76,175,80,0.1)"
        ))

        # AQI threshold lines
        for level, color, label in [
            (50,  "green",  "Good"),
            (100, "yellow", "Moderate"),
            (150, "orange", "Unhealthy"),
            (200, "red",    "Very Unhealthy"),
        ]:
            fig.add_hline(y=level, line_dash="dot",
                         line_color=color, opacity=0.5,
                         annotation_text=label, annotation_position="right")

        fig.update_layout(
            title="Hourly AQI Forecast — Next 3 Days",
            xaxis_title="Time",
            yaxis_title="AQI",
            height=400,
            template="plotly_dark",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── HISTORICAL TRENDS ────────────────────
    st.subheader("Historical AQI Trends")

    if not hist_df.empty:
        tab1, tab2, tab3 = st.tabs(["Daily Average", "By Hour of Day", "By Month"])

        with tab1:
            daily = hist_df.groupby(hist_df["timestamp"].dt.date)["aqi"].mean().reset_index()
            daily.columns = ["date", "avg_aqi"]
            fig1 = px.line(daily, x="date", y="avg_aqi",
                          title="Daily Average AQI",
                          labels={"avg_aqi": "AQI", "date": "Date"},
                          template="plotly_dark")
            fig1.update_traces(line_color="#4CAF50")
            st.plotly_chart(fig1, use_container_width=True)

        with tab2:
            hourly = hist_df.groupby("hour")["aqi"].mean().reset_index()
            fig2 = px.bar(hourly, x="hour", y="aqi",
                         title="Average AQI by Hour of Day",
                         labels={"aqi": "Avg AQI", "hour": "Hour"},
                         template="plotly_dark",
                         color="aqi",
                         color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            monthly = hist_df.groupby("month")["aqi"].mean().reset_index()
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            monthly["month_name"] = monthly["month"].map(month_names)
            fig3 = px.bar(monthly, x="month_name", y="aqi",
                         title="Average AQI by Month",
                         labels={"aqi": "Avg AQI", "month_name": "Month"},
                         template="plotly_dark",
                         color="aqi",
                         color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # ── SHAP FEATURE IMPORTANCE ──────────────
    st.subheader("What Drives AQI? (SHAP Feature Importance)")

    if shap_imp is not None:
        shap_df = shap_imp.reset_index()
        shap_df.columns = ["feature", "importance"]
        shap_df = shap_df.sort_values("importance", ascending=True).tail(10)

        fig4 = px.bar(shap_df, x="importance", y="feature",
                     orientation="h",
                     title="Top 10 Features by SHAP Importance",
                     labels={"importance": "Mean |SHAP value|", "feature": "Feature"},
                     template="plotly_dark",
                     color="importance",
                     color_continuous_scale="Blues")
        st.plotly_chart(fig4, use_container_width=True)

        st.info("""
        **How to read this:** Features with higher SHAP values have more influence on AQI predictions.
        - **wind_speed** — stronger winds disperse pollutants
        - **month** — seasonal smog patterns (worse in winter)
        - **is_rush_hour** — traffic emissions spike AQI
        - **humidity** — high humidity traps particles
        """)

    st.divider()

    # ── MODEL PERFORMANCE ────────────────────
    st.subheader("Model Performance")

    if metadata.get("all_results"):
        results_df = pd.DataFrame(metadata["all_results"])
        col1, col2 = st.columns(2)

        with col1:
            fig5 = px.bar(results_df, x="model", y="rmse",
                         title="RMSE by Model (lower is better)",
                         template="plotly_dark",
                         color="rmse",
                         color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            fig6 = px.bar(results_df, x="model", y="r2",
                         title="R² Score by Model (higher is better)",
                         template="plotly_dark",
                         color="r2",
                         color_continuous_scale="RdYlGn")
            st.plotly_chart(fig6, use_container_width=True)

        st.success(f"""
        **Best Model: {metadata.get('best_model')}**
        | RMSE: {metadata.get('rmse')} | MAE: {metadata.get('mae')} | R²: {metadata.get('r2')}
        | Trained: {metadata.get('trained_at')} UTC
        """)

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **AQI Predictor Dashboard**
        - City: {CITY}
        - Model: {metadata.get('best_model','N/A')}
        - Training rows: 2,161
        - Updates: Hourly (via GitHub Actions)
        """)

        st.divider()
        st.header("🎨 AQI Scale")
        scale = [
            (25,  "#00e400", "Good (0–50)"),
            (75,  "#ffff00", "Moderate (51–100)"),
            (125, "#ff7e00", "Unhealthy Sensitive (101–150)"),
            (175, "#ff0000", "Unhealthy (151–200)"),
            (250, "#8f3f97", "Very Unhealthy (201–300)"),
            (350, "#7e0023", "Hazardous (300+)"),
        ]
        for val, color, label in scale:
            st.markdown(
                f'<div style="background:{color};color:{"black" if val<100 else "white"};'
                f'padding:6px 10px;border-radius:5px;margin:3px 0;font-size:0.85rem">'
                f'{label}</div>',
                unsafe_allow_html=True
            )

        st.divider()
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()