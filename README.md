# aqi-predictor

# 🌬️ AQI Predictor — Islamabad

An end-to-end machine learning system that predicts Air Quality Index (AQI) for the next 3 days using a fully automated, serverless pipeline. Live data is fetched every hour, models retrain daily, and results are served through an interactive Streamlit dashboard.

---

## 📸 Dashboard Preview

> Real-time AQI display · 3-day forecast · Historical trends · SHAP feature importance · Model performance comparison

---

## 🏗️ Architecture

```
External APIs          Feature Pipeline         Feature Store
─────────────         ─────────────────        ─────────────
AQICN (AQI)    ──►   feature_pipeline.py  ──►  feature_store/
OpenWeather    ──►   (runs every hour)         aqi_features.csv
                      via GitHub Actions
                             │
                             ▼
                     Training Pipeline
                     ─────────────────
                     training_pipeline.py
                     (runs every 24h)
                      via GitHub Actions
                             │
                             ▼
                        Model Registry
                        ──────────────
                        models/
                        ├── best_model.pkl
                        ├── keras_model.keras
                        ├── model_metadata.json
                        ├── keras_metrics.json
                        └── shap_importance.csv
                             │
                             ▼
                         Dashboard
                         ─────────
                         dashboard.py (Streamlit)
                         · Current AQI & weather
                         · 3-day hourly forecast
                         · Historical trends
                         · SHAP explanations
```

---

## Features

- **Live AQI & weather data** fetched every hour automatically
- **3-day hourly AQI forecast** using trained ML models
- **Multiple models compared** — Random Forest, Gradient Boosting, Ridge Regression, Keras Neural Network
- **SHAP feature importance** — understand what drives AQI predictions
- **Hazard alerts** — automatic warnings when AQI exceeds 150
- **Historical trend analysis** — by day, hour, and month
- **Fully automated CI/CD** via GitHub Actions — no manual intervention needed
- **Historical backfill** — 90 days of weather data from Open-Meteo used for training

---

## 🤖 ML Models & Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Gradient Boosting** ⭐ | 9.45 | 7.15 | 0.8994 |
| Random Forest | 9.89 | 7.42 | 0.8898 |
| Keras Neural Network | 23.97 | 20.81 | 0.3536 |
| Ridge Regression | 39.47 | 31.96 | -0.7531 |

**Best model: Gradient Boosting** with R² = 0.8994

### Features Used
| Feature | Description |
|---|---|
| `temp`, `feels_like` | Temperature from OpenWeatherMap |
| `humidity`, `pressure` | Atmospheric conditions |
| `wind_speed`, `wind_direction` | Wind data |
| `precipitation` | Rainfall |
| `weather_code` | WMO weather condition code |
| `hour`, `day`, `month`, `dayofweek` | Time-based features |
| `is_weekend`, `is_rush_hour` | Engineered traffic proxies |
| `aqi_change_rate` | AQI delta from previous hour |

---

## ⚙️ Automated Pipelines (GitHub Actions)

| Pipeline | Schedule | What it does |
|---|---|---|
| `feature_pipeline.yml` | Every hour | Fetches AQI + weather, appends to CSV |
| `training_pipeline.yml` | Daily at 3 AM UTC | Retrains all models, saves best |

Model metrics are visible directly on the GitHub Actions **Summary** tab after each training run.

---

## 🚀 Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/FaiqaRashid99/aqi-predictor.git
cd aqi-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the root directory:
```env
AQICN_TOKEN=your_aqicn_token
OPENWEATHER_KEY=your_openweather_key
CITY=Islamabad
FEATURE_STORE_PATH=feature_store
```

Get your free API keys:
- AQICN: https://aqicn.org/data-platform/token/
- OpenWeather: https://openweathermap.org/api

### 5. Test your API keys
```bash
python test_apis.py
```

### 6. Run the historical backfill (generates training data)
```bash
python backfill.py
```

### 7. Train the models
```bash
python training_pipeline.py
```

### 8. Launch the dashboard
```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
aqi-predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml   # Hourly data collection
│       └── training_pipeline.yml  # Daily model retraining
│
├── feature_store/
│   └── aqi_features.csv           # Auto-updated every hour
│
├── models/
│   ├── best_model.pkl             # Best sklearn model
│   ├── keras_model.keras          # Neural network
│   ├── model_metadata.json        # Metrics + model info
│   ├── keras_metrics.json         # Keras-specific metrics
│   └── shap_importance.csv        # Feature importance scores
│
├── backfill.py                    # Generate 90-day historical data
├── feature_pipeline.py            # Hourly live data collection
├── training_pipeline.py           # Model training + evaluation
├── dashboard.py                   # Streamlit web dashboard
├── test_apis.py                   # API connectivity check
├── requirements.txt               # Full dependencies
├── requirements_ci.txt            # Lightweight CI dependencies
└── .env                           # API keys (not committed)
```

---

## 🔑 GitHub Actions Secrets Required

Go to `Settings → Secrets and variables → Actions` and add:

| Secret | Description |
|---|---|
| `AQICN_TOKEN` | Your AQICN API token |
| `OPENWEATHER_KEY` | Your OpenWeatherMap API key |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| ML Models | Scikit-learn, Keras / TensorFlow |
| Explainability | SHAP |
| Dashboard | Streamlit, Plotly |
| Data Sources | AQICN API, OpenWeatherMap API, Open-Meteo |
| CI/CD | GitHub Actions |
| Version Control | Git + GitHub |

---

## 📊 AQI Scale Reference

| AQI Range | Category | Health Implication |
|---|---|---|
| 0–50 | 🟢 Good | Air quality is satisfactory |
| 51–100 | 🟡 Moderate | Acceptable for most people |
| 101–150 | 🟠 Unhealthy for Sensitive Groups | At-risk groups should limit outdoor activity |
| 151–200 | 🔴 Unhealthy | Everyone may experience health effects |
| 201–300 | 🟣 Very Unhealthy | Health alert — avoid outdoor activity |
| 300+ | 🔴 Hazardous | Emergency conditions |