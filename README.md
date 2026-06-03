# 🌬️ Pearls AQI Predictor — Islamabad

> End-to-end Machine Learning pipeline for Air Quality Index (AQI) forecasting with automated data collection, feature engineering, model training, and real-time predictions through a live web dashboard.

## 🌐 Live Dashboard
**[https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app](https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app)**

---

## 📋 Project Overview

This project predicts the Air Quality Index (AQI) for Islamabad, Pakistan for the next 3 days using a fully serverless, automated ML pipeline. It fetches real-time data every hour, retrains models daily, and serves predictions through an interactive web dashboard.

---

## 🏗️ System Architecture

```
AQICN API ──────────────────────┐
                                 ├──► Feature Pipeline ──► Supabase Feature Store
OpenWeather API ─────────────────┘         (hourly)         (PostgreSQL cloud DB)
                                                                      │
Open-Meteo Archive ──► Historical Backfill ──────────────────────────┘
    (real weather)          (90 days)                                 │
                                                                      ▼
                                                     Training Pipeline (daily)
                                                     + Preprocessing Pipeline
                                                                      │
                                                     ┌────────────────┴──────────────┐
                                                     │        Model Registry         │
                                                     │  best_model.pkl (XGBoost)     │
                                                     │  keras_model.keras            │
                                                     └────────────────┬──────────────┘
                                                                      │
                                                          Streamlit Dashboard
                                                         (3-day AQI Forecast)
```

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | Scikit-learn, XGBoost, Keras |
| Feature Store | Supabase (PostgreSQL cloud database) |
| CI/CD | GitHub Actions (hourly + daily) |
| Dashboard | Streamlit + Plotly |
| APIs | AQICN, OpenWeatherMap, Open-Meteo, OpenAQ |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Version Control | Git + GitHub |
| Preprocessing | KNN Imputation, RobustScaler, Cyclical Encoding |

---

## 🔑 Key Features

### 1. Feature Pipeline (Hourly — Automated)
- Fetches live AQI and pollutant data from AQICN API
- Fetches weather data from OpenWeatherMap API
- Engineers time-based features (hour, day, month, rush hour, weekend)
- Computes derived features (AQI change rate)
- Stores to **Supabase cloud PostgreSQL database** + local CSV backup
- Runs automatically **every hour** via GitHub Actions

### 2. Historical Backfill
- Fetches 90 days of **real hourly weather** from Open-Meteo archive
- Estimates historical AQI using seasonal and meteorological patterns
- Generated 2,200+ training rows for model training
- Investigated OpenAQ API for real PM2.5 data (5,751 readings available from Islamabad sensors)

### 3. Advanced Preprocessing Pipeline (`preprocessing.py`)
- **Outlier removal** using IQR method on AQI, PM2.5, PM10
- **KNN Imputation** for missing pollutant values (better than median)
- **Cyclical encoding** of hour and month using sin/cos transforms
- **Strict leakage control** — only uses 72h+ lag features for 3-day forecasting
- **Meteorological interaction features** (temp×humidity, wind×humidity)
- **Stagnation detection** (low wind + high humidity = trapped pollution)
- **Season encoding** (winter smog patterns in Islamabad)

### 4. Training Pipeline (Daily — Automated)
- Loads all features from **Supabase** feature store
- Applies strict preprocessing pipeline
- Trains **5 models**: XGBoost, Gradient Boosting, Random Forest, Ridge Regression, Keras Neural Network
- Evaluates using RMSE, MAE, and R² metrics
- Computes **SHAP feature importance**
- Saves best model automatically and commits to GitHub
- Runs **daily at 3am UTC** via GitHub Actions

### 5. Web Dashboard
- Live current AQI with color-coded health category
- ⚠️ **Hazard alerts** when AQI exceeds 150
- **3-day hourly AQI forecast** using lag and rolling features
- Historical trend charts (daily, hourly, monthly)
- SHAP feature importance visualization
- Model performance comparison (all 5 models)
- **Refresh Data** button for manual cache clearing

### 6. CI/CD Automation
- Feature pipeline runs **every hour** automatically
- Training pipeline runs **every day** automatically
- New model committed to GitHub daily
- Dashboard updates automatically with fresh predictions
- GitHub Actions **job summary** shows model results after each run

---

## 📊 Model Performance

| Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| **XGBoost** | **11.05** | **7.47** | **0.8724** | 🏆 Best model |
| Gradient Boosting | 11.70 | 7.82 | 0.857 | ✅ Very good |
| Random Forest | 14.03 | 9.77 | 0.7943 | ✅ Good |
| Keras Neural Network | 37.81 | 30.64 | -0.50 | ❌ Needs more data |
| Ridge Regression | 82.90 | 37.35 | -6.19 | ❌ Too linear |

**Best Model: XGBoost** — predicts AQI within ±11 points on average (R²=0.87)

---

## 🔍 SHAP Feature Importance

Top factors driving AQI predictions in Islamabad:

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | 🌬️ wind_speed | 23.7 | Stronger winds disperse pollutants |
| 2 | 📅 month_cos | 7.2 | Seasonal smog cycles |
| 3 | 🚗 is_rush_hour | 5.5 | Traffic emissions spike AQI |
| 4 | 💧 humidity | 2.9 | High humidity traps particles |
| 5 | 🕐 hour_cos | 1.5 | Daily pollution cycle |
| 6 | 🌧️ precipitation | 2.3 | Rain cleans the air |
| 7 | 📊 aqi_rolling_72h | 1.7 | 3-day historical trend |

---

## 🗄️ Feature Store (Supabase PostgreSQL)

**2,200+ rows** stored in Supabase cloud database, growing hourly:

| Feature Category | Features |
|---|---|
| **Target** | `aqi` |
| **Pollutants** | `pm25`, `pm10`, `no2`, `o3`, `co`, `so2` |
| **Weather** | `temp`, `feels_like`, `humidity`, `pressure`, `wind_speed`, `wind_direction`, `precipitation` |
| **Time** | `hour`, `day`, `month`, `dayofweek` |
| **Binary** | `is_weekend`, `is_rush_hour`, `is_hot`, `is_cold`, `is_calm_wind`, `is_strong_wind`, `is_stagnant` |
| **Cyclical** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| **Lag** | `aqi_lag_72h`, `aqi_lag_96h` |
| **Rolling** | `aqi_rolling_72h`, `aqi_rolling_96h` |
| **Interaction** | `temp_humidity`, `wind_humidity`, `season` |
| **Derived** | `aqi_change_rate` |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/FaiqaRashid99/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create virtual environment
```bash# 🌬️ Pearls AQI Predictor — Islamabad

> End-to-end Machine Learning pipeline for Air Quality Index (AQI) forecasting with automated data collection, feature engineering, model training, and real-time predictions through a live web dashboard.

## 🌐 Live Dashboard
**[https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app](https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app)**

---

## 📋 Project Overview

This project predicts the Air Quality Index (AQI) for Islamabad, Pakistan for the next 3 days using a fully serverless, automated ML pipeline. It fetches real-time data every hour, retrains models daily, and serves predictions through an interactive web dashboard.

Key goals:
- Collect live AQI and weather data from public APIs (no paid tiers required)
- Build a reproducible feature store and training pipeline with CI/CD automation
- Produce explainable, real-time 3-day forecasts with SHAP-driven insights
- Surface everything in a polished Streamlit dashboard with hazard alerts

---

## 🏗️ System Architecture

```
AQICN API ──────────────────────┐
                                 ├──► Feature Pipeline ──► Supabase Feature Store
OpenWeather API ─────────────────┘         (hourly)         (PostgreSQL cloud DB)
                                                                      │
Open-Meteo Archive ──► Historical Backfill ──────────────────────────┘
OpenAQ API ──────────┘   (real PM2.5 data)                           │
                                                                      ▼
                                                     Training Pipeline (daily)
                                                     + Preprocessing Pipeline
                                                                      │
                                                     ┌────────────────┴──────────────┐
                                                     │        Model Registry         │
                                                     │  best_model.pkl (XGBoost)     │
                                                     │  keras_model.keras            │
                                                     │  shap_importance.csv          │
                                                     └────────────────┬──────────────┘
                                                                      │
                                                          Streamlit Dashboard
                                                         (3-day AQI Forecast)
```

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 / 3.13 |
| ML Models | Scikit-learn, XGBoost, Keras (TensorFlow) |
| Feature Store | Supabase (PostgreSQL cloud database) |
| CI/CD | GitHub Actions (hourly + daily) |
| Dashboard | Streamlit + Plotly |
| APIs | AQICN, OpenWeatherMap, Open-Meteo, OpenAQ |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Version Control | Git + GitHub |
| Preprocessing | KNN Imputation, RobustScaler, Cyclical Encoding |

---

## 🔑 Key Features

### 1. Feature Pipeline (Hourly — Automated)
- Fetches live AQI and pollutant data (PM2.5, PM10, NO₂, O₃, CO, SO₂) from AQICN
- Fetches weather data (temperature, humidity, pressure, wind, precipitation) from OpenWeatherMap
- Engineers time-based features: hour, day, month, rush-hour flag, weekend flag
- Computes derived features: AQI change rate, cyclical sin/cos encodings
- Stores to **Supabase cloud PostgreSQL** with local CSV fallback
- Runs automatically **every hour** via GitHub Actions

### 2. Historical Backfill
Three backfill strategies were developed and iterated on:

| Script | Data Source | AQI Method |
|---|---|---|
| `backfill.py` | Open-Meteo weather | Estimated from single AQICN snapshot |
| `backfill_up3.py` | Open-Meteo weather | Estimated using monthly baselines |
| `backfill_openaq.py` / `backfill_up.py` | OpenAQ PM2.5 + Open-Meteo | Real PM2.5 → EPA AQI formula |

The final approach uses **real measured PM2.5** from confirmed active OpenAQ sensors near Islamabad/Rawalpindi (sensor IDs: 4554236, 4565932, 4566426, 4567363, 4567364), averaged per hour and converted to AQI using the US EPA piecewise linear formula. This covers data from May 2025 onwards.

### 3. Advanced Preprocessing Pipeline (`preprocessing.py`)
- **Outlier removal** using IQR method (3× for AQI, 5× for pollutants)
- **KNN Imputation** (k=5 neighbors) for missing pollutant values
- **Cyclical encoding** of hour and month using sin/cos transforms
- **Strict leakage control** — only lag features ≥72 hours ahead (for 3-day forecasting)
- **Meteorological interaction features**: temp×humidity, wind×humidity
- **Stagnation detection**: low wind speed + high humidity = trapped pollution flag
- **Season encoding**: winter smog (0), spring (1), summer (2), autumn (3)

### 4. Training Pipeline (Daily — Automated)
- Loads full feature history from Supabase (paginated, up to unlimited rows)
- Applies strict preprocessing with temporal train/test split (80/20)
- Trains **5 models**: XGBoost, Gradient Boosting, Random Forest, Huber Regression, Keras Neural Network
- Evaluates using RMSE, MAE, and R² on held-out test data
- Computes **SHAP feature importance** using TreeExplainer
- Saves best sklearn model automatically and commits to GitHub
- Runs **daily at 3am UTC** via GitHub Actions

### 5. Web Dashboard (`dashboard.py`)
- Live current AQI with animated color-coded health category (green → hazardous)
- ⚠️ **Hazard alert banner** when AQI exceeds 150
- **3-day hourly AQI forecast** chart (72 hours ahead) with AQI band overlays
- **Daily forecast cards** showing avg/min/max per day with health label
- Historical trend charts: daily average, by hour of day, by month
- **SHAP feature importance bar chart** with plain-language explanations
- **Model comparison** table and RMSE/R² charts for all 5 models
- Light/dark theme toggle, sidebar with live AQI donut gauge and model metadata
- Arc gauges for temperature, humidity, pressure; compass gauge for wind

### 6. CI/CD Automation (GitHub Actions)
- Feature pipeline: **every hour** — fetches live data → stores to Supabase → commits CSV backup
- Training pipeline: **every day at 3am UTC** — retrains all models → commits best model + metadata
- Job summary reports model metrics in the GitHub Actions UI after each training run
- All API keys stored as GitHub Secrets

---

## 📊 Model Performance

| Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| **XGBoost** | **11.05** | **7.47** | **0.8724** | 🏆 Best model |
| Gradient Boosting | 11.70 | 7.82 | 0.857 | ✅ Very good |
| Random Forest | 14.03 | 9.77 | 0.7943 | ✅ Good |
| Keras Neural Network | 37.81 | 30.64 | -0.50 | ❌ Needs more data |
| Huber Regression | 82.90 | 37.35 | -6.19 | ❌ Too linear |

**Best Model: XGBoost** — predicts AQI within ±11 points on average (R² = 0.87)

> The neural network and linear models underperform due to limited training data (~1,400 effective rows post-preprocessing). Performance is expected to improve as the feature pipeline accumulates more real hourly readings.

---

## 🔍 SHAP Feature Importance

Top factors driving AQI predictions in Islamabad:

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | 🌬️ wind_speed | 23.7 | Stronger winds disperse pollutants |
| 2 | 📅 month_cos | 7.2 | Seasonal smog cycles (winter peak) |
| 3 | 🚗 is_rush_hour | 5.5 | Traffic emissions spike AQI |
| 4 | 💧 humidity | 2.9 | High humidity traps particles |
| 5 | 🕐 hour_cos | 1.5 | Daily pollution cycle |
| 6 | 🌧️ precipitation | 2.3 | Rain cleans the air |
| 7 | 📊 aqi_rolling_72h | 1.7 | 3-day historical trend |

---

## 🗄️ Feature Store Schema (Supabase)

**2,200+ rows** stored in Supabase, growing hourly:

| Category | Features |
|---|---|
| **Target** | `aqi` |
| **Pollutants** | `pm25`, `pm10`, `no2`, `o3`, `co`, `so2` |
| **Weather** | `temp`, `feels_like`, `humidity`, `pressure`, `wind_speed`, `wind_direction`, `precipitation`, `visibility`, `weather_code` |
| **Time** | `hour`, `day`, `month`, `dayofweek` |
| **Binary flags** | `is_weekend`, `is_rush_hour`, `is_hot`, `is_cold`, `is_calm_wind`, `is_strong_wind`, `is_stagnant` |
| **Cyclical** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| **Lag** | `aqi_lag_72h`, `aqi_lag_96h` |
| **Rolling** | `aqi_rolling_72h`, `aqi_rolling_96h` |
| **Interaction** | `temp_humidity`, `wind_humidity`, `season` |
| **Derived** | `aqi_change_rate`, `city`, `timestamp` |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/FaiqaRashid99/pearls-aqi-predictor.git
cd pearls-aqi-predictor
```

### 2. Create virtual environment
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
Create a `.env` file in the project root:
```
AQICN_TOKEN=your_aqicn_token
OPENWEATHER_KEY=your_openweather_key
OPENAQ_KEY=your_openaq_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
CITY=Islamabad
FEATURE_STORE_PATH=feature_store
```

### 5. Run the pipelines in order

```bash
# Step 1: Backfill historical data (run once)
python backfill_up.py

# Step 2: Collect current data point
python feature_pipeline.py

# Step 3: Train all models
python training_pipeline.py

# Step 4: Launch dashboard
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
pearls-aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     ← Runs every hour
│       └── training_pipeline.yml   ← Runs every day at 3am UTC
├── feature_store/
│   └── aqi_features.csv            ← Local CSV backup (auto-updated)
├── models/
│   ├── best_model.pkl              ← Best trained model (XGBoost)
│   ├── keras_model.keras           ← Trained Neural Network
│   ├── keras_imputer.pkl           ← Imputer for Keras pipeline
│   ├── keras_scaler.pkl            ← Scaler for Keras pipeline
│   ├── model_metadata.json         ← Metrics, feature list, training date
│   └── shap_importance.csv         ← SHAP feature importances
├── feature_pipeline.py             ← Hourly live data collection
├── backfill.py                     ← Backfill v1 (estimated AQI)
├── backfill_up3.py                 ← Backfill v2 (monthly baselines)
├── backfill_openaq.py              ← Backfill v3 (real OpenAQ PM2.5)
├── backfill_up.py                  ← Backfill v4 (active sensors, final)
├── preprocessing.py                ← Feature engineering + leakage-safe cleaning
├── training_pipeline.py            ← Trains 5 models, saves best + SHAP
├── dashboard.py                    ← Streamlit web application
├── app.py                          ← Simpler AQI monitoring app
├── find_sensors.py                 ← OpenAQ sensor discovery utility
├── test_apis.py                    ← API connectivity checks
├── test_stations.py                ← Multi-station AQI averaging test
├── restore_supabase.py             ← Restore Supabase from CSV backup
├── requirements.txt                ← Full pinned dependencies
├── requirements_ci.txt             ← Lightweight CI/CD dependencies
└── .env                            ← API keys (not committed)
```

---

## 🌍 APIs Used

| API | Purpose | Free Tier |
|---|---|---|
| [AQICN](https://aqicn.org/api/) | Real-time AQI & pollutants (hourly) | ✅ Yes |
| [OpenWeatherMap](https://openweathermap.org/api) | Live weather data (hourly) | ✅ Yes (1000 calls/day) |
| [Open-Meteo](https://open-meteo.com/) | Historical weather archive + forecast | ✅ Yes (unlimited) |
| [OpenAQ](https://openaq.org/) | Real historical PM2.5 measurements | ✅ Yes |
| [Supabase](https://supabase.com/) | Cloud feature store (PostgreSQL) | ✅ Yes (500MB) |

---

## 📝 Data Transparency

| Data | Source | Status |
|---|---|---|
| Live AQI (May 15, 2026 onwards) | AQICN API (US Embassy station) | ✅ 100% Real |
| Live weather (May 15, 2026 onwards) | OpenWeatherMap | ✅ 100% Real |
| Historical PM2.5 (May 2025 – May 2026) | OpenAQ sensors (5 Rawalpindi/Islamabad sites) | ✅ 100% Real |
| Historical weather (May 2025 – May 2026) | Open-Meteo archive | ✅ 100% Real |
| Historical AQI conversion | US EPA PM2.5→AQI formula | ✅ Standard method |

> Real-time AQICN data collection has been operational since May 15, 2026 and grows every hour. Historical training data uses real PM2.5 measurements from 5 confirmed active OpenAQ sensors near Islamabad, averaged per hour and converted to AQI using the official US EPA piecewise linear formula.

---

## 🤖 GitHub Actions CI/CD

```yaml
Feature Pipeline:  runs every hour  → fetches live data → stores to Supabase → commits CSV
Training Pipeline: runs every day   → trains 5 models   → commits best model + SHAP report
```

Both pipelines are configured with GitHub Secrets for secure API key access. The training run posts a Markdown summary table of all model results to the GitHub Actions job summary.

---

## 👩‍💻 Author

**Faiqa Rashid**
- GitHub: [@FaiqaRashid99](https://github.com/FaiqaRashid99)
- Project Due: June 8, 2026
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file:
```
AQICN_TOKEN=your_aqicn_token
OPENWEATHER_KEY=your_openweather_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
CITY=Islamabad
FEATURE_STORE_PATH=feature_store
```

### 5. Run pipelines
```bash
# Collect current data point
python feature_pipeline.py

# Train all models
python training_pipeline.py

# Launch dashboard
streamlit run dashboard.py
```

---

## 📁 Project Structure

```
pearls-aqi-predictor/
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     ← Runs every hour
│       └── training_pipeline.yml   ← Runs every day
├── feature_store/
│   └── aqi_features.csv            ← Local CSV backup
├── models/
│   ├── best_model.pkl              ← Trained XGBoost model
│   ├── keras_model.keras           ← Trained Neural Network
│   ├── keras_imputer.pkl           ← Keras preprocessor
│   ├── keras_scaler.pkl            ← Keras scaler
│   ├── model_metadata.json         ← Model metrics + feature list
│   └── shap_importance.csv         ← SHAP feature values
├── feature_pipeline.py             ← Hourly live data collection
├── backfill.py                     ← Historical data (Open-Meteo)
├── backfill_openaq.py              ← Real PM2.5 backfill (OpenAQ)
├── preprocessing.py                ← Feature engineering + cleaning
├── training_pipeline.py            ← Model training (5 models)
├── dashboard.py                    ← Streamlit web application
├── find_sensors.py                 ← OpenAQ sensor discovery
├── requirements.txt                ← Dependencies
├── requirements_ci.txt             ← CI/CD dependencies
└── .env                            ← API keys (not committed)
```

---

## 🌍 APIs Used

| API | Purpose | Cost |
|---|---|---|
| [AQICN](https://aqicn.org/api/) | Real-time AQI & pollutants (hourly) | Free |
| [OpenWeatherMap](https://openweathermap.org/api) | Live weather data (hourly) | Free tier |
| [Open-Meteo](https://open-meteo.com/) | Historical weather archive | Free |
| [OpenAQ](https://openaq.org/) | Real historical PM2.5 measurements | Free |
| [Supabase](https://supabase.com/) | Cloud feature store (PostgreSQL) | Free tier |

---

## 📝 Data Transparency

| Data | Source | Real? |
|---|---|---|
| Live AQI (May 15, 2026 onwards) | AQICN API | ✅ 100% Real |
| Live weather (May 15, 2026 onwards) | OpenWeatherMap | ✅ 100% Real |
| Historical weather (Feb–May 2026) | Open-Meteo archive | ✅ 100% Real |
| Historical AQI (backfill) | Estimated from real weather patterns | ⚠️ Estimated |

> Historical AQI values were estimated using real meteorological data and seasonal adjustment factors based on published research on South Asian air quality. Real-time AQICN data collection has been operational since May 15, 2026 and grows hourly.

---

## 🤖 GitHub Actions CI/CD

```yaml
Feature Pipeline:  runs every hour  → fetches live data → stores to Supabase
Training Pipeline: runs every day   → trains 5 models   → commits best model
```

Both pipelines add Supabase credentials via GitHub Secrets for secure access.

---

## 👩‍💻 Author

**Faiqa Rashid**
- GitHub: [@FaiqaRashid99](https://github.com/FaiqaRashid99)

---
