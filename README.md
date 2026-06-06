# 🌬️ Pearls AQI Predictor — Islamabad

> End-to-end Machine Learning pipeline for Air Quality Index (AQI) forecasting with automated data collection, feature engineering, model training, and real-time predictions through a live web dashboard.

## 🌐 Live Dashboard
**[https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app](https://pearls-aqi-predictor-gf5vcjmhgdibssgxha7erk.streamlit.app)**

---

## 📋 Project Overview

This project predicts the Air Quality Index (AQI) for Islamabad, Pakistan for the next 3 days using a fully serverless, automated ML pipeline. It fetches real-time data every hour, retrains models daily, and serves predictions through an interactive web dashboard.

---

## 🛰️ System Architecture

```
OpenWeatherMap Air Pollution API  ───┐
                                     ├──► Feature Pipeline ──► Supabase Feature Store
OpenWeatherMap Weather API ──────────┘         (hourly)         (PostgreSQL cloud DB)
                                                                        │
Open-Meteo Archive ──► Historical Backfill  ────────────────────────────┘
    (real weather)                                                      │
                                                                        ▼
                                                       Training Pipeline (daily)
                                                       + Preprocessing Pipeline
                                                                        │
                                                       ┌────────────────┴──────────────┐
                                                       │        Model Registry         │
                                                       │  best_model.pkl               │
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
| APIs | OpenWeatherMap (Air Pollution + Weather), Open-Meteo, OpenAQ |
| Explainability | SHAP (SHapley Additive exPlanations) |
| Version Control | Git + GitHub |
| Preprocessing | KNN Imputation, RobustScaler, Cyclical Encoding |

---

## ⚡ Key Features

### 1. Feature Pipeline (Hourly — Automated)
- Fetches live PM2.5 and pollutant data from **OpenWeatherMap Air Pollution API**
- Converts PM2.5 → AQI using the **US EPA standard formula**
- Fetches live weather data from OpenWeatherMap Current Weather API
- Engineers time-based features (hour, day, month, rush hour, weekend)
- Computes derived features (AQI change rate)
- Stores to **Supabase cloud PostgreSQL database** + local CSV backup
- Runs automatically **every hour** via GitHub Actions

### 2. Historical Backfill
- Fetches 1 year of **real hourly weather** from Open-Meteo archive
- Real PM2.5 data from **OpenAQ sensors** (Rawalpindi/Islamabad area)
- Converts real PM2.5 readings → AQI using US EPA formula
- 8,900+ training rows stored in Supabase

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
- Trains **5 models**: XGBoost, Gradient Boosting, Random Forest, Huber Regression, Keras Neural Network
- Evaluates using RMSE, MAE, and R² metrics
- Computes **SHAP feature importance**
- Saves best model automatically and commits to GitHub
- Runs **daily at 3am UTC** via GitHub Actions

### 5. Web Dashboard
- Live current AQI with color-coded health category (updates every 5 minutes)
- ⚠️ **Hazard alerts** when AQI exceeds 150
- **3-day hourly AQI forecast** using lag and rolling features
- Historical trend charts (daily, hourly, monthly)
- SHAP feature importance visualization
- Model performance comparison (all 5 models)
- Light/Dark theme toggle
- **Refresh Data** button for manual cache clearing

### 6. CI/CD Automation
- Feature pipeline runs **every hour** automatically
- Training pipeline runs **every day** automatically
- New model committed to GitHub daily
- Dashboard updates automatically with fresh predictions
- GitHub Actions **job summary** shows model results after each run

---

## 📊 Model Performance

> ⚠️ **Note:** Results below reflect the current trained model. Since the feature store grows every hour and the model retrains daily, these metrics will keep evolving as more real data accumulates. Check the live dashboard's Model Comparison section for the latest results.

| Model | RMSE | MAE | R² | Notes |
|---|---|---|---|---|
| **Gradient Boosting** | **11.2804** | **9.0554** | **0.8297** | 🏆 Best model |
| Random Forest | 11.605 | 9.372 | 0.8197 | ✅ Good |
| XGBoost | 11.7973 | 9.3638 | 0.8137 | ✅ Good |
| Keras Neural Network | 17.174 | 14.0294 | 0.6052 | ✅ Improving with more data |
| Huber Regression | 19.796 | 13.6271 | 0.4755 | ⚠️ Linear baseline |

**Best Model: Gradient Boosting** — predicts AQI within ±11 points on average (R²=0.83)

---

## 🔍 SHAP Feature Importance

Top factors driving AQI predictions in Islamabad:

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | 🌬️ wind_speed | Stronger winds disperse pollutants |
| 2 | 📅 month_cos | Seasonal smog cycles (winter worst) |
| 3 | 🚗 is_rush_hour | Traffic emissions spike AQI |
| 4 | 💧 humidity | High humidity traps particles |
| 5 | 🕐 hour_cos | Daily pollution cycle |
| 6 | 🌧️ precipitation | Rain cleans the air |
| 7 | 📊 aqi_rolling_72h | 3-day historical trend |

---

## 🗄️ Feature Store (Supabase PostgreSQL)

**8,900+ rows** stored in Supabase cloud database, growing hourly:

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
Create a `.env` file:
```
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
│   ├── best_model.pkl              ← Trained best model
│   ├── keras_model.keras           ← Trained Neural Network
│   ├── keras_imputer.pkl           ← Keras preprocessor
│   ├── keras_scaler.pkl            ← Keras scaler
│   ├── model_metadata.json         ← Model metrics + feature list
│   └── shap_importance.csv         ← SHAP feature values
├── feature_pipeline.py             ← Hourly live data collection
├── backfill_up.py                  ← Real PM2.5 backfill (OpenAQ)
├── backfill_up3.py                 ← Historical data (Open-Meteo)
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
| [OpenWeatherMap Air Pollution](https://openweathermap.org/api/air-pollution) | Real-time PM2.5 → AQI (primary source) | Free tier |
| [OpenWeatherMap Weather](https://openweathermap.org/api) | Live + forecast weather | Free tier |
| [Open-Meteo](https://open-meteo.com/) | Historical weather archive + forecast fallback | Free |
| [OpenAQ](https://openaq.org/) | Real historical PM2.5 measurements | Free |
| [Supabase](https://supabase.com/) | Cloud feature store (PostgreSQL) | Free tier |

---

## 📝 Data Transparency

| Data | Source | Real? |
|---|---|---|
| Live AQI (Jun 2026 onwards) | OpenWeatherMap Air Pollution API → EPA formula | ✅ 100% Real |
| Live weather (Jun 2026 onwards) | OpenWeatherMap Current Weather | ✅ 100% Real |
| Historical weather (May 2025–Jun 2026) | Open-Meteo archive | ✅ 100% Real |
| Historical AQI (May 2025–Jun 2026) | OpenAQ real PM2.5 sensors → EPA formula | ✅ Real measured data |

> **Previous AQICN dependency removed:** The AQICN US Embassy station was discovered to have stopped reporting in February 2026 while silently returning stale cached data (AQI=154). All AQI data collection now uses OpenWeatherMap Air Pollution API with PM2.5→AQI conversion via the US EPA formula, ensuring real-time accuracy.

---

## 🤖 GitHub Actions CI/CD

```yaml
Feature Pipeline:  runs every hour  → fetches live PM2.5 + weather → stores to Supabase
Training Pipeline: runs every day   → trains 5 models              → commits best model
```

Both pipelines use Supabase and OpenWeatherMap credentials via GitHub Secrets for secure access. No AQICN token required.

---

## 👩‍💻 Author

**Faiqa Rashid**
- GitHub: [@FaiqaRashid99](https://github.com/FaiqaRashid99)

---
