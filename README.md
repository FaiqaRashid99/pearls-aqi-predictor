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

**Faiq Rashid**
- GitHub: [@FaiqaRashid99](https://github.com/FaiqaRashid99)

---

## 📅 Project Timeline

- **Started:** May 14, 2026
- **Due:** June 8, 2026
- **Status:** ✅ Complete and deployed

---
