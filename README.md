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
OpenWeather API ─────────────────┘         (hourly)              (cloud DB)
                                                                      │
Open-Meteo Archive ──► Historical Backfill ──────────────────────────┘
                                                                      │
                                                                      ▼
                                                          Training Pipeline (daily)
                                                                      │
                                                          ┌───────────┴───────────┐
                                                          │    Model Registry     │
                                                          │  (GitHub + models/)   │
                                                          └───────────┬───────────┘
                                                                      │
                                                          Streamlit Dashboard
                                                         (3-day AQI Forecast)
```

---

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| ML Models | Scikit-learn, Keras |
| Feature Store | Supabase (PostgreSQL) |
| CI/CD | GitHub Actions |
| Dashboard | Streamlit + Plotly |
| APIs | AQICN, OpenWeatherMap, Open-Meteo |
| Explainability | SHAP |
| Version Control | Git + GitHub |

---

## 🔑 Key Features

### 1. Feature Pipeline (Hourly)
- Fetches live AQI and pollutant data from AQICN API
- Fetches weather data from OpenWeatherMap API
- Engineers time-based features (hour, day, month, rush hour, weekend)
- Computes derived features (AQI change rate)
- Stores to Supabase cloud database + local CSV backup
- Runs automatically every hour via GitHub Actions

### 2. Historical Backfill
- Fetches 90 days of real hourly weather from Open-Meteo archive
- Estimates historical AQI using seasonal and meteorological patterns
- Generated 2,197 training rows for model training

### 3. Training Pipeline (Daily)
- Loads all features from Supabase feature store
- Trains 4 models: Random Forest, Gradient Boosting, Ridge Regression, Keras Neural Network
- Evaluates using RMSE, MAE, and R² metrics
- Computes SHAP feature importance
- Saves best model automatically
- Runs daily at 3am UTC via GitHub Actions

### 4. Web Dashboard
- Live current AQI with color-coded health category
- ⚠️ Hazard alerts when AQI exceeds 150
- 3-day hourly AQI forecast
- Historical trend charts (daily, hourly, monthly)
- SHAP feature importance visualization
- Model performance comparison

### 5. CI/CD Automation
- Feature pipeline: runs every hour automatically
- Training pipeline: runs every day automatically
- New model committed to GitHub daily
- Dashboard updates automatically with fresh predictions

---

## 📊 Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Gradient Boosting** | **12.41** | **7.89** | **0.8363** |
| Random Forest | 14.26 | 9.48 | 0.7836 |
| Ridge Regression | 66.97 | 44.32 | -3.77 |
| Keras Neural Network | 32.39 | 25.55 | -0.12 |

**Best Model: Gradient Boosting** — predicts AQI within ±12 points on average.

---

## 🔍 SHAP Feature Importance

Top factors driving AQI predictions in Islamabad:

1. 🌬️ **Wind Speed** — stronger winds disperse pollutants
2. 📅 **Month** — seasonal smog patterns (worse Nov–Feb)
3. 🚗 **Rush Hour** — traffic emissions spike AQI
4. 💧 **Humidity** — high humidity traps particles
5. 🕐 **Hour of Day** — daily pollution cycle

---

## 🗄️ Feature Store (Supabase)

Features stored in Supabase PostgreSQL cloud database:

| Feature | Description |
|---|---|
| `aqi` | Air Quality Index (target) |
| `pm25`, `pm10` | Particulate matter |
| `no2`, `o3`, `co`, `so2` | Pollutant gases |
| `temp`, `feels_like` | Temperature (°C) |
| `humidity`, `pressure` | Atmospheric conditions |
| `wind_speed`, `wind_direction` | Wind data |
| `precipitation` | Rainfall (mm) |
| `hour`, `day`, `month`, `dayofweek` | Time features |
| `is_weekend`, `is_rush_hour` | Derived binary features |
| `aqi_change_rate` | AQI change from previous hour |

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
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
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
# Collect current data
python feature_pipeline.py

# Train models
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
│       ├── feature_pipeline.yml    ← Runs hourly
│       └── training_pipeline.yml  ← Runs daily
├── feature_store/
│   └── aqi_features.csv           ← Local CSV backup
├── models/
│   ├── best_model.pkl             ← Trained Gradient Boosting
│   ├── keras_model.keras          ← Trained Neural Network
│   ├── model_metadata.json        ← Model metrics
│   └── shap_importance.csv        ← SHAP values
├── feature_pipeline.py            ← Hourly data collection
├── backfill.py                    ← Historical data generation
├── training_pipeline.py           ← Model training
├── dashboard.py                   ← Streamlit web app
├── requirements.txt               ← Full dependencies
├── requirements_ci.txt            ← CI/CD dependencies
└── .env                           ← API keys (not committed)
```

---

## 🌍 APIs Used

| API | Purpose | Cost |
|---|---|---|
| [AQICN](https://aqicn.org/api/) | Real-time AQI & pollutants | Free |
| [OpenWeatherMap](https://openweathermap.org/api) | Live weather data | Free tier |
| [Open-Meteo](https://open-meteo.com/) | Historical weather archive | Free |
| [Supabase](https://supabase.com/) | Cloud feature store (PostgreSQL) | Free tier |

---

## 👩‍💻 Author

**Faiq Rashid**
- GitHub: [@FaiqaRashid99](https://github.com/FaiqaRashid99)

---

## 📚 References

- [OpenWeatherMap](https://openweathermap.org/)
- [Open-Meteo](https://open-meteo.com/)
- [Supabase](https://supabase.com/)
- [Streamlit](https://streamlit.io/)
- [SHAP](https://shap.readthedocs.io/en/latest/)
- [Air Quality Index](https://en.wikipedia.org/wiki/Air_quality_index)

---
