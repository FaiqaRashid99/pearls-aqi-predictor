import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

def preprocess_features(df: pd.DataFrame, scale_features: bool = False) -> pd.DataFrame:
    """STRICT Leakage Control - Optimized for True 3-Day Forecasting"""
    
    print("🚀 Starting STRICT leakage-controlled preprocessing...")
    original_rows = len(df)
    
    df = df.copy()
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print(f"   Initial rows: {original_rows} | After dedup: {len(df)}")
    
    # Outlier Removal
    for col in ["aqi", "pm25", "pm10"]:
        if col in df.columns and df[col].notna().sum() > 20:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = 5 if col != "aqi" else 3
            before = len(df)
            df = df[df[col].between(Q1 - multiplier*IQR, Q3 + multiplier*IQR)]
            if before - len(df) > 0:
                print(f"   Removed {before - len(df)} outliers from {col}")
    
    # Imputation
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    weather_cols = ["temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_direction"]
    
    valid_pollutants = [col for col in pollutant_cols if col in df.columns and df[col].notna().sum() > 10]
    if valid_pollutants:
        try:
            imputer = KNNImputer(n_neighbors=5)
            df[valid_pollutants] = imputer.fit_transform(df[valid_pollutants])
            print(f"   KNN imputation on {len(valid_pollutants)} pollutants")
        except:
            print("   KNN imputation skipped")
    
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Time & Cyclical Features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.weekday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    df["season"] = df["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    
    # === STRICT LEAKAGE CONTROL ===
    # Only long-term historical signals (2-3 days+)
    df["aqi_lag_72h"] = df["aqi"].shift(72)      # 3 days ago
    df["aqi_lag_96h"] = df["aqi"].shift(96)      # 4 days ago
    
    df["aqi_rolling_72h"] = df["aqi"].rolling(window=72, min_periods=24).mean()   # 3-day average
    df["aqi_rolling_96h"] = df["aqi"].rolling(window=96, min_periods=24).mean()
    
    # Remove change feature completely (biggest source of leakage)
    # df["aqi_change_48h"] = ...  ← Removed
    
    # Meteorological Interactions
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["wind_humidity"] = df["wind_speed"] * df["humidity"]
    df["is_stagnant"] = ((df["wind_speed"] < 2) & (df["humidity"] > 70)).astype(int)
    
    df["is_hot"] = (df["temp"] > 35).astype(int)
    df["is_cold"] = (df["temp"] < 10).astype(int)
    df["is_calm_wind"] = (df["wind_speed"] < 2).astype(int)
    df["is_strong_wind"] = (df["wind_speed"] > 8).astype(int)
    
    # Drop NaNs
    lag_cols = ["aqi_lag_72h"]
    df.dropna(subset=lag_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"   Final dataset: {len(df)} rows")
    print(f"   Total features: {len(df.columns)}")
    
    if scale_features:
        scaler = RobustScaler()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["aqi", "timestamp"]
        scale_cols = [c for c in numeric_cols if c not in exclude]
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        print("   Applied RobustScaler")
        return df, scaler
    
    return df


def get_feature_columns():
    """Safe features for 3-day forecasting"""
    return [
        "temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_direction",
        "precipitation", "weather_code", 
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "dayofweek", "is_weekend", "is_rush_hour", "season",
        "is_hot", "is_cold", "is_calm_wind", "is_strong_wind", "is_stagnant",
        "temp_humidity", "wind_humidity",
        "aqi_lag_72h", "aqi_lag_96h",
        "aqi_rolling_72h", "aqi_rolling_96h"
    ]