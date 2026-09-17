import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

def _to_naive_ns(series: pd.Series) -> np.ndarray:
    """Convert a (possibly tz-aware) datetime Series to naive datetime64[ns].
    numpy can't do timedelta arithmetic on tz-aware Timestamps -- .to_numpy()
    silently returns dtype('O') object arrays in that case, which is what
    broke the subtraction below. Safe to drop tz here since everything is UTC."""
    if series.dt.tz is not None:
        series = series.dt.tz_localize(None)
    return series.to_numpy(dtype="datetime64[ns]")

def _real_only_lag_rolling(df: pd.DataFrame, roll_half_window_hours: int = 12) -> pd.DataFrame:
    """
    Compute aqi_lag_* and aqi_rolling_* using ONLY genuinely measured (is_real=1)
    readings as the source signal — never the interpolated/resampled series.

    Why this exists: with real-data coverage well under 100%, the previous
    .shift()/.rolling() calls on the resampled `aqi` column were drawing most
    of their values from linear interpolation, not observation. SHAP has
    consistently ranked aqi_rolling_72h as the single most important feature
    across training runs, which meant the model was largely learning to
    reproduce a smoothed, partly-fabricated trend rather than real dynamics.

    aqi_lag_{h}h  = most recent REAL reading at/before (t - h), if one exists
                    within a small tolerance window (else NaN -> row dropped
                    downstream, same as the old lag-based NaN drop).
    aqi_rolling_{h}h = mean of REAL readings within +/- roll_half_window_hours
                       of (t - h) -- i.e. "what did genuinely measured AQI
                       look like around h hours ago", not a smoothed guess.

    Implemented with searchsorted + prefix sums for O(n log m) instead of a
    naive O(n * m) scan, since m (real readings) can be a few thousand rows.
    """
    real = (
        df.loc[df["is_real"] == 1, ["timestamp", "aqi"]]
        .dropna(subset=["aqi"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if real.empty:
        for hours in (72, 96):
            df[f"aqi_lag_{hours}h"] = np.nan
            df[f"aqi_rolling_{hours}h"] = np.nan
        return df

    # real_ts = real["timestamp"].to_numpy()
    real_ts = _to_naive_ns(real["timestamp"])
    real_val = real["aqi"].to_numpy(dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(real_val)))

    def mean_in_window(center_times, half_window):
        lo = np.searchsorted(real_ts, center_times - half_window, side="left")
        hi = np.searchsorted(real_ts, center_times + half_window, side="right")
        counts = hi - lo
        sums = prefix[hi] - prefix[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        return means

    def nearest_before_or_at(target_times, tolerance):
        idx = np.searchsorted(real_ts, target_times, side="right") - 1
        idx_clipped = np.clip(idx, 0, len(real_ts) - 1)
        vals = real_val[idx_clipped]
        ok = (idx >= 0) & (np.abs(real_ts[idx_clipped] - target_times) <= tolerance)
        return np.where(ok, vals, np.nan)

    # ts = df["timestamp"].to_numpy()
    ts = _to_naive_ns(df["timestamp"])
    half_window = np.timedelta64(roll_half_window_hours, "h")
    lag_tolerance = np.timedelta64(6, "h")

    for hours in (72, 96):
        target = ts - np.timedelta64(hours, "h")
        df[f"aqi_lag_{hours}h"] = nearest_before_or_at(target, lag_tolerance)
        df[f"aqi_rolling_{hours}h"] = mean_in_window(target, half_window)

    return df


def preprocess_features(df: pd.DataFrame, scale_features: bool = False) -> pd.DataFrame:
    print("🚀 Starting STRICT leakage-controlled preprocessing...")
    original_rows = len(df)
    df = df.copy()
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    print(f"   Initial rows: {original_rows} | After dedup: {len(df)}")

    # ── snap onto a true hourly grid before any lag/rolling math ──
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Mark which hourly slots are backed by an actual collected row,
    # BEFORE resample manufactures empty slots for missing hours.
    df["is_real"] = 1
    df = df.resample("1h").mean(numeric_only=True)
    fill_cols = [c for c in numeric_cols if c != "is_real"]
    df[fill_cols] = df[fill_cols].interpolate(limit=6, limit_direction="forward")
    df["is_real"] = df["is_real"].fillna(0)  # NaN here = no real row that hour
    df.reset_index(inplace=True)
    print(f"   After hourly resample: {len(df)} rows")

    # Outlier Removal
    for col in ["aqi", "pm25", "pm10"]:
        if col in df.columns and df[col].notna().sum() > 20:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = 5 if col != "aqi" else 3
            before = len(df)
            df = df[df[col].between(Q1 - multiplier * IQR, Q3 + multiplier * IQR)]
            if before - len(df) > 0:
                print(f"   Removed {before - len(df)} outliers from {col}")

    # Imputation
    pollutant_cols = ["pm25", "pm10", "no2", "o3", "co", "so2"]
    weather_cols = ["temp", "feels_like", "humidity", "pressure", "wind_speed", "wind_direction"]
    valid_pollutants = [c for c in pollutant_cols if c in df.columns and df[c].notna().sum() > 10]
    if valid_pollutants:
        try:
            imputer = KNNImputer(n_neighbors=5)
            df[valid_pollutants] = imputer.fit_transform(df[valid_pollutants])
            print(f"   KNN imputation on {len(valid_pollutants)} pollutants")
        except Exception:
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
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    # === STRICT LEAKAGE CONTROL ===
    # Lag / rolling — now built from REAL readings only (see _real_only_lag_rolling)
    df = _real_only_lag_rolling(df)

    # Meteorological Interactions
    df["temp_humidity"] = df["temp"] * df["humidity"]
    df["wind_humidity"] = df["wind_speed"] * df["humidity"]
    df["is_stagnant"] = ((df["wind_speed"] < 2) & (df["humidity"] > 70)).astype(int)
    df["is_hot"] = (df["temp"] > 35).astype(int)
    df["is_cold"] = (df["temp"] < 10).astype(int)
    df["is_calm_wind"] = (df["wind_speed"] < 2).astype(int)
    df["is_strong_wind"] = (df["wind_speed"] > 8).astype(int)

    # Drop rows with no anchoring real data nearby
    lag_cols = ["aqi_lag_72h"]
    before_drop = len(df)
    df.dropna(subset=lag_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"   Dropped {before_drop - len(df)} rows lacking a real aqi_lag_72h anchor")

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
        "aqi_rolling_72h", "aqi_rolling_96h",
    ]