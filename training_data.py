"""
Builds ONE consistent training frame:
  - ow_history.csv   (OpenWeather history)  for everything before LIVE_START
  - Supabase rows    (your hourly pipeline) from LIVE_START onward
Synthetic rows and the pre-June-2026 rows of unknown source are dropped.

In training_pipeline.py, at the end of load_features() replace
    return df
with
    from training_data import clean_training_frame
    return clean_training_frame(df)
(do the same in the CSV-fallback branch).
"""
import os
import pandas as pd

LIVE_START = pd.Timestamp("2026-06-06")   # first row with full OpenWeather pollutant columns
HISTORY_CSV = os.path.join(os.getenv("FEATURE_STORE_PATH", "feature_store"), "ow_history.csv")


def _naive(ts):
    return pd.to_datetime(ts, utc=True).dt.tz_localize(None)


def clean_training_frame(live: pd.DataFrame) -> pd.DataFrame:
    live = live.copy()
    live["timestamp"] = _naive(live["timestamp"])
    # NULL tag = rows written by the hourly pipeline after tagging was added
    src = live["data_source"] if "data_source" in live else pd.Series("openweather_live", index=live.index)
    live = live[(src.fillna("openweather_live") != "synthetic_backfill")
                & (live["timestamp"] >= LIVE_START)]

    parts = [live]
    if os.path.exists(HISTORY_CSV):
        hist = pd.read_csv(HISTORY_CSV)
        hist["timestamp"] = _naive(hist["timestamp"])
        parts.insert(0, hist[hist["timestamp"] < LIVE_START])
    else:
        print(f"  WARNING: {HISTORY_CSV} not found - training on live rows only")

    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp").reset_index(drop=True))
    print(f"  Clean frame: {len(df)} rows "
          f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")
    return df
