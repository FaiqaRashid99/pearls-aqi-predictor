"""
Walk-forward evaluation for the AQI 72h forecast.

Run:  python eval_walkforward.py

- Loads data exactly like training (Supabase + ow_history.csv, via load_features)
- Tests on several periods (winter peaks included), always training only on the past
- Scores only hours that have a real reading
- Prints baselines next to the models so you can see what actually adds value
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training_pipeline import load_features
from preprocessing import preprocess_features, get_feature_columns

warnings.filterwarnings("ignore")

# Test periods: each is trained ONLY on data before its start.
FOLDS = [
    ("Winter 24/25 (Dec-Jan)", "2024-12-01", "2025-02-01"),
    ("Spring 2025 (Apr-May)", "2025-04-01", "2025-06-01"),
    ("Winter 25/26 (Dec-Jan)", "2025-12-01", "2026-02-01"),
    ("Last 30 days",           None,         None),
]


def add_origin_features(df):
    """Features known at forecast time (72h before the target hour)."""
    d = df.copy()
    a72 = d["aqi"].shift(72)
    d["origin_mean24"] = a72.rolling(24, min_periods=12).mean()
    d["origin_max24"] = a72.rolling(24, min_periods=12).max()
    d["origin_std24"] = a72.rolling(24, min_periods=12).std()
    d["origin_trend"] = d["aqi"].shift(72) - d["aqi"].shift(96)
    d["origin_mean7d"] = a72.rolling(168, min_periods=72).mean()
    # weather over the forecast horizon (comes from the weather forecast in production)
    d["wind_mean72"] = d["wind_speed"].rolling(72, min_periods=36).mean()
    d["rain_sum72"] = d["precipitation"].rolling(72, min_periods=36).sum()
    d["hum_mean24"] = d["humidity"].rolling(24, min_periods=12).mean()
    d["dpress24"] = d["pressure"] - d["pressure"].shift(24)
    return d


EXTRA = ["origin_mean24", "origin_max24", "origin_std24", "origin_trend", "origin_mean7d",
         "wind_mean72", "rain_sum72", "hum_mean24", "dpress24"]


def metrics(y, p):
    return (np.sqrt(mean_squared_error(y, p)), mean_absolute_error(y, p), r2_score(y, p))


def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Huber": Pipeline([("i", imp()), ("s", StandardScaler()),
                           ("m", HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000))]),
        "GradBoost": Pipeline([("i", imp()), ("m", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            min_samples_leaf=20, random_state=42))]),
        "RandForest": Pipeline([("i", imp()), ("m", RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=15, max_features=0.5,
            n_jobs=-1, random_state=42))]),
    }


def main():
    df = load_features()
    df = preprocess_features(df)
    df = add_origin_features(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None) \
        if pd.to_datetime(df["timestamp"]).dt.tz is not None else pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    base = [c for c in get_feature_columns() if c in df.columns]
    ext = base + [c for c in EXTRA if c in df.columns]
    last = df["timestamp"].max()

    rows = []
    for name, s, e in FOLDS:
        if s is None:
            s, e = last - pd.Timedelta(days=30), last + pd.Timedelta(days=1)
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        tr = df[df["timestamp"] < s - pd.Timedelta(hours=72)].dropna(subset=["aqi", "aqi_lag_72h"])
        te = df[(df["timestamp"] >= s) & (df["timestamp"] < e)].dropna(subset=["aqi", "aqi_lag_72h"])
        if "is_real" in te:
            te = te[te["is_real"] == 1]
        if len(tr) < 2000 or len(te) < 100:
            print(f"skip {name}: train={len(tr)} test={len(te)}")
            continue
        print(f"\n{name}: train={len(tr)}  test={len(te)}  test mean AQI={te.aqi.mean():.0f} std={te.aqi.std():.0f}")

        res = {}
        res["Persistence (lag72)"] = te["aqi_lag_72h"].values
        clim = tr.groupby(["month", "hour"])["aqi"].mean()
        res["Climatology (month x hour)"] = [clim.get((m, h), tr.aqi.mean())
                                             for m, h in zip(te["month"], te["hour"])]
        preds = {}
        for fname, feats in [("base", base), ("ext", ext)]:
            for mname, model in make_models().items():
                model.fit(tr[feats], tr["aqi"])
                preds[f"{mname}_{fname}"] = model.predict(te[feats])
        res.update(preds)
        res["Blend Huber+GB (ext)"] = (preds["Huber_ext"] + preds["GradBoost_ext"]) / 2
        res["Blend Huber+GB (base)"] = (preds["Huber_base"] + preds["GradBoost_base"]) / 2

        print(f"  {'model':<28}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
        for k, p in res.items():
            rmse, mae, r2 = metrics(te["aqi"].values, np.asarray(p))
            print(f"  {k:<28}{rmse:>8.1f}{mae:>8.1f}{r2:>8.3f}")
            rows.append({"fold": name, "model": k, "r2": r2, "rmse": rmse})

    if rows:
        summ = pd.DataFrame(rows).pivot(index="model", columns="fold", values="r2")
        summ["mean R2"] = summ.mean(axis=1)
        print("\n=== R2 summary (higher is better) ===")
        print(summ.sort_values("mean R2", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
