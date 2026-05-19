import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# SHAP
import shap

# Keras standalone (works on Python 3.13, no TensorFlow needed)
try:
    import keras
    from keras import layers, callbacks
    KERAS_AVAILABLE = True
    print("Keras available — neural network will be trained")
except ImportError:
    KERAS_AVAILABLE = False
    print(" Keras not available — skipping neural network")

load_dotenv()

FEATURE_STORE = os.getenv("FEATURE_STORE_PATH", "feature_store")
MODEL_DIR     = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    # Remove pm25 — it equals aqi in our data (leakage!)
    # Keep only real weather + time features
    "temp", "feels_like", "humidity", "pressure",
    "wind_speed", "wind_direction",
    "precipitation", "weather_code",
    "hour", "day", "month", "dayofweek",
    "is_weekend", "is_rush_hour", "aqi_change_rate",
]

# FEATURE_COLS = [
#     "pm25", "pm10", "no2", "o3", "co", "so2",
#     "temp", "feels_like", "humidity", "pressure",
#     "wind_speed", "wind_direction", "visibility",
#     "precipitation", "weather_code",
#     "hour", "day", "month", "dayofweek",
#     "is_weekend", "is_rush_hour", "aqi_change_rate",
# ]
TARGET_COL = "aqi"


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_features() -> pd.DataFrame:
    csv_path = os.path.join(FEATURE_STORE, "aqi_features.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Feature store not found: {csv_path}. Run backfill.py first.")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(df)} rows")
    print(f"   Range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────────
def prepare_data(df: pd.DataFrame):
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\nFeatures ({len(available)}): {available}")

    X = df[available].copy()
    y = df[TARGET_COL].copy()

    # Time-based split — no data leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test, available


# ─────────────────────────────────────────────
# 3. METRICS
# ─────────────────────────────────────────────
def evaluate(name: str, y_test, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n   {name}:")
    print(f"      RMSE: {rmse:.2f}  |  MAE: {mae:.2f}  |  R²: {r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


# ─────────────────────────────────────────────
# 4. SKLEARN MODELS
# ─────────────────────────────────────────────
def train_sklearn_models(X_train, X_test, y_train, y_test):
    results = []

    # Random Forest
    print("\nTraining Random Forest...")
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    results.append({**evaluate("Random Forest", y_test, rf.predict(X_test)), "pipeline": rf})

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=42))
    ])
    gb.fit(X_train, y_train)
    results.append({**evaluate("Gradient Boosting", y_test, gb.predict(X_test)), "pipeline": gb})

    # Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=10.0))
    ])
    ridge.fit(X_train, y_train)
    results.append({**evaluate("Ridge Regression", y_test, ridge.predict(X_test)), "pipeline": ridge})

    return results


# ─────────────────────────────────────────────
# 5. KERAS NEURAL NETWORK
# ─────────────────────────────────────────────
def train_keras_model(X_train, X_test, y_train, y_test, feature_cols):
    print("\nTraining Keras Neural Network...")

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc  = scaler.transform(imputer.transform(X_test))

    model = keras.Sequential([
        layers.Input(shape=(X_train_sc.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse", metrics=["mae"]
    )

    history = model.fit(
        X_train_sc, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5)
        ],
        verbose=0
    )

    epochs_done = len(history.history["loss"])
    print(f"   Trained {epochs_done} epochs (early stopping)")

    preds   = model.predict(X_test_sc, verbose=0).flatten()
    metrics = evaluate("Keras Neural Network", y_test, preds)

    # Save
    model.save(os.path.join(MODEL_DIR, "keras_model.keras"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "keras_imputer.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "keras_scaler.pkl"))
    print(f"   Keras model saved to models/keras_model.keras")
    
    #####
    import json
    keras_metrics = {"rmse": round(metrics["rmse"], 4), "mae": round(metrics["mae"], 4), "r2": round(metrics["r2"], 4)}
    with open(os.path.join(MODEL_DIR, "keras_metrics.json"), "w") as f:
        json.dump(keras_metrics, f)
    ####

    return {**metrics, "pipeline": None}


# ─────────────────────────────────────────────
# 6. SHAP IMPORTANCE
# ─────────────────────────────────────────────
def compute_shap(pipeline, X_test, feature_cols):
    print("\nComputing SHAP feature importance...")
    try:
        model   = pipeline.named_steps["model"]
        imputer = pipeline.named_steps["imputer"]
        X_imp   = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_imp.iloc[:200])

        importance = pd.Series(
            np.abs(shap_values).mean(0), index=feature_cols
        ).sort_values(ascending=False)

        print("\n   Top 10 features:")
        for feat, val in importance.head(10).items():
            bar = "█" * int(val / importance.max() * 20)
            print(f"   {feat:<22} {bar} {val:.3f}")

        importance.to_csv(
            os.path.join(MODEL_DIR, "shap_importance.csv"),
            header=["shap_value"]
        )
        print(f"   💾 SHAP saved to models/shap_importance.csv")
        return importance

    except Exception as e:
        print(f"    SHAP skipped: {e}")
        return None


# ─────────────────────────────────────────────
# 7. SAVE BEST MODEL
# ─────────────────────────────────────────────
def save_best_model(results, feature_cols):
    sklearn_results = [r for r in results if r.get("pipeline") is not None]
    best = min(sklearn_results, key=lambda x: x["rmse"])
    print(f"\nBest model: {best['model']} (RMSE={best['rmse']:.2f}, R²={best['r2']:.4f})")

    joblib.dump(best["pipeline"], os.path.join(MODEL_DIR, "best_model.pkl"))

    metadata = {
        "best_model":   best["model"],
        "rmse":         round(best["rmse"], 4),
        "mae":          round(best["mae"],  4),
        "r2":           round(best["r2"],   4),
        "feature_cols": feature_cols,
        "trained_at":   datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "all_results": [
            {"model": r["model"], "rmse": round(r["rmse"], 4),
             "mae": round(r["mae"], 4), "r2": round(r["r2"], 4)}
            for r in results
        ]
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   best_model.pkl + model_metadata.json saved")
    return best, metadata


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AQI PREDICTOR — TRAINING PIPELINE")
    print("="*55)

    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

    # Sklearn models
    print("\n" + "─"*55)
    print("SKLEARN MODELS")
    print("─"*55)
    all_results = train_sklearn_models(X_train, X_test, y_train, y_test)

    # Keras neural network
    if KERAS_AVAILABLE:
        print("\n" + "─"*55)
        print("DEEP LEARNING (KERAS)")
        print("─"*55)
        keras_result = train_keras_model(
            X_train, X_test, y_train, y_test, feature_cols)
        all_results.append(keras_result)

    # SHAP
    print("\n" + "─"*55)
    print("FEATURE IMPORTANCE (SHAP)")
    print("─"*55)
    best_sklearn = min(
        [r for r in all_results if r.get("pipeline")],
        key=lambda x: x["rmse"]
    )
    compute_shap(best_sklearn["pipeline"], X_test, feature_cols)

    # Save
    print("\n" + "─"*55)
    print("SAVING MODELS")
    print("─"*55)
    save_best_model(all_results, feature_cols)

    # Summary
    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("─"*55)
    for r in all_results:
        print(f"{r['model']:<25} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.4f}")
    print("─"*55)
    print("\nTraining complete! Run next: python dashboard.py")