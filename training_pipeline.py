import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import shap

from preprocessing import preprocess_features, get_feature_columns

try:
    import keras
    from keras import layers, callbacks
    KERAS_AVAILABLE = True
    print("Keras available")
except ImportError:
    KERAS_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGB_AVAILABLE = False

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_DIR    = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

supabase   = create_client(SUPABASE_URL, SUPABASE_KEY)
TARGET_COL = "aqi"


def load_features() -> pd.DataFrame:
    print("Loading features from Supabase...")
    try:
        all_rows = []
        page = 0
        while True:
            result = (supabase.table("aqi_features")
                      .select("*")
                      .order("timestamp")
                      .range(page * 1000, (page + 1) * 1000 - 1)
                      .execute())
            if not result.data:
                break
            all_rows.extend(result.data)
            if len(result.data) < 1000:
                break
            page += 1
        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"   Loaded {len(df)} rows from Supabase")
        return df
    except Exception as e:
        print(f"   Supabase failed, loading CSV: {e}")
        path = os.path.join(os.getenv("FEATURE_STORE_PATH", "feature_store"), "aqi_features.csv")
        return pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")


def prepare_data(df: pd.DataFrame):
    FEATURE_COLS = get_feature_columns()
    available    = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\nFeatures ({len(available)}): {available}")

    X = df[available].copy()
    y = df[TARGET_COL].copy()

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, available


def evaluate(name: str, y_test, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"   {name}:  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def train_sklearn_models(X_train, X_test, y_train, y_test):
    results = []

    # ── Random Forest ──────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=4, max_features=0.7,
            random_state=42, n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    results.append({**evaluate("Random Forest", y_test, rf.predict(X_test)), "pipeline": rf})

    # ── Gradient Boosting ──────────────────────────────────────────
    print("\nTraining Gradient Boosting...")
    gb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.04,
            max_depth=5, subsample=0.8,
            min_samples_leaf=4, random_state=42
        ))
    ])
    gb.fit(X_train, y_train)
    results.append({**evaluate("Gradient Boosting", y_test, gb.predict(X_test)), "pipeline": gb})

    # ── XGBoost ────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ))
        ])
        xgb.fit(X_train, y_train)
        results.append({**evaluate("XGBoost", y_test, xgb.predict(X_test)), "pipeline": xgb})
        
    # ── Huber Regression (simple — no poly, can't handle 351 features) ──
    print("\nTraining Huber Regression...")
    huber = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=1000))
    ])
    huber.fit(X_train, y_train)
    results.append({**evaluate("Huber Regression", y_test, huber.predict(X_test)), "pipeline": huber})

    return results


def train_keras_model(X_train, X_test, y_train, y_test, feature_cols):
    print("\nTraining Keras Neural Network...")
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc  = scaler.transform(imputer.transform(X_test))

    # KEY FIX: .to_numpy() before slicing — avoids pandas index mismatch
    y_np      = y_train.to_numpy()
    val_size  = int(len(X_train_sc) * 0.2)
    X_tr, X_val = X_train_sc[:-val_size], X_train_sc[-val_size:]
    y_tr, y_val = y_np[:-val_size],        y_np[-val_size:]

    # Simpler architecture for ~1400 effective training samples
    model = keras.Sequential([
        layers.Input(shape=(X_train_sc.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="huber",       # robust to AQI outliers vs MSE
        metrics=["mae"]
    )
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=16,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss", patience=25,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
            )
        ],
        verbose=0
    )
    preds   = np.maximum(0, model.predict(X_test_sc, verbose=0).flatten())
    metrics = evaluate("Keras Neural Network", y_test, preds)
    model.save(os.path.join(MODEL_DIR, "keras_model.keras"))
    joblib.dump(imputer, os.path.join(MODEL_DIR, "keras_imputer.pkl"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "keras_scaler.pkl"))
    return {**metrics, "pipeline": None}


def compute_shap(pipeline, X_test, feature_cols):
    print("\nComputing SHAP feature importance...")
    try:
        model   = pipeline.named_steps["model"]
        imputer = pipeline.named_steps.get("imputer")
        X_imp   = pd.DataFrame(
            imputer.transform(X_test), columns=feature_cols
        ) if imputer else X_test

        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_imp.iloc[:200])
        importance = (pd.Series(np.abs(shap_vals).mean(0), index=feature_cols)
                      .sort_values(ascending=False))

        print("\n   Top 10 features:")
        for feat, val in importance.head(10).items():
            bar = "█" * int(val / importance.max() * 20)
            print(f"   {feat:<22} {bar} {val:.3f}")

        importance.to_csv(os.path.join(MODEL_DIR, "shap_importance.csv"),
                          header=["shap_value"])
        print("   SHAP saved")
        return importance
    except Exception as e:
        print(f"   SHAP skipped: {e}")
        return None


def save_best_model(results, feature_cols, n_train: int):
    sklearn_results = [r for r in results if r.get("pipeline") is not None]
    best = min(sklearn_results, key=lambda x: x["rmse"])

    print(f"\nBest model: {best['model']}  RMSE={best['rmse']:.2f}  R²={best['r2']:.4f}")
    joblib.dump(best["pipeline"], os.path.join(MODEL_DIR, "best_model.pkl"))

    metadata = {
        "best_model":    best["model"],
        "rmse":          round(best["rmse"], 4),
        "mae":           round(best["mae"],  4),
        "r2":            round(best["r2"],   4),
        "feature_cols":  feature_cols,
        "training_rows": n_train,           # ← now dynamic
        "trained_at":    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "all_results": [
            {"model": r["model"], "rmse": round(r["rmse"], 4),
             "mae": round(r["mae"], 4), "r2": round(r["r2"], 4)}
            for r in results
        ]
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("   Model saved")
    return best, metadata

#######################
# Main
#######################


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  AQI PREDICTOR — TRAINING PIPELINE")
    print("="*65)

    df = load_features()
    print("\nApplying preprocessing...")
    df = preprocess_features(df)

    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df)

    print("\n" + "─"*65)
    print("TRAINING MODELS")
    print("─"*65)
    all_results = train_sklearn_models(X_train, X_test, y_train, y_test)

    if KERAS_AVAILABLE:
        all_results.append(
            train_keras_model(X_train, X_test, y_train, y_test, feature_cols)
        )

    print("\n" + "─"*65)
    print("FEATURE IMPORTANCE (SHAP)")
    print("─"*65)
    best_tree = min(
        [r for r in all_results
         if r.get("pipeline") and hasattr(r["pipeline"].named_steps["model"], "feature_importances_")],
        key=lambda x: x["rmse"]
    )
    compute_shap(best_tree["pipeline"], X_test, feature_cols)

    print("\n" + "─"*65)
    print("SAVING BEST MODEL")
    print("─"*65)
    # save_best_model(all_results, feature_cols)
    save_best_model(all_results, feature_cols, n_train=len(X_train))

    print("\n" + "="*65)
    print("  FINAL RESULTS")
    print("="*65)
    print(f"\n{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("─"*65)
    for r in all_results:
        print(f"{r['model']:<25} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.4f}")
    print("─"*65)
    print("\nDone! Run: streamlit run dashboard.py")