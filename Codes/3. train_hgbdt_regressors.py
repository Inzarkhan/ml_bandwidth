#!/usr/bin/env python3
import os, json
import numpy as np
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "dataset_reg"
MODEL_DIR = "models"

LOG_MIN = np.log1p(0.0)
LOG_MAX = np.log1p(60000.0)  # 60s cap

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse(y_true, y_pred)}

def fit_log_model(X_train, y_ms_train):
    y_t = np.log1p(y_ms_train)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_depth=8,
        max_iter=1200,
        random_state=42,
    )
    model.fit(X_train, y_t)
    return model

def predict_ms(model, X):
    pred_t = model.predict(X)
    pred_t = np.clip(pred_t, LOG_MIN, LOG_MAX)
    return np.expm1(pred_t)

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    yL_train = np.load(os.path.join(DATA_DIR, "y_latency_train.npy"))
    yL_test  = np.load(os.path.join(DATA_DIR, "y_latency_test.npy"))
    yE_train = np.load(os.path.join(DATA_DIR, "y_energy_train.npy"))
    yE_test  = np.load(os.path.join(DATA_DIR, "y_energy_test.npy"))

    # We need cold_start column index from meta.json features
    meta = json.load(open(os.path.join(DATA_DIR, "meta.json")))
    feats = meta["features"]
    if "cold_start" not in feats:
        raise ValueError("cold_start not found in features. Check dataset_reg/meta.json")
    cold_i = feats.index("cold_start")

    # Masks
    cold_train = X_train[:, cold_i] == 1
    warm_train = ~cold_train
    cold_test = X_test[:, cold_i] == 1
    warm_test = ~cold_test

    # ---- Train two latency models ----
    m_cold = fit_log_model(X_train[cold_train], yL_train[cold_train])
    m_warm = fit_log_model(X_train[warm_train], yL_train[warm_train])

    # Predict
    pred_cold = predict_ms(m_cold, X_test[cold_test]) if cold_test.any() else np.array([])
    pred_warm = predict_ms(m_warm, X_test[warm_test]) if warm_test.any() else np.array([])

    # Combine into one array aligned with X_test order
    y_pred_all = np.empty_like(yL_test, dtype=float)
    if cold_test.any():
        y_pred_all[cold_test] = pred_cold
    if warm_test.any():
        y_pred_all[warm_test] = pred_warm

    # Metrics
    overall = metrics(yL_test, y_pred_all)
    cold_m = metrics(yL_test[cold_test], y_pred_all[cold_test]) if cold_test.any() else None
    warm_m = metrics(yL_test[warm_test], y_pred_all[warm_test]) if warm_test.any() else None

    # ---- Energy model (still placeholder) ----
    energy_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        random_state=42,
    )
    energy_model.fit(X_train, yE_train)
    yE_pred = energy_model.predict(X_test)
    energy_m = metrics(yE_test, yE_pred)

    print("\n==== Evaluation Results ====")
    print("Latency OVERALL MAE/RMSE:", overall)
    print("Latency COLD    MAE/RMSE:", cold_m)
    print("Latency WARM    MAE/RMSE:", warm_m)
    print("Energy          MAE/RMSE:", energy_m)

    # Save models
    dump(m_cold, os.path.join(MODEL_DIR, "latency_cold_hgbdt.joblib"))
    dump(m_warm, os.path.join(MODEL_DIR, "latency_warm_hgbdt.joblib"))
    dump(energy_model, os.path.join(MODEL_DIR, "energy_hgbdt.joblib"))

    out = {
        "latency_overall": overall,
        "latency_cold": cold_m,
        "latency_warm": warm_m,
        "energy": energy_m,
        "latency_target": "log1p(duration_ms) with cold/warm split",
        "prediction_postprocess": f"clip log to [{float(LOG_MIN):.3f},{float(LOG_MAX):.3f}] then expm1",
        "features": feats,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Models saved to: {os.path.abspath(MODEL_DIR)}")
    print(f"[OK] Metrics saved to: {os.path.abspath(os.path.join(MODEL_DIR,'metrics.json'))}")

if __name__ == "__main__":
    main()