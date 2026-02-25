#!/usr/bin/env python3
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------
# USER SETTINGS (EDIT HERE)
# ---------------------------------------------------

DATA_DIR = "dataset_reg"   # output of build_dataset_regression.py
OUT_DIR  = "models"        # where to save joblib models + metrics
MAX_ITER = 300             # training iterations
RANDOM_STATE = 42

# Optional tuning knobs (set to None to use defaults)
LEARNING_RATE = 0.08
MAX_DEPTH = None           # e.g. 6, 10, None

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def evaluate(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    data_dir = Path(DATA_DIR)
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load arrays
    X_train = np.load(data_dir / "X_train.npy")
    X_test  = np.load(data_dir / "X_test.npy")

    yL_train = np.load(data_dir / "y_latency_train.npy")
    yL_test  = np.load(data_dir / "y_latency_test.npy")

    yE_train = np.load(data_dir / "y_energy_train.npy")
    yE_test  = np.load(data_dir / "y_energy_test.npy")

    # Models: 2 regressors
    latency_model = HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )

    energy_model = HistGradientBoostingRegressor(
        max_iter=MAX_ITER,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE + 1
    )

    # Train
    latency_model.fit(X_train, yL_train)
    energy_model.fit(X_train, yE_train)

    # Predict
    yL_pred = latency_model.predict(X_test)
    yE_pred = energy_model.predict(X_test)

    # Evaluate
    results = {
        "latency": evaluate(yL_test, yL_pred),
        "energy": evaluate(yE_test, yE_pred),
        "config": {
            "max_iter": MAX_ITER,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "random_state": RANDOM_STATE
        },
        "shapes": {
            "X_train": list(X_train.shape),
            "X_test": list(X_test.shape)
        }
    }

    print("\n==== Evaluation Results ====")
    print("Latency  MAE/RMSE:", results["latency"])
    print("Energy   MAE/RMSE:", results["energy"])

    # Save models
    joblib.dump(latency_model, outdir / "latency_model.joblib")
    joblib.dump(energy_model, outdir / "energy_model.joblib")

    # Save metrics as json
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n[OK] Models saved to:", outdir.resolve())
    print("[OK] Metrics saved to:", (outdir / "metrics.json").resolve())


if __name__ == "__main__":
    main()