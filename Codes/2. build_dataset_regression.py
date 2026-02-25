#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# ---------------------------------------------------
# USER SETTINGS (EDIT HERE)
# ---------------------------------------------------

INPUT_CSV = "prepared.csv"     # CSV created from prepare_serverless_csv.py
OUTPUT_DIR = "dataset_reg"     # Folder where .npy files will be saved
TEST_SIZE = 0.2                # 20% test split
RANDOM_STATE = 42

# ---------------------------------------------------
# FEATURES USED FOR TRAINING
# ---------------------------------------------------

FEATURES = [
    "cpu_time_ms",
    "rss_mb",
    "peak_rss_mb",
    "io_read_bytes",
    "io_write_bytes",
    "cold_start",
    "concurrency",
    "queue_delay_ms",
    "mem_limit_mb"
]

LATENCY_LABEL = "duration_ms"
ENERGY_LABEL = "energy_joules"
GROUP_COLUMN = "run_id"   # prevents data leakage

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path.resolve()}")

    df = pd.read_csv(input_path)

    print("Total rows before cleaning:", len(df))

    # Drop rows with missing feature or label values
    df = df.dropna(subset=FEATURES + [LATENCY_LABEL, ENERGY_LABEL, GROUP_COLUMN])
    df = df.reset_index(drop=True)

    print("Rows after cleaning:", len(df))

    # Feature matrix
    X = df[FEATURES].astype(float).values

    # Labels
    y_latency = df[LATENCY_LABEL].astype(float).values
    y_energy = df[ENERGY_LABEL].astype(float).values

    # Group split to avoid leakage
    groups = df[GROUP_COLUMN].astype(str).values

    splitter = GroupShuffleSplit(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_idx, test_idx = next(splitter.split(X, y_latency, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    yL_train, yL_test = y_latency[train_idx], y_latency[test_idx]
    yE_train, yE_test = y_energy[train_idx], y_energy[test_idx]

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "X_train.npy", X_train)
    np.save(outdir / "X_test.npy", X_test)

    np.save(outdir / "y_latency_train.npy", yL_train)
    np.save(outdir / "y_latency_test.npy", yL_test)

    np.save(outdir / "y_energy_train.npy", yE_train)
    np.save(outdir / "y_energy_test.npy", yE_test)

    print("\nDataset built successfully.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Saved in:", outdir.resolve())


if __name__ == "__main__":
    main()