#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

INPUT_CSV = "prepared.csv"
OUT_DIR = "dataset_reg"

# Raw numeric features (already in prepared.csv)
BASE_FEATURES = [
    "mem_limit_mb",
    "cold_start",
    "concurrency",
    "queue_delay_ms",
    "cpu_time_ms",
    "peak_rss_mb",
    "rss_mb",
    "io_read_bytes",
    "io_write_bytes",
]

TARGET_LATENCY = "duration_ms"
TARGET_ENERGY = "energy_joules"
GROUP_COL = "run_id"
WORKLOAD_COL = "workload"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows before cleaning: {len(df)}")

    required = set(BASE_FEATURES + [TARGET_LATENCY, TARGET_ENERGY, GROUP_COL, WORKLOAD_COL])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prepared.csv: {missing}")

    # Convert numeric columns
    for c in BASE_FEATURES + [TARGET_LATENCY, TARGET_ENERGY]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=BASE_FEATURES + [TARGET_LATENCY, TARGET_ENERGY, GROUP_COL, WORKLOAD_COL])
    df = df[df[TARGET_LATENCY] >= 0]

    print(f"Rows after cleaning: {len(df)}")

    # ---- SAFE engineered features (NO duration_ms used) ----
    eps = 1e-9
    df["cold_start"] = df["cold_start"].astype(int)

    df["rss_ratio"] = df["peak_rss_mb"] / (df["mem_limit_mb"] + eps)
    df["cpu_per_mem"] = df["cpu_time_ms"] / (df["mem_limit_mb"] + eps)
    df["log_mem"] = np.log1p(df["mem_limit_mb"].astype(float))

    # ---- One-hot encode workload identity (VALID: provider knows function/workload ID) ----
    # Creates columns like: workload__cpuintensive, workload__memory_touch, ...
    workload_dummies = pd.get_dummies(
        df[WORKLOAD_COL].astype(str),
        prefix="workload",
        prefix_sep="__"
    )

    # Build feature dataframe (numeric + engineered + one-hot)
    feature_df = df[BASE_FEATURES].copy()
    feature_df["rss_ratio"] = df["rss_ratio"]
    feature_df["cpu_per_mem"] = df["cpu_per_mem"]
    feature_df["log_mem"] = df["log_mem"]

    # Append one-hot columns (sorted to keep stable order)
    workload_dummies = workload_dummies.reindex(sorted(workload_dummies.columns), axis=1)
    feature_df = pd.concat([feature_df, workload_dummies], axis=1)

    FEATURES = feature_df.columns.tolist()

    # Targets and groups
    X = feature_df.astype(float).values
    y_latency = df[TARGET_LATENCY].astype(float).values
    y_energy = df[TARGET_ENERGY].astype(float).values
    groups = df[GROUP_COL].astype(str).values

    idx = np.arange(len(df))
    unique_groups = np.unique(groups)

    # Group-aware split preferred
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y_latency, groups))
    else:
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    yL_train, yL_test = y_latency[train_idx], y_latency[test_idx]
    yE_train, yE_test = y_energy[train_idx], y_energy[test_idx]
    g_train, g_test = groups[train_idx], groups[test_idx]

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save arrays
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_latency_train.npy"), yL_train)
    np.save(os.path.join(OUT_DIR, "y_latency_test.npy"), yL_test)
    np.save(os.path.join(OUT_DIR, "y_energy_train.npy"), yE_train)
    np.save(os.path.join(OUT_DIR, "y_energy_test.npy"), yE_test)
    np.save(os.path.join(OUT_DIR, "groups_train.npy"), g_train.astype(object))
    np.save(os.path.join(OUT_DIR, "groups_test.npy"), g_test.astype(object))

    # Save metadata (important for reproducibility + later inference)
    meta = {
        "input_csv": os.path.abspath(INPUT_CSV),
        "n_rows": int(len(df)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "features": FEATURES,
        "base_features": BASE_FEATURES,
        "engineered_features": ["rss_ratio", "cpu_per_mem", "log_mem"],
        "workload_onehot_columns": workload_dummies.columns.tolist(),
        "target_latency": TARGET_LATENCY,
        "target_energy": TARGET_ENERGY,
        "group_col": GROUP_COL,
        "workload_col": WORKLOAD_COL,
        "unique_groups": int(len(unique_groups)),
        "unique_workloads": sorted(df[WORKLOAD_COL].astype(str).unique().tolist()),
    }

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDataset built successfully (with workload one-hot).")
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples: {len(test_idx)}")
    print(f"Saved in: {os.path.abspath(OUT_DIR)}")
    print(f"Total features: {len(FEATURES)}")
    print("Workload one-hot columns:", workload_dummies.columns.tolist())


if __name__ == "__main__":
    main()