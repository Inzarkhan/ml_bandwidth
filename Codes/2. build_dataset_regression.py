import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from leakage_audit import audit_target_matrix, summarize_audit

INPUT_CSV = os.environ.get("DATASET_INPUT_CSV", "prepared_known_full_plusfb.csv")
OUT_DIR = os.environ.get("DATASET_OUT_DIR", "dataset_reg_known_full_plusfb")
WORKLOAD_COL = "workload"
GROUP_COL = WORKLOAD_COL

# Latency leakage: these columns either directly reconstruct the per-window
# target duration or are derived from full-run timing counters that are not a
# fair input for latency prediction.
LATENCY_EXCLUDED_FEATURES = [
    "service_time_ms",
    "throughput_ops_per_s",
    "elapsed_seconds",
    "iterations_completed",
    "window_start_ms",
    "window_end_ms",
    "progress_ratio",
]

# Features expected in the CSV
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
OPTIONAL_NUMERIC_FEATURES = [
    "omp_threads",
    "cpu_limit",
    "target_seconds",
    "target_iterations",
    "elapsed_seconds",
    "iterations_completed",
    "command_runs",
    "idle_gap_ms",
    "launch_overhead_ms",
    "cpu_user_time_ms",
    "cpu_system_time_ms",
    "cpu_util_pct",
    "cpu_nr_periods",
    "cpu_nr_throttled",
    "cpu_throttled_ms",
    "cpu_throttled_pct",
    "memory_current_mb",
    "memory_peak_mb",
    "memory_avg_mb",
    "memory_util_pct",
    "memory_peak_util_pct",
    "memory_avg_util_pct",
    "memory_max_events",
    "memory_high_events",
    "memory_oom_events",
    "memory_oom_kill_events",
    "cpu_pressure_some_avg10",
    "cpu_pressure_full_avg10",
    "memory_pressure_some_avg10",
    "memory_pressure_full_avg10",
    "queue_length",
    "queue_signal_available",
    "monitor_sample_count",
    "service_time_ms",
    "throughput_ops_per_s",
    "window_index",
    "window_start_ms",
    "window_end_ms",
]
OPTIONAL_CATEGORICAL_FEATURES = [
    "suite",
    "workload_type",
    "input_profile",
    "work_mode",
]

TARGET_LATENCY = "duration_ms"
TARGET_ENERGY = "energy_joules"

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows read: {len(df)}")

    # 1. FIX: Handle Boolean "True"/"False" strings safely
    # Map string variants to 1/0 immediately
    bool_map = {
        "True": 1, "False": 0, 
        "true": 1, "false": 0, 
        True: 1, False: 0,
        1: 1, 0: 0
    }
    if "cold_start" in df.columns:
        df["cold_start"] = df["cold_start"].map(bool_map)

    # 2. Check for missing columns
    required = set(BASE_FEATURES + [TARGET_LATENCY, TARGET_ENERGY, GROUP_COL, WORKLOAD_COL])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 3. Convert ONLY strictly numeric columns (exclude cold_start which is already fixed)
    numeric_cols = [c for c in BASE_FEATURES if c != "cold_start"] + [TARGET_LATENCY, TARGET_ENERGY]
    numeric_cols += [c for c in OPTIONAL_NUMERIC_FEATURES if c in df.columns]
    
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. Debug: Check what is becoming NaN
    # If rows are still dropped, uncomment lines below to see WHY
    # print(df[df.isna().any(axis=1)])

    # 5. Clean Data
    initial_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=BASE_FEATURES + [TARGET_LATENCY, TARGET_ENERGY, GROUP_COL, WORKLOAD_COL])
    
    # Filter out bad latency
    df = df[df[TARGET_LATENCY] > 0]

    print(f"Rows after cleaning: {len(df)} (Dropped {initial_count - len(df)})")

    if len(df) == 0:
        raise ValueError("Dataframe is empty! Check if columns like 'energy_joules' or 'cpu_time_ms' are all NaNs.")

    # ---- Feature Engineering ----
    eps = 1e-9
    df["cold_start"] = df["cold_start"].astype(int)
    elapsed_seconds = pd.to_numeric(df.get("elapsed_seconds", 0.0), errors="coerce").fillna(0.0)
    iterations_completed = pd.to_numeric(df.get("iterations_completed", 0.0), errors="coerce").fillna(0.0)

    df["rss_ratio"] = df["peak_rss_mb"] / (df["mem_limit_mb"] + eps)
    df["current_rss_ratio"] = df["rss_mb"] / (df["mem_limit_mb"] + eps)
    df["cpu_per_mem"] = df["cpu_time_ms"] / (df["mem_limit_mb"] + eps)
    df["cpu_util_fraction"] = pd.to_numeric(df.get("cpu_util_pct", 0.0), errors="coerce").fillna(0.0) / 100.0
    df["memory_headroom_pct"] = 100.0 - pd.to_numeric(df.get("memory_peak_util_pct", 0.0), errors="coerce").fillna(0.0)
    df["throttle_fraction"] = pd.to_numeric(df.get("cpu_throttled_pct", 0.0), errors="coerce").fillna(0.0) / 100.0
    safe_iterations = iterations_completed.where(iterations_completed > 0, np.nan)
    safe_elapsed = elapsed_seconds.where(elapsed_seconds > 0, np.nan)
    df["service_time_ms"] = ((elapsed_seconds * 1000.0) / safe_iterations).fillna(df[TARGET_LATENCY])
    df["throughput_ops_per_s"] = (iterations_completed / safe_elapsed).fillna(0.0)
    df["log_mem"] = np.log1p(df["mem_limit_mb"].astype(float))
    df["progress_ratio"] = (
        pd.to_numeric(df.get("window_end_ms", 0.0), errors="coerce").fillna(0.0)
        / np.maximum((elapsed_seconds * 1000.0).replace(0.0, np.nan), 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0, upper=1.0)
    df["is_fixed_work"] = (pd.to_numeric(df.get("target_iterations", 0.0), errors="coerce").fillna(0.0) > 0).astype(float)
    
    # Build final feature set
    feature_df = df[BASE_FEATURES].copy()
    feature_df["rss_ratio"] = df["rss_ratio"]
    feature_df["current_rss_ratio"] = df["current_rss_ratio"]
    feature_df["cpu_per_mem"] = df["cpu_per_mem"]
    feature_df["cpu_util_fraction"] = df["cpu_util_fraction"]
    feature_df["memory_headroom_pct"] = df["memory_headroom_pct"]
    feature_df["throttle_fraction"] = df["throttle_fraction"]
    feature_df["service_time_ms"] = df["service_time_ms"]
    feature_df["throughput_ops_per_s"] = df["throughput_ops_per_s"]
    feature_df["log_mem"] = df["log_mem"]
    feature_df["progress_ratio"] = df["progress_ratio"]
    feature_df["is_fixed_work"] = df["is_fixed_work"]

    for col in OPTIONAL_NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().all():
            continue
        feature_df[col] = series.fillna(series.median())
    
    for col in OPTIONAL_CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        valid = df[col].replace({"nan": np.nan}).dropna()
        if valid.empty:
            continue
        dummies = pd.get_dummies(df[col].astype(str), prefix=col, prefix_sep="__")
        dummies = dummies.reindex(sorted(dummies.columns), axis=1)
        feature_df = pd.concat([feature_df, dummies], axis=1)

    ENERGY_FEATURES = feature_df.columns.tolist()
    latency_feature_df = feature_df.drop(
        columns=[col for col in LATENCY_EXCLUDED_FEATURES if col in feature_df.columns]
    ).copy()
    LATENCY_FEATURES = latency_feature_df.columns.tolist()

    # Split Data
    X_energy = feature_df.astype(float).values
    X_latency = latency_feature_df.astype(float).values
    y_latency = df[TARGET_LATENCY].astype(float).values
    y_energy = df[TARGET_ENERGY].astype(float).values
    groups = df[GROUP_COL].astype(str).values

    idx = np.arange(len(df))
    unique_groups = np.unique(groups)

    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X_energy, y_latency, groups))
    else:
        # Fallback if only 1 run_id exists
        print("Warning: Only 1 Group/Run ID found. Using random split instead of group split.")
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_energy_train, X_energy_test = X_energy[train_idx], X_energy[test_idx]
    X_latency_train, X_latency_test = X_latency[train_idx], X_latency[test_idx]
    yL_train, yL_test = y_latency[train_idx], y_latency[test_idx]
    yE_train, yE_test = y_energy[train_idx], y_energy[test_idx]

    os.makedirs(OUT_DIR, exist_ok=True)
    # Backward compatibility: keep X_train/X_test as the energy feature matrix.
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_energy_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_energy_test)
    np.save(os.path.join(OUT_DIR, "X_energy_train.npy"), X_energy_train)
    np.save(os.path.join(OUT_DIR, "X_energy_test.npy"), X_energy_test)
    np.save(os.path.join(OUT_DIR, "X_latency_train.npy"), X_latency_train)
    np.save(os.path.join(OUT_DIR, "X_latency_test.npy"), X_latency_test)
    np.save(os.path.join(OUT_DIR, "y_latency_train.npy"), yL_train)
    np.save(os.path.join(OUT_DIR, "y_latency_test.npy"), yL_test)
    np.save(os.path.join(OUT_DIR, "y_energy_train.npy"), yE_train)
    np.save(os.path.join(OUT_DIR, "y_energy_test.npy"), yE_test)

    # Save Meta
    meta = {
        "features": ENERGY_FEATURES,
        "features_energy": ENERGY_FEATURES,
        "features_latency": LATENCY_FEATURES,
        "latency_excluded_features": [col for col in LATENCY_EXCLUDED_FEATURES if col in ENERGY_FEATURES],
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "workload_onehot": []
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    leakage_report = {
        "latency_model_features": audit_target_matrix(
            X_latency,
            y_latency,
            LATENCY_FEATURES,
            target_name="duration_ms",
            run_pairwise=True,
        ),
        "latency_full_feature_set": audit_target_matrix(
            X_energy,
            y_latency,
            ENERGY_FEATURES,
            target_name="duration_ms_full_feature_set",
            run_pairwise=True,
        ),
        "energy_model_features": audit_target_matrix(
            X_energy,
            y_energy,
            ENERGY_FEATURES,
            target_name="energy_joules",
            run_pairwise=False,
        ),
    }
    with open(os.path.join(OUT_DIR, "leakage_audit.json"), "w") as f:
        json.dump(leakage_report, f, indent=2)

    print("\n" + summarize_audit("Latency model feature set", leakage_report["latency_model_features"]))
    print(summarize_audit("Latency full feature set", leakage_report["latency_full_feature_set"]))
    print(summarize_audit("Energy model feature set", leakage_report["energy_model_features"]))

    print(f"\n[OK] Dataset built. Train size: {len(train_idx)}, Test size: {len(test_idx)}")

if __name__ == "__main__":
    main()
