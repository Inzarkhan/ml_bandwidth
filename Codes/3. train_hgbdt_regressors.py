#!/usr/bin/env python3
import os, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from resource_decision_features import build_projected_decision_dataframe, build_resource_decision_feature_df, derive_service_time_ms

DATA_DIR = os.environ.get("DATASET_DIR", "dataset_reg")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
PREPARED_FILE = os.environ.get("PREPARED_CSV", "prepared.csv")

LOG_MIN = np.log1p(0.0)
LOG_MAX = np.log1p(300000.0)  # allow up to 300s runs for 120-200s benchmarking

DECISION_ENERGY_MODEL_FILE = "energy_hgbdt_decision.joblib"
DECISION_BENEFICIAL_MODEL_FILE = "energy_hgbdt_decision_classifier.joblib"
DECISION_ENERGY_META_FILE = "energy_hgbdt_decision_meta.json"
DECISION_GROUP_COL = "workload"
DECISION_WORKLOAD_COL = "workload"
DECISION_WORKLOAD_TYPE_COL = "workload_type"
DECISION_SOURCE_TARGET = "energy_joules"
DECISION_TARGET = "energy_delta_vs_1024_joules"
DECISION_BASELINE_MEM = 1024
DECISION_BASELINE_CPU = float(os.environ.get("SEBS_DECISION_BASELINE_CPU_LIMIT", "1.0"))
DECISION_REPEAT_AGG_MODE = os.environ.get("SEBS_DECISION_REPEAT_AGG_MODE", "median")
DECISION_SLO_MULTIPLIER = float(os.environ.get("SEBS_DECISION_SLO_MULTIPLIER", "1.30"))
DECISION_BENEFICIAL_MIN_SAVINGS_PCT = float(os.environ.get("SEBS_DECISION_MIN_SAVINGS_PCT", "0.50"))
DECISION_BENEFICIAL_THRESHOLD = float(os.environ.get("SEBS_DECISION_CLASSIFIER_THRESHOLD", "0.60"))
DECISION_PARAM_GRID = [
    {
        "learning_rate": 0.05,
        "max_depth": 4,
        "max_iter": 300,
        "min_samples_leaf": 5,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 400,
        "min_samples_leaf": 5,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.03,
        "max_depth": 8,
        "max_iter": 800,
        "min_samples_leaf": 3,
        "l2_regularization": 0.0,
    },
    {
        "learning_rate": 0.03,
        "max_depth": 6,
        "max_iter": 1000,
        "min_samples_leaf": 10,
        "l2_regularization": 0.1,
    },
]

WORKLOAD_TYPE_MAP = {
    "sebs_sleep_known": "web",
    "sebs_dynamic_html_known": "web",
    "sebs_thumbnailer_known": "multimedia",
    "sebs_graph_bfs_known": "scientific",
    "sebs_compression_known": "utility",
    "sebs_graph_pagerank_known": "scientific",
    "sebs_crud_api_known": "web",
    "sebs_uploader_known": "web",
    "sebs_video_processing_known": "multimedia",
    "sebs_dna_visualisation_known": "scientific",
    "sebs_compression_unseen": "utility",
    "sebs_graph_pagerank_unseen": "scientific",
    "sebs_graph_mst_unseen": "scientific",
    "sebs_uploader_unseen": "web",
    "sebs_video_processing_unseen": "multimedia",
    "sebs_dna_visualisation_unseen": "scientific",
}

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred):
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": rmse(y_true, y_pred)}


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

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


def select_decision_model(X_train, y_train, groups_train):
    if len(np.unique(groups_train)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
        inner_train_idx, val_idx = next(splitter.split(X_train, y_train, groups_train))
    else:
        idx = np.arange(len(X_train))
        inner_train_idx, val_idx = train_test_split(idx, test_size=0.25, random_state=7)

    search_results = []
    best_result = None

    for params in DECISION_PARAM_GRID:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=42,
            **params,
        )
        model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
        y_val_pred = model.predict(X_train[val_idx])
        val_metrics = metrics(y_train[val_idx], y_val_pred)
        record = {
            "params": params,
            "metrics": val_metrics,
        }
        search_results.append(record)
        if best_result is None or val_metrics["rmse"] < best_result["metrics"]["rmse"]:
            best_result = record

    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=42,
        **best_result["params"],
    )
    final_model.fit(X_train, y_train)
    return final_model, best_result, search_results


def train_decision_classifier(X_train, y_train):
    if len(np.unique(y_train)) < 2:
        return None

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=3,
        random_state=42,
    )
    classifier.fit(X_train, y_train)
    return classifier


def build_decision_time_energy_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = build_projected_decision_dataframe(
        df,
        baseline_mem_mb=DECISION_BASELINE_MEM,
        baseline_cpu_limit=DECISION_BASELINE_CPU,
        agg_mode=DECISION_REPEAT_AGG_MODE,
    )

    bool_map = {
        "True": 1,
        "False": 0,
        "true": 1,
        "false": 0,
        True: 1,
        False: 0,
        1: 1,
        0: 0,
    }

    required_cols = [
        "mem_limit_mb",
        "cold_start",
        DECISION_GROUP_COL,
        DECISION_WORKLOAD_COL,
        "observed_energy_joules",
        "baseline_energy_1024_joules",
        DECISION_TARGET,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for decision-time energy model: {missing}")

    df["cold_start"] = df["cold_start"].map(bool_map)
    df["mem_limit_mb"] = pd.to_numeric(df["mem_limit_mb"], errors="coerce")
    df["observed_energy_joules"] = pd.to_numeric(df["observed_energy_joules"], errors="coerce")
    df["baseline_energy_1024_joules"] = pd.to_numeric(df["baseline_energy_1024_joules"], errors="coerce")
    df[DECISION_TARGET] = pd.to_numeric(df[DECISION_TARGET], errors="coerce")
    if "baseline_service_time_ms" in df.columns:
        df["baseline_service_time_ms"] = pd.to_numeric(df["baseline_service_time_ms"], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)
    df = df[df["mem_limit_mb"] > 0]

    df["cold_start"] = df["cold_start"].astype(int)
    source_workload_type = df[DECISION_WORKLOAD_TYPE_COL] if DECISION_WORKLOAD_TYPE_COL in df.columns else None
    if source_workload_type is not None and not source_workload_type.isna().any():
        df[DECISION_WORKLOAD_TYPE_COL] = source_workload_type.astype(str)
    else:
        df[DECISION_WORKLOAD_TYPE_COL] = df[DECISION_WORKLOAD_COL].astype(str).map(WORKLOAD_TYPE_MAP)
    if df[DECISION_WORKLOAD_TYPE_COL].isna().any():
        missing_workloads = sorted(
            df.loc[df[DECISION_WORKLOAD_TYPE_COL].isna(), DECISION_WORKLOAD_COL].astype(str).unique().tolist()
        )
        raise ValueError(f"Missing workload_type mapping for: {missing_workloads}")

    workload_type_map = (
        df[[DECISION_WORKLOAD_COL, DECISION_WORKLOAD_TYPE_COL]]
        .drop_duplicates()
        .sort_values([DECISION_WORKLOAD_TYPE_COL, DECISION_WORKLOAD_COL])
    )
    workload_type_map = dict(
        zip(
            workload_type_map[DECISION_WORKLOAD_COL].astype(str),
            workload_type_map[DECISION_WORKLOAD_TYPE_COL].astype(str),
        )
    )

    workload_baseline_map = (
        df.groupby(DECISION_WORKLOAD_COL)["baseline_energy_1024_joules"]
        .median()
        .to_dict()
    )
    missing_baselines = sorted(
        set(df[DECISION_WORKLOAD_COL].astype(str).unique()) - set(workload_baseline_map)
    )
    if missing_baselines:
        raise ValueError(
            f"Missing {DECISION_BASELINE_MEM}MB baselines for decision-time target: {missing_baselines}"
        )
    df["baseline_energy_1024_joules"] = df[DECISION_WORKLOAD_COL].astype(str).map(workload_baseline_map)

    feature_df = build_resource_decision_feature_df(df, WORKLOAD_TYPE_MAP)

    observed_energy = pd.to_numeric(df["observed_energy_joules"], errors="coerce").fillna(0.0)
    baseline_energy = pd.to_numeric(df["baseline_energy_1024_joules"], errors="coerce").fillna(0.0)
    service_time_ms = derive_service_time_ms(df)
    baseline_service_time = pd.to_numeric(df.get("baseline_service_time_ms", service_time_ms), errors="coerce").fillna(service_time_ms)
    mem_limit = pd.to_numeric(df["mem_limit_mb"], errors="coerce").fillna(DECISION_BASELINE_MEM)
    cpu_limit = pd.to_numeric(df.get("cpu_limit", DECISION_BASELINE_CPU), errors="coerce").fillna(DECISION_BASELINE_CPU)
    peak_mem_util = pd.to_numeric(df.get("memory_peak_util_pct", 0.0), errors="coerce").fillna(0.0)
    oom_events = pd.to_numeric(df.get("memory_oom_events", 0.0), errors="coerce").fillna(0.0)
    oom_kill_events = pd.to_numeric(df.get("memory_oom_kill_events", 0.0), errors="coerce").fillna(0.0)

    savings_pct = ((baseline_energy - observed_energy) / np.maximum(baseline_energy, 1e-9)) * 100.0
    non_baseline_mask = ~(
        (np.isclose(mem_limit.astype(float), DECISION_BASELINE_MEM))
        & (np.isclose(cpu_limit.astype(float), DECISION_BASELINE_CPU))
    )
    slo_safe_mask = service_time_ms <= (baseline_service_time * DECISION_SLO_MULTIPLIER)
    resource_safe_mask = (peak_mem_util < 98.0) & (oom_events <= 0.0) & (oom_kill_events <= 0.0)
    beneficial_mask = (
        non_baseline_mask
        & slo_safe_mask
        & resource_safe_mask
        & (savings_pct >= DECISION_BENEFICIAL_MIN_SAVINGS_PCT)
    )

    X = feature_df.astype(float).to_numpy()
    y = df[DECISION_TARGET].astype(float).to_numpy()
    y_beneficial = beneficial_mask.astype(int).to_numpy()
    groups = df[DECISION_GROUP_COL].astype(str).to_numpy()
    features = feature_df.columns.tolist()

    return X, y, y_beneficial, groups, features, workload_type_map, workload_baseline_map


def train_decision_time_energy_model():
    X, y, y_beneficial, groups, features, workload_type_map, workload_baseline_map = build_decision_time_energy_dataset(PREPARED_FILE)

    if len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
    else:
        idx = np.arange(len(X))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_beneficial_train, y_beneficial_test = y_beneficial[train_idx], y_beneficial[test_idx]

    model, best_search_result, search_results = select_decision_model(
        X_train,
        y_train,
        groups[train_idx],
    )

    y_pred = model.predict(X_test)
    energy_metrics = metrics(y_test, y_pred)

    dump(model, os.path.join(MODEL_DIR, DECISION_ENERGY_MODEL_FILE))

    classifier = train_decision_classifier(X_train, y_beneficial_train)
    classifier_metrics = None
    if classifier is not None:
        classifier_prob = classifier.predict_proba(X_test)[:, 1]
        classifier_pred = (classifier_prob >= DECISION_BENEFICIAL_THRESHOLD).astype(int)
        classifier_metrics = classification_metrics(y_beneficial_test, classifier_pred)
        dump(classifier, os.path.join(MODEL_DIR, DECISION_BENEFICIAL_MODEL_FILE))

    decision_meta = {
        "model_file": DECISION_ENERGY_MODEL_FILE,
        "classifier_model_file": DECISION_BENEFICIAL_MODEL_FILE if classifier is not None else None,
        "target": DECISION_TARGET,
        "description": "Resource-aware decision-time energy delta model trained on repeated-run baseline-to-target projections. Features come from the default baseline telemetry projected onto the candidate action, and the target is the observed aggregated run-level energy delta versus the default baseline. An auxiliary classifier predicts whether a candidate action is beneficial and SLO-safe.",
        "features": features,
        "workload_type_map": workload_type_map,
        "workload_baseline_mem_mb": DECISION_BASELINE_MEM,
        "workload_baseline_cpu_limit": DECISION_BASELINE_CPU,
        "workload_baseline_map": workload_baseline_map,
        "group_column": DECISION_GROUP_COL,
        "repeat_aggregation_mode": DECISION_REPEAT_AGG_MODE,
        "beneficial_min_savings_pct": DECISION_BENEFICIAL_MIN_SAVINGS_PCT,
        "beneficial_probability_threshold": DECISION_BENEFICIAL_THRESHOLD,
        "decision_slo_multiplier": DECISION_SLO_MULTIPLIER,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "selected_params": best_search_result["params"],
        "validation_metrics": best_search_result["metrics"],
        "search_results": search_results,
        "metrics": energy_metrics,
        "classifier_metrics": classifier_metrics,
        "n_beneficial_train": int(y_beneficial_train.sum()),
        "n_beneficial_test": int(y_beneficial_test.sum()),
    }
    with open(os.path.join(MODEL_DIR, DECISION_ENERGY_META_FILE), "w") as f:
        json.dump(decision_meta, f, indent=2)

    return decision_meta

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

    # ---- Train latency model(s) ----
    if cold_train.any() and warm_train.any():
        m_cold = fit_log_model(X_train[cold_train], yL_train[cold_train])
        m_warm = fit_log_model(X_train[warm_train], yL_train[warm_train])

        pred_cold = predict_ms(m_cold, X_test[cold_test]) if cold_test.any() else np.array([])
        pred_warm = predict_ms(m_warm, X_test[warm_test]) if warm_test.any() else np.array([])

        y_pred_all = np.empty_like(yL_test, dtype=float)
        if cold_test.any():
            y_pred_all[cold_test] = pred_cold
        if warm_test.any():
            y_pred_all[warm_test] = pred_warm
    else:
        # Benchmark-only datasets can be all warm starts. Fall back to one
        # latency model rather than failing on an empty split.
        fallback_model = fit_log_model(X_train, yL_train)
        m_cold = fallback_model
        m_warm = fallback_model
        y_pred_all = predict_ms(fallback_model, X_test)

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

    decision_energy_meta = train_decision_time_energy_model()

    print("\n==== Evaluation Results ====")
    print("Latency OVERALL MAE/RMSE:", overall)
    print("Latency COLD    MAE/RMSE:", cold_m)
    print("Latency WARM    MAE/RMSE:", warm_m)
    print("Energy          MAE/RMSE:", energy_m)
    print("Energy (Decision-Time) MAE/RMSE:", decision_energy_meta["metrics"])
    print("Decision Beneficial Classifier:", decision_energy_meta["classifier_metrics"])

    # Save models
    dump(m_cold, os.path.join(MODEL_DIR, "latency_cold_hgbdt.joblib"))
    dump(m_warm, os.path.join(MODEL_DIR, "latency_warm_hgbdt.joblib"))
    dump(energy_model, os.path.join(MODEL_DIR, "energy_hgbdt.joblib"))

    out = {
        "latency_overall": overall,
        "latency_cold": cold_m,
        "latency_warm": warm_m,
        "energy": energy_m,
        "energy_decision_time": decision_energy_meta["metrics"],
        "energy_decision_time_model": DECISION_ENERGY_MODEL_FILE,
        "energy_decision_time_classifier": decision_energy_meta["classifier_model_file"],
        "energy_decision_time_meta": DECISION_ENERGY_META_FILE,
        "energy_decision_time_features": decision_energy_meta["features"],
        "latency_target": "log1p(duration_ms) with cold/warm split",
        "prediction_postprocess": f"clip log to [{float(LOG_MIN):.3f},{float(LOG_MAX):.3f}] then expm1",
        "features": feats,
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] Models saved to: {os.path.abspath(MODEL_DIR)}")
    print(f"[OK] Metrics saved to: {os.path.abspath(os.path.join(MODEL_DIR,'metrics.json'))}")
    print(f"[OK] Decision-time energy model saved to: {os.path.abspath(os.path.join(MODEL_DIR, DECISION_ENERGY_MODEL_FILE))}")
    print(f"[OK] Decision-time energy metadata saved to: {os.path.abspath(os.path.join(MODEL_DIR, DECISION_ENERGY_META_FILE))}")

if __name__ == "__main__":
    main()
