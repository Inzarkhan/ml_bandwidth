import json
import os

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from resource_decision_features import (
    aggregate_repeated_measurements,
    aggregate_policy_rows,
    build_resource_decision_feature_df,
    compute_resource_safety_penalty,
    derive_service_time_ms,
    project_baseline_rows_to_memory,
)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})


def parse_int_list_env(name, default_values):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default_values
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

DATA_FILE = os.environ.get("EVAL_KNOWN_CSV", "prepared.csv")
POLICY_FILE = os.environ.get("EVAL_KNOWN_POLICY_FILE", DATA_FILE)
ACTUAL_FILE = os.environ.get("EVAL_KNOWN_ACTUAL_FILE", "")
MODEL_FILE = os.environ.get("EVAL_MODEL_FILE", "models/energy_hgbdt_decision.joblib")
MODEL_META_FILE = os.environ.get("EVAL_MODEL_META_FILE", "models/energy_hgbdt_decision_meta.json")
MEMORY_OPTIONS = parse_int_list_env("SEBS_MEMORY_SIZES", [128, 256, 384, 512, 768, 1024])
CPU_OPTIONS = [
    float(item.strip())
    for item in os.environ.get("SEBS_CPU_LIMITS", "1.0").split(",")
    if item.strip()
]
SLO_MULTIPLIER = float(os.environ.get("SEBS_SLO_MULTIPLIER", "1.30"))
CURRENT_MEM_MB = int(os.environ.get("SEBS_POLICY_CURRENT_MEM_MB", "1024"))
CURRENT_CPU_LIMIT = float(os.environ.get("SEBS_POLICY_CURRENT_CPU_LIMIT", "1.0"))
SWITCH_MARGIN_J = float(os.environ.get("SEBS_POLICY_SWITCH_MARGIN_J", "0.01"))
SWITCH_MARGIN_PCT = float(os.environ.get("SEBS_POLICY_SWITCH_MARGIN_PCT", "5.0"))
PRESSURED_SWITCH_MARGIN_PCT = float(os.environ.get("SEBS_POLICY_PRESSURED_SWITCH_MARGIN_PCT", "1.0"))
PRESSURED_PEAK_UTIL_PCT = float(os.environ.get("SEBS_POLICY_PRESSURED_PEAK_UTIL_PCT", "60.0"))
REPEAT_AGG_MODE = os.environ.get("EVAL_REPEAT_AGG_MODE", "median")
CLASSIFIER_FILE = os.environ.get(
    "EVAL_CLASSIFIER_FILE",
    os.path.join(os.path.dirname(MODEL_FILE), "energy_hgbdt_decision_classifier.joblib"),
)

def load_model_meta():
    with open(MODEL_META_FILE, "r") as f:
        return json.load(f)


def load_dataframe_auto(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def build_feature_matrix(df, feature_names, workload_type_map):
    feature_df = build_resource_decision_feature_df(df, workload_type_map, feature_names=feature_names)
    return feature_df.to_numpy(dtype=float)


def choose_hgbdt_action(model, classifier, feature_names, workload_type_map, model_meta, workload_df):
    workload_df = aggregate_repeated_measurements(workload_df, agg_mode=REPEAT_AGG_MODE)
    candidate_scores = {}
    baseline_rows = workload_df[
        (pd.to_numeric(workload_df["mem_limit_mb"], errors="coerce") == CURRENT_MEM_MB)
        & (pd.to_numeric(workload_df.get("cpu_limit", CURRENT_CPU_LIMIT), errors="coerce").fillna(CURRENT_CPU_LIMIT) == CURRENT_CPU_LIMIT)
    ].copy()
    if not baseline_rows.empty:
        baseline_service_time_ms = float(derive_service_time_ms(baseline_rows).median())
        baseline_energy = float(pd.to_numeric(baseline_rows["energy_joules"], errors="coerce").dropna().median())
    else:
        baseline_service_time_ms = None
        baseline_energy = None

    baseline_cpu_limit = float(model_meta.get("workload_baseline_cpu_limit", CURRENT_CPU_LIMIT))
    beneficial_threshold = float(model_meta.get("beneficial_probability_threshold", 0.60))

    for cpu_limit in CPU_OPTIONS:
        for mem in MEMORY_OPTIONS:
            if baseline_rows.empty:
                continue
            candidate_runs = project_baseline_rows_to_memory(
                baseline_rows,
                target_mem_mb=mem,
                baseline_mem_mb=CURRENT_MEM_MB,
                target_cpu_limit=cpu_limit,
                baseline_cpu_limit=baseline_cpu_limit,
            )
            if candidate_runs.empty:
                continue

            X_candidate = build_feature_matrix(candidate_runs, feature_names, workload_type_map)
            predicted_delta = float(np.median(model.predict(X_candidate)))
            absolute_predicted_energy = float((baseline_energy or 0.0) + predicted_delta)
            safety_penalty = compute_resource_safety_penalty(
                candidate_runs,
                slo_reference_ms=baseline_service_time_ms,
                slo_multiplier=SLO_MULTIPLIER,
                current_mem_mb=CURRENT_MEM_MB,
            )
            beneficial_proba = 1.0
            if classifier is not None and not (mem == CURRENT_MEM_MB and float(cpu_limit) == float(CURRENT_CPU_LIMIT)):
                beneficial_proba = float(np.median(classifier.predict_proba(X_candidate)[:, 1]))
                if beneficial_proba < beneficial_threshold:
                    continue

            candidate_scores[(mem, float(cpu_limit))] = {
                "predicted_delta": predicted_delta,
                "predicted_energy": absolute_predicted_energy,
                "score": absolute_predicted_energy + safety_penalty,
                "projected_peak_util_pct": float(pd.to_numeric(candidate_runs["memory_peak_util_pct"], errors="coerce").median()) if "memory_peak_util_pct" in candidate_runs.columns else 0.0,
                "beneficial_proba": beneficial_proba,
            }

    if not candidate_scores:
        return None, None, None

    baseline_key = (CURRENT_MEM_MB, float(CURRENT_CPU_LIMIT))
    if baseline_key not in candidate_scores and not baseline_rows.empty:
        X_baseline = build_feature_matrix(baseline_rows, feature_names, workload_type_map)
        baseline_predicted_delta = float(np.median(model.predict(X_baseline)))
        baseline_predicted_energy = float((baseline_energy or 0.0) + baseline_predicted_delta)
        baseline_penalty = compute_resource_safety_penalty(
            baseline_rows,
            slo_reference_ms=baseline_service_time_ms,
            slo_multiplier=SLO_MULTIPLIER,
            current_mem_mb=CURRENT_MEM_MB,
        )
        candidate_scores[baseline_key] = {
            "predicted_delta": baseline_predicted_delta,
            "predicted_energy": baseline_predicted_energy,
            "score": baseline_predicted_energy + baseline_penalty,
            "projected_peak_util_pct": float(pd.to_numeric(baseline_rows["memory_peak_util_pct"], errors="coerce").median()) if "memory_peak_util_pct" in baseline_rows.columns else 0.0,
            "beneficial_proba": 1.0,
        }

    best_action = min(candidate_scores, key=lambda action: (candidate_scores[action]["score"], action[0], action[1]))
    baseline_score = candidate_scores.get(baseline_key, {}).get("score")
    best_score = candidate_scores[best_action]["score"]
    best_peak_util_pct = candidate_scores[best_action].get("projected_peak_util_pct", 0.0)
    required_pct_margin = (
        PRESSURED_SWITCH_MARGIN_PCT
        if best_peak_util_pct >= PRESSURED_PEAK_UTIL_PCT
        else SWITCH_MARGIN_PCT
    )
    if (
        baseline_score is not None
        and best_action != baseline_key
        and (
            (baseline_score - best_score) < SWITCH_MARGIN_J
            or (
                baseline_score > 0
                and ((baseline_score - best_score) / baseline_score) * 100.0 < required_pct_margin
            )
        )
    ):
        best_action = baseline_key
    return best_action[0], best_action[1], candidate_scores[best_action]["predicted_energy"]


# 1. Load policy data and real aggregate energy data
policy_df = load_dataframe_auto(POLICY_FILE)
actual_df = load_dataframe_auto(ACTUAL_FILE) if ACTUAL_FILE else policy_df.copy()
policy_df = aggregate_repeated_measurements(policy_df, agg_mode=REPEAT_AGG_MODE)
actual_df = aggregate_repeated_measurements(actual_df, agg_mode=REPEAT_AGG_MODE)

# Load the trained model
model = joblib.load(MODEL_FILE)
model_meta = load_model_meta()
feature_names = model_meta["features"]
workload_type_map = model_meta["workload_type_map"]
classifier = joblib.load(CLASSIFIER_FILE) if os.path.exists(CLASSIFIER_FILE) else None

# 2. Define the Candidates
workloads = sorted(policy_df['workload'].unique())

total_energy_default = 0.0
total_energy_hgbdt = 0.0

results = []

print(f"{'WORKLOAD':<32} | {'DEFAULT (1024MB/1.0)':<22} | {'HgBDT CHOICE':<24} | {'SAVINGS':<10}")
print("-" * 92)

# 3. The Loop
for wl in workloads:
    workload_df = policy_df[policy_df['workload'] == wl].copy()
    actual_workload_df = actual_df[actual_df['workload'] == wl].copy()
    display_name = workload_df["display_name"].dropna().iloc[0] if "display_name" in workload_df.columns and workload_df["display_name"].notna().any() else wl
    plot_workload_name = (
        workload_df["plot_workload_name"].dropna().iloc[0]
        if "plot_workload_name" in workload_df.columns and workload_df["plot_workload_name"].notna().any()
        else display_name
    )

    # Get Real Average Energy for this workload at 1024MB (Baseline)
    default_runs = actual_workload_df[
        (pd.to_numeric(actual_workload_df['mem_limit_mb'], errors='coerce') == CURRENT_MEM_MB)
        & ((pd.to_numeric(actual_workload_df['cpu_limit'], errors='coerce').fillna(CURRENT_CPU_LIMIT) if 'cpu_limit' in actual_workload_df.columns else pd.Series(CURRENT_CPU_LIMIT, index=actual_workload_df.index)) == CURRENT_CPU_LIMIT)
    ]
    if len(default_runs) == 0:
        continue
    real_energy_default = pd.to_numeric(default_runs['energy_joules'], errors='coerce').median()

    # Use the trained HGBDT model to choose the memory size,
    # then evaluate the choice with the real measured energy.
    optimal_mem, optimal_cpu, _ = choose_hgbdt_action(model, classifier, feature_names, workload_type_map, model_meta, workload_df)
    if optimal_mem is None:
        continue

    optimal_runs = actual_workload_df[
        (pd.to_numeric(actual_workload_df['mem_limit_mb'], errors='coerce') == optimal_mem)
        & ((pd.to_numeric(actual_workload_df['cpu_limit'], errors='coerce').fillna(optimal_cpu) if 'cpu_limit' in actual_workload_df.columns else pd.Series(optimal_cpu, index=actual_workload_df.index)) == optimal_cpu)
    ]
    real_energy_optimal = pd.to_numeric(optimal_runs['energy_joules'], errors='coerce').median()

    # Add to totals
    total_energy_default += real_energy_default
    total_energy_hgbdt += real_energy_optimal

    saving = real_energy_default - real_energy_optimal

    print(
        f"{plot_workload_name:<32} | {real_energy_default:.2f} J               | "
        f"{real_energy_optimal:.2f} J ({optimal_mem}MB/{optimal_cpu:.2f}CPU) | {saving:+.2f} J"
    )
    results.append(
        {
            "workload": wl,
            "display_name": display_name,
            "plot_workload_name": plot_workload_name,
            "Default": real_energy_default,
            "HgBDT": real_energy_optimal,
        }
    )

# 4. Final Results
print("-" * 92)
total_savings = total_energy_default - total_energy_hgbdt
percent_savings = (total_savings / total_energy_default) * 100

print(f"TOTAL ENERGY (Default):  {total_energy_default:.2f} Joules")
print(f"TOTAL ENERGY (HgBDT):    {total_energy_hgbdt:.2f} Joules")
print(f"REAL SAVINGS (Per Run):  {total_savings:.2f} Joules")
print(f"IMPROVEMENT:             {percent_savings:.1f}%")

# --- NEW: PROJECTION SECTION (Longer Time Interval) ---
print("\n" + "="*40)
print("      PROJECTED LONG-TERM SAVINGS      ")
print("="*40)
# Assuming 1 Million Invocations (Standard Cloud Scale)
scale_factor = 1_000_000
saved_kwh = (total_savings * scale_factor) / 3_600_000  # Convert Joules to kWh

print(f"If running 1 Million Invocations:")
print(f"  - Energy Saved: {total_savings * scale_factor:,.0f} Joules")
print(f"  - Electricity:  {saved_kwh:.2f} kWh saved")
print(f"  - CO2 Equivalent: {saved_kwh * 0.4:.2f} kg CO2e (approx)")
print("="*40)

# 5. Plot
labels = ["Default (1024MB)", "HgBDT-Optimized"]
values = [total_energy_default, total_energy_hgbdt]
fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, color=["gray", "#2ecc71"], edgecolor="black")

ax.set_ylabel("Total Energy (Joules)", fontsize=21, labelpad=8)
ax.set_title(f"Real-World Power Savings\n(-{percent_savings:.1f}% with HgBDT)", fontsize=22, pad=12)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f} J",
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold",
    )

plt.tight_layout()
plt.show()

save_plot = input("Do you want to save the plot? (y/n): ").strip().lower()
if save_plot == "y":
    filename = input("Enter filename [fig_real_savings.png]: ").strip()
    if not filename:
        filename = "fig_real_savings.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved plot: {filename}")
else:
    print("Plot not saved.")
