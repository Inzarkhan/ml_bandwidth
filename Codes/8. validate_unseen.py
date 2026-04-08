import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from matplotlib.patches import Patch

from resource_decision_features import (
    aggregate_repeated_measurements,
    aggregate_policy_rows,
    build_resource_decision_feature_df,
    compute_resource_safety_penalty,
    derive_service_time_ms,
    project_baseline_rows_to_memory,
)


def parse_int_list_env(name, default_values):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default_values
    return [int(item.strip()) for item in raw.split(",") if item.strip()]

# CONFIGURATION
INPUT_FILE = os.environ.get("EVAL_UNSEEN_INPUT", "raw_sebs_unseen.jsonl")
POLICY_FILE = os.environ.get("EVAL_UNSEEN_POLICY_INPUT", INPUT_FILE)
ACTUAL_FILE = os.environ.get("EVAL_UNSEEN_ACTUAL_INPUT", INPUT_FILE)
MODEL_FILE = os.environ.get("EVAL_MODEL_FILE", "models/energy_hgbdt_decision.joblib")
MODEL_META_FILE = os.environ.get("EVAL_MODEL_META_FILE", "models/energy_hgbdt_decision_meta.json")
DEFAULT_MEM = 1024
DEFAULT_CPU = float(os.environ.get("SEBS_POLICY_CURRENT_CPU_LIMIT", "1.0"))
COLD_START_DEFAULT = 0
SLO_MULTIPLIER = float(os.environ.get("SEBS_SLO_MULTIPLIER", "1.30"))
UNSEEN_WORKLOAD_TYPE_MAP = {
    "sebs_graph_mst_unseen": "scientific",
    "sebs_uploader_unseen": "web",
    "sebs_video_processing_unseen": "multimedia",
    "sebs_dna_visualisation_unseen": "scientific",
}
WORKLOAD_LABELS = {
    "sebs_graph_mst_unseen": "Scientific (graph-mst)",
    "sebs_uploader_unseen": "Web (uploader)",
    "sebs_video_processing_unseen": "Multimedia (video-processing)",
    "sebs_dna_visualisation_unseen": "Scientific (dna-visualisation)",
}
MEMORY_OPTIONS = parse_int_list_env("SEBS_MEMORY_SIZES", [128, 256, 384, 512, 768, 1024])
CURRENT_MEM_MB = int(os.environ.get("SEBS_POLICY_CURRENT_MEM_MB", "1024"))
CPU_OPTIONS = [
    float(item.strip())
    for item in os.environ.get("SEBS_CPU_LIMITS", "1.0").split(",")
    if item.strip()
]
SWITCH_MARGIN_J = float(os.environ.get("SEBS_POLICY_SWITCH_MARGIN_J", "0.01"))
SWITCH_MARGIN_PCT = float(os.environ.get("SEBS_POLICY_SWITCH_MARGIN_PCT", "5.0"))
PRESSURED_SWITCH_MARGIN_PCT = float(os.environ.get("SEBS_POLICY_PRESSURED_SWITCH_MARGIN_PCT", "1.0"))
PRESSURED_PEAK_UTIL_PCT = float(os.environ.get("SEBS_POLICY_PRESSURED_PEAK_UTIL_PCT", "60.0"))
REPEAT_AGG_MODE = os.environ.get("EVAL_REPEAT_AGG_MODE", "median")
CLASSIFIER_FILE = os.environ.get(
    "EVAL_CLASSIFIER_FILE",
    os.path.join(os.path.dirname(MODEL_FILE), "energy_hgbdt_decision_classifier.joblib"),
)

def add_bar_labels(ax, rects, fontsize=16, suffix=""):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}{suffix}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold"
        )


def load_model_meta():
    with open(MODEL_META_FILE, "r") as f:
        return json.load(f)


def load_dataframe_auto(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)

    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data)


def build_feature_matrix(df, feature_names, workload_type_map):
    if "cold_start" not in df.columns:
        df = df.copy()
        df["cold_start"] = COLD_START_DEFAULT
    feature_df = build_resource_decision_feature_df(df, workload_type_map, feature_names=feature_names)
    return feature_df.to_numpy(dtype=float)


def choose_hgbdt_action(model, classifier, feature_names, workload_type_map, model_meta, workload_df):
    workload_df = aggregate_repeated_measurements(workload_df, agg_mode=REPEAT_AGG_MODE)
    candidate_scores = {}
    baseline_rows = workload_df[
        (pd.to_numeric(workload_df["mem_limit_mb"], errors="coerce") == CURRENT_MEM_MB)
        & (pd.to_numeric(workload_df.get("cpu_limit", DEFAULT_CPU), errors="coerce").fillna(DEFAULT_CPU) == DEFAULT_CPU)
    ].copy()
    if not baseline_rows.empty:
        baseline_service_time_ms = float(derive_service_time_ms(baseline_rows).median())
        baseline_energy = float(pd.to_numeric(baseline_rows["energy_joules"], errors="coerce").dropna().median())
    else:
        baseline_service_time_ms = None
        baseline_energy = None

    baseline_cpu_limit = float(model_meta.get("workload_baseline_cpu_limit", DEFAULT_CPU))
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
            if classifier is not None and not (mem == CURRENT_MEM_MB and float(cpu_limit) == float(DEFAULT_CPU)):
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

    baseline_key = (CURRENT_MEM_MB, float(DEFAULT_CPU))
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


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    try:
        policy_df = load_dataframe_auto(POLICY_FILE)
        actual_df = load_dataframe_auto(ACTUAL_FILE)
    except Exception as e:
        print(f"Error reading unseen evaluation inputs: {e}")
        return

    try:
        model = joblib.load(MODEL_FILE)
        model_meta = load_model_meta()
        feature_names = model_meta["features"]
        workload_type_map = {
            **model_meta.get("workload_type_map", {}),
            **UNSEEN_WORKLOAD_TYPE_MAP,
        }
        classifier = joblib.load(CLASSIFIER_FILE) if os.path.exists(CLASSIFIER_FILE) else None
    except Exception as e:
        print(f"Error loading HGBDT model inputs: {e}")
        return

    # Use median to reduce spike effects
    policy_df = aggregate_repeated_measurements(policy_df, agg_mode=REPEAT_AGG_MODE)
    grouped = aggregate_repeated_measurements(actual_df, agg_mode=REPEAT_AGG_MODE)

    print("\n--- HEAVY GENERALIZATION TEST (MEDIAN FILTERED) ---")
    print(f"{'WORKLOAD':<22} | {'DEFAULT (1024MB/1.0)':<22} | {'HgBDT CHOICE':<24} | {'SAVINGS':<10}")
    print("-" * 88)

    total_default = 0
    total_hgbdt = 0
    workloads = sorted(policy_df["workload"].unique())

    results = {"workload": [], "display_name": [], "plot_workload_name": [], "Default": [], "HgBDT": []}

    for wl in workloads:
        workload_df = policy_df[policy_df["workload"] == wl].copy()
        display_name = workload_df["display_name"].dropna().iloc[0] if "display_name" in workload_df.columns and workload_df["display_name"].notna().any() else WORKLOAD_LABELS.get(wl, wl.replace("_", " ").title())
        plot_workload_name = (
            workload_df["plot_workload_name"].dropna().iloc[0]
            if "plot_workload_name" in workload_df.columns and workload_df["plot_workload_name"].notna().any()
            else display_name
        )

        def_row = grouped[
            (grouped["workload"] == wl) &
            (pd.to_numeric(grouped["mem_limit_mb"], errors="coerce") == DEFAULT_MEM) &
            ((pd.to_numeric(grouped["cpu_limit"], errors="coerce").fillna(DEFAULT_CPU) if "cpu_limit" in grouped.columns else pd.Series(DEFAULT_CPU, index=grouped.index)) == DEFAULT_CPU)
        ]
        if def_row.empty:
            continue
        def_energy = def_row["energy_joules"].values[0]

        choice_mem, choice_cpu, _ = choose_hgbdt_action(model, classifier, feature_names, workload_type_map, model_meta, workload_df)
        if choice_mem is None:
            continue

        opt_row = grouped[
            (grouped["workload"] == wl) &
            (pd.to_numeric(grouped["mem_limit_mb"], errors="coerce") == choice_mem) &
            ((pd.to_numeric(grouped["cpu_limit"], errors="coerce").fillna(choice_cpu) if "cpu_limit" in grouped.columns else pd.Series(choice_cpu, index=grouped.index)) == choice_cpu)
        ]
        if opt_row.empty:
            continue
        opt_energy = opt_row["energy_joules"].values[0]

        savings = def_energy - opt_energy

        print(f"{plot_workload_name:<22} | {def_energy:.2f} J               | {opt_energy:.2f} J ({choice_mem}MB/{choice_cpu:.2f}CPU) | {savings:+.2f} J")

        total_default += def_energy
        total_hgbdt += opt_energy

        results["workload"].append(wl)
        results["display_name"].append(display_name)
        results["plot_workload_name"].append(plot_workload_name)
        results["Default"].append(def_energy)
        results["HgBDT"].append(opt_energy)

    print("-" * 88)

    if total_default == 0:
        print("No valid data found for plotting.")
        return

    total_savings_pct = (total_default - total_hgbdt) / total_default * 100
    print(f"TOTAL SAVINGS: {total_savings_pct:.1f}% ({total_default - total_hgbdt:.2f} J)")

    # Plot
    plot_labels = results["plot_workload_name"]
    x = np.arange(len(results["workload"]))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    rects1 = ax.bar(
        x - width / 2,
        results["Default"],
        width,
        color="#7F7F7F",
        edgecolor="black",
        linewidth=1.0,
        hatch="///",
        label="Default (1024MB)"
    )

    rects2 = ax.bar(
        x + width / 2,
        results["HgBDT"],
        width,
        color="#55A868",
        edgecolor="black",
        linewidth=1.0,
        hatch="xxx",
        label="HgBDT Choice"
    )

    ax.set_title("Unseen Energy Comparison", fontsize=22, pad=12)
    ax.set_ylabel("Energy (Joules)", fontsize=21, labelpad=8)
    ax.set_xlabel("Workload", fontsize=21, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, rotation=12, ha="right")
    max_value = max(results["Default"] + results["HgBDT"])
    ax.set_ylim(0, max_value * 1.15)

    ax.grid(False)

    add_bar_labels(ax, rects1, fontsize=17)
    add_bar_labels(ax, rects2, fontsize=17)

    legend_handles = [
        Patch(facecolor="#7F7F7F", edgecolor="black", hatch="///", label="Default (1024MB)"),
        Patch(facecolor="#55A868", edgecolor="black", hatch="xxx", label="HgBDT Choice")
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=18,
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.90),
    )

    plt.tight_layout()
    plt.show()

    save_plot = input("Do you want to save the plot? (y/n): ").strip().lower()
    if save_plot == "y":
        filename = input("Enter filename [fig_heavy_generalization.png]: ").strip()
        if not filename:
            filename = "fig_heavy_generalization.png"
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[OK] Saved plot: {filename}")
    else:
        print("Plot not saved.")

if __name__ == "__main__":
    main()
