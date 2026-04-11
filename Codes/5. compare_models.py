import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from plot_style import (
    PAPER_DPI,
    apply_paper_style,
    finalize_figure,
    make_figure,
    maybe_show,
    prompt_filename,
    prompt_yes_no,
    style_axes,
    wrap_label,
)

DATA_DIR = os.environ.get("COMPARE_DATASET_DIR", "dataset_reg_known_full_plusfb")
OUTPUT_CSV = os.environ.get("COMPARE_OUTPUT_CSV", "model_comparison_full_plusfb.csv")


def load_array_with_fallback(preferred_name, fallback_name):
    preferred_path = os.path.join(DATA_DIR, preferred_name)
    if os.path.exists(preferred_path):
        return np.load(preferred_path)
    return np.load(os.path.join(DATA_DIR, fallback_name))


def evaluate(target_name, name, model, X_train, y_train, X_test, y_test):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "Target": target_name,
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "Train Time (s)": train_time
    }


def add_bar_labels(ax, rects, fontsize=16):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold"
        )


def plot_comparison(df, target_name, unit):

    df_sub = df[df["Target"] == target_name].copy()

    model_order = ["Linear Regression", "Random Forest", "HGBDT (Ours)"]
    df_sub["Model"] = pd.Categorical(df_sub["Model"], categories=model_order, ordered=True)
    df_sub = df_sub.sort_values("Model")

    models = df_sub["Model"].tolist()
    mae_scores = df_sub["MAE"].to_numpy()
    rmse_scores = df_sub["RMSE"].to_numpy()

    x = np.arange(len(models))
    width = 0.34

    fig, ax = make_figure()

    rects1 = ax.bar(
        x - width / 2,
        mae_scores,
        width,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.0,
        hatch="///",
        label=f"MAE ({unit})"
    )

    rects2 = ax.bar(
        x + width / 2,
        rmse_scores,
        width,
        color="#C44E52",
        edgecolor="black",
        linewidth=1.0,
        hatch="xxx",
        label=f"RMSE ({unit})"
    )

    style_axes(
        ax,
        title=f"{target_name} Prediction Accuracy",
        xlabel="Model",
        ylabel=f"Error ({unit})",
        title_width=28,
        title_pad=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(name, width=18) for name in models], rotation=10, ha="right")
    max_value = max(np.max(mae_scores), np.max(rmse_scores))
    ax.set_ylim(0, max_value * 1.35)

    ax.grid(False)

    add_bar_labels(ax, rects1, fontsize=15)
    add_bar_labels(ax, rects2, fontsize=15)

    legend_handles = [
        Patch(facecolor="#4C72B0", edgecolor="black", hatch="///", label=f"MAE ({unit})"),
        Patch(facecolor="#C44E52", edgecolor="black", hatch="xxx", label=f"RMSE ({unit})")
    ]

    ax.legend(
        handles=legend_handles,
        fontsize=14,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
    )

    finalize_figure(fig, top=0.90)
    return fig


# ---------------- NEW FUNCTION ----------------
def plot_training_overhead(df):
    """
    Uses Energy training time for comparison
    (one value per model).
    Color + grayscale friendly version.
    """

    df_sub = df[df["Target"] == "Energy"].copy()

    model_order = ["Linear Regression", "Random Forest", "HGBDT (Ours)"]
    df_sub["Model"] = pd.Categorical(df_sub["Model"], categories=model_order, ordered=True)
    df_sub = df_sub.sort_values("Model")

    models = df_sub["Model"].tolist()
    train_times = df_sub["Train Time (s)"].to_numpy()

    x = np.arange(len(models))

    fig, ax = make_figure()

    # Strong distinct colors + unique hatch patterns
    styles = [
        {"color": "#4C72B0", "hatch": "///", "edgecolor": "black"},  # Blue
        {"color": "#DD8452", "hatch": "xxx", "edgecolor": "black"},  # Orange
        {"color": "#55A868", "hatch": "...", "edgecolor": "black"},  # Green
    ]

    rects = []
    for i in range(len(models)):
        r = ax.bar(
            x[i],
            train_times[i],
            width=0.55,
            linewidth=1.5,
            **styles[i]
        )
        rects.append(r[0])

    style_axes(
        ax,
        title="Training Overhead Comparison",
        xlabel="Model",
        ylabel="Training Time (seconds)",
        title_width=28,
        title_pad=10,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(name, width=18) for name in models], rotation=10, ha="right")
    ax.set_ylim(0, float(np.max(train_times)) * 1.20)

    ax.grid(False)

    add_bar_labels(ax, rects, fontsize=15)

    finalize_figure(fig, top=0.96)
    return fig
# ------------------------------------------------


def main():

    apply_paper_style()

    print("Loading data...")
    try:
        X_energy_train = load_array_with_fallback("X_energy_train.npy", "X_train.npy")
        X_energy_test = load_array_with_fallback("X_energy_test.npy", "X_test.npy")
        X_latency_train = load_array_with_fallback("X_latency_train.npy", "X_train.npy")
        X_latency_test = load_array_with_fallback("X_latency_test.npy", "X_test.npy")

        yE_train = np.load(os.path.join(DATA_DIR, "y_energy_train.npy"))
        yE_test = np.load(os.path.join(DATA_DIR, "y_energy_test.npy"))

        yL_train = np.load(os.path.join(DATA_DIR, "y_latency_train.npy"))
        yL_test = np.load(os.path.join(DATA_DIR, "y_latency_test.npy"))

    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    model_defs = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("HGBDT (Ours)", HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        ))
    ]

    results = []

    print("--- Evaluating Energy Models ---")
    for name, clf in model_defs:
        results.append(evaluate("Energy", name, clone(clf),
                                X_energy_train, yE_train, X_energy_test, yE_test))

    print("--- Evaluating Latency Models ---")
    for name, clf in model_defs:
        results.append(evaluate("Latency", name, clone(clf),
                                X_latency_train, yL_train, X_latency_test, yL_test))

    df_res = pd.DataFrame(results)

    print("\nMODEL COMPARISON RESULTS\n")
    print(df_res.to_string(index=False))

    df_res.to_csv(OUTPUT_CSV, index=False)

    print("\nGenerating plots...")

    fig1 = plot_comparison(df_res, "Energy", "Joules")
    fig2 = plot_comparison(df_res, "Latency", "ms")
    fig3 = plot_training_overhead(df_res)

    maybe_show()

    save_plots = prompt_yes_no("Do you want to save the plots? (y/n): ", default="n")
    if save_plots == "y":
        energy_file = prompt_filename("Enter energy plot filename", "fig_benchmark_energy.png")
        latency_file = prompt_filename("Enter latency plot filename", "fig_benchmark_latency.png")
        time_file = prompt_filename("Enter training overhead filename", "fig_model_comparison_time.png")

        fig1.savefig(energy_file, dpi=PAPER_DPI, bbox_inches="tight")
        fig2.savefig(latency_file, dpi=PAPER_DPI, bbox_inches="tight")
        fig3.savefig(time_file, dpi=PAPER_DPI, bbox_inches="tight")

        print(f"[OK] Saved {energy_file}")
        print(f"[OK] Saved {latency_file}")
        print(f"[OK] Saved {time_file}")
    else:
        print("Plots were not saved.")


if __name__ == "__main__":
    main()
