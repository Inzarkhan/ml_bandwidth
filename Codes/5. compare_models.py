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

DATA_DIR = "dataset_reg"


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

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

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

    ax.set_title(f"{target_name} Prediction Accuracy", fontsize=22, pad=12)
    ax.set_ylabel(f"Error ({unit})", fontsize=21, labelpad=8)
    ax.set_xlabel("Model", fontsize=21, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right")
    if target_name == "Energy":
        ax.set_ylim(0, 7)
    elif target_name == "Latency":
        ax.set_ylim(0, 6)

    ax.grid(False)

    add_bar_labels(ax, rects1, fontsize=17)
    add_bar_labels(ax, rects2, fontsize=17)

    legend_handles = [
        Patch(facecolor="#4C72B0", edgecolor="black", hatch="///", label=f"MAE ({unit})"),
        Patch(facecolor="#C44E52", edgecolor="black", hatch="xxx", label=f"RMSE ({unit})")
    ]

    legend_y = 0.75 if target_name == "Energy" else 0.65

    ax.legend(
        handles=legend_handles,
        fontsize=18,
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.98, legend_y),
    )

    plt.tight_layout()
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

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

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

    ax.set_title("Training Overhead Comparison", fontsize=20, pad=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=21, labelpad=8)
    ax.set_xlabel("Model", fontsize=21, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right")
    ax.set_ylim(0, 1)

    ax.grid(False)

    add_bar_labels(ax, rects, fontsize=17)

    plt.tight_layout()
    return fig
# ------------------------------------------------


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

    print("Loading data...")
    try:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))

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
                                X_train, yE_train, X_test, yE_test))

    print("--- Evaluating Latency Models ---")
    for name, clf in model_defs:
        results.append(evaluate("Latency", name, clone(clf),
                                X_train, yL_train, X_test, yL_test))

    df_res = pd.DataFrame(results)

    print("\nMODEL COMPARISON RESULTS\n")
    print(df_res.to_string(index=False))

    df_res.to_csv("model_comparison_full.csv", index=False)

    print("\nGenerating plots...")

    fig1 = plot_comparison(df_res, "Energy", "Joules")
    fig2 = plot_comparison(df_res, "Latency", "ms")
    fig3 = plot_training_overhead(df_res)

    plt.show()

    save_plots = input("Do you want to save the plots? (y/n): ").strip().lower()
    if save_plots == "y":
        energy_file = input("Enter energy plot filename [fig_benchmark_energy.png]: ").strip()
        latency_file = input("Enter latency plot filename [fig_benchmark_latency.png]: ").strip()
        time_file = input("Enter training overhead filename [fig_model_comparison_time.png]: ").strip()

        if not energy_file:
            energy_file = "fig_benchmark_energy.png"
        if not latency_file:
            latency_file = "fig_benchmark_latency.png"
        if not time_file:
            time_file = "fig_model_comparison_time.png"

        fig1.savefig(energy_file, dpi=300, bbox_inches="tight")
        fig2.savefig(latency_file, dpi=300, bbox_inches="tight")
        fig3.savefig(time_file, dpi=300, bbox_inches="tight")

        print(f"[OK] Saved {energy_file}")
        print(f"[OK] Saved {latency_file}")
        print(f"[OK] Saved {time_file}")
    else:
        print("Plots were not saved.")


if __name__ == "__main__":
    main()
