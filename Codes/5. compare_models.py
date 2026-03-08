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

    ax.tick_params(axis="y", labelsize=19)
    ax.tick_params(axis="x", labelsize=17)

    ax.grid(False)

    add_bar_labels(ax, rects1, fontsize=17)
    add_bar_labels(ax, rects2, fontsize=17)

    legend_handles = [
        Patch(facecolor="#4C72B0", edgecolor="black", hatch="///", label=f"MAE ({unit})"),
        Patch(facecolor="#C44E52", edgecolor="black", hatch="xxx", label=f"RMSE ({unit})")
    ]

    ax.legend(handles=legend_handles, fontsize=18, frameon=True)

    plt.tight_layout()
    return fig


def main():

    # Global MICRO-style font configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 21,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18
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

    plt.show()

    save = input("Do you want to save the plots? (y/n): ").strip().lower()
    if save == "y":
        fig1.savefig("fig_benchmark_energy.png", dpi=300, bbox_inches="tight")
        fig2.savefig("fig_benchmark_latency.png", dpi=300, bbox_inches="tight")
        print("[OK] Plots saved.")
    else:
        print("Plots not saved.")


if __name__ == "__main__":
    main()