import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

# Settings
MODEL_DIR = "models"
DATA_DIR = "dataset_reg"

def main():
    # Global paper-style font settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "legend.title_fontsize": 18
    })

    # 1. Load Data
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    yE_test = np.load(os.path.join(DATA_DIR, "y_energy_test.npy"))

    print("Loading Energy Model...")
    energy_model = joblib.load(os.path.join(MODEL_DIR, "energy_hgbdt.joblib"))

    # 2. Predict
    yE_pred = energy_model.predict(X_test)

    # 3. Calculate R²
    correlation = np.corrcoef(yE_test, yE_pred)[0, 1]
    r2 = correlation ** 2
    print(f"Energy R² Score: {r2:.4f}")

    # -----------------------------
    # PLOT 1: PREDICTED VS ACTUAL
    # -----------------------------
    fig1, ax1 = plt.subplots(figsize=(8.5, 7.0))

    ax1.scatter(
        yE_test,
        yE_pred,
        alpha=0.65,
        color="#4C72B0",
        edgecolors="black",
        linewidths=0.5,
        s=50,
        label="Test Samples"
    )

    min_val = min(yE_test.min(), yE_pred.min())
    max_val = max(yE_test.max(), yE_pred.max())

    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2.2,
        color="#C44E52",
        label="Perfect Prediction"
    )

    ax1.set_title(f"Energy Prediction Accuracy (R² = {r2:.3f})", fontsize=22, pad=12)
    ax1.set_xlabel("Actual Energy (Joules)", fontsize=21, labelpad=8)
    ax1.set_ylabel("Predicted Energy (Joules)", fontsize=21, labelpad=8)
    ax1.tick_params(axis="both", labelsize=18)
    ax1.legend(frameon=True, fontsize=18)
    ax1.grid(False)

    plt.tight_layout()

    # -----------------------------
    # PLOT 2: FEATURE IMPORTANCE
    # -----------------------------
    with open(os.path.join(DATA_DIR, "meta.json"), "r") as f:
        meta = json.load(f)
    feature_names = np.array(meta["features"])

    print("Calculating Feature Importance (this takes a moment)...")
    result = permutation_importance(
        energy_model,
        X_test,
        yE_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()

    fig2, ax2 = plt.subplots(figsize=(16, 10))

    bp = ax2.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=feature_names[sorted_idx],
        patch_artist=True,
        widths=0.75
    )

    for box in bp["boxes"]:
        box.set(facecolor="#DDDDDD", edgecolor="black", linewidth=7.6)

    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=7.6)

    for cap in bp["caps"]:
        cap.set(color="black", linewidth=7.6)

    for median in bp["medians"]:
        median.set(color="#C44E52", linewidth=7.6)

    for flier in bp["fliers"]:
        flier.set(marker="o",
                markerfacecolor="gray",
                markeredgecolor="black",
                alpha=0.6,
                markersize=10)

    ax2.set_title(
        "Feature Importance for Energy Consumption",
        fontsize=26,
        pad=14
    )

    ax2.set_xlabel(
        "Permutation Importance (Decrease in Accuracy)",
        fontsize=24,
        labelpad=10
    )

    ax2.set_ylabel(
        "Features",
        fontsize=24,
        labelpad=12
    )

    ax2.tick_params(axis="x", labelsize=22)
    ax2.tick_params(axis="y", labelsize=22)

    ax2.grid(False)

    plt.tight_layout(pad=2.0)

    # Show both plots first
    plt.show()

    # Ask whether to save
    save_plots = input("Do you want to save the plots? (y/n): ").strip().lower()
    if save_plots == "y":
        acc_file = input("Enter accuracy plot filename [fig_energy_accuracy.png]: ").strip()
        imp_file = input("Enter importance plot filename [fig_feature_importance.png]: ").strip()

        if not acc_file:
            acc_file = "fig_energy_accuracy.png"
        if not imp_file:
            imp_file = "fig_feature_importance.png"

        fig1.savefig(acc_file, dpi=300, bbox_inches="tight")
        fig2.savefig(imp_file, dpi=300, bbox_inches="tight")

        print(f"[OK] Saved {acc_file}")
        print(f"[OK] Saved {imp_file}")
    else:
        print("Plots were not saved.")

if __name__ == "__main__":
    main()
