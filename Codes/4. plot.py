import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance

from plot_style import (
    PAPER_DPI,
    apply_paper_style,
    finalize_figure,
    format_feature_label,
    make_figure,
    maybe_show,
    prompt_filename,
    prompt_yes_no,
    style_axes,
)

# Settings
MODEL_DIR = os.environ.get("PLOT_MODEL_DIR", "models_known_full_plusfb_slo14")
DATA_DIR = os.environ.get("PLOT_DATASET_DIR", "dataset_reg_known_full_plusfb")
TOP_FEATURES = int(os.environ.get("PLOT_TOP_FEATURES", "12"))

def main():
    apply_paper_style()

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
    fig1, ax1 = make_figure()

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

    style_axes(
        ax1,
        title=f"Energy Prediction Accuracy (R² = {r2:.3f})",
        xlabel="Actual Energy (Joules)",
        ylabel="Predicted Energy (Joules)",
        title_width=28,
        title_pad=10,
    )
    ax1.legend(frameon=True)
    ax1.grid(False)

    finalize_figure(fig1, top=0.96)

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

    mean_importance = result.importances_mean
    std_importance = result.importances_std
    top_idx = np.argsort(mean_importance)[-TOP_FEATURES:]
    top_idx = top_idx[np.argsort(mean_importance[top_idx])]
    top_features = [format_feature_label(name) for name in feature_names[top_idx]]
    top_means = mean_importance[top_idx]
    top_stds = std_importance[top_idx]

    fig2, ax2 = make_figure()

    ax2.barh(
        top_features,
        top_means,
        xerr=top_stds,
        color="#4C72B0",
        edgecolor="black",
        linewidth=1.0,
        capsize=4,
        alpha=0.9
    )

    style_axes(
        ax2,
        title=f"Top {len(top_features)} Feature Importance for Energy Prediction",
        xlabel="Permutation Importance",
        ylabel="Features",
        title_width=30,
        title_pad=10,
    )

    ax2.tick_params(axis="y", pad=4)
    ax2.grid(False)
    ax2.set_xlim(left=min(-0.02, float(top_means.min()) * 1.10))

    finalize_figure(fig2, top=0.95)

    maybe_show()

    save_plots = prompt_yes_no("Do you want to save the plots? (y/n): ", default="n")
    if save_plots == "y":
        acc_file = prompt_filename("Enter accuracy plot filename", "fig_energy_accuracy.png")
        imp_file = prompt_filename("Enter importance plot filename", "fig_feature_importance.png")

        fig1.savefig(acc_file, dpi=PAPER_DPI, bbox_inches="tight")
        fig2.savefig(imp_file, dpi=PAPER_DPI, bbox_inches="tight")

        print(f"[OK] Saved {acc_file}")
        print(f"[OK] Saved {imp_file}")
    else:
        print("Plots were not saved.")

if __name__ == "__main__":
    main()
