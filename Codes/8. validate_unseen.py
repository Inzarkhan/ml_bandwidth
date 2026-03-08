import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# CONFIGURATION
INPUT_FILE = "raw_unseen.jsonl"
DEFAULT_MEM = 1024
HGBDT_CHOICES = {
    "disk_io": 128,
    "matrix": 128
}

def add_bar_labels(ax, rects, fontsize=17, suffix=" J"):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.1f}{suffix}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold"
        )

def main():
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

    try:
        data = []
        with open(INPUT_FILE, "r") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading {INPUT_FILE}: {e}")
        return

    # Use median to reduce spike effects
    grouped = (
        df.groupby(["workload", "mem_limit_mb"])["energy_joules"]
          .median()
          .reset_index()
    )

    print("\n--- HEAVY GENERALIZATION TEST (MEDIAN FILTERED) ---")
    print(f"{'WORKLOAD':<12} | {'DEFAULT (1024MB)':<18} | {'HgBDT CHOICE':<20} | {'SAVINGS':<10}")
    print("-" * 75)

    total_default = 0
    total_hgbdt = 0
    workloads = list(HGBDT_CHOICES.keys())

    results = {"workload": [], "Default": [], "HgBDT": []}

    for wl in workloads:
        def_row = grouped[
            (grouped["workload"] == wl) &
            (grouped["mem_limit_mb"] == DEFAULT_MEM)
        ]
        if def_row.empty:
            continue
        def_energy = def_row["energy_joules"].values[0]

        choice = HGBDT_CHOICES[wl]
        opt_row = grouped[
            (grouped["workload"] == wl) &
            (grouped["mem_limit_mb"] == choice)
        ]
        if opt_row.empty:
            continue
        opt_energy = opt_row["energy_joules"].values[0]

        savings = def_energy - opt_energy

        print(f"{wl:<12} | {def_energy:.2f} J           | {opt_energy:.2f} J ({choice}MB)    | {savings:+.2f} J")

        total_default += def_energy
        total_hgbdt += opt_energy

        results["workload"].append(wl)
        results["Default"].append(def_energy)
        results["HgBDT"].append(opt_energy)

    print("-" * 75)

    if total_default == 0:
        print("No valid data found for plotting.")
        return

    total_savings_pct = (total_default - total_hgbdt) / total_default * 100
    print(f"TOTAL SAVINGS: {total_savings_pct:.1f}% ({total_default - total_hgbdt:.2f} J)")

    # Plot
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

    ax.set_title(f"Heavy Generalization Test (Total Savings: {total_savings_pct:.1f}%)", fontsize=20, pad=12)
    ax.set_ylabel("Real Energy (Joules) [Median]", fontsize=21, labelpad=8)
    ax.set_xlabel("Workload", fontsize=21, labelpad=8)

    ax.set_xticks(x)
    ax.set_xticklabels(results["workload"], rotation=10, ha="right")

    ax.tick_params(axis="y", labelsize=17)
    ax.tick_params(axis="x", labelsize=17)

    ax.grid(False)

    add_bar_labels(ax, rects1, fontsize=17, suffix=" J")
    add_bar_labels(ax, rects2, fontsize=17, suffix=" J")

    legend_handles = [
        Patch(facecolor="#7F7F7F", edgecolor="black", hatch="///", label="Default (1024MB)"),
        Patch(facecolor="#55A868", edgecolor="black", hatch="xxx", label="HgBDT Choice")
    ]
    ax.legend(handles=legend_handles, fontsize=18, frameon=True)

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