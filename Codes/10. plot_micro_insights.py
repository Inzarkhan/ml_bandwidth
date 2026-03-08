import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load data
df = pd.read_csv("micro_metrics.csv")

# ----- Fixed orders -----
workload_order = ["disk_io", "matrix", "memory_touch"]
mem_order = [128, 256, 512, 1024]

# Ensure correct types
df["mem_limit"] = df["mem_limit"].astype(int)

# Derived metrics
df["IPC"] = df["instructions"] / df["cycles"]
df["Cache_Miss_Rate_Per_KInstr"] = (df["LLC-misses"] / df["instructions"]) * 1000

# Aggregate duplicate rows for each (workload, mem_limit)
df_plot = (
    df.groupby(["workload", "mem_limit"], as_index=False)
      .agg({
          "IPC": "mean",
          "LLC-misses": "mean",
          "page-faults": "mean"
      })
)

# Fixed style per memory size
mem_palette = {
    128: "#4C72B0",
    256: "#55A868",
    512: "#C44E52",
    1024: "#8172B3"
}

hatch_map = {
    128: "",
    256: "///",
    512: "xxx",
    1024: "..."
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "legend.title_fontsize": 17
})

fig, axes = plt.subplots(1, 3, figsize=(22, 6.8))

x = np.arange(len(workload_order))
bar_width = 0.18

# Fixed left-to-right order
offsets = {
    128: -1.5 * bar_width,
    256: -0.5 * bar_width,
    512:  0.5 * bar_width,
    1024: 1.5 * bar_width
}

metrics = [
    ("IPC", "Instructions Per Cycle", "IPC (Higher is Better)"),
    ("LLC-misses", "Last-Level Cache Pressure", "Total LLC Misses"),
    ("page-faults", "Memory Thrashing", "Page Faults")
]

for ax, (metric, title, ylabel), panel in zip(
    axes, metrics, ["(a)", "(b)", "(c)"]
):
    for mem in mem_order:
        sub = df_plot[df_plot["mem_limit"] == mem]

        # Build y-values in exact workload order
        y = []
        for wl in workload_order:
            row = sub[sub["workload"] == wl]
            if row.empty:
                y.append(np.nan)
            else:
                y.append(row.iloc[0][metric])

        ax.bar(
            x + offsets[mem],
            y,
            width=bar_width,
            color=mem_palette[mem],
            edgecolor="black",
            linewidth=0.9,
            hatch=hatch_map[mem],
            label=str(mem)
        )

    ax.set_title(title)
    ax.set_xlabel("Workload")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(workload_order, rotation=15, ha="right")
    ax.tick_params(axis="both", labelsize=17)

    ax.text(
        0.02, 0.98, panel,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left"
    )

# Log scale for page faults
axes[2].set_yscale("log")
max_pf = df_plot["page-faults"].max()
min_pf = max(1, df_plot["page-faults"].min())
axes[2].set_ylim(min_pf * 0.9, max_pf * 1.25)

# One legend only, in correct order
legend_handles = [
    Patch(
        facecolor=mem_palette[mem],
        edgecolor="black",
        hatch=hatch_map[mem],
        label=str(mem)
    )
    for mem in mem_order
]

axes[2].legend(
    handles=legend_handles,
    title="Memory Limit",
    loc="upper right",
    frameon=True,
    fontsize=16,
    title_fontsize=17
)

plt.tight_layout()
plt.show()

save_plot = input("Do you want to save the figure? (y/n): ").strip().lower()
if save_plot == "y":
    filename = input("Enter filename [fig_micro_insights.png]: ").strip()
    if not filename:
        filename = "fig_micro_insights.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Figure saved as {filename}")
else:
    print("Figure was not saved.")