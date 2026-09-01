
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path
csv_path = r"C:\Users\Chenxi Zhou\Desktop\VPT\chenxi_segmentation_delivery\masks\vptlike_cp2.2.3_02\summary_endo_IoU_thresholds_02.csv"
region_title = "Endoderm"
out_path = r"C:\Users\Chenxi Zhou\Desktop\VPT\chenxi_segmentation_delivery\masks\vptlike_cp2.2.3_02\plots\lineplot_endo_IoU_shaded.png"

band_alpha = 0.10       # shade transparecy

# line adjustment
axis_linewidth = 2.5     # axis
tick_linewidth = 2.5     # tick
line_linewidth = 2.5     # line
marker_size = 8          # dot

# font adjust
tick_fontsize = 22       # axis
axis_label_fontsize = 22 # axis title
title_fontsize = 22      # title
legend_fontsize = 14     # legend

category_map = [
    ("mosaic_Cellbound1_z3_DAPI_mask_endo.tif", "Cellbound1"),
    ("mosaic_Cellbound2_z3_DAPI_mask_endo.tif", "Cellbound2"),
    ("mosaic_PolyT_z3_DAPI_mask_endo.tif",      "PolyT"),
    ("added_image_z3_DAPI_mask_endo.tif",       "Sum(Cellbound 1, 3)"),
    ("max_image_z3_DAPI_mask_endo.tif",         "Max(Cellbound 1, 3)"),
]

colors = {
    "Cellbound1":           {"line": "#3B5D91", "marker": "#7CA0D4"},
    "Cellbound2":           {"line": "#2A1550", "marker": "#4F2B8E"},
    "PolyT":                {"line": "#5F9A2E", "marker": "#BADE86"},
    "Sum(Cellbound 1, 3)":  {"line": "#E0006E", "marker": "#FF8080"},
    "Max(Cellbound 1, 3)":  {"line": "#C15C00", "marker": "#F5A623"},  
}

# read data 
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(8, 6))

# figure
all_stats = []

for fname, label in category_map:
    sub = df[df["Test image"] == fname]
    stats = sub.groupby("Object IoU threshold")["Object Jaccard Index"].agg(["mean", "std"])

    # save mean/std
    stats_to_save = stats.reset_index()
    stats_to_save["Category"] = label
    all_stats.append(stats_to_save)

    x = stats.index.values
    mean = stats["mean"].values
    std = stats["std"].values

    line_color = colors[label]["line"]
    marker_color = colors[label]["marker"]

    ax.fill_between(x, mean - std, mean + std, color=line_color, alpha=band_alpha, linewidth=0)

    ax.plot(
        x, mean,
        color=line_color, marker="o", markersize=marker_size,
        markerfacecolor=marker_color, markeredgecolor=line_color,
        linewidth=line_linewidth, label=label
    )

ax.set_xlabel("IoU threshold", fontsize=axis_label_fontsize, fontweight="bold")
ax.set_ylabel("Jaccard Index (IoU)", fontsize=axis_label_fontsize, fontweight="bold")
ax.set_title(region_title, fontsize=title_fontsize, fontweight="bold")
ax.set_xlim(0.48, 1.02)
ax.set_ylim(0, 1.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(axis_linewidth)
ax.spines["bottom"].set_linewidth(axis_linewidth)
ax.tick_params(width=tick_linewidth, labelsize=tick_fontsize)
ax.legend(fontsize=legend_fontsize, frameon=False, loc="upper right")

plt.tight_layout()
plt.savefig(out_path, dpi=300)
print(f"saved: {out_path}")

# save source data
stats_df = pd.concat(all_stats, ignore_index=True)
stats_df = stats_df.rename(columns={"mean": "IoU_mean", "std": "IoU_std"})
stats_csv_path = out_path.replace(".png", "_stats.csv")
stats_df.to_csv(stats_csv_path, index=False)
print(f"stats saved: {stats_csv_path}")
