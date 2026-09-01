# -*- coding: utf-8 -*-
"""
Blank probes are barcodes that were not assigned to any real gene during panel
design. Any transcript decoded as a blank probe is, by definition, a decoding
error, so blank-probe counts are the standard reference used in MERFISH papers
to estimate the background / misidentification rate
(Moffitt et al. 2016, PNAS 113:11046; Xia et al. 2019, PNAS 116:19490).
"""

"""
@author: Chenxi Zhou
"""


# %% 
## Blank probe (negative control) statistics

# %%
import pandas as pd
import numpy as np
import os
import anndata as ad
import squidpy as sq
import scanpy as sc
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# %%

print(os.getcwd())
this_path = os.getcwd()

# %%

# get the adata

adata_all = ad.read_h5ad("MER1-3-merged.h5ad")

## select the MERO2 dataset

adata_all.obs["batch"]

MER02 = adata_all[adata_all.obs["batch"]=="MER02"]

adata = MER02[MER02.obs["stage"] == 22].copy()

adata

adata.X = adata.layers["raw_counts"]

path_save = str(os.getcwd()) + "//gene_panel_QC//stage22//"
Path.mkdir(Path(path_save), parents=True, exist_ok = True)


#%%

## statistics of gene expression matrix

adata.X = adata.layers["raw_counts"]
stat = sc.pp.calculate_qc_metrics(adata,percent_top=(1,5,10,15))
stat[0].to_csv(path_save + "cell_stat.csv")  
stat[1].to_csv(path_save + "gene_stat.csv")
 

# %% 

## statistics of blank expression matrix

blank = adata.obsm["blank_genes"]

blank_totol = blank.sum(axis=0)
blank_totol.name = "total_blank_count"   
blank_totol.to_csv(str(path_save) + "\\blank_total.csv")

## frequency of single transcript event (where only one transcript of a gene is detected in a cell)

blank_l = np.where(blank.to_numpy() == 1, 1, 0)
blank_count_1 = blank_l.sum(axis=0)
blank_count_1 = pd.DataFrame(blank_count_1, index=blank_totol.index, columns=["single_detection_event"])
blank_count_1.to_csv(str(path_save) + "\\blank_count1.csv")

blank_l = np.where(blank.to_numpy() == 0, 1, 0)
blank_count_0 = blank_l.sum(axis=0)
blank_count_0 = pd.DataFrame(blank_count_0, index=blank_totol.index, columns=["dropout_event"])
blank_count_0.to_csv(str(path_save) + "\\blank_count0.csv")


blank_merge = pd.merge(blank_totol, blank_count_1, left_index=True, right_index=True)
blank_merge = pd.merge(blank_merge, blank_count_0, left_index=True, right_index=True)
blank_merge.to_csv(str(path_save) + "\\blank_statistics.csv")


#blank_l = np.where(blank.to_numpy() >= 1, 1, 0)
#blank_count_exist = blank_l.sum(axis=0)
#blank_count_exist = pd.DataFrame(blank_count_exist, index=blank_totol.index)
#blank_count_exist.to_csv(str(path_save) + "\\blank_count_exist.csv")


#blank_l = np.where(blank.to_numpy() == 2, 1, 0)
#blank_count_2 = blank_l.sum(axis=0)
#blank_count_2 = pd.DataFrame(blank_count_2, index=blank_totol.index)
#blank_count_2.to_csv(str(path_save) + "\\blank_count2.csv")


#blank_l = np.where(blank.to_numpy() == 3, 1, 0)
#blank_count_3 = blank_l.sum(axis=0)
#blank_count_3 = pd.DataFrame(blank_count_3, index=blank_totol.index)
#blank_count_3.to_csv(str(path_save) + "\\blank_count3.csv")



print("frequence of count == 0:", np.where(blank.to_numpy() == 0, 1, 0).sum())
print("frequence of count == 1:", np.where(blank.to_numpy() == 1, 1, 0).sum())
print("frequence of count == 2:", np.where(blank.to_numpy() == 2, 1, 0).sum())
print("frequence of count == 3:", np.where(blank.to_numpy() == 3, 1, 0).sum())
print("frequence of count >= 1:", np.where(blank.to_numpy() >= 1, 1, 0).sum())
print("frequence of count >= 0:", np.where(blank.to_numpy() >= 0, 1, 0).sum())



# %% 
## Evaluate the spatial specificity of blank probes

## for stage22

print(adata.obs["stage"])
adata

stage22 = adata[adata.obs["stage"] == 22].copy()
# stage18.X = stage18.obsm['blank_genes'].copy()
stage22

blank = stage22.obsm["blank_genes"]

blank
blank_adata = ad.AnnData(blank)
# blank.obs_names = [f"Cell_{i:d}" for i in range(blank.n_obs)]
# blank.var_names = [f"Gene_{i:d}" for i in range(blank.n_vars)]

assert blank_adata.n_obs == stage22.obsm['spatial'].shape[0]
blank_adata.obsm['spatial'] = stage22.obsm['spatial'].copy()

stage22.obsm['spatial']
blank_adata.obsm['spatial']

blank_adata.to_df()

sq.gr.spatial_neighbors(blank_adata, coord_type="generic")
sq.gr.spatial_autocorr(blank_adata, mode="moran")
blank_adata.uns["moranI"].to_csv(str(path_save) + "\\stage22_blank_moranI.csv")
blank_adata.uns["moranI"]

blank_merge = pd.merge(blank_merge, blank_adata.uns["moranI"], left_index=True, right_index=True)
blank_merge.to_csv(str(path_save) + "\\blank_statistics.csv")




# %%

# figure of blank genes

blank = adata.obsm["blank_genes"]
n_blank_probes = blank.shape[1]
n_gene_probes = adata.n_vars

# %%

# 1) total blank counts per cell -- histogram

total_blank_per_cell = blank.sum(axis=1)
total_blank_per_cell.to_csv(path_save + "total_blank_counts_per_cell.csv")

plt.figure(figsize=(6, 4))
sns.histplot(total_blank_per_cell, kde=False)
plt.title("Total blank (negative control) counts per cell")
plt.xlabel("total blank counts per cell")
plt.tight_layout()
plt.savefig(path_save + "total_blank_per_cell_hist.png", dpi=300, bbox_inches="tight")
plt.show()

# %%

# 2) total counts per blank probe -- bar plot
#   (checks whether any single blank probe is systematically over-represented,
#    which would suggest cross-hybridization rather than random noise)

total_blank_per_probe = blank.sum(axis=0).sort_values(ascending=False)
total_blank_per_probe.to_csv(path_save + "total_counts_per_blank_probe.csv")

plt.figure(figsize=(10, 5))
sns.barplot(x=total_blank_per_probe.index, y=total_blank_per_probe.values, color="grey")
plt.xticks(rotation=90, fontsize=6)
plt.ylabel("total counts")
plt.xlabel("blank probe")
plt.title("Total counts per blank (negative control) probe")
plt.tight_layout()
plt.savefig(path_save + "total_counts_per_blank_probe_barplot.png", dpi=300, bbox_inches="tight")
plt.show()


#%%

# 3) frequency of blank probe -- histo plot
#    (check the frequency of blank probes)

### histoplot

plt.figure(figsize=(10, 6))

for probe in blank.columns:
    values = blank[probe]
    # values = values[values > 0]  # ignore cells with 0 expression
    if len(values) > 0:
        sns.histplot(values, label=probe, linewidth=1, kde=True)

plt.title("Blank Expression histplot")
plt.xlabel("Raw counts")
plt.ylabel("Frequency")
# plt.axvline(x = 2, color = "red", linestyle = "--")
# plt.axvline(x = 3, color = "red", linestyle = "--")
plt.xlim(left=0)
plt.xticks(list(plt.xticks()[0]) + [1, 2, 3])
plt.savefig(path_save + "Blank histplot.png", dpi=300, bbox_inches="tight")


# %%
# 4) blank total vs real-gene total, per cell -- correlation scatter
#   (checks whether background scales with total expression / cell size,
#    as expected for genuine noise)

gene_total_per_cell = stat[0]["total_counts"]  # per-cell total real-gene counts, already computed above

blank_vs_gene = pd.DataFrame({
    "blank_total": total_blank_per_cell,
    "gene_total": gene_total_per_cell,
})
blank_vs_gene.to_csv(path_save + "blank_vs_gene_total_per_cell.csv")

corr = blank_vs_gene[["blank_total", "gene_total"]].corr(method="pearson").iloc[0, 1]

plt.figure(figsize=(6, 6))
sns.scatterplot(data=blank_vs_gene, x="gene_total", y="blank_total", s=8)
plt.xlabel("total real-gene counts per cell")
plt.ylabel("total blank counts per cell")
plt.title(f"Blank vs real-gene counts per cell (Pearson r = {corr:.3f})")
plt.tight_layout()
plt.savefig(path_save + "blank_vs_gene_correlation.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# 5) misidentification rate
#   gross misidentification rate = mean count per blank probe / mean count per barcode (blank+gene)
#   gene-only misidentification rate = mean count per blank probe / mean count per gene probe
#   definitions follow Moffitt et al. 2016 PNAS 113:11046 (SI, Fig. S2C) and
#   Xia et al. 2019 PNAS 116:19490 (SI Materials and Methods)


gene_total_per_probe = stat[1]["total_counts"]  # per-gene total counts, already computed above

mean_count_per_blank_probe = total_blank_per_probe.sum() / n_blank_probes
mean_count_per_gene_probe = gene_total_per_probe.sum() / n_gene_probes
mean_count_per_all_probe = (total_blank_per_probe.sum() + gene_total_per_probe.sum()) / (n_blank_probes + n_gene_probes)

gross_misID_rate = mean_count_per_blank_probe / mean_count_per_all_probe
gene_only_misID_rate = mean_count_per_blank_probe / mean_count_per_gene_probe

print(f"number of blank probes: {n_blank_probes}")
print(f"number of gene probes: {n_gene_probes}")
print(f"mean count per blank probe: {mean_count_per_blank_probe:.3f}")
print(f"mean count per gene probe: {mean_count_per_gene_probe:.3f}")
print(f"gross misidentification rate (blank+gene as denominator): {gross_misID_rate:.3%}")
print(f"misidentification rate (gene-only denominator): {gene_only_misID_rate:.3%}")

misID_summary = pd.DataFrame({
    "metric": [
        "n_blank_probes",
        "n_gene_probes",
        "mean_count_per_blank_probe",
        "mean_count_per_gene_probe",
        "gross_misidentification_rate",
        "gene_only_misidentification_rate",
    ],
    "value": [
        n_blank_probes,
        n_gene_probes,
        mean_count_per_blank_probe,
        mean_count_per_gene_probe,
        gross_misID_rate,
        gene_only_misID_rate,
    ],
})
misID_summary.to_csv(path_save + "misidentification_rate_summary.csv", index=False)
misID_summary
