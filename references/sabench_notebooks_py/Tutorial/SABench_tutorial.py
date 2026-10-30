# Converted from /Users/yiqun/Downloads/SABench/Tutorial/SABench_tutorial.ipynb
# Notebook cell count: 20

# %% [cell 1: markdown]
# # SABench tutorial

# %% [cell 2: markdown]
# This tutorial is designed to enable developers to quickly compare the results of their own methods with those of other methods. By running it, various evaluation metrics used in our paper will be automatically calculated for your aligned results, and visualizations will be generated. Here we use DLPFC data as example.

# %% [cell 3: code]
import sys
sys.path.append('Tutorial/')
import SABench

import os
import re
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% [cell 4: markdown]
# ## Step 1: Load Your Alignment Results

# %% [cell 5: markdown]
# Please load the slices aligned using your method in sequential order to form a list. You can refer to the code below.

# %% [cell 6: code]
def load_results(data_dir,pattern):
    slices = []
    files = []
    for filename in os.listdir(data_dir):
        if pattern in filename :
            files.append(filename)
    files.sort(key=lambda x: int(re.search(r'(\d+)\.h5ad$', x).group(1)))
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        adata = sc.read_h5ad(file_path)
        slices.append(adata)

    return slices

DLPFC_results=load_results('Tutorial/Data_DLPFC','.h5ad')
DLPFC_results

# %% [cell 7: markdown]
# ## Step 2: Calculate gene-based metrics

# %% [cell 8: markdown]
# ### Load marker gene list

# %% [cell 9: markdown]
# You can use known marker genes from the dataset, or alternatively, calculate using all genes, highly variable genes (HVG), spatially variable genes (SVG), etc., depending on the dataset you wish to test. In the absence of relevant literature, we recommend calculating HVG or SVG as the genes to use. Here, we utilized marker genes from the literature.

# %% [cell 10: code]
df = pd.read_csv('Tutorial/Methods_Results_DLPFC/DLPFC_gene_list.csv')
gene_list = df["marker gene"].tolist()
len(gene_list)

# %% [cell 11: markdown]
# ### Calculate gene-based metrics

# %% [cell 12: code]
pcc_values = SABench.Avg_pcc_values_for_method(DLPFC_results, gene_list)
cos_sim_values = SABench.Avg_cos_sim_values_for_method(DLPFC_results, gene_list)
ssim_values = SABench.Avg_ssim_values_for_method(DLPFC_results, gene_list)
MI_values = SABench.Avg_MI_values_for_method(DLPFC_results, gene_list)

# %% [cell 13: markdown]
# ### Compare your results with other methods.

# %% [cell 14: code]
def Custom_boxplot(data, labels, colors=None, 
                   title='Boxplot', xlabel='Method', ylabel='Values'):
    if colors is None:
        colors = ['#f7fcf0','#e0f3db','#ccebc5','#daf5fbff','#caf0f8ff','#ade8f4ff','#50a9d7','#2b8cbe','#225ea8','#9D9ECD','#ffc6a0','#ffb19f','#ff968b','#f28682']
    colors = (colors * ((len(data) // len(colors)) + 1))[:len(data)]

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(data, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1)

    for median in bp['medians']:
        median.set(color='black', linewidth=1)

    for whisker in bp['whiskers']:
        whisker.set(color='#333333', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='#333333', linewidth=1)

    for flier in bp['fliers']:
        flier.set(marker='o', color='#333333', alpha=0.5)

    plt.xticks(range(1, len(labels) + 1), labels, rotation=70)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


csv_files = [
    "Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_PCC.csv",
    "Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_Cos_Sim.csv",
    "Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_SSIM.csv",
    "Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_MI.csv"
]

new_method_lists = [
    pcc_values ,
    cos_sim_values,
    ssim_values,
    MI_values
]

plot_titles = [
    "PCC",
    "Cos_Sim",
    "SSIM",
    "MI"
]

for i in range(4):
    csv_path = csv_files[i]
    add_list = new_method_lists[i]
    title = plot_titles[i]

    df = pd.read_csv(csv_path, index_col=0)
    data_list = df.values.tolist()
    data_list.append(add_list)

    labels = df.index.tolist() + ["new method"]

    # plot
    Custom_boxplot(
        data=data_list,
        labels=labels,
        title=title,
        xlabel="Method",
        ylabel="Value"
    )

# %% [cell 15: markdown]
# ## Step 3: Calculate landmark-based metrics

# %% [cell 16: markdown]
# ### Calculate landmark-based metrics

# %% [cell 17: markdown]
# For the ROI overlap metrics, you need to specify which regions to calculate the metrics for. For accuracy, it is calculated by default based on adata.obs['Region'].

# %% [cell 18: code]
# ROI
region = ['WM', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6']
landmark_results = {}

for r in region:
    result = SABench.Average_ROI_Overlap(DLPFC_results, r)
    landmark_results[f"Average_ROI_Overlap_{r}"] = result

# Accuracy
accuracy = SABench.Average_Accuracy(DLPFC_results)
landmark_results["Average_Accuracy"] = accuracy

# check
for k, v in landmark_results.items():
    print(k, ":", v)

# %% [cell 19: markdown]
# ### Compare your results with other methods.

# %% [cell 20: code]
df = pd.read_csv("Tutorial/Methods_Results_DLPFC/DLPFCS_LandmarkBased_Heatmap_PlotFig.csv", index_col=0)  
landmark_df = pd.DataFrame([landmark_results], index=["new method"])
df_combined = pd.concat([df, landmark_df], axis=0)


plt.figure(figsize=(12, 6))
ax = sns.heatmap(df_combined, annot=True, cmap='viridis', linewidths=0.5)

plt.xticks(rotation=70)
plt.title('Heatmap of Landmark-based Metrics')
plt.tight_layout()
plt.show()

