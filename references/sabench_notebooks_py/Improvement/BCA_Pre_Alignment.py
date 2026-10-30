# Converted from /Users/yiqun/Downloads/SABench/Improvement/BCA_Pre_Alignment.ipynb
# Notebook cell count: 12

# %% [cell 1: code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq
import anndata as ad

# %% [cell 2: markdown]
# ### raw slices

# %% [cell 3: code]
slice_Rep1 = sc.read_h5ad('/SABench/Data/BCA_processed/XeniumBCARep1_processed.h5ad')
slice_Rep2 = sc.read_h5ad('/SABench/Data/BCA_processed/XeniumBCARep2_processed.h5ad')

slice_namesX=['Xenium_Rep1','Xenium_Rep2']
groupX=[slice_Rep1,slice_Rep2]

# %% [cell 4: code]
slice_Rep1

# %% [cell 5: code]
slice_Rep2

# %% [cell 6: code]
## plot style setup
fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

ax.scatter(
    slice_Rep1.obsm['spatial'][:,0],
    slice_Rep1.obsm['spatial'][:,1],
    s=2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color='#B8474D'
)

ax.scatter(
    slice_Rep2.obsm['spatial'][:,0],
    slice_Rep2.obsm['spatial'][:,1],
    s=0.5,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="gold"
)

# %% [cell 7: markdown]
# ### pre-alignment

# %% [cell 8: code]
slice_Rep1_PA=slice_Rep1.copy()
slice_Rep2_PA=slice_Rep2.copy()

# %% [cell 9: code]
spatial_coords = slice_Rep2.obsm["spatial"].copy()
spatial_coords[:, 0] -= 500  
spatial_coords[:, 1] += 2000  

slice_Rep2_PA.obsm["spatial"] = spatial_coords

# %% [cell 10: code]
## plot style setup
fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

ax.scatter(
    slice_Rep1_PA.obsm['spatial'][:,0],
    slice_Rep1_PA.obsm['spatial'][:,1],
    s=2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="skyblue"
)

ax.scatter(
    slice_Rep2_PA.obsm['spatial'][:,0],
    slice_Rep2_PA.obsm['spatial'][:,1],
    s=0.5,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="gold"
)

# %% [cell 11: markdown]
# ### Save data

# %% [cell 12: code]
slice_Rep1_PA.write_h5ad('/SABench/Improvement/BCA_PA/Data_PA/slice_Rep1_PA.h5ad')
slice_Rep2_PA.write_h5ad('/SABench/Improvement/BCA_PA/Data_PA/slice_Rep2_PA.h5ad')

