# Converted from /Users/yiqun/Downloads/SABench/Improvement/BCA_Get_DifferentResolutions.ipynb
# Notebook cell count: 17

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
    color="skyblue"
)

ax.scatter(
    slice_Rep2.obsm['spatial'][:,0],
    slice_Rep2.obsm['spatial'][:,1],
    s=0.2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="gold"
)

# %% [cell 7: markdown]
# ### change resolutions

# %% [cell 8: code]
def Change_slices_resolution(adata, nx, ny):

    spatial_coords = adata.obsm['spatial']
    x_min, x_max = np.min(spatial_coords[:, 0]), np.max(spatial_coords[:, 0])
    y_min, y_max = np.min(spatial_coords[:, 1]), np.max(spatial_coords[:, 1])
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    
    new_spatial_coords = []
    aggregated_X = []
    
    for i in range(nx):
        for j in range(ny):
            x_center = x_min + (i + 0.5) * dx
            y_center = y_min + (j + 0.5) * dy

            x_start, x_end = x_min + i*dx, x_min + (i+1)*dx
            y_start, y_end = y_min + j*dy, y_min + (j+1)*dy
            
            rect_mask = (spatial_coords[:, 0] >= x_start) & (spatial_coords[:, 0] < x_end) & \
                        (spatial_coords[:, 1] >= y_start) & (spatial_coords[:, 1] < y_end)
            
            if np.any(rect_mask):
                agg_X = np.sum(adata.X[rect_mask], axis=0)
                new_spatial_coords.append([x_center, y_center])
                aggregated_X.append(agg_X)
    
    new_obs = pd.DataFrame(index=range(len(new_spatial_coords)), columns=['x_center', 'y_center'])
    new_obs[['x_center', 'y_center']] = np.array(new_spatial_coords)
    
    new_adata = ad.AnnData(X=np.squeeze(aggregated_X), obs=new_obs, var=adata.var)
    new_adata.obsm['spatial'] = np.array(new_spatial_coords)
    
    return new_adata

# %% [cell 9: code]
slice_Rep1_10x10 = Change_slices_resolution(slice_Rep1, 10, 10)
slice_Rep2_10x10 = Change_slices_resolution(slice_Rep2, 10, 10)

# %% [cell 10: markdown]
# ### check

# %% [cell 11: code]
## plot style setup
fig = plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.invert_yaxis()

ax.scatter(
    slice_Rep1.obsm['spatial'][:,0],
    slice_Rep1.obsm['spatial'][:,1],
    s=2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="skyblue"
)

ax.scatter(
    slice_Rep1_10x10.obsm['spatial'][:,0],
    slice_Rep1_10x10.obsm['spatial'][:,1],
    s=100,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="gray"
)

# %% [cell 12: code]
fig, axs = plt.subplots(3, 1, figsize=(30, 20))  
     
sq.pl.spatial_scatter(slice_Rep1,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[0],size=1,colorbar=False)
sq.pl.spatial_scatter(slice_Rep1_10x10,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[1],size=500,colorbar=False)
sq.pl.spatial_scatter(slice_Rep1,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[2],size=1,colorbar=False)
sq.pl.spatial_scatter(slice_Rep1_10x10,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[2],size=500,colorbar=False)

axs[2].invert_yaxis()
plt.show()

# %% [cell 13: code]
## plot style setup
fig = plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.invert_yaxis()

ax.scatter(
    slice_Rep2.obsm['spatial'][:,0],
    slice_Rep2.obsm['spatial'][:,1],
    s=0.2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="gold"
)
ax.scatter(
    slice_Rep2_10x10.obsm['spatial'][:,0],
    slice_Rep2_10x10.obsm['spatial'][:,1],
    s=100,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="orange"
)

# %% [cell 14: code]
fig, axs = plt.subplots(3, 1, figsize=(30, 20))  
     
sq.pl.spatial_scatter(slice_Rep2,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[0],size=1,colorbar=False)
sq.pl.spatial_scatter(slice_Rep2_10x10,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[1],size=500,colorbar=False)
sq.pl.spatial_scatter(slice_Rep2,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[2],size=1,colorbar=False)
sq.pl.spatial_scatter(slice_Rep2_10x10,library_id="spatial",shape=None,color="TACSTD2",wspace=1,ax=axs[2],size=500,colorbar=False)

axs[2].invert_yaxis()
plt.show()

# %% [cell 15: markdown]
# ### Save data

# %% [cell 16: code]
for i in range(1,26):
    slice_Rep1_i0xi0 = Change_slices_resolution(slice_Rep1, i*25, i*25)
    slice_Rep2_i0xi0 = Change_slices_resolution(slice_Rep2, i*25, i*25)
    print(slice_Rep1_i0xi0)
    print(slice_Rep2_i0xi0)
    slice_Rep1_i0xi0.write_h5ad('/SABench/Improvement/BCA/Data_DR/slice_Rep1_'+str(i*25)+'x'+str(i*25)+'.h5ad')
    slice_Rep2_i0xi0.write_h5ad('/SABench/Improvement/BCA/Data_DR/slice_Rep2_'+str(i*25)+'x'+str(i*25)+'.h5ad')

# %% [cell 17: code]


