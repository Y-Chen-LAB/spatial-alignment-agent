# Converted from /Users/yiqun/Downloads/SABench/Robustness/MHPR_crop.ipynb
# Notebook cell count: 4

# %% [cell 1: code]
import anndata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import squidpy as sq

# %% [cell 2: code]
slice1 = anndata.read_h5ad('/SABench/Data/MHPR_processed/MERFISH_0.04.h5ad')
slice2 = anndata.read_h5ad('/SABench/Data/MHPR_processed/MERFISH_0.09.h5ad')

# %% [cell 3: code]
def crop_adata(slice,overlap_percentage,save_path):
 
    ad = slice.copy()
    
    x_coords = ad.obsm['spatial'][:, 0]

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    
    x_threshold = x_min + (x_max - x_min) * (overlap_percentage/100)
        
    # 
    indices = np.where(x_coords < x_threshold)[0]
    cropped_ad = ad[indices]
    cropped_ad.write_h5ad(save_path)
    return cropped_ad

percentage = [i for i in range(10, 101, 10)]
fig, axs = plt.subplots(2, 5, figsize=(25, 10))
axs=axs.flatten()
for index,ratio in enumerate(percentage):
    new_slice = crop_adata(slice2, ratio, f'/SABench/Data/MHPR_cropped/MERFISH_009_cropped_{ratio}%.h5ad')
   
    # 
    sq.pl.spatial_scatter(new_slice,library_id="spatial",shape=None,color="Region",wspace=1,ax=axs[index])

    #axs[index].invert_yaxis()

    axs[index].set_title(f'Overlap percentage :{ratio} %')

    x_limits = (0,4000)
    y_limits = (1000,5000)
    axs[index].set_xlim(x_limits)
    axs[index].set_ylim(y_limits)
plt.tight_layout()
plt.show()

crop_adata(slice1, 100, f'/SABench/Data/MHPR_cropped/MERFISH_004_cropped_100%.h5ad')

# %% [cell 4: markdown]


