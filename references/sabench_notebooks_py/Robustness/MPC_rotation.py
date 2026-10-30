# Converted from /Users/yiqun/Downloads/SABench/Robustness/MPC_rotation.ipynb
# Notebook cell count: 4

# %% [cell 1: code]
import anndata
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import squidpy as sq

# %% [cell 2: code]
slice1 = anndata.read_h5ad('/SABench/Data/MPC_processed/BaristaSeq_1_processed.h5ad')
slice2 = anndata.read_h5ad('/SABench/Data/MPC_processed/BaristaSeq_2_processed.h5ad')

# %% [cell 3: code]
def rotate_adata(slice, degree,save_path):
    adata=slice.copy()
    coords = adata.obsm['spatial']
    # 
    center = np.mean(coords, axis=0)

    # 
    coords_centered = coords - center

    theta = np.radians(degree)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_coords = np.dot(coords_centered, rotation_matrix)

    # 
    rotated_coords = rotated_coords + center

    adata.obsm['spatial'] = rotated_coords
    adata.write_h5ad(save_path)
    return adata

angles = [i for i in range(0, 361, 15)]
fig, axs = plt.subplots(5, 5, figsize=(25, 25)) 

for index,degree in enumerate(angles):
    new_slice = rotate_adata(slice2, degree, f'/SABench/Data/MPC_rotated/MPC_2_rotated_{degree}.h5ad')
    i = index // 5  
    j = index % 5  

    # 
    sq.pl.spatial_scatter(new_slice,library_id="spatial",shape=None,color="Region",wspace=1,ax=axs[i,j])

    #axs[index].invert_yaxis()
  
    axs[i,j].set_title(f'Rotation {degree} degrees')

    #x_limits = (0,55)
    #y_limits = (-5,50)
    #axs[i,j].set_xlim(x_limits)
    #axs[i,j].set_ylim(y_limits)
plt.tight_layout()
plt.show()

rotate_adata(slice1, 0, f'/SABench/Data/MPC_rotated/MPC_1_rotated_0.h5ad')

# %% [cell 4: code]


