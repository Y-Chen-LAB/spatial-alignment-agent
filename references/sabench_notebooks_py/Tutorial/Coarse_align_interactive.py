# Converted from /Users/yiqun/Downloads/SABench/Tutorial/Coarse_align_interactive.ipynb
# Notebook cell count: 12

# %% [cell 1: markdown]
# # Interactive manual coarse alignment

# %% [cell 2: markdown]
# The following code provides a simple interactive tool for manual alignment, designed to help users perform coarse alignment before formal (automated) alignment. Since it is interactive, please run it on your local computer.

# %% [cell 3: code]
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
matplotlib.use('TkAgg')   
from skimage.transform import estimate_transform

#import sys
#sys.path.append('')
import SABench

# %% [cell 4: markdown]
# Here we use shuffled DLPFC data (151673 and 151674) as example data, loading these two adjacent slices respectively. You can choose the key to use for visualization (e.g., clustering labels, etc.), or leave it unspecified, in which case it will default to using total counts for plotting.  
#
# After running the code, plotting windows for the two slices will pop up. You can then alternately click on the two images to select corresponding landmark point pairs for alignment (at least three pairs are required: first click a point on the fixed image, then click the corresponding point on the moving image, then select the second pair, and so on). Once you have selected all desired landmark pairs, simply close the windows. The program will automatically compute the affine transformation for the moving image and finally return the two aligned slices.

# %% [cell 5: code]
fixed_path  = "coarse_align_interactive_data/151673_shuffled.h5ad"
moving_path = "coarse_align_interactive_data/151674_shuffled.h5ad"
fixed_adata = ad.read_h5ad(fixed_path )
moving_adata = ad.read_h5ad(moving_path)

# specify key in obs for plotting; if None, use total counts
PLOT_KEY = "Region"
#PLOT_KEY = None    #if you don't want to use obs for coloring
print(f"Start manual coarse alignment, using obs['{PLOT_KEY}'] for coloring")
ad_fixed, ad_moving_aligned = SABench.interactive_coarse_align(fixed_adata, moving_adata, PLOT_key=PLOT_KEY)

# %% [cell 6: markdown]
# Now we can visualize the alignment results.
# We will plot the slices before and after alignment respectively. Similarly, you can specify which key (field) to use for visualization.

# %% [cell 7: code]
def plot_aligned_comparison(
    adata_fixed=None,
    adata_moving_aligned=None,
    region_key="Region",
    spot_size=80
):

    # ================= Checks =================
    if adata_fixed is None or adata_moving_aligned is None:
        print("Error: Please run interactive_coarse_align() first to obtain ad_fixed and ad_moving_aligned.")
        return

    if 'spatial' not in adata_fixed.obsm or 'spatial' not in adata_moving_aligned.obsm:
        print("Error: obsm['spatial'] does not exist in one of the AnnData objects.")
        return

    # ================= Determine Coloring Mode =================
    use_region = (
        region_key in adata_fixed.obs.columns and
        region_key in adata_moving_aligned.obs.columns and
        adata_fixed.obs[region_key].dtype.name == 'category'
    )

    if use_region:
        
        fixed_colors = adata_fixed.obs[region_key].cat.codes.values
        moving_colors = adata_moving_aligned.obs[region_key].cat.codes.values
        categories = adata_fixed.obs[region_key].cat.categories
        n_cats = len(categories)
        cmap = plt.get_cmap('tab20' if n_cats <= 20 else 'tab20b')
        colorbar_label = region_key
    else:
        # Fallback: total counts
        print(
            f"Warning: '{region_key}' not found in obs columns or not category dtype, Using total counts instead."
        )
        fixed_counts = np.ravel(adata_fixed.X.sum(axis=1))
        moving_counts = np.ravel(adata_moving_aligned.X.sum(axis=1))
        if hasattr(fixed_counts, "toarray"):
            fixed_counts = fixed_counts.toarray().ravel()
        if hasattr(moving_counts, "toarray"):
            moving_counts = moving_counts.toarray().ravel()
        fixed_colors = fixed_counts
        moving_colors = moving_counts
        cmap = 'viridis'
        colorbar_label = 'Total counts'

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    ax1, ax2, ax3 = axes  

    # ----- 1. Fixed -----
    sc1 = ax1.scatter(
        adata_fixed.obsm['spatial'][:, 0],
        adata_fixed.obsm['spatial'][:, 1],
        c=fixed_colors,
        cmap=cmap,
        s=spot_size,
        alpha=0.9,
        edgecolor='none'
    )
    ax1.set_title("Fixed (Reference)", fontsize=18, pad=20)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()

    # ----- 2. Moving (Aligned) -----
    sc2 = ax2.scatter(
        adata_moving_aligned.obsm['spatial'][:, 0],
        adata_moving_aligned.obsm['spatial'][:, 1],
        c=moving_colors,
        cmap=cmap,
        s=spot_size,
        alpha=0.9,
        edgecolor='none'
    )
    ax2.set_title("Moving (Aligned)", fontsize=18, pad=20)
    ax2.set_aspect('equal')
    ax2.invert_yaxis()

    # ----- 3. Overlay -----
    ax3.scatter(
        adata_fixed.obsm['spatial'][:, 0],
        adata_fixed.obsm['spatial'][:, 1],
        c=fixed_colors,
        cmap=cmap,
        s=spot_size,
        alpha=0.5,
        label='Fixed'
    )
    ax3.scatter(
        adata_moving_aligned.obsm['spatial'][:, 0],
        adata_moving_aligned.obsm['spatial'][:, 1],
        c=moving_colors,
        cmap=cmap,
        s=spot_size,
        alpha=0.95,
        edgecolor='black',
        linewidth=0.3,
        label='Moving (aligned)'
    )
    ax3.set_title("Overlay Comparison", fontsize=18, pad=20)
    ax3.set_aspect('equal')
    ax3.invert_yaxis()
    ax3.legend(loc='upper left')

    # ================= Legend for Region =================
    if use_region:
        fig_leg = plt.figure(figsize=(3, 6))
        ax_legend = fig_leg.add_subplot(111)
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=cmap(i / max(n_cats - 1, 1)),
                markersize=12,
                label=cat
            )
            for i, cat in enumerate(categories)
        ]
        ax_legend.legend(
            handles=handles,
            title=region_key,
            loc='center',
            fontsize=12,
            title_fontsize=14
        )
        ax_legend.axis('off')

    # ================= Dedicated Colorbar =================
    if not use_region:
       
        cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  
        fig.colorbar(sc1, cax=cbar_ax, label=colorbar_label)

    plt.suptitle("Final Coarse Manual Alignment Result", fontsize=26, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  
    plt.show()

# %% [cell 8: markdown]
# raw slices

# %% [cell 9: code]
plot_aligned_comparison(fixed_adata, moving_adata, region_key="Region", spot_size=50)

# %% [cell 10: markdown]
# slices after alignment

# %% [cell 11: code]
plot_aligned_comparison(ad_fixed, ad_moving_aligned, region_key="Region", spot_size=50)

# %% [cell 12: code]
plot_aligned_comparison(ad_fixed, ad_moving_aligned,region_key="total counts" ,spot_size=50)

