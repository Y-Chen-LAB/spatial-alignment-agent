# Converted from /Users/yiqun/Downloads/SABench/DownstreamTasks/DLPFC_3D_Spatial_Clustering.ipynb
# Notebook cell count: 18

# %% [cell 1: code]
import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
import multiprocessing as mp
import matplotlib.pyplot as plt
from GraphST import GraphST
import anndata
import numpy as np

# %% [cell 2: code]
# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace it with local R installation path
os.environ['R_HOME'] = '/anaconda3/envs/GraphST/lib/R'

# %% [cell 3: code]
# the number of clusters
n_clusters = 7

# %% [cell 4: markdown]
# ## Reading data

# %% [cell 5: code]
def prepare_slices(anndata_list,slicenames):
    merged_adata = anndata.concat(anndata_list, label='data', keys=slicenames  )
    merged_adata.var_names_make_unique()
    return merged_adata

# %% [cell 6: markdown]
# ### Raw data

# %% [cell 7: code]
#read h5ad slices
def load_slices(data_dir, slice_names):
    slices = []
    for slice_name in slice_names:
        slice_i = sc.read_h5ad(data_dir+slice_name+'_shuffled.h5ad')

        slices.append(slice_i)
    return slices

# %% [cell 8: code]
data_dir = '/SABench/Data/DLPFC_shuffled/'
slice_names = ["151673","151674", "151675", "151676"]
DLPFCS_raw = load_slices(data_dir,slice_names)
adata=prepare_slices(DLPFCS_raw,slice_names)

# %% [cell 9: code]
adata.obs['data']

# %% [cell 10: code]
plt.rcParams["figure.figsize"] = (3, 3)
ax = sc.pl.embedding(adata, basis='spatial',
                color='data',
                show=False)
ax.set_title('Raw data')
#ax.axis('off')

# %% [cell 11: markdown]
# ### alignment results

# %% [cell 12: code]
import re
def load_results(data_dir,pattern):
    slices = []
    files = []
    for filename in os.listdir(data_dir):
        if pattern in filename :
            files.append(filename)
    files.sort(key=lambda x: int(re.search(r'_(\d+)\.h5ad', x).group(1)))
   
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        adata = sc.read_h5ad(file_path)
        slices.append(adata)
 
    return slices

# %% [cell 13: code]
PASTE_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/PASTE_DLPFCS/','.h5ad'),slice_names)
PASTE_p0_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/PASTE_p0_DLPFCS/','.h5ad'),slice_names)
PASTE2_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/PASTE2_DLPFCS/','.h5ad'),slice_names)
STAligner_DLPFCS_landmark0=prepare_slices(load_results('/SABench/AlignmentResults/STAligner_DLPFCS/','landmark0'),slice_names)
STAligner_DLPFCS_landmark1=prepare_slices(load_results('/SABench/AlignmentResults/STAligner_DLPFCS/','landmark1'),slice_names)
STAligner_DLPFCS_landmark2=prepare_slices(load_results('/SABench/AlignmentResults/STAligner_DLPFCS/','landmark2'),slice_names)
STAligner_DLPFCS_landmark3=prepare_slices(load_results('/SABench/AlignmentResults/STAligner_DLPFCS/','landmark3'),slice_names)
SLAT_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/SLAT_DLPFCS/','.h5ad'),slice_names)
STalign_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/STalign_DLPFCS/','.h5ad'),slice_names)
CAST_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/CAST_DLPFCS/','.h5ad'),slice_names)
STAIR_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/STAIR_DLPFCS/','.h5ad'),slice_names)
SPACEL_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/SPACEL_DLPFCS/','.h5ad'),slice_names)
Spateo_rigid_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/Spateo_rigid_DLPFCS/','.h5ad'),slice_names)
Spateo_nonrigid_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/Spateo_nonrigid_DLPFCS/','.h5ad'),slice_names)
SANTO_DLPFCS=prepare_slices(load_results('/SABench/AlignmentResults/SANTO_DLPFCS/','.h5ad'),slice_names)

# %% [cell 14: code]
groups=[adata,PASTE_DLPFCS,PASTE_p0_DLPFCS,PASTE2_DLPFCS,STAligner_DLPFCS_landmark0,STAligner_DLPFCS_landmark1,STAligner_DLPFCS_landmark2,STAligner_DLPFCS_landmark3,SLAT_DLPFCS,STalign_DLPFCS,CAST_DLPFCS,STAIR_DLPFCS,SPACEL_DLPFCS,Spateo_rigid_DLPFCS,Spateo_nonrigid_DLPFCS,SANTO_DLPFCS]
groups_names=['RAW_DLPFCS','PASTE_DLPFCS','PASTE_p0_DLPFCS','PASTE2_DLPFCS','STAligner_DLPFCS_landmark0','STAligner_DLPFCS_landmark1','STAligner_DLPFCS_landmark2','STAligner_DLPFCS_landmark3','SLAT_DLPFCS','STalign_DLPFCS','CAST_DLPFCS','STAIR_DLPFCS','SPACEL_DLPFCS','Spateo_rigid_DLPFCS','Spateo_nonrigid_DLPFCS','SANTO_DLPFCS']

# %% [cell 15: code]
plt.rcParams["figure.figsize"] = (3, 3)
ax = sc.pl.embedding(SLAT_DLPFCS, basis='spatial',
                color='data',
                show=False)
ax.set_title('SLAT_DLPFCS')

# %% [cell 16: markdown]
# ## Run GraphST

# %% [cell 17: code]
def RunGraphST(raw_adata,idx):
    adata =raw_adata.copy()
    # define model
    model = GraphST.GraphST(adata, device=device)
    # run model
    adata = model.train()

    # clustering
    from GraphST.utils import clustering
    tool = 'mclust' # mclust, leiden, and louvain

    if tool == 'mclust':
        clustering(adata, n_clusters, method=tool) 
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, method=tool, start=0.1, end=2.0, increment=0.01)

    adata.obs["domain"] = adata.obs["domain"].astype(int)-1
    adata.obs["domain"] = pd.Categorical(adata.obs["domain"])
  
    adata_section1 = adata[adata.obs['data']=='151673', :]
    adata_section2 = adata[adata.obs['data']=='151674', :]
    adata_section3 = adata[adata.obs['data']=='151675', :]
    adata_section4 = adata[adata.obs['data']=='151676', :]
    results=[adata_section1,adata_section2,adata_section3,adata_section4]
    #Compute ARI & Visualization
    ARI_list=[]
    NMI_list=[]
    HOM_list=[]
    COM_list=[]
    for i, adata_st_i in enumerate(results):
        
        print("Slice {} spatial domain detection result:".format(idx[i]))
        adata_st_i = adata_st_i[~pd.isnull(adata_st_i.obs['Region'])]# filter out NA nodes

        # calculate metric ARI
        ARI = metrics.adjusted_rand_score(adata_st_i.obs['domain'],adata_st_i.obs['Region'])
        adata_st_i.uns['ARI'] = ARI
        sc.pl.spatial(adata_st_i,color=["Region", 'domain'],title=["Ground truth", "ARI=%.4f"%ARI],legend_fontsize='x-small',spot_size=100,show=True)
        ARI_list.append(ARI)
        #NMI
        NMI=metrics.normalized_mutual_info_score(adata_st_i.obs['domain'],adata_st_i.obs['Region'])
        adata_st_i.uns['NMI'] = NMI
        NMI_list.append(NMI)
        #HOM
        HOM=metrics.homogeneity_score(adata_st_i.obs['domain'],adata_st_i.obs['Region'])
        adata_st_i.uns['HOM'] = HOM
        HOM_list.append(HOM)
        #COM
        COM=metrics.completeness_score(adata_st_i.obs['domain'],adata_st_i.obs['Region'])
        adata_st_i.uns['COM'] = COM
        COM_list.append(COM)
    
    return ARI_list,NMI_list,HOM_list,COM_list

# %% [cell 18: code]
results = {
    "ARI": [],
    "NMI": [],
    "HOM": [],
    "COM": []
}

for i,adatas in enumerate(groups):
    print(groups_names[i])
    ARI_list, NMI_list, HOM_list, COM_list = RunGraphST(adatas, slice_names)
    results["ARI"].append(ARI_list)
    results["NMI"].append(NMI_list)
    results["HOM"].append(HOM_list)
    results["COM"].append(COM_list)

for metric in results:
    df = pd.DataFrame(results[metric], index=groups_names)
    df.to_csv(f"/SABench/DownstreamTasks/DLPFC_3D_Spatial_Clustering_metrics/{metric}.csv", index=True, header=False)

