# Converted from /Users/yiqun/Downloads/SABench/Plot/BCA_plot.ipynb
# Notebook cell count: 20

# %% [cell 1: code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import squidpy as sq

# %% [cell 2: code]
Raw_Xenium_Rep1 = sc.read_h5ad('/SABench/Data/BCA_processed/XeniumBCARep1_processed.h5ad')
Raw_Xenium_Rep2 = sc.read_h5ad('/SABench/Data/BCA_processed/XeniumBCARep2_processed.h5ad')
Raw=[Raw_Xenium_Rep1,Raw_Xenium_Rep2]

# %% [cell 3: code]
SPACEL_BCA_rep1 = sc.read_h5ad('/SABench/AlignmentResults/SPACEL_BCA/SPACEL_BCA_new_slices_Xenium_Rep1.h5ad')
SPACEL_BCA_rep2 = sc.read_h5ad('/SABench/AlignmentResults/SPACEL_BCA/SPACEL_BCA_new_slices_Xenium_Rep2.h5ad')
SPACEL=[SPACEL_BCA_rep1,SPACEL_BCA_rep2]

# %% [cell 4: code]
PA_Xenium_Rep1 = sc.read_h5ad('/SABench/Improvement/BCA_PA/Data_PA/slice_Rep1_PA.h5ad')
PA_Xenium_Rep2 = sc.read_h5ad('/SABench/Improvement/BCA_PA/Data_PA/slice_Rep2_PA.h5ad')
PA=[PA_Xenium_Rep1,PA_Xenium_Rep2]

# %% [cell 5: code]
STalign_rep1 = sc.read_h5ad('/SABench/AlignmentResults/STalign_BCA/STalign_BCA_new_slices_Xenium_Rep1.h5ad')
STalign_rep2 = sc.read_h5ad('/SABench/AlignmentResults/STalign_BCA/STalign_BCA_new_slices_Xenium_Rep2.h5ad')
STalign=[STalign_rep1,STalign_rep2]

# %% [cell 6: code]
STalign_PA_rep1 = sc.read_h5ad('/SABench/Improvement/BCA_PA/AlignmentResults/STalign_BCA_PA/STalign_BCA_PA_new_slices_Xenium_Rep1.h5ad')
STalign_PA_rep2 = sc.read_h5ad('/SABench/Improvement/BCA_PA/AlignmentResults/STalign_BCA_PA/STalign_BCA_PA_new_slices_Xenium_Rep2.h5ad')
STalign_PA=[STalign_PA_rep1,STalign_PA_rep2]

# %% [cell 7: code]
STAligner_PADR_Rep1 = sc.read_h5ad('/SABench/Improvement/BCA_PADR/FinalSlices/STAligner_BCA/STAligner_BCA_new_lanmark6_425x425_Xenium_Rep1.h5ad')
STAligner_PADR_Rep2 = sc.read_h5ad('/SABench/Improvement/BCA_PADR/FinalSlices/STAligner_BCA/STAligner_BCA_new_lanmark6_425x425_Xenium_Rep2.h5ad')
STAligner_PADR=[STAligner_PADR_Rep1,STAligner_PADR_Rep2]

# %% [cell 8: code]
Spateo_rep1 = sc.read_h5ad('/SABench/AlignmentResults/Spateo_nonrigid_BCA/Spateo_nonrigid_BCA_new_slices_Xenium_Rep1.h5ad')
Spateo_rep2 = sc.read_h5ad('/SABench/AlignmentResults/Spateo_nonrigid_BCA/Spateo_nonrigid_BCA_new_slices_Xenium_Rep2.h5ad')
Spateo=[Spateo_rep1,Spateo_rep2]

# %% [cell 9: code]
Spateo_BestPara_rep1 = sc.read_h5ad('/SABench/Improvement/BCA_UseBestParameter/AlignmentResults/Spateo_nonrigid_BCA_BP/Spateo_nonrigid_BCA_BP_new_slices_Xenium_Rep1.h5ad')
Spateo_BestPara_rep2 = sc.read_h5ad('/SABench/Improvement/BCA_UseBestParameter/AlignmentResults/Spateo_nonrigid_BCA_BP/Spateo_nonrigid_BCA_BP_new_slices_Xenium_Rep2.h5ad')
Spateo_BestPara=[Spateo_BestPara_rep1,Spateo_BestPara_rep2]

# %% [cell 10: code]
def plot_BCA(slices,title,figsize=(8, 8),s1=0.2,s2=0.2):
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.title(title, fontsize=20)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])

    ax.scatter(
    slices[0].obsm['spatial'][:,0],
    slices[0].obsm['spatial'][:,1],
    s=s1,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="#2064ab"
    )

    ax.scatter(
    slices[1].obsm['spatial'][:,0],
    slices[1].obsm['spatial'][:,1],
    s=s2,
    linewidth=0,
    edgecolor=None,
    marker="o",
    color="#de4951"
    )

# %% [cell 11: code]
plot_BCA(Raw, 'Raw data',(8,6))

# %% [cell 12: code]
plot_BCA(SPACEL, 'SPACEL')

# %% [cell 13: code]
plot_BCA(PA, 'PA',(7.5,7))

# %% [cell 14: code]
plot_BCA(STalign, 'STalign',(8,6))

# %% [cell 15: code]
plot_BCA(STalign_PA, 'STalign_PA')

# %% [cell 16: code]
plot_BCA(STAligner_PADR, 'STAligner_PADR')

# %% [cell 17: code]
plot_BCA(Spateo, 'Spateo',(7.5,7))

# %% [cell 18: code]
plot_BCA(Spateo_BestPara, 'Spateo_BestPara',(7.5,7))

# %% [cell 19: code]
STAIR_rep1 = sc.read_h5ad('/SABench/AlignmentResults/STAIR_BCA/STAIR_BCA_new_slices_Xenium_Rep1.h5ad')
STAIR_rep2 = sc.read_h5ad('/SABench/AlignmentResults/STAIR_BCA/STAIR_BCA_new_slices_Xenium_Rep2.h5ad')
STAIR=[STAIR_rep1,STAIR_rep2]

# %% [cell 20: code]
plot_BCA(STAIR, 'STAIR',(7.5,7))

