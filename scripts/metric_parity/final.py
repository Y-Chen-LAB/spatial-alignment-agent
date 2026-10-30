import sys, numpy as np, pandas as pd
SP=str(__import__("pathlib").Path(__file__).parent)
sys.path.insert(0,SP); sys.path.insert(0,'src')
import shim
from pathlib import Path
from alpha_shapes import Alpha_Shaper
from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench import metrics as M
src=open(SP+"/xcheck.py").read()
exec(src.split("# ---------------- ours")[0].split("import shim")[1])

P1='/Users/yiqun/Downloads/paper2bench-data/DLPFC/151673.h5ad'; P2='/Users/yiqun/Downloads/paper2bench-data/DLPFC/151674.h5ad'
a1=shim.read(P1); a2=shim.read(P2)
s1=H5ADSlice(Path(P1),slice_id='151673'); s2=H5ADSlice(Path(P2),slice_id='151674')
c1,c2=s1.spatial,s2.spatial
ga1,ga2=overlap_bygrids(a1,a2,'spatial','spatial',10,10)
genes=pd.read_csv('benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/data/marker_genes.csv').iloc[:,0].tolist()[:20]
genes=[g for g in genes if g in a1.var_names and g in a2.var_names]

print("=== aggregate gene_mi: authors' unseeded spread over 10 repeats vs ours ===")
reps=[float(np.mean([author_metrics(ga1,ga2,g)['mi'] for g in genes])) for _ in range(10)]
ours=M._gene_grid_metrics(s1,s2,c1,c2,genes,10)['gene_mi']
print(f"  authors: min={min(reps):.6f} max={max(reps):.6f} spread={max(reps)-min(reps):.6f}")
print(f"  ours (seeded): {ours:.6f}   inside authors' range: {min(reps)<=ours<=max(reps)}")

print("\n=== overlap_ratio_region: authors' ROI_overlap per region vs ours ===")
def roi(label):
    x_min=max(c1[:,0].min(),c2[:,0].min()); x_max=min(c1[:,0].max(),c2[:,0].max())
    y_min=max(c1[:,1].min(),c2[:,1].min()); y_max=min(c1[:,1].max(),c2[:,1].max())
    o1=(c1[:,0]>=x_min)&(c1[:,0]<=x_max)&(c1[:,1]>=y_min)&(c1[:,1]<=y_max)
    o2=(c2[:,0]>=x_min)&(c2[:,0]<=x_max)&(c2[:,1]>=y_min)&(c2[:,1]<=y_max)
    A=a1[o1]; B=a2[o2]
    m1=np.where(np.asarray(A.obs['Region'])==label); m2=np.where(np.asarray(B.obs['Region'])==label)
    if len(m1[0])==0 or len(m2[0])==0: return 0.0
    s_1=Alpha_Shaper(A.obsm['spatial'][m1,].squeeze()).get_shape(alpha=12.0)
    s_2=Alpha_Shaper(B.obsm['spatial'][m2,].squeeze()).get_shape(alpha=12.0)
    return float(s_1.intersection(s_2).area/s_1.union(s_2).area)
mean_ours, per = M._overlap_ratio_region(s1,s2,c1,c2,'Region',12.0)
print(f"  {'region':8s} {'authors':>9s} {'ours':>9s} {'diff':>10s}")
for r in sorted(per):
    a=roi(r); print(f"  {r:8s} {a:9.6f} {per[r]:9.6f} {abs(a-per[r]):10.2e}")

print("\n=== paper's 6 named DLPFC regions only (Layer1 excluded, as in the notebook) ===")
cfg=M.AlignmentScoreConfig(label_key='Region', regions=('Layer2','Layer3','Layer4','Layer5','Layer6','WM'))
mean6, per6 = M._overlap_ratio_region(s1,s2,c1,c2,'Region',12.0,cfg.regions)
print(f"  ours restricted mean={mean6:.6f}   all-label mean={mean_ours:.6f}")
