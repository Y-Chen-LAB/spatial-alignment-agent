"""Run the SABench authors' metric functions verbatim against ours on one real pair."""
import numpy as np, pandas as pd
import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).parent))
import shim
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial import cKDTree

# ---------------- authors' code, copied verbatim from FeatureSimilarity/GeneBased.ipynb
def overlap_bygrids(slice1, slice2, spatial_key1, spatial_key2, m, n):
    adata1=slice1.copy(); adata2=slice2.copy()
    coords1 = adata1.obsm[spatial_key1]; coords2 = adata2.obsm[spatial_key2]
    x_min = max(np.min(coords1[:,0]), np.min(coords2[:,0])); x_max = min(np.max(coords1[:,0]), np.max(coords2[:,0]))
    y_min = max(np.min(coords1[:,1]), np.min(coords2[:,1])); y_max = min(np.max(coords1[:,1]), np.max(coords2[:,1]))
    def points_per_grid(coords, x_min,x_max,y_min,y_max,m,n):
        c=[]
        for i in range(n):
            for j in range(m):
                xs=x_min+i*(x_max-x_min)/n; ys=y_min+j*(y_max-y_min)/m
                xe=xs+(x_max-x_min)/n; ye=ys+(y_max-y_min)/m
                c.append(np.sum((coords[:,0]>=xs)&(coords[:,0]<xe)&(coords[:,1]>=ys)&(coords[:,1]<ye)))
        return np.mean(c)
    average_cells = min(points_per_grid(coords1,x_min,x_max,y_min,y_max,m,n),
                        points_per_grid(coords2,x_min,x_max,y_min,y_max,m,n))
    threshold_points = average_cells/2
    def check(ac,xs,ys,xe,ye,t):
        return len(ac[(ac[:,0]>=xs)&(ac[:,0]<xe)&(ac[:,1]>=ys)&(ac[:,1]<ye)])>=t
    adata1.obs['grid_label']=None; adata2.obs['grid_label']=None
    gc=1
    for i in range(n):
        for j in range(m):
            xs=x_min+i*(x_max-x_min)/n; ys=y_min+j*(y_max-y_min)/m
            xe=xs+(x_max-x_min)/n; ye=ys+(y_max-y_min)/m
            in1=check(coords1,xs,ys,xe,ye,threshold_points); in2=check(coords2,xs,ys,xe,ye,threshold_points)
            label=f"Grid_{gc}" if in1 and in2 else "NonOverlap"
            adata1.obs.loc[(coords1[:,0]>=xs)&(coords1[:,0]<xe)&(coords1[:,1]>=ys)&(coords1[:,1]<ye),'grid_label']=label
            adata2.obs.loc[(coords2[:,0]>=xs)&(coords2[:,0]<xe)&(coords2[:,1]>=ys)&(coords2[:,1]<ye),'grid_label']=label
            if in1 and in2: gc+=1
    return adata1,adata2

def _means(adata1, adata2, gene_name):
    g1 = adata1.X[:, adata1.var_names==gene_name]
    g2 = adata2.X[:, adata2.var_names==gene_name]
    g1 = g1.toarray().squeeze() if hasattr(g1,'toarray') else np.asarray(g1).squeeze()
    g2 = g2.toarray().squeeze() if hasattr(g2,'toarray') else np.asarray(g2).squeeze()
    df1=pd.DataFrame({'expression':g1,'grid_label':adata1.obs['grid_label'].values})
    df2=pd.DataFrame({'expression':g2,'grid_label':adata2.obs['grid_label'].values})
    df1=df1[(df1['grid_label']!='NonOverlap')&(df1['grid_label'].notnull())]
    df2=df2[(df2['grid_label']!='NonOverlap')&(df2['grid_label'].notnull())]
    return df1.groupby('grid_label').mean().sort_index(), df2.groupby('grid_label').mean().sort_index()

def author_metrics(adata1, adata2, gene_name, L=1):
    m1,m2=_means(adata1,adata2,gene_name)
    a=np.array(m1).squeeze(); b=np.array(m2).squeeze()
    if m1.size<2 or m2.size<2: return dict(pcc=0,cos=0,ssim=0,mi=0)
    pcc=pearsonr(m1['expression'],m2['expression'])[0]
    cos=a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))
    a2,b2=a/a.max(),b/b.max()
    mu1,mu2=a2.mean(),b2.mean()
    s1=np.sqrt(((a2-mu1)**2).mean()); s2=np.sqrt(((b2-mu2)**2).mean()); s12=((a2-mu1)*(b2-mu2)).mean()
    C1,C2=(0.01*L)**2,(0.03*L)**2; C3=C2/2
    ssim=((2*mu1*mu2+C1)/(mu1**2+mu2**2+C1))*((2*s1*s2+C2)/(s1**2+s2**2+C2))*((s12+C3)/(s1*s2+C3))
    mi=mutual_info_regression(np.array(m1), b)[0]
    return dict(pcc=pcc,cos=cos,ssim=ssim,mi=mi)

def author_accuracy(adata1, adata2, key='Region'):
    c1=adata1.obsm['spatial']; c2=adata2.obsm['spatial']
    x_min=max(c1[:,0].min(),c2[:,0].min()); x_max=min(c1[:,0].max(),c2[:,0].max())
    y_min=max(c1[:,1].min(),c2[:,1].min()); y_max=min(c1[:,1].max(),c2[:,1].max())
    o1=(c1[:,0]>=x_min)&(c1[:,0]<=x_max)&(c1[:,1]>=y_min)&(c1[:,1]<=y_max)
    o2=(c2[:,0]>=x_min)&(c2[:,0]<=x_max)&(c2[:,1]>=y_min)&(c2[:,1]<=y_max)
    a1=adata1[o1,:]; a2=adata2[o2,:]
    tree=cKDTree(a1.obsm['spatial']); _,idx=tree.query(a2.obsm['spatial'])
    if len(idx)==0: return 0.0
    l1=np.asarray(a1.obs[key]); l2=np.asarray(a2.obs[key])
    return float(np.mean(l1[idx]==l2))

# ---------------- ours
import sys; sys.path.insert(0,'src')
from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench import metrics as M

P1='/Users/yiqun/Downloads/paper2bench-data/DLPFC/151673.h5ad'
P2='/Users/yiqun/Downloads/paper2bench-data/DLPFC/151674.h5ad'
a1=shim.read(P1); a2=shim.read(P2)
s1=H5ADSlice(__import__('pathlib').Path(P1), slice_id='151673')
s2=H5ADSlice(__import__('pathlib').Path(P2), slice_id='151674')
c1=s1.spatial; c2=s2.spatial

genes=pd.read_csv('benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/data/marker_genes.csv').iloc[:,0].tolist()[:20]
genes=[g for g in genes if g in a1.var_names and g in a2.var_names]
print(f"comparing {len(genes)} genes on 151673 x 151674\n")

# grid labels
ga1,ga2=overlap_bygrids(a1,a2,'spatial','spatial',10,10)
ol,orr=M._pair_grid_labels(c1,c2,10)
auth_l=np.array([x if x else 'NonOverlap' for x in ga1.obs['grid_label']])
auth_r=np.array([x if x else 'NonOverlap' for x in ga2.obs['grid_label']])
print("grid labels identical (left) :", bool((auth_l==np.asarray(ol)).all()))
print("grid labels identical (right):", bool((auth_r==np.asarray(orr)).all()))
print("n grids authors:", len(set(auth_l))-1, " ours:", len(set(np.asarray(ol)))-1)

rows=[]
lg,le=s1.expression_for_genes(genes); rg,re_=s2.expression_for_genes(genes)
li={g:i for i,g in enumerate(lg)}; ri={g:i for i,g in enumerate(rg)}
for g in genes:
    A=author_metrics(ga1,ga2,g)
    df1=pd.DataFrame({'expression':le[:,li[g]],'grid_label':ol})
    df2=pd.DataFrame({'expression':re_[:,ri[g]],'grid_label':orr})
    df1=df1[df1.grid_label!='NonOverlap']; df2=df2[df2.grid_label!='NonOverlap']
    m1=df1.groupby('grid_label').mean().sort_index(); m2=df2.groupby('grid_label').mean().sort_index()
    cl=sorted(set(m1.index)&set(m2.index))
    aa=m1.loc[cl,'expression'].to_numpy(); bb=m2.loc[cl,'expression'].to_numpy()
    rows.append(dict(gene=g,
        d_pcc=abs(A['pcc']-M._safe_pcc(aa,bb)), d_cos=abs(A['cos']-M._safe_cosine(aa,bb)),
        d_ssim=abs(A['ssim']-M._safe_ssim(aa,bb)), d_mi=abs(A['mi']-M._safe_mi(aa,bb)),
        mi_auth=A['mi'], mi_ours=M._safe_mi(aa,bb)))
d=pd.DataFrame(rows)
print("\nmax |authors - ours| per metric over", len(d), "genes:")
print(d[['d_pcc','d_cos','d_ssim','d_mi']].max().to_string())
print("\nMI side by side (first 8):")
print(d[['gene','mi_auth','mi_ours']].head(8).to_string(index=False))

print("\nmatching_accuracy  authors:", round(author_accuracy(a1,a2),8),
      " ours:", round(M._matching_accuracy(s1,s2,c1,c2,'Region'),8))

print("\n=== whole-pipeline aggregate: authors vs our _gene_grid_metrics, same gene set ===")
auth = {k: float(np.mean([author_metrics(ga1,ga2,g)[k] for g in genes])) for k in ("pcc","cos","ssim","mi")}
ours = M._gene_grid_metrics(s1,s2,c1,c2,genes,10)
for k,v in [("pcc","gene_pcc"),("cos","gene_cosine"),("ssim","gene_ssim"),("mi","gene_mi")]:
    print(f"  {v:12s} authors={auth[k]:.6f}  ours={ours[v]:.6f}  diff={abs(auth[k]-ours[v]):.2e}")
print("  dtype of our grid means:", le.dtype)
