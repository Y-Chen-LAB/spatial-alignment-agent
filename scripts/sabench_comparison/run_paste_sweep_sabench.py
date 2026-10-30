"""PASTE (alpha=0.1, norm=True, NumPy backend) on each level of a DLPFC/MHPR robustness sweep task,
reference slice first, then Overlap_accuracy on the stacked output, as SABench's robustness
evaluation does. Writes one CSV row per level. Run with venv-paste."""
import argparse, os, sys, tomllib, time
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, ot, paste as pst
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "src"))
from genomics_agent_bench import sabench_exact as SX
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--method", default="PASTE", choices=["PASTE", "PASTE_p0"])
a = ap.parse_args()
spec = tomllib.load(open(REPO / f"benchmarks/spatial_alignment/tasks/{a.task}/task.toml", "rb"))
items = [s.split("=", 1) for s in spec["slices"]]
import anndata as ad, h5py
from scipy import sparse
from genomics_agent_bench.h5ad import _read_obs_column, _read_string_dataset
def rd(p):
    """Our compiled sweep files lack anndata encoding-version attrs, so read the needed fields by hand."""
    p = Path(p); p = p if p.is_absolute() else REPO / p
    with h5py.File(p, "r") as h:
        X = h["X"]
        if isinstance(X, h5py.Group):
            enc = X.attrs.get("encoding-type"); shape = tuple(int(v) for v in X.attrs["shape"])
            M = sparse.csr_matrix if enc == "csr_matrix" else sparse.csc_matrix
            X = M((X["data"][()], X["indices"][()], X["indptr"][()]), shape=shape).tocsr()
        else:
            X = sparse.csr_matrix(np.asarray(X[()]))
        obs = pd.DataFrame({"Region": _read_obs_column(h, "Region")}, index=_read_string_dataset(h, "obs/_index").astype(str))
        var = pd.DataFrame(index=_read_string_dataset(h, "var/_index").astype(str))
        s = ad.AnnData(X=X if os.environ.get("KEEP_X_DTYPE") else X.astype(np.float32), obs=obs, var=var, obsm={"spatial": np.asarray(h["obsm/spatial"][()], float)})
    s.var_names_make_unique(); return s
ref = rd(items[0][1]); rows = []
# MHPR sweep slices carry Cell_class in obs/Region (see score_sabench_paper_metrics.REGION_OVERRIDE)
relabel = None
if "mhpr" in a.task:
    d6 = rd("/users/ychen5/p2b-data/paper2bench-data/MHPR_paper/MHPR_animal1_n009.h5ad")
    relabel = dict(zip(d6.obs_names.astype(str), d6.obs["Region"].astype(str)))
for sid, p in items[1:]:
    t0 = time.time(); other = rd(p)
    kw = {}
    if a.method == "PASTE_p0":
        kw["G_init"] = pst.match_spots_using_spatial_heuristic(ref.obsm["spatial"], other.obsm["spatial"], use_ot=True)
    try:
        pi = pst.pairwise_align(ref, other, alpha=0.1, norm=True, verbose=False, backend=ot.backend.NumpyBackend(), **kw)
        new = pst.stack_slices_pairwise([ref, other], [pi])
        regs = [np.asarray(new[0].obs["Region"]).astype(str),
                np.array([relabel[n] for n in new[1].obs_names.astype(str)]) if relabel else np.asarray(new[1].obs["Region"]).astype(str)]
        e = [SX.ExactSlice(s, n.obsm["spatial"], region=r) for s, n, r in zip([items[0][0], sid], new, regs)]
        acc = SX.overlap_accuracy(e[0], e[1]); err = ""
    except Exception as exc:
        acc, err = float("nan"), repr(exc)[:200]
    rows.append({"task": a.task, "method": a.method, "level": sid.split("_", 1)[1], "Average_Accuracy": acc, "seconds": time.time() - t0, "error": err})
    print(rows[-1], flush=True)
out = Path("/dcs07/chen/data/alignment-agent/sabench_comparison/validation/sweeps"); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / f"{a.task}_{a.method}{os.environ.get('OUT_SUFFIX', '')}.csv", index=False)
