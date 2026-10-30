"""PASTE / PASTE_p0 (SABench settings: alpha=0.1, norm=True; NumPy backend) on a serial task's
slices in task order, stacked with stack_slices_pairwise. Used to validate the MHPR table.
Run with venv-paste."""
import argparse, os, sys, time, tomllib
from pathlib import Path
import numpy as np, ot, paste as pst
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sabench_comparison"))
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--method", default="PASTE", choices=["PASTE", "PASTE_p0"])
a = ap.parse_args()
sys.argv = ["x", "--task", a.task]
src = (REPO / "scripts/sabench_comparison/run_paste_sweep_sabench.py").read_text().split("ref = rd(")[0]
exec(src)  # defines rd() (hand-rolled AnnData reader) and `items`
slices = [rd(p) for _, p in items]
pis = []
for i in range(len(slices) - 1):
    t0 = time.time(); kw = {}
    if a.method == "PASTE_p0":
        kw["G_init"] = pst.match_spots_using_spatial_heuristic(slices[i].obsm["spatial"], slices[i + 1].obsm["spatial"], use_ot=True)
    pis.append(pst.pairwise_align(slices[i], slices[i + 1], alpha=0.1, norm=True, verbose=False, backend=ot.backend.NumpyBackend(), **kw))
    print(i, time.time() - t0, flush=True)
new = pst.stack_slices_pairwise(slices, pis)
od = Path(f"/dcs07/chen/data/alignment-agent/sabench_comparison/validation/{a.method}_{a.task}{os.environ.get('OUT_SUFFIX', '')}"); od.mkdir(parents=True, exist_ok=True)
for (sid, _), s in zip(items, new):
    s.write(od / f"{sid}.h5ad")
print("wrote", od)
