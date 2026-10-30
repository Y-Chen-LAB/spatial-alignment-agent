"""PASTE2 on DLPFC as in SABench SpatialAlignment/PASTE2.py: partial_pairwise_align(s=0.8, alpha=0.1)
per consecutive pair, then projection.partial_stack_slices_pairwise. Slices 1-2 in the SABench
shuffled pose (see run_paste_dlpfc_sabench.py). Run with .method-venv-ot (paste2 1.0.1, POT 0.9.7)."""
import sys, time
from pathlib import Path
import numpy as np, scanpy as sc
from paste2 import PASTE2, projection
np.random.seed(0)
NATIVE = Path("/users/ychen5/p2b-data/paper2bench-data/DLPFC")
SHUF = Path("/dcs07/chen/data/alignment-agent/sabench_ref/zip/SABench/Tutorial/coarse_align_interactive_data")
IDS = ["151673", "151674", "151675", "151676"]
slices = []
for i, sid in enumerate(IDS):
    p = SHUF / f"{sid}_shuffled.h5ad" if i < 2 else NATIVE / f"{sid}.h5ad"
    s = sc.read_h5ad(p); s.var_names_make_unique(); slices.append(s)
pis = []
for i in range(3):
    t0 = time.time(); pis.append(PASTE2.partial_pairwise_align(slices[i], slices[i + 1], s=0.8, alpha=0.1)); print(i, time.time() - t0, flush=True)
new = projection.partial_stack_slices_pairwise(slices, pis)
od = Path("/dcs07/chen/data/alignment-agent/sabench_comparison/validation/PASTE2_shuffled"); od.mkdir(parents=True, exist_ok=True)
for sid, s in zip(IDS, new):
    s.write(od / f"PASTE2_new_{sid}.h5ad")
