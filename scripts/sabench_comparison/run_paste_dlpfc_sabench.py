"""Re-run PASTE / PASTE_p0 on DLPFC 151673-151676 the way SABench's SpatialAlignment/PASTE*.py does.

SABench: pst.pairwise_align(alpha=0.1, norm=True) per consecutive pair (PASTE_p0 adds
G_init = match_spots_using_spatial_heuristic(use_ot=True)), then stack_slices_pairwise, then
write the stacked AnnData. They ran the Torch GPU backend; we run the NumPy CPU backend.

--pose shuffled uses the SABench archive's 151673_shuffled / 151674_shuffled for slices 1-2
(the pose the paper's DLPFC inputs were in; the archive does not ship 151675/151676 shuffled,
which does not matter for PASTE because its output frame is slice 1's frame and the transport
plan is invariant to rigid motions of the inputs). --pose native uses the unshuffled slices.

Run with the venv at /dcs07/chen/data/alignment-agent/venv-paste (paste-bio 1.4.0, POT 0.9.5).
"""
import argparse, time
from pathlib import Path
import numpy as np, scanpy as sc, ot, paste as pst

NATIVE = Path("/users/ychen5/p2b-data/paper2bench-data/DLPFC")
SHUF = Path("/dcs07/chen/data/alignment-agent/sabench_ref/zip/SABench/Tutorial/coarse_align_interactive_data")
IDS = ["151673", "151674", "151675", "151676"]

ap = argparse.ArgumentParser()
ap.add_argument("--method", choices=["PASTE", "PASTE_p0"], default="PASTE")
ap.add_argument("--pose", choices=["shuffled", "native"], default="shuffled")
ap.add_argument("--out", type=Path, default=Path("/dcs07/chen/data/alignment-agent/sabench_comparison/validation"))
a = ap.parse_args()

slices = []
for i, sid in enumerate(IDS):
    p = SHUF / f"{sid}_shuffled.h5ad" if (a.pose == "shuffled" and i < 2) else NATIVE / f"{sid}.h5ad"
    s = sc.read_h5ad(p); s.var_names_make_unique(); slices.append(s); print("read", p, s.shape, flush=True)

pis = []
for i in range(len(slices) - 1):
    t0 = time.time()
    kw = {}
    if a.method == "PASTE_p0":
        kw["G_init"] = pst.match_spots_using_spatial_heuristic(slices[i].obsm["spatial"], slices[i + 1].obsm["spatial"], use_ot=True)
    pis.append(pst.pairwise_align(slices[i], slices[i + 1], alpha=0.1, norm=True, verbose=False,
                                  use_gpu=False, backend=ot.backend.NumpyBackend(), **kw))
    print(f"pair {i}: {time.time()-t0:.1f}s", flush=True)

new = pst.stack_slices_pairwise(slices, pis)
od = a.out / f"{a.method}_{a.pose}"; od.mkdir(parents=True, exist_ok=True)
for sid, s in zip(IDS, new):
    s.write(od / f"{a.method}_new_{sid}.h5ad")
np.save(od / "pis.npy", np.array([p for p in pis], dtype=object), allow_pickle=True)
print("wrote", od)
