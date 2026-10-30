# Cutaneous SCC Patient 5 ST 3-Slice Serial Alignment (Data13-15)

Three serial Spatial Transcriptomics sections from a single human cutaneous squamous cell carcinoma patient (P5). Compiled from GSE144240 (Ji et al. 2020). The SABench paper (Yan et al. 2026) uses these as Data13-15.

## Task

Align the three ST slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- ~520-590 spots × ~17k genes per slice. No curated cluster annotations in the raw GEO bundle; SABench landmark/region metrics are not requested for this task.
- Compiled by `scripts/compile_scc_geo.py` from `~/Downloads/paper2bench-data/sabench_full/Data10-15_SCC/GSE144240_RAW.tar`.
