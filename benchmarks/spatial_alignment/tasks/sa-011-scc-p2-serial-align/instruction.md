# Cutaneous SCC Patient 2 ST 3-Slice Serial Alignment (Data10-12)

Three serial Spatial Transcriptomics (Spatial Transcriptomics v1, 100 µm spots) sections from a single human cutaneous squamous cell carcinoma patient (P2). Compiled from GSE144240 (Ji et al. 2020) by `scripts/compile_scc_geo.py`.

The SABench paper (Yan et al. 2026) uses these as Data10-12 in its sequencing-based, low-spot-count alignment evaluation.

## Task

Align the three ST slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- Each slice has ~640-670 spots × ~17k genes. The raw GEO data ships no curated cluster annotations, so SABench landmark/region metrics are not requested for this task; gene-grid metrics (PCC/Cos_Sim/SSIM/MI) score the alignment.
- Source tar: `~/Downloads/paper2bench-data/sabench_full/Data10-15_SCC/GSE144240_RAW.tar`.
- Compiled slices: `~/Downloads/paper2bench-data/SCC/SCC_P2_rep*.h5ad`.
