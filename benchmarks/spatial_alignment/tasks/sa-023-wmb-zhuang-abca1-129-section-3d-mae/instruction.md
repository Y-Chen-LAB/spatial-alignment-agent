# Whole Mouse Brain Zhuang-ABCA-1 129-Section Serial Alignment with 3D Ground-Truth MAE (Data99-227)

The full whole-mouse-brain MERFISH atlas from Zhang et al. 2023, sourced from the Allen Institute Brain Cell Atlas (Zhuang-ABCA-1 subset of the WMB collection). This is **the largest single alignment task in the SABench benchmark** (Yan et al. 2026, paper Methods page 13, Data99-227): 129 serial coronal sections spanning the full mouse brain, ~2.6 million cells with full transcriptome coverage at MERFISH single-cell resolution, scored against the Allen CCF (Common Coordinate Framework) 3D atlas using the paper's MAE metric.

## Source filter (147 → 129)

The Allen ABC `Zhuang-ABCA-1` raw release contains 147 sections. Per paper Methods: *"annotations—primarily small sections from the brain edges—were excluded, resulting in a final total of 129 slices."* The exact filter is "keep sections with any CCF parcellation coverage", which drops 9 anterior (`Zhuang-ABCA-1.001`–`009`) and 9 posterior (`Zhuang-ABCA-1.142`–`150`) edge sections. Recoverable deterministically via `scripts/compile_wmb_zhuang_abca1.py`.

## Task

Align the 129 MERFISH sections into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` (cell labels) values must match the source `.h5ad` observations.

## Per-slice contract

Each compiled h5ad exposes:

| Field | Content |
|---|---|
| `obsm/spatial` | deterministic random-rigid-perturbed (rotation + translation, seed = `990 + section_idx`) version of the CCF (x, y) — what the agent sees |
| `obs/x_GT, y_GT, z_GT` | held-out CCF atlas coordinates used by the MAE scorer |
| `obs/Region` | CCF parcellation substructure (paper-aligned label key, ≈688-category vocabulary) |
| `obs/_index` | unique cell label (string-encoded large integer) |
| `var/_index` | gene symbol from the 1,122-gene MERFISH panel |
| `X` | dense float32 raw counts |

## Scoring (paper-aligned, with paper-text MAE)

`metric_profile = "sabench_mae_gene_efficiency_v1"`. Aggregated metrics:

- `mae` — paper Methods page 13 formula `MAE = (1/(2N)) Σ (|x_pred − x_GT| + |y_pred − y_GT|)`, computed after **optimal rigid Procrustes** alignment of the agent's submitted frame to the held-out (x_GT, y_GT) frame to factor out global rotation/translation ambiguity. Lower is better, unbounded.
- `gene_pcc`, `gene_cosine`, `gene_ssim`, `gene_mi` — paper gene-grid metrics on the top-100 highly variable Zhuang-ABCA-1 genes.

`overall_mean` excludes MAE (lower-is-better, unbounded) and averages only the four gene metrics.

## Notes

- **Heavy task**: 2.6M cells, 1,122 genes, 128 adjacent-pair scoring passes. Allocate ≥1 hour for a single full scoring run. For development, the existing partial-WMB sa-028 / sa-029 (5- and 10-section windows from a different upstream subset) are smaller smoke targets.
- Cells/section: min 5,505, mean 19,366, max 37,068 (z-axis: anterior to posterior; the smallest sections are at the brain edges that survived the CCF filter).
- Companion fetch script: `scripts/fetch_wmb_abc_atlas.py` (downloads the 1.2 GB `Zhuang-ABCA-1-raw.h5ad` + metadata).
- Compiled slice catalog with per-section perturbation parameters: `~/Downloads/paper2bench-data/wmb_zhuang_abca1/compiled_catalog.json`.
- The paper places this dataset in the "challenging cases — large data" bucket (Fig 4) with explicit pre-alignment + downsampling + resolution-recovery as recommended mitigation strategies.
