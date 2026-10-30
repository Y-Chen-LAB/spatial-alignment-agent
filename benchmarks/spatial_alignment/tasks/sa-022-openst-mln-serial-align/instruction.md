# Open-ST Human Metastatic Lymph Node 19-Section Serial Alignment (Data80-98)

Nineteen high-resolution Open-ST sections from a human metastatic lymph node tumor sample (Schott et al. 2024). Each section contains ~55,000 cells with the full ~28,943-gene transcriptome. The SABench paper (Yan et al. 2026) uses these as **Data80-98** — its Open-ST large-scale alignment task family.

The compiled assets carry curated `obs/Region` annotations covering 17 cell-type / tissue-region categories (CAF, Cortex_CCL21, Dendritic_cells, Fibroblasts, Germinal_Center_Plasma_IgM_B_cell, Macrophages, M1_macrophages, …) — full SABench landmark scoring (matching accuracy + alpha-shape region overlap) is available alongside the gene-grid metrics.

## Task

Align the 19 Open-ST sections into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`.

## Notes

- Section IDs are **non-contiguous** (02, 03, 04, 05, 06, 07, 09, 11, 17, 18, 19, 23, 24, 25, 26, 28, 33, 34, 36). They are real serial-section indices from the source bundle; missing numbers reflect sections excluded by the original Open-ST QC pipeline.
- **Large data**: 19 × 55k ≈ 1M cells × 28,943 genes. Expect significant runtime; the task is one of the paper's large-scale stress tests.
- Profile is `sabench_light_alignment_v1`: 4 gene-grid metrics + matching_accuracy + overlap_ratio_region (alpha-shapes), all paper-aligned.
- Compiled by the parallel agent at `(withheld)`.
