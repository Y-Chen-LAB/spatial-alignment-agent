# Notes

These files were extracted from the downloaded Nature Computational Science source-data workbooks:

- `~/Downloads/43588_2026_977_MOESM6_ESM.xlsx`
- `~/Downloads/43588_2026_977_MOESM7_ESM.xlsx`
- `~/Downloads/43588_2026_977_MOESM8_ESM.xlsx`
- `~/Downloads/43588_2026_977_MOESM9_ESM.xlsx`
- `~/Downloads/43588_2026_977_MOESM10_ESM.xlsx`

## MMPC / STARmap Data39-41

The STARmap medial prefrontal cortex dataset does appear in the extracted source data, but only as dataset metadata in:

- `43588_2026_977_MOESM6_ESM/Fig1C.csv`

Relevant rows:

- `MMPC_processed,STARmap_1_processed.h5ad,1049,166,0.6943158716850242`
- `MMPC_processed,STARmap_2_processed.h5ad,1053,166,0.7655694001075527`
- `MMPC_processed,STARmap_3_processed.h5ad,1088,166,0.695788669383416`

## What Is Not Present In These Workbooks

After extracting and grepping the downloaded workbooks, there is no per-method MMPC / Data39-41 score table in these files.

The available sheets cover:

- dataset inventory (`Fig1C`)
- DLPFC metrics
- MHPR metrics
- DLPFC and MHPR robustness
- aggregate NGS/image rank summaries
- MOB metrics
- Flysta3D and MLN MAE tables

## Closest Comparable Paper-Side Data In These Files

The closest paper-side comparison available for imaging datasets in the downloaded workbooks is:

- `43588_2026_977_MOESM9_ESM/Image_datasets_rank.csv`

This is an aggregate rank summary across imaging datasets, not a raw MMPC-only score table.

## Best Local Task Matches For Paper Comparison

If the goal is a clean apples-to-apples local-agent-versus-paper comparison, the best currently available local tasks are:

- `sa-001-dlpfc-serial-align`
- `sa-009-mhpr-merfish-serial-align`

Why these are preferable:

- DLPFC has direct local source-data sheets for gene-based and landmark-based metrics:
  - `43588_2026_977_MOESM7_ESM/DLPFC_GeneBased_PCC.csv`
  - `43588_2026_977_MOESM7_ESM/DLPFC_GeneBased_Cos_Sim.csv`
  - `43588_2026_977_MOESM7_ESM/DLPFC_GeneBased_SSIM.csv`
  - `43588_2026_977_MOESM7_ESM/DLPFC_GeneBased_MI.csv`
  - `43588_2026_977_MOESM7_ESM/DLPFC_LandmarkBased_Heatmap.csv`
- MHPR likewise has direct local source-data sheets:
  - `43588_2026_977_MOESM7_ESM/MHPR_GeneBased_PCC.csv`
  - `43588_2026_977_MOESM7_ESM/MHPR_GeneBased_Cos_Sim.csv`
  - `43588_2026_977_MOESM7_ESM/MHPR_GeneBased_SSIM.csv`
  - `43588_2026_977_MOESM7_ESM/MHPR_GeneBased_MI.csv`
  - `43588_2026_977_MOESM7_ESM/MHPR_LandmarkBased_Heatmap.csv`

Recommended ordering:

1. Use `sa-001-dlpfc-serial-align` first for prompt and method A/B tests because it is a starter task and has the cleanest local paper-side comparison tables.
2. Use `sa-009-mhpr-merfish-serial-align` next for a second paper-comparable imaging-family task.
3. Treat `sa-018-starmap-mmpc-serial-align` as useful for solver exploration, but not as the first paper-comparison showcase because the downloaded local workbooks do not include a direct per-method MMPC score table.
