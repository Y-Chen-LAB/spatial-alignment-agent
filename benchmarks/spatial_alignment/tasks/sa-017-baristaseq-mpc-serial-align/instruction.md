# BaristaSeq Mouse Primary Cortex 3-Slice Serial Alignment (Data36-38)

Three BaristaSeq slices from the mouse primary cortex from the SABench paper (Yan et al. 2026, Data36-38), sourced from the SDMBench processed release.

In the paper's evaluation pipeline, random rigid perturbations are applied to disrupt the original coordinate frame before alignment benchmarking. This local task uses deterministic seeded rigid perturbations compiled into the repo-local `.h5ad` files referenced by `task.toml`.

## Task

Align the three slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- This family retains published `obs/Region` and `obs/layer` annotations, so the paper-aligned SABench landmark and region-overlap metrics are enabled in addition to the gene-based metrics.
- Source bundle: `~/Downloads/22550074.zip` from the SDMBench dataset release.
- The exact perturbation parameters are recorded in `(withheld)`.
