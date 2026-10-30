# STARmap Mouse Medial Prefrontal Cortex 3-Slice Serial Alignment (Data39-41)

Three STARmap slices from the mouse medial prefrontal cortex from the SABench paper (Yan et al. 2026, Data39-41), sourced from the SDMBench processed release.

As described in the SABench paper, random rigid perturbations are applied before benchmarking. This local task uses deterministic seeded rigid perturbations compiled into the repo-local `.h5ad` files referenced by `task.toml`.

## Task

Align the three slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- `obs/Region` is preserved from the SDMBench STARmap bundle, so the task supports the paper-aligned SABench landmark and region-overlap metrics in addition to gene-based metrics.
- Source bundle: `~/Downloads/22565200.zip` from the SDMBench dataset release.
- The exact perturbation parameters are recorded in `(withheld)`.
