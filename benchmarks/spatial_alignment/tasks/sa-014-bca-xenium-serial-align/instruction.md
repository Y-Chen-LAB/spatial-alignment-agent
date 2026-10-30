# Human Breast Cancer Adjacent Xenium Serial Alignment (Data229-230)

Two adjacent Xenium slices from the SABench human breast cancer cross-platform family. This task focuses on the adjacent Xenium pair (`Data229`, `Data230`) rather than the separate Visium slice (`Data228`).

## Task

Align the two Xenium slices into a shared coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- This task is the paper-aligned raw-Xenium counterpart to the existing downsampled and resolution-recovery BCA tasks.
- The local `BCA_raw/slice_Rep*.h5ad` files already carry `obs/Cluster`, so the deterministic scorer can use both the SABench gene-based metrics and the landmark/region metrics.
- The direct raw-bundle compilation at `(withheld)` is retained as provenance, but the task uses the richer SABench-prepared raw slices because they preserve the paper's annotation-bearing structure.
