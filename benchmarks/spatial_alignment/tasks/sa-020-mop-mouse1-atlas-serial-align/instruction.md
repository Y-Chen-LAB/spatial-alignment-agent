# MOp MERFISH 33-Slice Atlas Serial Alignment (Data45-77, mouse 1)

The full mouse 1 atlas of the MERFISH primary motor cortex (MOp) study (Zhang et al. 2021), spanning 33 serial coronal sections from anterior to posterior. The SABench paper (Yan et al. 2026) enumerates this exact set as **Data45-Data77** — its largest single-family alignment task.

This task runs serial alignment over all 33 mouse-1 slices in one shot, complementing the existing six 6-slice "sample" subtasks (sa-101…sa-106) which use the same data partitioned into smaller groups.

## Task

Align the 33 MERFISH slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- ~2,300–4,300 cells per slice across the published 254-gene MERFISH panel.
- Compiled slices live at `(withheld)`. The integer suffix is the original brain-section index from the source bundle, **not** sequential — slices are coronal sections sampled across the full MOp atlas.
- The compiled bundle now includes `obs/Cluster` derived from the original MOp `cell_labels.csv` column `class_label`. This enables SABench-style landmark metrics as a **cluster-label proxy**, not a true anatomical-region score.
- Same proxy-label scoring profile as the existing sa-101…sa-106 group tasks. This task differs only in scope: 33 slices in one alignment vs 6 in each group task.
