# MOp MERFISH 31-Slice Atlas Serial Alignment (companion mouse 2)

A companion atlas built from the second biological replicate of the Zhang et al. 2021 MERFISH MOp study, spanning 31 serial coronal sections. The SABench paper's Data45-Data77 enumeration covers **mouse 1 only** (33 slices); this task is the analogous full-atlas alignment over the **mouse 2** slices that are also locally compiled. It is paper-aligned in scoring but not in unit definition.

## Task

Align the 31 MERFISH slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`.

## Notes

- ~2,000–4,500 cells per slice across the 254-gene MOp MERFISH panel.
- The compiled bundle now includes `obs/Cluster` derived from the original MOp `cell_labels.csv` column `class_label`. This enables SABench-style landmark metrics as a **cluster-label proxy**, not a true anatomical-region score.
- For atomic adjacent-pair scoring this task internally yields 30 pairs.
