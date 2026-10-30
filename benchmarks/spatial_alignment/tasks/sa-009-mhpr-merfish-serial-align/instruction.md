# MHPR MERFISH 5-Slice Serial Alignment (Data5-9)

Five consecutive coronal MERFISH sections of the mouse hypothalamic preoptic region (MHPR), Animal 1, Bregma -0.14 → +0.06 mm at 0.05 mm spacing. Compiled from the Moffitt et al. 2018 Dryad release (`Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv`) by `scripts/compile_mhpr_csv.py`.

The SABench paper (Yan et al. 2026) uses this dataset as Data5-9 for imaging-based MERFISH alignment evaluation (paper Fig 2c-d).

## Task

Align the five MERFISH slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- Each slice has 5,488 to 5,926 cells across 155 panel genes (Blank_* control probes and the all-NaN `Fos` probe dropped). Slices are Animal 1 at Bregma -0.04, -0.09, -0.14, -0.19, -0.24; cells with `Cell_class == Ambiguous` are removed, matching the paper's Data5-9.
- `obs/Region` carries the published `Cell_class` annotation (Inhibitory, Excitatory, Astrocyte, Endothelial, …) — used as the label key for SABench landmark/region metrics.
- Source CSV: `~/Downloads/paper2bench-data/sabench_full/Data5-9_MHPR/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv`.
- Compiled slices: `~/Downloads/paper2bench-data/MHPR_paper/MHPR_animal1_*.h5ad` (paper-faithful rebuild, 2026-09-14; the earlier `MHPR/` directory used a different Bregma set and kept all cells).
