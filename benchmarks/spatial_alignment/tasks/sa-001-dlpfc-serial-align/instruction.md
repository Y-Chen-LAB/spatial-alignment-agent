# DLPFC Serial Alignment

Align the four 10x Visium DLPFC slices into a common coordinate system.

Submit one coordinate CSV per slice with columns `obs_id,x,y`. The row order and `obs_id` values must match the source `.h5ad` file.

The benchmark scores adjacent slice pairs using gene-grid similarity over DLPFC marker genes, landmark matching by `Region`, region-grid overlap, runtime, and memory.
