# P22 mouse brain RNA to H3K4me3 cross-modality alignment

Two neighbouring sections of postnatal day 22 mouse brain measured on barcoded grids with different assays. `rna` holds RNA counts per spot (genes). `h3k4me3` holds normalised histone H3K4me3 signal per spot over genomic peaks (features are genomic intervals, not genes). The sections were placed independently. Bring `h3k4me3` into the coordinate system of `rna`.

Measured at 20 µm barcoded pixels (spatial CUT&Tag-RNA-seq). Coordinates are in microns, and each section's physical scale is correct as given.

- `rna`: 9,370 observations
- `h3k4me3`: 9,548 observations

Keep `rna` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
