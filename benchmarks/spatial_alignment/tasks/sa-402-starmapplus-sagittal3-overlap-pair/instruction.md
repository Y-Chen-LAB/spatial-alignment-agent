# STARmap PLUS sagittal sections with 30% overlap

Two neighbouring thin sections from one block of mouse brain tissue that only partly overlap: each covers a different part of the tissue, and they share a band of roughly 30%. The second section was placed on the slide independently. Bring `s01` into the coordinate system of `s00`.

Measured at single-cell resolution (STARmap PLUS, 1,022 genes). Coordinates are in microns, and each section's physical scale is correct as given.

- `s00`: 32,770 observations
- `s01`: 31,281 observations

Keep `s00` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
