# STARmap PLUS sagittal section, four-slice depth stack

Four thin sections from one block of mouse brain tissue, in order, each measured at single-cell resolution. Each section was placed on the slide independently, so they do not share a coordinate system. Bring every section into the coordinate system of `s00`.

Measured at single-cell resolution (STARmap PLUS, 1,022 genes). Coordinates are in microns, and each section's physical scale is correct as given.

- `s00`: 51,897 observations
- `s01`: 51,897 observations
- `s02`: 51,897 observations
- `s03`: 51,897 observations

Keep `s00` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
