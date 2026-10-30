# Albis visium spot sparse serial alignment

10 serial sections of one simulated tissue block, measured at 55 um capture spots on a 100 um lattice (Visium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

The data passed standard QC: observations with fewer than 3 detected genes were removed.

- `s00`: 383 observations
- `s01`: 766 observations
- `s02`: 1,054 observations
- `s03`: 1,224 observations
- `s04`: 1,309 observations
- `s05`: 1,314 observations
- `s06`: 1,227 observations
- `s07`: 1,042 observations
- `s08`: 769 observations
- `s09`: 378 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
