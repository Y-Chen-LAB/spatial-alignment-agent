# Albis single-cell serial alignment

10 serial sections of one simulated tissue block, measured at single-cell resolution (Xenium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 683 observations
- `s01`: 1,816 observations
- `s02`: 2,767 observations
- `s03`: 3,282 observations
- `s04`: 3,674 observations
- `s05`: 3,544 observations
- `s06`: 3,227 observations
- `s07`: 2,730 observations
- `s08`: 1,860 observations
- `s09`: 624 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
