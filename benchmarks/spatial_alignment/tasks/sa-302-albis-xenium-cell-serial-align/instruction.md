# Albis xenium cell serial alignment

10 serial sections of one simulated tissue block, measured at single-cell resolution (Xenium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 16,682 observations
- `s01`: 45,325 observations
- `s02`: 67,345 observations
- `s03`: 82,224 observations
- `s04`: 88,776 observations
- `s05`: 88,242 observations
- `s06`: 81,906 observations
- `s07`: 67,267 observations
- `s08`: 45,354 observations
- `s09`: 16,879 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between adjacent sections on a shared grid.
