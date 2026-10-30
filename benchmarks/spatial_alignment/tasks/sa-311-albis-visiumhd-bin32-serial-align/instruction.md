# Albis Visium HD 32um bin serial alignment

10 serial sections of one simulated tissue block, measured at 32 um square bins (Visium HD-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 4,344 observations
- `s01`: 8,033 observations
- `s02`: 10,683 observations
- `s03`: 12,320 observations
- `s04`: 12,955 observations
- `s05`: 12,980 observations
- `s06`: 12,305 observations
- `s07`: 10,676 observations
- `s08`: 7,998 observations
- `s09`: 4,356 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
