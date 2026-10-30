# Albis visium spot slide serial alignment

10 serial sections of one simulated tissue block, measured at 55 um capture spots on a 100 um lattice (Visium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 459 observations
- `s01`: 840 observations
- `s02`: 1,105 observations
- `s03`: 1,274 observations
- `s04`: 1,346 observations
- `s05`: 1,341 observations
- `s06`: 1,277 observations
- `s07`: 1,116 observations
- `s08`: 839 observations
- `s09`: 460 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
