# Albis visiumhd bin16 serial alignment

10 serial sections of one simulated tissue block, measured at 16 um square bins (Visium HD-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 15,558 observations
- `s01`: 30,503 observations
- `s02`: 41,514 observations
- `s03`: 48,370 observations
- `s04`: 51,157 observations
- `s05`: 51,190 observations
- `s06`: 48,320 observations
- `s07`: 41,509 observations
- `s08`: 30,452 observations
- `s09`: 15,620 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between adjacent sections on a shared grid.
