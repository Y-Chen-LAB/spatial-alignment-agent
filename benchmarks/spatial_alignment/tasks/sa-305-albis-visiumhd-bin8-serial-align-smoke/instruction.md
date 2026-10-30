# Albis visiumhd bin8 serial alignment (smoke)

4 serial sections of one simulated tissue block, measured at 8 um square bins (Visium HD-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

- `s00`: 3,899 observations
- `s01`: 6,920 observations
- `s02`: 6,968 observations
- `s03`: 3,898 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between adjacent sections on a shared grid.
