# Albis strong mix cell gap serial alignment

9 sections of one simulated tissue block, measured at single-cell resolution (Xenium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

The sections are listed in block order, but their spacing is irregular and not given: some
consecutive sections in this list are adjacent in the block, others have sections missing between them.

- `s00`: 699 observations
- `s01`: 2,168 observations
- `s02`: 2,991 observations
- `s03`: 3,456 observations
- `s04`: 3,281 observations
- `s05`: 2,898 observations
- `s06`: 2,691 observations
- `s07`: 1,064 observations
- `s08`: 675 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
