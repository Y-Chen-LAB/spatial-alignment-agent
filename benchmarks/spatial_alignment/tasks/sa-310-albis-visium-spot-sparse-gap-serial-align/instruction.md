# Albis visium spot sparse gap serial alignment

9 sections of one simulated tissue block, measured at 55 um capture spots on a 100 um lattice (Visium-like). Each section was
placed on the slide independently, so the sections do not share a coordinate system.

The sections are listed in block order, but their spacing is irregular and not given: some
consecutive sections in this list are adjacent in the block, others have sections missing between them.

The data passed standard QC: observations with fewer than 3 detected genes were removed.

- `s00`: 294 observations
- `s01`: 823 observations
- `s02`: 1,093 observations
- `s03`: 1,254 observations
- `s04`: 1,217 observations
- `s05`: 1,091 observations
- `s06`: 1,013 observations
- `s07`: 446 observations
- `s08`: 299 observations

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
