# STARmap PLUS stitching at 80% overlap (30, 60 and 90 degrees)

A target section and three rotated copies of a neighbouring section from the same block of mouse brain tissue. The two sections only partly overlap (about 80% of the target's height), and each copy was rotated by a different, unknown angle. Align every `source_*` section to `target`, independently of the others.

Measured at single-cell resolution (STARmap PLUS, 1,022 genes). Coordinates are in microns, and each section's physical scale is correct as given.

- `target`: 46,519 observations
- `source_rot030`: 43,857 observations
- `source_rot060`: 43,857 observations
- `source_rot090`: 43,857 observations

Keep `target` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
