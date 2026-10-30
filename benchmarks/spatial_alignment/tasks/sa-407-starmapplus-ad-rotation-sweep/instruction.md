# STARmap PLUS cross-animal angle sweep (30 to 180 degrees)

A target section and seven rotated copies of a matching section from a second mouse (cortex and hippocampus). Each copy was rotated by a different, unknown angle. Align every `source_*` section to `target`, independently of the others.

Measured at single-cell resolution (STARmap PLUS, 2,766 genes; expression is log-normalised as released). Coordinates are in microns, and each section's physical scale is correct as given.

- `target`: 9,803 observations
- `source_rot030`: 8,506 observations
- `source_rot060`: 8,506 observations
- `source_rot090`: 8,506 observations
- `source_rot120`: 8,506 observations
- `source_rot135`: 8,506 observations
- `source_rot150`: 8,506 observations
- `source_rot180`: 8,506 observations

Keep `target` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
