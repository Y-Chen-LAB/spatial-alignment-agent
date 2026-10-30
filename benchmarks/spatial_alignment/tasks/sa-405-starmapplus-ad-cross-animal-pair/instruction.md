# STARmap PLUS cross-animal alignment (two 8-month control mice)

Matching sections of the cortex and hippocampus from two different mice. The two sections were placed on their slides independently. Bring `mouse_cr2` into the coordinate system of `mouse_cr1`.

Measured at single-cell resolution (STARmap PLUS, 2,766 genes; expression is log-normalised as released). Coordinates are in microns, and each section's physical scale is correct as given.

- `mouse_cr1`: 8,506 observations
- `mouse_cr2`: 9,803 observations

Keep `mouse_cr1` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
