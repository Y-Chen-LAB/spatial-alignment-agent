# STARmap PLUS overlap sweep (10 to 100 percent)

A reference section and ten versions of the neighbouring section from the same block of mouse brain tissue. Each version keeps a different fraction of the neighbouring section (10% to 100% of its height) and was placed on the slide independently. Align every `moving_*` section to `reference`, independently of the others.

Measured at single-cell resolution (STARmap PLUS, 1,022 genes). Coordinates are in microns, and each section's physical scale is correct as given.

- `reference`: 51,897 observations
- `moving_overlap010`: 2,430 observations
- `moving_overlap020`: 6,887 observations
- `moving_overlap030`: 12,123 observations
- `moving_overlap040`: 18,372 observations
- `moving_overlap050`: 25,298 observations
- `moving_overlap060`: 31,594 observations
- `moving_overlap070`: 36,769 observations
- `moving_overlap080`: 42,336 observations
- `moving_overlap090`: 47,061 observations
- `moving_overlap100`: 51,897 observations

Keep `reference` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
