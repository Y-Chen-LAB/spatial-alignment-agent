# STARmap (RNA) to RIBOmap (ribosome-bound RNA) cross-modality alignment

Two adjacent sections of mouse brain. `starmap` measures RNA; `ribomap` measures ribosome-bound RNA (protein synthesis) for the same genes. The two sections were placed on their slides independently. Bring `ribomap` into the coordinate system of `starmap`.

Measured at single-cell resolution (STARmap and RIBOmap, 5,413 genes). Coordinates are in microns, and each section's physical scale is correct as given.

- `starmap`: 59,165 observations
- `ribomap`: 58,692 observations

Keep `starmap` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
