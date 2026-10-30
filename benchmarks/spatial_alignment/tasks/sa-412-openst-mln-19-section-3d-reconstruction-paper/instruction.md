# Open-ST lymph node 3D reconstruction, GEASO protocol (19 sections)

19 consecutive sections of a human metastatic lymph node, in order. Each section was placed on its slide independently, so they do not share a coordinate system. Reconstruct the block: bring every section into the coordinate system of the first section.

Measured at near single-cell resolution (Open-ST). Coordinates are in microns; sections may differ slightly in scale from one another.

- `section_02`: 55,400 observations
- `section_03`: 52,405 observations
- `section_04`: 77,868 observations
- `section_05`: 47,217 observations
- `section_06`: 38,555 observations
- `section_07`: 27,542 observations
- `section_09`: 54,899 observations
- `section_11`: 67,206 observations
- `section_17`: 70,879 observations
- `section_18`: 80,288 observations
- `section_19`: 39,167 observations
- `section_23`: 66,389 observations
- `section_24`: 68,177 observations
- `section_25`: 39,646 observations
- `section_26`: 82,583 observations
- `section_28`: 63,661 observations
- `section_33`: 68,138 observations
- `section_34`: 57,057 observations
- `section_36`: 40,692 observations

Keep `section_02` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
