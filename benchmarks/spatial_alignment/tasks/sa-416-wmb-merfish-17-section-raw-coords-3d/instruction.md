# MERFISH mouse brain 3D reconstruction from raw imaging coordinates (17 sections)

17 coronal sections through one mouse brain, in order (about 0.7 mm apart), each measured at single-cell resolution and given in its own imaging coordinates. The sections were cut, mounted and imaged separately, so besides orientation and position they can differ slightly in scale and are locally deformed. Reconstruct the block: bring every section into the coordinate system of `s000`.

Measured at single-cell resolution (MERFISH, 1,120 genes). Coordinates are in microns; sections may differ slightly in scale from one another.

- `s000`: 12,448 observations
- `s001`: 16,563 observations
- `s002`: 12,755 observations
- `s003`: 14,588 observations
- `s004`: 20,852 observations
- `s005`: 30,457 observations
- `s006`: 34,983 observations
- `s007`: 36,788 observations
- `s008`: 38,566 observations
- `s009`: 47,597 observations
- `s010`: 41,050 observations
- `s011`: 38,452 observations
- `s012`: 35,568 observations
- `s013`: 20,905 observations
- `s014`: 23,811 observations
- `s015`: 23,111 observations
- `s016`: 13,041 observations

Keep `s000` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
