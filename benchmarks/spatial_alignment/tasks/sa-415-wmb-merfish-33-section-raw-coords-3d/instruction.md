# MERFISH mouse brain 3D reconstruction from raw imaging coordinates (33 sections)

33 coronal sections through one mouse brain, in order (about 0.35 mm apart), each measured at single-cell resolution and given in its own imaging coordinates. The sections were cut, mounted and imaged separately, so besides orientation and position they can differ slightly in scale and are locally deformed. Reconstruct the block: bring every section into the coordinate system of `s000`.

Measured at single-cell resolution (MERFISH, 1,120 genes). Coordinates are in microns; sections may differ slightly in scale from one another.

- `s000`: 12,448 observations
- `s001`: 16,717 observations
- `s002`: 16,563 observations
- `s003`: 18,689 observations
- `s004`: 12,755 observations
- `s005`: 14,305 observations
- `s006`: 14,588 observations
- `s007`: 15,502 observations
- `s008`: 20,852 observations
- `s009`: 27,591 observations
- `s010`: 30,457 observations
- `s011`: 34,141 observations
- `s012`: 34,983 observations
- `s013`: 40,127 observations
- `s014`: 36,788 observations
- `s015`: 36,527 observations
- `s016`: 38,566 observations
- `s017`: 44,930 observations
- `s018`: 47,597 observations
- `s019`: 41,186 observations
- `s020`: 41,050 observations
- `s021`: 35,361 observations
- `s022`: 38,452 observations
- `s023`: 38,441 observations
- `s024`: 35,568 observations
- `s025`: 29,197 observations
- `s026`: 20,905 observations
- `s027`: 23,931 observations
- `s028`: 23,811 observations
- `s029`: 25,781 observations
- `s030`: 23,111 observations
- `s031`: 20,643 observations
- `s032`: 13,041 observations

Keep `s000` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
