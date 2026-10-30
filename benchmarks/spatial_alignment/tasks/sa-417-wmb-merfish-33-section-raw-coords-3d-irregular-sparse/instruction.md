# MERFISH mouse brain 3D reconstruction from raw imaging coordinates (33 sections)

33 coronal sections through one mouse brain, in order (irregular spacing, between about 0.09 and 0.9 mm apart; the spacing is not given), each measured at single-cell resolution and given in its own imaging coordinates (sections differ in how many cells were captured). The sections were cut, mounted and imaged separately, so besides orientation and position they can differ slightly in scale and are locally deformed. Reconstruct the block: bring every section into the coordinate system of `s000`.

Measured at single-cell resolution (MERFISH, 1,120 genes). Coordinates are in microns; sections may differ slightly in scale from one another.

- `s000`: 9,603 observations
- `s001`: 7,431 observations
- `s002`: 5,064 observations
- `s003`: 8,870 observations
- `s004`: 9,920 observations
- `s005`: 7,026 observations
- `s006`: 6,110 observations
- `s007`: 7,847 observations
- `s008`: 5,134 observations
- `s009`: 9,557 observations
- `s010`: 17,158 observations
- `s011`: 10,483 observations
- `s012`: 11,339 observations
- `s013`: 14,570 observations
- `s014`: 15,353 observations
- `s015`: 26,922 observations
- `s016`: 17,641 observations
- `s017`: 26,724 observations
- `s018`: 18,228 observations
- `s019`: 27,827 observations
- `s020`: 14,469 observations
- `s021`: 20,638 observations
- `s022`: 14,198 observations
- `s023`: 17,468 observations
- `s024`: 15,667 observations
- `s025`: 24,896 observations
- `s026`: 19,976 observations
- `s027`: 21,925 observations
- `s028`: 21,368 observations
- `s029`: 19,062 observations
- `s030`: 16,953 observations
- `s031`: 13,915 observations
- `s032`: 10,704 observations

Keep `s000` where it is: its submitted coordinates should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the row order of the source `.h5ad` file.
