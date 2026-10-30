# MOp MERFISH Serial Alignment: mouse1sample6

Align the serial MERFISH slices from this MOp sample into a common coordinate system.

Submit one coordinate CSV per slice with columns `obs_id,x,y`. The row order and `obs_id` values must match the source `.h5ad` file.

This task uses `obs/Cluster` as a proxy landmark label derived from the original MOp `cell_labels.csv` column `class_label`. These are not anatomical region annotations, so the landmark metrics should be interpreted as cluster-consistency proxies rather than exact SABench region scores.
