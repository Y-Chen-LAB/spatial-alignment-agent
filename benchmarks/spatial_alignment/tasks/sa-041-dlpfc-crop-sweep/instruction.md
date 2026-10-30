# DLPFC overlap robustness sweep (10 to 100 percent overlap)

Align the slices below into a shared coordinate system.

## Slices

- `data1` at `151673.h5ad`
- `data2_crop010` at `data2_crop010.h5ad`
- `data2_crop020` at `data2_crop020.h5ad`
- `data2_crop030` at `data2_crop030.h5ad`
- `data2_crop040` at `data2_crop040.h5ad`
- `data2_crop050` at `data2_crop050.h5ad`
- `data2_crop060` at `data2_crop060.h5ad`
- `data2_crop070` at `data2_crop070.h5ad`
- `data2_crop080` at `data2_crop080.h5ad`
- `data2_crop090` at `data2_crop090.h5ad`
- `data2_crop100` at `data2_crop100.h5ad`

## Scoring

Each perturbed slice is compared against `data1`, the unperturbed reference, so one score is produced per perturbation level. Region labels are held by the scorer under `obs/Region`; they are not in your workspace.

Metric profile `sabench_light_robustness_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (124 genes, source
`sabench_panel`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
