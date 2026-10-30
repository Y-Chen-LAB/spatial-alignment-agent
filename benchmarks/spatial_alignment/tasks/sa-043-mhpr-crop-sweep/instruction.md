# MHPR MERFISH overlap robustness sweep (10 to 100 percent overlap)

Align the slices below into a shared coordinate system.

## Slices

- `data5` at `MHPR_animal1_n004.h5ad`
- `data6_crop010` at `data6_crop010.h5ad`
- `data6_crop020` at `data6_crop020.h5ad`
- `data6_crop030` at `data6_crop030.h5ad`
- `data6_crop040` at `data6_crop040.h5ad`
- `data6_crop050` at `data6_crop050.h5ad`
- `data6_crop060` at `data6_crop060.h5ad`
- `data6_crop070` at `data6_crop070.h5ad`
- `data6_crop080` at `data6_crop080.h5ad`
- `data6_crop090` at `data6_crop090.h5ad`
- `data6_crop100` at `data6_crop100.h5ad`

## Scoring

Each perturbed slice is compared against `data5`, the unperturbed reference, so one score is produced per perturbation level. Region labels are held by the scorer under `obs/Region`; they are not in your workspace.

Metric profile `sabench_light_robustness_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (155 genes, source
`paper_source_data`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
