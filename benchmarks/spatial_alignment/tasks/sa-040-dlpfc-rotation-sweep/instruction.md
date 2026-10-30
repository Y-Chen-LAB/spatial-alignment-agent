# DLPFC rotation robustness sweep (0 to 360 degrees, 15 degree steps)

Align the slices below into a shared coordinate system.

## Slices

- `data1` at `151673.h5ad`
- `data2_rotation000` at `data2_rotation000.h5ad`
- `data2_rotation015` at `data2_rotation015.h5ad`
- `data2_rotation030` at `data2_rotation030.h5ad`
- `data2_rotation045` at `data2_rotation045.h5ad`
- `data2_rotation060` at `data2_rotation060.h5ad`
- `data2_rotation075` at `data2_rotation075.h5ad`
- `data2_rotation090` at `data2_rotation090.h5ad`
- `data2_rotation105` at `data2_rotation105.h5ad`
- `data2_rotation120` at `data2_rotation120.h5ad`
- `data2_rotation135` at `data2_rotation135.h5ad`
- `data2_rotation150` at `data2_rotation150.h5ad`
- `data2_rotation165` at `data2_rotation165.h5ad`
- `data2_rotation180` at `data2_rotation180.h5ad`
- `data2_rotation195` at `data2_rotation195.h5ad`
- `data2_rotation210` at `data2_rotation210.h5ad`
- `data2_rotation225` at `data2_rotation225.h5ad`
- `data2_rotation240` at `data2_rotation240.h5ad`
- `data2_rotation255` at `data2_rotation255.h5ad`
- `data2_rotation270` at `data2_rotation270.h5ad`
- `data2_rotation285` at `data2_rotation285.h5ad`
- `data2_rotation300` at `data2_rotation300.h5ad`
- `data2_rotation315` at `data2_rotation315.h5ad`
- `data2_rotation330` at `data2_rotation330.h5ad`
- `data2_rotation345` at `data2_rotation345.h5ad`
- `data2_rotation360` at `data2_rotation360.h5ad`

## Scoring

Each perturbed slice is compared against `data1`, the unperturbed reference, so one score is produced per perturbation level. Region labels are held by the scorer under `obs/Region`; they are not in your workspace.

Metric profile `sabench_light_robustness_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (124 genes, source
`sabench_panel`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
