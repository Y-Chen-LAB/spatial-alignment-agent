# MHPR MERFISH rotation robustness sweep (0 to 360 degrees, 15 degree steps)

Align the slices below into a shared coordinate system.

## Slices

- `data5` at `MHPR_animal1_n004.h5ad`
- `data6_rotation000` at `data6_rotation000.h5ad`
- `data6_rotation015` at `data6_rotation015.h5ad`
- `data6_rotation030` at `data6_rotation030.h5ad`
- `data6_rotation045` at `data6_rotation045.h5ad`
- `data6_rotation060` at `data6_rotation060.h5ad`
- `data6_rotation075` at `data6_rotation075.h5ad`
- `data6_rotation090` at `data6_rotation090.h5ad`
- `data6_rotation105` at `data6_rotation105.h5ad`
- `data6_rotation120` at `data6_rotation120.h5ad`
- `data6_rotation135` at `data6_rotation135.h5ad`
- `data6_rotation150` at `data6_rotation150.h5ad`
- `data6_rotation165` at `data6_rotation165.h5ad`
- `data6_rotation180` at `data6_rotation180.h5ad`
- `data6_rotation195` at `data6_rotation195.h5ad`
- `data6_rotation210` at `data6_rotation210.h5ad`
- `data6_rotation225` at `data6_rotation225.h5ad`
- `data6_rotation240` at `data6_rotation240.h5ad`
- `data6_rotation255` at `data6_rotation255.h5ad`
- `data6_rotation270` at `data6_rotation270.h5ad`
- `data6_rotation285` at `data6_rotation285.h5ad`
- `data6_rotation300` at `data6_rotation300.h5ad`
- `data6_rotation315` at `data6_rotation315.h5ad`
- `data6_rotation330` at `data6_rotation330.h5ad`
- `data6_rotation345` at `data6_rotation345.h5ad`
- `data6_rotation360` at `data6_rotation360.h5ad`

## Scoring

Each perturbed slice is compared against `data5`, the unperturbed reference, so one score is produced per perturbation level. Region labels are held by the scorer under `obs/Region`; they are not in your workspace.

Metric profile `sabench_light_robustness_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (155 genes, source
`paper_source_data`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
