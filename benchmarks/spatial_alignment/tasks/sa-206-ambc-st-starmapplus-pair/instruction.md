# Adult mouse brain coronal cross-platform alignment: st to starmapplus

Align the slices below into a shared coordinate system.

## Slices

- `ambc_st` at `data233_ambc_st_21A.h5ad`
- `ambc_starmapplus` at `data237_starmapplus_well11.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. Region labels are held by the scorer under `obs/Region`; they are not in your workspace.

Metric profile `sabench_light_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (100 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
