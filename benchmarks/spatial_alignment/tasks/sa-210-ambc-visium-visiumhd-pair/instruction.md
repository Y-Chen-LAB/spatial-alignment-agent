# Adult mouse brain coronal cross-platform alignment: visium to visiumhd

Align the slices below into a shared coordinate system.

## Slices

- `ambc_visium` at `Data231_adult_mouse_brain.h5ad`
- `ambc_visiumhd` at `data232_visiumhd_08um.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (100 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
