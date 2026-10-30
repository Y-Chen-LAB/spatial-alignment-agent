# Adult mouse brain coronal cross-platform alignment: xenium to merfish

Align the slices below into a shared coordinate system.

## Slices

- `ambc_xenium` at `xenium_v1_ff_mouse_brain_multisection_1.h5ad`
- `ambc_merfish` at `data238_ambc_merfish_S2R3_right.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (27 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
