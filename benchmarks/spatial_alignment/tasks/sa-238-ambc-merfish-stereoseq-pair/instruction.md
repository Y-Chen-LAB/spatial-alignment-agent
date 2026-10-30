# Adult mouse brain coronal cross-platform alignment: merfish to stereoseq

Align the slices below into a shared coordinate system.

## Slices

- `ambc_merfish` at `data238_ambc_merfish_S2R3_right.h5ad`
- `ambc_stereoseq` at `data239_ambc_stereoseq_T312.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (100 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
