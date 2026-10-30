# Vizgen MERFISH mouse brain receptor map serial alignment (S1R1 to S1R3)

Align the slices below into a shared coordinate system.

## Slices

- `mbrm_s1r1` at `data42_mbrm_S1R1_right.h5ad`
- `mbrm_s1r2` at `data43_mbrm_S1R2_right.h5ad`
- `mbrm_s1r3` at `data44_mbrm_S1R3_right.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (100 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
