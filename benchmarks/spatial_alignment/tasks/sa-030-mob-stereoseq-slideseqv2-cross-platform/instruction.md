# Mouse olfactory bulb cross-platform alignment: Stereo-seq to Slide-seqV2

Align the slices below into a shared coordinate system.

## Slices

- `mob_stereoseq` at `data78_mob_stereoseq.h5ad`
- `mob_slideseqv2` at `data79_mob_slideseqv2.h5ad`

## Scoring

Consecutive slices are compared in the order listed above. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (7 genes, source
`paper_source_data`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
