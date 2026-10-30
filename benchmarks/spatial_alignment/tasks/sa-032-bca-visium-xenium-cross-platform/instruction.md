# Human breast cancer cross-platform alignment: Visium to both Xenium replicates

Align the slices below into a shared coordinate system.

## Slices

- `bca_visium` at `data228_bca_visium.h5ad`
- `bca_xenium_rep1` at `slice_Rep1.h5ad`
- `bca_xenium_rep2` at `slice_Rep2.h5ad`

## Scoring

Each perturbed slice is compared against `bca_visium`, the unperturbed reference, so one score is produced per perturbation level. This task is scored on gene-expression agreement only; no region labels are used.

Metric profile `sabench_gene_only_alignment_v1`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer (100 genes, source
`log1p_variance_ranking`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
