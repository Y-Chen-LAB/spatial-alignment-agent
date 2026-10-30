# Mouse Brain Sagittal-Anterior Serial Alignment (Data32 ↔ Data33)

Two consecutive 10x Visium sagittal-anterior mouse brain sections from the SABench paper (Yan et al. 2026, Data32-33). The slices were generated on 10x's public datasets portal and contain roughly 2,696 and 2,825 spots over the full 31k+ gene panel.

In the paper's evaluation pipeline, random rigid transformations are applied to disrupt the original 10x coordinate frame; methods are scored on how well they recover an aligned coordinate system across the pair.

## Task

Align the two slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- Each slice carries a rigid perturbation of its original coordinate frame; the transform is withheld.
- The SABench paper states that cluster annotations are provided for this pair, but the current local 10x-derived materialization still lacks `obs/Cluster`. This task therefore uses the honest gene-only SABench metric profile until label-enriched source files are resolved.
- Source: 10x Genomics filtered_feature_bc_matrix.h5 + spatial.tar.gz, materialized via `scripts/preprocess_10x_visium_to_h5ad.py`.
