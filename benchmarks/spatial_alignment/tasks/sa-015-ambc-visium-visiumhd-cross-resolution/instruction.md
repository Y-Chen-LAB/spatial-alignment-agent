# AMBC Visium ↔ Visium HD Cross-Resolution Alignment (Data231 ↔ Data232)

This is the within-vendor cross-resolution pair from the SABench adult mouse brain coronal family. `Data231` is standard 10x Visium; `Data232` is 10x Visium HD compiled into a 16 µm bin-level sparse `h5ad`.

## Task

Align the two slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- The Visium HD source is not a standard Cell Ranger matrix. It was compiled from `feature_slice.h5` by aggregating the sparse per-feature slices onto the public `square_016um` grid with `scripts/preprocess_10x_visiumhd_to_h5ad.py`.
- This task is paper-aligned in family and modality, but the local assets do not include paper-ready region labels, so it runs under the honest SABench gene-only profile.
- Marker genes are a deterministic top-100 HVG list derived from the Visium slice and shared with the Visium HD slice.
