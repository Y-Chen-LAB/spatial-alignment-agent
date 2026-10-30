# Flysta3D E14-16h Embryo 16-Slice Serial Alignment with 3D Ground-Truth MAE (Data16-31)

Sixteen consecutive Stereo-seq z-slices of a Drosophila E14-16h embryo, spanning the full anterior-posterior extent at 0.7 µm spacing (z = 0, 0.7, 1.4, …, 10.5). Compiled from the CNGB Flysta3D atlas (Wang et al. 2022) by `scripts/compile_flysta3d_slices.py`.

The SABench paper (Yan et al. 2026) uses this dataset as **Data16-31**, calling out that it is one of only two task families with **true 3D ground-truth coordinates** (Methods page 13). The paper consequently reports the **MAE** metric on this family in addition to gene-grid metrics — see PAPER_MANUSCRIPT.md lines 2161-2168 for the formal MAE definition:

> *MAE = (1/(2N)) Σᵢ (|x_pred,ᵢ − x_GT,ᵢ| + |y_pred,ᵢ − y_GT,ᵢ|)*

The compiled per-slice h5ads expose:
- `obsm/spatial`: a deterministic random-rigid-perturbed (rotation + translation, seed = `160 + slice_id`) version of the atlas (x, y) — what the agent sees and tries to align
- `obs/x_GT`, `obs/y_GT`, `obs/z_GT`: the held-out atlas coordinates used by the MAE scorer (the agent should treat these as hidden)
- `obs/Region`: paper-aligned tissue annotation (categorical), available to the agent for landmark-style approaches

## Task

Align the sixteen Stereo-seq z-slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Scoring (paper-aligned)

`metric_profile = "sabench_mae_gene_efficiency_v1"`. Aggregated metrics:
- `mae` — SABench MAE against held-out 3D atlas coordinates: `(1/2N) sum(|x - x_GT| + |y - y_GT|)`, computed on your submitted coordinates with no fitting step. Submit in the input coordinate frame: the first slice is the reference and should not move. Leaving every slice untouched scores 143.9; the reference slice alone scores 102.3.
- `gene_pcc`, `gene_cosine`, `gene_ssim`, `gene_mi` — paper gene-grid metrics on the top-100 highly variable Stereo-seq genes

The `Region` annotation is also stored on each slice; the SABench landmark-based metrics are *not* in this profile (paper uses MAE primarily for 3D-GT datasets), but solver authors may use `obs/Region` for landmark heuristics.

## Notes

- ~6,000–15,295 cells per slice, 13,668 panel genes.
- Per-slice perturbation parameters are recorded in `obs.attrs["perturbation"]` (JSON) for reproducibility.
- Source: `~/Downloads/paper2bench-data/sabench_full/Data16-31_Flysta3D/E14-16h_a_count_normal_stereoseq.h5ad` (Stereo-seq, 1.6 GB).
- Compiled slices: `~/Downloads/paper2bench-data/Flysta3D/flysta3d_e14_16_slice{00..15}.h5ad`.
