# Adult Mouse Brain Coronal Xenium Multisection Serial Alignment (Data234 family)

Three serial coronal sections of the adult mouse brain profiled by 10x Xenium V1 (mouse-brain-replicates-1-standard). The compiled assets are derived from the public 10x Xenium `outs.zip` bundles using `scripts/compile_xenium_10x_bundle.py`.

The SABench paper (Yan et al. 2026) used a single Xenium slice from this family (Data234) for its ten-platform mouse brain coronal alignment task (Data231-240). This task uses all three multisection replicates as a serial sequence (multisection 1 → 2 → 3) under the paper's gene-based + landmark-based evaluation profile.

## Task

Align the three Xenium slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- Each compiled slice has ~150-160k cells across 541 panel genes. There are no cluster annotations in the local `obs/` (raw 10x outs.zip does not ship them), so `landmark_accuracy` and `region_grid_iou` will return NaN; gene-based metrics (`gene_pcc`, `gene_cosine`, `gene_ssim`, `gene_mi`) computed over the top-100 highly variable genes still fire.
- Source bundles: `~/Downloads/paper2bench-data/sabench_full/Data234_AMBC_Xenium/Xenium_V1_FF_Mouse_Brain_MultiSection_*_outs.zip`.
- Compiled slices: `(withheld)`.
