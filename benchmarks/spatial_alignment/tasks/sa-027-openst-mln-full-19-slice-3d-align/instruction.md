# Open-ST Human Metastatic Lymph Node Alignment

This task is compiled from the Open-ST human metastatic lymph node dataset used as SABench Data80-98.

Included sections: section_02, section_03, section_04, section_05, section_06, section_07, section_09, section_11, section_17, section_18, section_19, section_23, section_24, section_25, section_26, section_28, section_33, section_34, section_36

The source data provide:
- `obsm/spatial`: the observed 2D section coordinates to align
- `obs/Region`: cell-type / tissue annotations derived from the published `annotation`
- `obs/x_GT`, `obs/y_GT`, `obs/z_GT` and `obsm/spatial_3d_aligned`: the published aligned 3D coordinate frame used for the paper-style MAE scorer

## Task

Align the provided sections into a shared coordinate frame. Submit one coordinate CSV per section with columns `obs_id,x,y`.

## Scoring

`metric_profile = "sabench_mae_gene_efficiency_v1"`

Aggregated metrics:
- `mae` against the published aligned 3D frame after optimal rigid frame matching
- `gene_pcc`, `gene_cosine`, `gene_ssim`, `gene_mi` on the held-out eval split

This mirrors the SABench paper's treatment of Open-ST as a multi-consecutive-slice dataset with 3D ground truth.

Coordinates are in microns; sections may differ slightly in scale from one another.
