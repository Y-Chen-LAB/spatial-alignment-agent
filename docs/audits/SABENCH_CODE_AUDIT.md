# SABench Code Audit

This note maps the public SABench repository to the metrics, scores, perturbations, and utilities described in the paper.

Scope of search:
- `/Users/yiqun/Downloads/SABench`
- `/tmp/SABench-official`
- converted notebook exports under `references/sabench_notebooks_py`

## Bottom line

- The public SABench repo does ship executable code for the core gene-based metrics, landmark-based metrics, robustness perturbations, coarse alignment, and downstream clustering metrics.
- I do **not** find executable public code for **MAE** in the repo, even though the paper and README say MAE was used for datasets with known 3D coordinates.
- I also do **not** find a public runtime/peak-memory measurement harness in the repo.

## Metric map

| Metric family | Public implementation | Notes |
| --- | --- | --- |
| Overlap-grid construction | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:19-80` | Defines shared overlapping rectangle, 10x10 grids, and `grid_label` assignment using a density threshold of half the smaller average cells-per-grid count. |
| PCC | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:133-156` | Aggregates gene expression by overlapping grid, then uses `scipy.stats.pearsonr`. |
| Cosine similarity | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:158-181` | Same grid aggregation, then cosine on per-grid mean expression vectors. |
| SSIM | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:184-219` | Same grid aggregation, explicit SSIM-style formula with `k1=0.01`, `k2=0.03`, `L=1`. |
| Mutual information | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:221-243` | Same grid aggregation, then `sklearn.feature_selection.mutual_info_regression`. |
| ASSD-like expression discrepancy | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:246-268` | Despite the name, this is implemented as mean squared difference of per-grid mean expression, not a surface-distance metric. |
| Average gene-metric aggregation across a serial stack | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:270-333` | Loops over adjacent slice pairs and averages per-gene metric values across pairs. |
| Landmark accuracy | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:336-386` | Restricts to overlap bbox, then nearest-neighbor match with `cKDTree`; score is fraction of matched `obs['Region']`. |
| Region overlap ratio | `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:388-440` | Uses `alpha_shapes.Alpha_Shaper` with `alpha = 12.0`, then IoU of region-specific alpha shapes. |
| MAE | No executable code found in public SABench repo | The paper defines it, but repo grep found only README mentions. Paper definition is in `PAPER_MANUSCRIPT.md:2157-2183`. |
| ARI / NMI / HOM / COM | `references/sabench_notebooks_py/DownstreamTasks/DLPFC_3D_Spatial_Clustering.py:141-169` | Uses `sklearn.metrics.adjusted_rand_score`, `normalized_mutual_info_score`, `homogeneity_score`, `completeness_score`. |

## Tutorial entry points

These are the public-facing examples showing how SABench expects users to call the metric code:

- `references/sabench_notebooks_py/Tutorial/SABench_tutorial.py:48-57`
  Uses `Avg_pcc_values_for_method`, `Avg_cos_sim_values_for_method`, `Avg_ssim_values_for_method`, `Avg_MI_values_for_method`.
- `references/sabench_notebooks_py/Tutorial/SABench_tutorial.py:154-170`
  Uses `Average_ROI_Overlap` and `Average_Accuracy`.

The tutorial compares new-method outputs against the CSVs under:

- `/Users/yiqun/Downloads/SABench/Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_PCC.csv`
- `/Users/yiqun/Downloads/SABench/Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_Cos_Sim.csv`
- `/Users/yiqun/Downloads/SABench/Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_SSIM.csv`
- `/Users/yiqun/Downloads/SABench/Tutorial/Methods_Results_DLPFC/DLPFCS_GeneBased_MI.csv`
- `/Users/yiqun/Downloads/SABench/Tutorial/Methods_Results_DLPFC/DLPFCS_LandmarkBased_Heatmap_PlotFig.csv`

## Robustness generation

The public robustness notebooks are simple and explicit.

### Rotation

Example:
- `references/sabench_notebooks_py/Robustness/DLPFC_rotation.py:16-33`

What it does:
- centers the target slice at its centroid
- rotates clockwise/counterclockwise through the standard 2D rotation matrix
- adds the centroid back
- writes a perturbed H5AD

Sweep:
- `references/sabench_notebooks_py/Robustness/DLPFC_rotation.py:36-40`
- angles are `0, 15, 30, ..., 360`

The same pattern appears for:
- `Robustness/MHPR_rotation.py`
- `Robustness/SCC_rotation.py`
- `Robustness/MPC_rotation.py`
- `Robustness/MMPC_rotation.py`
- `Robustness/Flysta_rotation.py`

### Crop / overlap-ratio perturbation

Example:
- `references/sabench_notebooks_py/Robustness/DLPFC_crop.py:15-30`

What it does:
- uses the x-coordinate span of the target slice
- keeps points with `x < x_threshold`
- where `x_threshold = x_min + (x_max - x_min) * overlap_percentage / 100`

Sweep:
- `references/sabench_notebooks_py/Robustness/DLPFC_crop.py:32-36`
- overlap percentages are `10, 20, ..., 100`

The same pattern appears for:
- `Robustness/MHPR_crop.py`
- `Robustness/SCC_crop.py`
- `Robustness/MPC_crop.py`
- `Robustness/MMPC_crop.py`
- `Robustness/Flysta_crop.py`

## Large-scale / preprocessing utilities

These are in the public tutorial helper library, not hidden in the paper:

- `Change_slices_resolution`
  `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:445-490`
  Grid-based downsampling by spatial binning and summing counts within each rectangle.

- `restore_initial_slices`
  `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:493-522`
  Fits `skimage.transform.SimilarityTransform` between downsampled raw slices and aligned downsampled outputs, then lifts that transform back to the original-resolution slices.

- `interactive_coarse_align`
  `/Users/yiqun/Downloads/SABench/Tutorial/SABench.py:565-649`
  Manual landmark clicking followed by `estimate_transform('affine', ...)`.

These utilities matter because the paper’s “three solutions” for large datasets are not just prose; the public repo exposes the core mechanics here.

## Method wrappers versus metric code

The files under:

- `/Users/yiqun/Downloads/SABench/SpatialAlignment/*.py`

are method runners and adapters, for example:
- `PASTE.py`
- `PASTE2.py`
- `STAligner.py`
- `CAST.py`
- `SPACEL.py`
- `Spateo.py`

They are **not** where SABench computes the benchmark scores. The scoring logic is concentrated in `Tutorial/SABench.py` and the feature-similarity notebooks.

## What is missing from the public repo

### 1. MAE implementation

I searched the public SABench tree for:

- `MAE`
- `mean_absolute_error`
- `absolute_error`
- `sklearn.metrics`

Result:
- only README mentions were found
- no executable MAE helper or notebook was found

So the paper definition exists, but the public code path does not appear to be shipped.

Paper definition:
- `PAPER_MANUSCRIPT.md:2157-2183`

Important detail:
- the paper says they used `mean_absolute_error` from scikit-learn

### 2. Runtime / peak-memory evaluation harness

The README and paper mention runtime and peak memory as benchmark axes, but I do not find a public measurement script that computes and records those values across methods in the repo.

### 3. Overall ranking-score code

The paper says scores were averaged across tasks and methods were ranked to derive final overall ranking scores, but I do not find a single public script in the repo that reproduces the full paper-wide ranking table end to end.

## Practical implication for paper2bench

The public SABench code is reliable enough to anchor:

- gene-grid metrics
- nearest-label landmark accuracy
- alpha-shape region overlap
- crop and rotation robustness generation
- coarse alignment and resolution-recovery helpers
- downstream ARI/NMI/HOM/COM

But for full paper-faithful parity, we still need to implement ourselves:

- MAE scorer
- runtime / peak-memory tracking
- any paper-wide overall-rank aggregation that is not explicitly released
