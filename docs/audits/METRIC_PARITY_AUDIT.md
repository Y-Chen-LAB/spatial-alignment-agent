# Scoring parity with the SABench reference implementation

Checked on 2026-09-15. The comparison runs the authors' notebook functions
(`FeatureSimilarity/GeneBased.ipynb`, `FeatureSimilarity/LandmarkBased.ipynb`, copied
verbatim) and our `src/genomics_agent_bench/metrics.py` over the same pair, 151673 against
151674, using the same gene list and the slices' native coordinates.

## Result

| Quantity | Authors | Ours | Difference |
| --- | --- | --- | --- |
| grid labels, left slice | 88 grids | 88 grids | identical, element by element |
| grid labels, right slice | 88 grids | 88 grids | identical, element by element |
| `gene_pcc`, mean over 20 genes | 0.646592 | 0.646593 | 2.4e-08 |
| `gene_cosine` | 0.850849 | 0.850849 | 1.8e-08 |
| `gene_ssim` | 0.631680 | 0.631680 | 1.3e-08 |
| `gene_mi` | 0.350957 to 0.355064 over 10 runs | 0.355292 | see below |
| `matching_accuracy` | 0.7921286 | 0.7921286 | 0 |
| `overlap_ratio_region`, each of 7 regions | per region | per region | 0.0 exactly, every region |

The residual on PCC, Cos_Sim and SSIM is float32 rounding in the expression matrix.

## Mutual information does not reproduce to the digit

`mutual_info_regression` is a k-nearest-neighbour estimator that adds random noise to
break ties. SABench calls it without `random_state`, so their own MI changes from run to
run: ten repeats of the 20-gene mean span 0.3510 to 0.3551. We pass `random_state=0`,
because a benchmark that returns a different score for the same submission is not usable,
and our value of 0.3553 sits just above the top of their observed span, 0.0002 outside a
spread of 0.0041.

MI is also sensitive to the input dtype in a way the other three metrics are not. On grid
means that are byte-identical, float32 and float64 inputs give MI values up to 0.03 apart.
`H5ADSlice.expression_for_genes` therefore no longer upcasts to float64, so our grid means
carry the same float32 the authors get from `adata.X[...].toarray()`. Before that change
the per-gene gap reached 0.070; after it, 0.032, with the aggregate gap at 0.0045.

Treat `gene_mi` as reproducible within our own runs and comparable to the paper only to
about two decimals.

## Changes made to reach parity

1. `_overlap_ratio_region` now follows `ROI_overlap`. A region missing from either slice
   scores 0 instead of being dropped from the average, which is what the notebook's
   `if len(mask1[0])==0 or len(mask2[0])==0: overlap_ratio=0` does. Dropping it rewarded a
   method that lost a whole region. The function also returns the per-region values, which
   is how the paper reports them, alongside the mean our aggregate uses.
2. `AlignmentScoreConfig.regions` and the task key `scored_regions` name the regions to
   score. SABench picks these per dataset rather than deriving them, and
   `DLPFC_LandmarkBased_Heatmap.csv` has columns for Layer2 to Layer6 and WM with no
   Layer1 column. The four DLPFC tasks now carry that list.
3. Marker genes now prefer a paper source-data header over the curated panel.
   `GeneBased.ipynb` reads the raw panel, drops genes absent from any slice, then drops
   three more by hard-coded index, so the panel's 124 DLPFC genes become the 117 the paper
   published. Five tasks changed: sa-001 and sa-002 scored 25 genes and now score 117,
   sa-040 and sa-041 scored 124 and now score 117, and sa-009 scored 100 and now scores
   the paper's full 155.
4. Removed `AlignmentScoreConfig.max_genes`, which two scripts set and nothing read, and
   the unused single-slice `_grid_labels_for_points`.

## MAE: the Procrustes fit is gone, and the frame convention is now known

We used to fit the submission onto the ground truth with an optimal rigid Procrustes
transform before measuring error, on the reasoning that the output frame was ambiguous. The
published tables show it is not, and that SABench applies no fit.

`Flysta3D_MAE.csv` reports 102.295148 in its first row for all ten methods, and
`MLN_MAE.csv` reports 0.250563 in its first row for all five. Those rows are the fixed
reference slice, which no method moves, so they record the error of the untouched input.
Both reproduce exactly from the source files as the plain MAE between the per-slice centred
input coordinates and the ground truth:

| Dataset | Reference slice | Published | Recomputed here |
| --- | --- | --- | --- |
| Flysta3D | slice 0 | 102.295148 | 102.295148 |
| MLN | section_02 | 0.250563 | 0.250563 |

So the convention is: centre each slice's input coordinates, compare to ground truth, no
fitting step. A Procrustes fit would drive both reference values to zero. The frame is not
free, the reference slice defines it, and the task instruction now says so.

`_mae_against_3d_ground_truth` no longer fits, and reports MAE per slice as well as overall,
because the published tables are per slice.

Flysta3D was rebuilt to the paper's construction. Our compile had been discarding the
source's `raw_x`, `raw_y` and substituting a seeded random rigid transform of the ground
truth, which put the input in a frame the paper never used. `compile_flysta3d_slices.py`
now takes `--coordinate-source`, defaulting to `paper_raw`. On the rebuilt sa-019 a
do-nothing submission scores 102.295148 on slice 0, matching the published row, and sits
just above PASTE on every later slice, which is where a do-nothing baseline belongs.

**MLN is not rebuilt yet.** Its input needs the same per-slice centring, but the four MLN
tasks (sa-022, sa-025, sa-026, sa-027) share one set of compiled slices, and centring
changes which regions overlap and so changes every metric, not only MAE. It would also
invalidate sa-022 as the regression anchor whose April scores we reproduce to four
decimals. Flysta3D was safe to rebuild because sa-019 is its only consumer.

## Where we still differ on purpose

- **Degenerate guards.** Where a gene has zero variance we return 0.0 and `pearsonr`
  returns NaN, which would poison the whole gene average. Alpha shapes need a
  triangulation, so a region with fewer than 4 cells scores 0 rather than raising. Neither
  case arises in the authors' data; both arise in a crop sweep.
- **`overall_mean`** is our own convenience average of the non-MAE metrics. The paper's
  "overall ranking score" instead ranks methods per metric and averages the ranks, which
  needs a full method field and cannot be computed for a single submission.

## MHPR landmark labels: solved

SABench scores MHPR `overlap_ratio_region` on eight anatomical structures: BST, MPA, MPN,
PV, PVH, PVT, V3 and fx. The Moffitt release carries only cell-type labels, and our build
had put `Cell_class` in `obs/Region` as a stand-in, which made the MHPR landmark metrics
incomparable to the paper.

The paper says where the labels come from: Data5 to Data9 are "five slices of mouse
hypothalamic preoptic region with annotation labels[54]", and reference 54 is BASS (Li and
Zhou, Genome Biology 2022). The BASS analysis repository ships them in
`data/MERFISH_Animal1.RData`, as column `z` of `info_mult`, for exactly the five bregma
sections -0.04 to -0.24 and with cell counts 5488, 5557, 5926, 5803 and 5543, which are
exactly the counts the paper reports for Data5 to Data9.

`scripts/attach_mhpr_bass_regions.py` joins them on (x, y) at three decimals, which matches
every cell on all five slices, and writes them into `obs/Region` while keeping the cell
types in `obs/Cell_class`. sa-009, sa-042 and sa-043 now carry the eight regions in
`scored_regions`.

Sanity check on sa-009: the raw MHPR frames are disjoint, so a do-nothing submission scores
0 on every metric, as it should. A centroid-centring baseline scores `matching_accuracy`
0.721 against a published range of 0.0 to 0.758, and mean region overlap 0.506 against a
published range of 0.0 to 0.538, which is where a translation-only baseline belongs.

## Reproducing this check

`scripts/metric_parity/{shim,xcheck,final}.py` hold the comparison. `shim.py` is a
minimal AnnData stand-in, because the notebook code needs `adata.X`, `adata.obs` and
`adata.obsm` and neither scanpy nor anndata is installed in this environment.

## What this does to the 531 existing runs

522 runs kept their submission, so the effect was measured by re-scoring rather than
estimated. `scripts/metric_parity/rescore.py` re-scores a task's retained submissions under
the current scorer and diffs against the stored `scores.json`.

| Task | Runs | Effect |
| --- | --- | --- |
| sa-016, sa-017 | 34 | every metric identical to 0.0000; `gene_mi` to 0.0002 |
| sa-003 | 14 | identical except `gene_mi` +0.0137, rank correlation 0.994 |
| sa-001, sa-002 | 46 | levels move (`gene_pcc` -0.082, `gene_ssim` -0.078, `gene_cosine` -0.054, `overall` -0.031) but rank correlation is 0.94 to 0.99 and `matching_accuracy` is unchanged |
| sa-009 | 32 | submissions no longer load; already invalid before this work, from the 2026-09-14 bregma rebuild |
| sa-019 | 26 | invalid, input frame rebuilt |

The change to `overlap_ratio_region` scoring a missing region 0 rather than dropping it
altered nothing retroactively: no existing submission ever lost a region from the overlap
entirely. It is a correctness fix for failures that have not happened yet.

The variance decomposition that the run-matrix design rests on does not move. Recomputing
it with the corrected sa-001 scores gives task omega^2 0.448 to 0.444, prompt omega^2
0.0064 to 0.0065, an unchanged residual SD, and an unchanged MDE of 0.054 at 45 tasks by 3
replicates. Within-task replicate SD on sa-001 fell slightly, 0.0354 to 0.0307.

## What this does to the paper comparisons

`agent_runs/analysis/spatial_common_prompt_latest/` calls sa-001 (DLPFC) and sa-009 (MHPR)
"the most comparable, exact direct-table plots". Both were the least comparable of the set.
sa-001 scored 25 of the paper's 117 genes, and sa-009 ran on cell-type labels against a
published table of anatomical structures, on the wrong bregma set. Those figures need
regenerating.

`mae_apples_to_apples_notes.md` names sa-027 as the one strict raw paper MAE overlay. Its
published reference row, 0.250563, reproduces only after per-slice centring, which the MLN
slices do not yet have, so that overlay is currently on uncentred coordinates. The same
file's caveat that "the local compiled Flysta3D MAE coordinate scale differs from the
extracted paper table" is now resolved: the rebuilt sa-019 reproduces 102.295148 exactly.

All five tasks carrying MAE are affected: sa-019 (26 runs, rebuilt), sa-023 (18, already on
the re-run list from the WMB rebuild), and sa-025, sa-026, sa-027 (40, pending the MLN
centring). Any conclusion resting on MAE needs recomputation. Conclusions resting on
within-task contrasts between arms survive, because the ordering is preserved.

## What the agent can see

Checked on 2026-09-15 across all 92 tasks, two ways.

**Structurally.** `_write_sanitized_h5ad` is an allowlist, not a redaction pass: it builds a
fresh H5AD and copies only `X`, `var`, `layers`, `obs/_index` and `obsm/spatial`. Every obs
column, every other `obsm` entry, and `uns`, `obsp` and `varm` are never copied, so they
cannot survive by being missed. `--blind-solve` refuses `--link-source-data`, so the raw-file
bypass cannot be combined with a blind run.

**Empirically.** `scripts/metric_parity/verify_agent_visible.py` sanitises a slice from every
task and asserts the result exposes nothing else. 92 of 92 pass, and 49 of them start from a
source slice that carries ground truth or labels.

A separate sweep over all 359 unique source files found that 175 carry ground truth in obs,
332 carry labels, and 160 carry extra `obsm` keys, none of which the allowlist copies. The
only allowlist surface beyond the expression matrix is `layers/raw` on the 19 Open-ST MLN
sections, which is a (55400, 28943) CSR expression matrix, not coordinates.

## What containment does not close

The container boundary is filesystem, not network, and this is deliberate: the prompt tells
agents they may use "web search, `curl`, GitHub READMEs, ReadTheDocs pages, PyPI pages" to
learn how to call method packages, and 112 of 527 historical runs used it. The hosts those
runs reached were documentation and package indexes, with no data repository among them.

The underlying datasets are public, so an agent that recognises which dataset it holds could
download the original file, and the originals do carry the ground truth: Flysta3D's source has
`new_x`/`new_y`, the Open-ST source has `spatial_3d_aligned`, and DLPFC layer labels are on
spatialLIBD. One instruction still names an accession (`GSE144240`). Nothing currently
prevents this, and no container will, because the opening is the network we granted on
purpose. The options are to detect it after the fact by flagging data-repository hosts in the
leakage audit, to put an allowlisting proxy in front of the container, or to run with
`--net --network=none` and give up the documentation affordance for every arm.

## Scoring the full slice, not a random half

Until 2026-09-15 every task declared `score_split = "eval"` against a 50/50 split manifest, so
the graded number came from a random half of the observations in each slice while SABench
computes on the whole slice. Measured on the same submissions:

| Metric | eval half | full slice | difference |
| --- | --- | --- | --- |
| `gene_pcc` | 0.433 | 0.538 | -0.104 |
| `gene_cosine` | 0.677 | 0.763 | -0.086 |
| `gene_ssim` | 0.417 | 0.523 | -0.106 |
| `gene_mi` | 0.199 | 0.307 | -0.108 |
| `matching_accuracy` | 0.744 | 0.748 | -0.004 |
| `overlap_ratio_region` | 0.503 | 0.537 | -0.034 |

The cause is attenuation rather than anything about the eval half: on sa-001 the train half
scored 0.381, the eval half 0.397, and the full slice 0.509, so both halves sit about 0.12
low. Halving the cells halves each subgrid's occupancy, which makes the per-grid mean
expression noisier and dilutes the correlation between the two slices' grid-mean vectors. The
four gene metrics aggregate within grids and are hit hard; the landmark metrics do not and
barely move.

That bias is larger than most of the between-method differences the paper reports, so the
gene metrics were not comparable to MOESM7. The split protected the solve-time self-scorer,
which blind runs withhold and which 7 of 527 historical runs ever invoked.

The split keys are gone from all 92 tasks and from the builder. With no `split_manifest`
declared, `TaskSpec.score_split` defaults to `all` and the scorer applies no mask. Rankings
were preserved across the change (Spearman 0.87 to 0.99), so within-benchmark conclusions
from the old scores still hold in order, but no absolute number carries over.
