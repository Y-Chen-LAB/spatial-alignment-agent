# Agents against the SABench paper tables

This directory compares our agent runs with the per-method tables in Yan et al. 2026,
*Nature Computational Science* ("SABench"). Every agent number here was scored with a new
SABench-exact path. That path was checked against the paper by re-running PASTE and PASTE2
and reproducing their published rows. No existing `scores.json` and no default metric profile
was changed.

## How to regenerate

```
# 1. score agent runs with the SABench-exact path (one SLURM job per dataset)
python scripts/score_sabench_paper_metrics.py --dataset {dlpfc,mhpr,mob,dlpfc_rot,dlpfc_crop,mhpr_rot,mhpr_crop} \
       --runs-csv /dcs07/chen/data/alignment-agent/sabench_comparison/runs_target.csv
# 2. flag submissions that are not spatial mappings of the input
python scripts/sabench_comparison/geometry_check.py
# 3. build the tables (primary), and the sensitivity view that keeps flagged runs
python scripts/sabench_comparison/build_sabench_comparison.py
python scripts/sabench_comparison/build_sabench_comparison.py --nonspatial keep
```

The validation re-runs are in `scripts/sabench_comparison/run_paste_*_sabench.py`,
`run_paste2_dlpfc_sabench.py` and `run_paste_serial_task.py`. Raw scores and intermediates
are under `/dcs07/chen/data/alignment-agent/sabench_comparison/`. The SABench code and Zenodo
archive are under `/dcs07/chen/data/alignment-agent/sabench_ref/`.

## What the scoring path does

The code is `src/genomics_agent_bench/sabench_exact.py`, driven by
`scripts/score_sabench_paper_metrics.py`. It follows the public SABench code
(`Tutorial/SABench.py`, `FeatureSimilarity/GeneBased.ipynb` and `LandmarkBased.ipynb`) line
for line:

- **Grid.** The grid sits on the shared bounding rectangle, 10 x 10 cells, with half-open
  cells. The density threshold is half the smaller mean number of points per cell, and a cell
  is kept only if both slices reach it.
- **Gene metrics.** PCC, cosine, SSIM (L = 1) and MI are computed per gene on the grid-mean
  stored `X`, then averaged over consecutive pairs. The paper's gene column order is kept,
  duplicates included: the DLPFC table has 123 columns for 117 unique genes.
- **Landmark metrics.** Nearest-neighbour label accuracy and alpha-shape (alpha = 12)
  region overlap are computed per named region, then averaged over pairs.
- **Robustness.** Matching accuracy is computed per perturbation level against the reference
  slice, which is what MOESM8 reports.
- **Frame.** The whole submitted stack is moved rigidly so that slice 1 sits in the pose
  the paper's PASTE-family outputs were in. This is the `canonical` frame, and the tables
  use it. For DLPFC that pose is SABench's own `151673_shuffled.h5ad` from the Zenodo
  archive. For the other datasets it is the task's input pose. Scores in the `submitted`
  frame are kept alongside.

### Differences from our default scorer (`metrics.py`)

None of the defaults changed. Each item says what the new path does instead.

| # | Difference | Default scorer | SABench / new path | Size of effect |
|---|---|---|---|---|
| 1 | Reporting unit | one scalar per pair, the mean over genes | one value per gene, the mean over pairs (paper layout) | layout only |
| 2 | Duplicate genes | removed (117 DLPFC genes) | kept, so the paper's 123 columns weight 6 genes twice | small, on row means only |
| 3 | Degenerate inputs | constant vector or zero norm gives 0; MI needs 4 or more grids | NaN propagates, as `pearsonr` and the zero norm do in SABench | none observed: 0 NaN across all rescored runs |
| 4 | MI | `random_state=0` | SABench calls it unseeded; we keep seed 0 | about 0.01 per gene (see validation) |
| 5 | Coordinate frame | the submitted frame | the method's output frame, i.e. the slice-1 input pose; DLPFC inputs were randomly rotated | **large on DLPFC**: moving the same 38 agent submissions into the paper pose raises PCC by 0.049 on average (max 0.060), cosine by 0.037 and MI by 0.040. MHPR 0.002, MOB 0.006 (`frame_sensitivity.csv`) |
| 6 | Robustness tasks | gene and landmark metrics averaged over all levels | matching accuracy per level only (MOESM8) | different quantity |
| 7 | Rank aggregation (`ranking.py`) | rank within each task, average the ranks; ties get the average rank | Methods text: average the scores over tasks, then rank. MOESM9 shows shared (min) ranks, and its overall columns equal the rank of the mean of the metric ranks | `ranking.py` left unchanged; the builder uses min-tie ranks |
| 8 | `overall_mean` | the default aggregate | not a paper quantity | not used here |

The grid labels, matching accuracy and region overlap were already identical to SABench
(`docs/audits/METRIC_PARITY_AUDIT.md`). The new path also reproduces the
SABench tutorial output to every printed digit. On the unaligned tutorial DLPFC slices it
gives `Average_Accuracy` 0.7585041490444177 and WM overlap 0.7302734867472861, which are
SABench's own printed values.

## Validation: re-running the paper's methods

Raw files are `validation/rerun_vs_paper_summary.csv` (per metric) and
`validation/rerun_vs_paper_per_gene.csv` (per gene).

| Re-run (our CPU, POT 0.9.1, paste-bio 1.4.0) | Paper row | Landmark metrics (7 columns) | PCC / Cos / SSIM mean, ours vs paper | Genes within 0.01 (PCC/Cos/SSIM) | MI |
|---|---|---|---|---|---|
| PASTE, SABench shuffled pose | PASTE_DLPFCS | **all 7 identical to 4 decimals** (accuracy 0.6412 = 0.6412) | 0.4921/0.4917, 0.7784/0.7780, 0.4805/0.4802; per-gene r = 0.9998 | 97.6 % / 99.2 % / 98.4 % of 123 | mean abs diff 0.012, r = 0.996 (unseeded in SABench) |
| PASTE2, shuffled pose | PASTE2_DLPFCS | accuracy 0.8611 vs 0.8619; regions within 0.01 | 0.598/0.593, 0.821/0.819, 0.582/0.576; r = 0.998 | median abs diff 0.008 | 0.357 vs 0.350 |
| PASTE on MHPR (sa-009 data), X cast to float32 | PASTE_MHPR | accuracy 0.209 vs 0.182 | 0.037/0.071, 0.727/0.742, 0.048/0.080 | not reproduced | 0.041 vs 0.050 |
| PASTE on MHPR, X kept float64 as stored | PASTE_MHPR | accuracy 0.437 vs 0.182 | 0.234/0.071 PCC | not reproduced | |
| PASTE on the MHPR crop and rotation sweeps | MOESM8 MHPR_* PASTE | not reproduced per level; rotation mean 0.280 vs 0.270, crop mean 0.141 vs 0.252 | | | |
| PASTE on the DLPFC crop sweep | MOESM8 DLPFC_crop PASTE | 9 of 10 levels within 0.004 (10 % to 90 %); 100 % gives 0.504 vs 0.169 | | | |
| PASTE on the DLPFC rotation sweep | MOESM8 DLPFC_rotation PASTE | not reproduced per angle: mean 0.355 vs 0.391 over 25 angles, with per-angle swings of ±0.66 | | | |
| PASTE_p0, shuffled pose | PASTE_p0_DLPFCS | accuracy 0.378 vs 0.416 | 0.037 vs 0.105 PCC | not reproduced | |

What this shows:

1. **The scoring path is exact.** Where the method's output can be regenerated, the paper's
   numbers come back to four decimals on the landmark metrics and to within 0.001 on the
   gene-metric means. The remaining per-gene residuals are on a handful of sparse genes, for
   example TH.
2. **The method outputs are not stable, even though the metric is.** PASTE in exact
   arithmetic does not depend on the input pose. In practice it lands in different basins:
   PASTE on the *native* DLPFC pose scores PCC 0.416 against 0.492 in the shuffled pose. The
   paper's own MOESM8 shows the same chaos, with PASTE scoring 0.864 at 0° and 0.169 at a
   100 % crop of almost the same pair. PASTE_p0 cannot be reproduced at all. Its spatial
   initialisation depends on the input pose of slices 3 and 4, whose shuffled versions
   SABench did not release.
3. **MHPR has no exact row reproduction.** Our MHPR slices have the paper's expression
   matrix: the sparsity of Data5 matches to 10 digits, and the BASS labels match the paper's
   cell counts. PASTE is the right kind of failure on MHPR in both re-runs, but merely casting
   X from float64 to float32 moves its accuracy from 0.437 to 0.209 (paper 0.182). With
   method noise that large, the MHPR rows cannot confirm the metric to better than "same
   regime". The DLPFC rows do confirm it, and the same code scores MHPR.

## Headline comparison

The primary statistic is the median over runs of each model and prompt arm (`bare` =
`full-basic`, `equipped` = `full-package-paper-prior`); the best run is a secondary view. The
full layouts are in `<dataset>/<DATASET>_GeneBased_<metric>_with_agents.csv` (per gene) and
`<DATASET>_LandmarkBased_Heatmap_with_agents.csv` (per region). Means and ranks are in
`<dataset>/<DATASET>_headline.csv`.

The rank columns place the agent row among the 14 paper methods only. STAligner is collapsed
to `STAligner_best` and `STAligner_worst` by per-metric max and min, as in MOESM9. Rank 1 is
best.

### MHPR (sa-009; Data5 to 9; MOESM7). The most comparable table.

| Row | PCC | Cos | SSIM | MI | Accuracy | ROI mean | Gene rank | Landmark rank |
|---|---|---|---|---|---|---|---|---|
| Spateo_rigid (best paper) | 0.533 | 0.921 | 0.525 | 0.241 | 0.754 | 0.538 | 1 | 1 |
| SPACEL | 0.531 | 0.867 | 0.517 | 0.246 | 0.758 | 0.527 | 3 | 3 |
| PASTE2 | 0.523 | 0.862 | 0.512 | 0.240 | 0.742 | 0.530 | 5 | 4 |
| gpt-6-astra bare / equipped | 0.541 / 0.545 | 0.870 | 0.527 / 0.531 | 0.253 / 0.258 | 0.765 / 0.762 | 0.544 / 0.546 | 1 / 1 | 1 / 1 |
| gemini-3.5-flash equipped | 0.534 | 0.867 | 0.519 | 0.250 | 0.755 | 0.532 | 1 | 1 |
| gpt-5.5 equipped | 0.532 | 0.865 | 0.517 | 0.242 | 0.754 | 0.534 | 4 | 1 |
| claude-sonnet-5-5 bare / equipped | 0.533 / 0.527 | 0.866 / 0.863 | 0.519 / 0.515 | 0.244 / 0.242 | 0.750 / 0.742 | 0.533 / 0.534 | 3 / 4 | 4 / 4 |
| claude-opus-5-5 bare / equipped | 0.527 / 0.494 | 0.863 / 0.852 | 0.514 / 0.482 | 0.241 / 0.214 | 0.743 / 0.710 | 0.533 / 0.495 | 4 / 11 | 4 / 9 |
| claude-sonnet-4-6 bare / equipped | 0.511 / 0.496 | 0.858 / 0.853 | | | 0.723 / 0.718 | 0.499 / 0.495 | 9 / 11 | 7 / 9 |
| gpt-5.5 bare | 0.505 | 0.856 | 0.493 | 0.223 | 0.718 | 0.503 | 11 | 9 |
| gemini-3.5-flash bare (1 of 4 runs failed) | 0.233 | 0.422 | 0.228 | 0.094 | 0.327 | 0.203 | 11 | 11 |

On MHPR the median agent sits inside the paper's top cluster, which is tight: the top eight
methods are within 0.03 on every metric. gpt-6-astra beats every paper method on 5 of 6
metrics, but the margins are 0.01 or less.

### DLPFC (sa-001; Data1 to 4; MOESM7). The metric is exact, but the task input is not the paper's.

| Row | PCC | Cos | SSIM | MI | Accuracy | ROI mean | Gene rank | Landmark rank |
|---|---|---|---|---|---|---|---|---|
| SPACEL (best paper) | 0.596 | 0.822 | 0.581 | 0.352 | 0.869 | 0.684 | 1 | 1 |
| PASTE2 | 0.593 | 0.819 | 0.576 | 0.350 | 0.862 | 0.659 | 2 | 2 |
| CAST | 0.580 | 0.816 | 0.565 | 0.327 | 0.821 | 0.562 | 3 | 4 |
| PASTE | 0.492 | 0.778 | 0.480 | 0.262 | 0.641 | 0.450 | 5 | 8 |
| **identity (no alignment), paper pose** | 0.577 | 0.814 | 0.560 | 0.311 | 0.759 | 0.487 | 4 | 7 |
| gpt-6-astra bare / equipped | 0.595 / 0.593 | 0.820 / 0.819 | 0.579 / 0.578 | 0.346 / 0.334 | **0.876** / 0.869 | **0.685** / 0.669 | 2 / 2 | 1 / 1 |
| gpt-5.5 equipped | 0.593 | 0.820 | 0.577 | 0.344 | 0.861 | 0.664 | 2 | 2 |
| claude-sonnet-5-5 bare / equipped | 0.586 / 0.591 | 0.817 / 0.818 | | | 0.846 / 0.854 | 0.636 / 0.654 | 3 / 3 | 3 / 3 |
| claude-opus-5-5 bare / equipped | 0.571 / 0.589 | | | | 0.785 / 0.821 | 0.531 / 0.595 | 4 / 3 | 7 / 4 |
| gemini-3.5-flash bare / equipped | 0.584 / 0.578 | | | | 0.805 / 0.819 | 0.572 / 0.587 | 3 / 4 | 5 / 4 |
| claude-sonnet-4-6 bare / equipped, gpt-5.5 bare | about 0.573 | | | | about 0.755 | about 0.49 | 4 to 5 | 7 |

**Not comparable as a benchmark result.** The paper applied random rigid transforms to all
four DLPFC slices before alignment. sa-001 gives the agents the native spatialLIBD
coordinates, which are already roughly registered. Leaving them untouched scores accuracy
0.759, above PASTE (0.641) and eight other paper methods. Three agent arms sit at that
identity level, so they effectively did nothing. The comparison is fair on the metric and
unfair on the task. A paper-faithful DLPFC task would need the four shuffled slices, and
SABench released only 151673 and 151674 shuffled.

### MOB (sa-030; Data78 to 79; MOESM10). Gene metrics only, partly comparable.

| Row | PCC | Cos | SSIM | MI | Gene rank |
|---|---|---|---|---|---|
| STAligner_best (landmark3) | 0.479 | 0.841 | 0.456 | 0.231 | 1 |
| CAST | 0.398 | 0.782 | 0.390 | 0.193 | 2 |
| SLAT | 0.154 | 0.687 | 0.156 | 0.094 | 3 |
| gpt-6-astra equipped / bare | 0.611 / 0.531 | 0.864 / 0.839 | 0.566 / 0.512 | 0.317 / 0.264 | 1 / 1 |
| claude-sonnet-5-5 equipped / bare | 0.581 / 0.559 | | | | 1 / 1 |
| claude-opus-5-5 bare / equipped | 0.564 / 0.551 | | | | 1 / 1 |
| gemini-3.5-flash equipped | 0.183 | | | | 3 |
| claude-sonnet-4-6 bare / equipped | 0.144 / -0.010 | | | | 5 / 9 |
| gpt-5.5 bare and equipped, gemini bare | 0 (median run zeroed) | | | | 9 |

Two caveats apply. First, our Slide-seqV2 slice has 21,724 beads where the paper's has
20,835, because the paper trimmed edge outliers; the Stereo-seq slice matches exactly.
Second, **11 of 31 MOB submissions are not spatial alignments**. They are coordinates that no
similarity transform of the input explains (`geometry_check.csv`, residual ratio > 0.5). One
gpt-5.5 run wrote a 90 % expression-PCA, 10 % spatial blend into x and y (its
`report.json`: `"method": "auto_hvg_expression_pca_spatial_blend"`). The SABench gene metrics
compare only per-grid mean expression, so such an output scores PCC 0.85 to 0.89, far above
every paper method. In the primary tables these runs are scored 0, like failures.
`sensitivity_nonspatial_kept/` scores them as submitted; there gpt-5.5 bare has median PCC
0.846, which shows how gameable these metrics are rather than how well the agent aligned. The
same rule flags 1 of 38 DLPFC runs and 3 of 31 DLPFC-crop runs.

### Robustness (sa-040 to sa-043; MOESM8). Mean matching accuracy over levels.

`robustness_<sheet>_with_agents.csv` is in the paper's layout, one row per level.
`robustness_summary.csv` gives the means and SDs.

| Row (median over runs) | DLPFC rotation | DLPFC crop | MHPR rotation | MHPR crop |
|---|---|---|---|---|
| SPACEL (best paper overall) | 0.888 | 0.866 | 0.737 | 0.780 |
| PASTE2 | 0.873 | 0.721 | 0.728 | 0.460 |
| Spateo_rigid | 0.876 | 0.664 | 0.733 | 0.714 |
| PASTE (paper / our re-run) | 0.391 / 0.349 | 0.365 / 0.398 | 0.270 / 0.280 | 0.252 / 0.141 |
| gpt-6-astra bare / equipped | 0.895 / 0.903 | 0.907 / 0.894 | 0.778 / 0.762 | 0.744 / 0.739 |
| claude-opus-5-5 bare / equipped | 0.880 / 0.835 | 0.913 / 0.893 | 0.729 / 0.731 | 0.718 / 0.713 |
| claude-sonnet-5-5 bare / equipped | 0.835 / 0.835 | 0.859 / 0.894 | 0.726 / 0.732 | 0.712 / 0.707 |
| gpt-5.5 bare / equipped | 0.857 / 0.835 | 0.859 / 0.859 | 0.734 / 0.727 | 0.710 / 0.709 |
| gemini-3.5-flash bare / equipped | 0.835 / 0.835 | 0.858 / 0.420 | 0.722 / 0.722 | 0.708 / 0.710 |
| claude-sonnet-4-6 bare / equipped | 0.814 / 0.875 | 0.859 / 0.782 | 0.356 / 0.718 | 0.714 / 0.167 |

Agents are nearly flat across angles because they undo the rotation exactly; the
best-performing agents match or beat SPACEL everywhere except the MHPR crop sweep. The DLPFC
sweeps are comparable: same slices, same reference pose, and PASTE's crop column reproduces.
The MHPR sweep numbers exist only because of the label fix in bug 1 below.

## SABench-style overall rank (MOESM9)

We **cannot** place the agents into the published NGS or imaging rank tables. Those tables
average over many datasets: NGS uses Data1 to 4, 10 to 12, 16 to 31, 32 to 33 and 34 to 35;
imaging uses Data5 to 9, 34 to 35, 39 to 41 and 45 to 77. The paper publishes per-method
scores only for DLPFC, MHPR and MOB (plus MAE for Flysta3D and MLN). What we provide instead
is in `sabench_style_rank_by_dataset.csv`:

- per-dataset SABench-style ranks with agents inserted;
- `rank_concordance_single_dataset_vs_moesm9.csv`, which shows how well a single dataset
  stands in for its group. For the paper methods, the DLPFC-only ranks agree with the NGS
  ranks at Kendall tau 0.42 to 0.58, and the MHPR-only ranks agree with the imaging ranks at
  tau 0.63 to 0.85.

So MHPR is a fair, if noisy, proxy for the imaging rank, and DLPFC is a weak proxy for the
NGS rank. On that MHPR proxy, gpt-6-astra (both arms) and gemini-3.5-flash-equipped tie at
rank 1, and gpt-5.5-equipped is rank 1 on landmark. None of this is a reproduction of
MOESM9.

## Bugs and issues found (not fixed; flagged)

1. **sa-042 and sa-043 (MHPR sweeps) have the wrong labels.** The perturbed Data6 slices in
   `compiled_data/sweeps/sa-04{2,3}-*` carry `Cell_class` in `obs/Region`, while the Data5
   reference carries the BASS anatomical regions. The default scorer therefore gives
   `matching_accuracy = 0` and `overlap_ratio_region = 0` to every run on these two tasks;
   every stored `scores.json` there is wrong on the landmark metrics. The perturbed slices
   hold the same 5,557 cells as `MHPR_animal1_n009.h5ad`, so the new path takes `Region` from
   there by obs id. The sweep h5ads should be rebuilt.
2. **The compiled sweep h5ads lack the anndata `encoding-version` attributes**, so
   `anndata.read_h5ad` refuses them. Our own h5py readers are unaffected, but method code run
   by an agent would fail to load them.
3. **The sa-001 input is unshuffled**; see the DLPFC section. This is a task-design issue, not
   a scorer bug.
4. **The gene metrics can be gamed** by non-spatial coordinates; see the MOB section.
   `geometry_check.py` is a proposed guard.
5. **`ranking.py` uses rank-then-average with average-rank ties.** That differs from the
   paper's stated procedure (average the scores, then rank) and from MOESM9's shared ranks.
   For a single task the two conventions coincide.

## What remains non-comparable

- **MOESM9 overall ranks**: the per-dataset scores behind them are not published.
- **DLPFC**: the inputs differ (native versus randomly transformed slices). The metric is
  exact.
- **MOB**: the Slide-seqV2 bead set differs (21,724 versus 20,835), and the input pose of the
  paper's methods is unknown.
- **Robustness per level**: the paper's own PASTE values swing chaotically between angles, so
  only the means over levels are meaningful. The MHPR sweep values rely on the label fix
  described above.
- **MI**: SABench's estimator is unseeded, so expect about 0.01 per-gene noise, and treat MI
  as comparable to about two decimals.
- **STAligner best and worst**: we assume a per-metric max and min over the landmark
  variants; the paper does not say how it picked them.
- **Agent sample sizes are small**: 1 to 10 runs per cell. claude-opus-5-5-equipped and
  claude-sonnet-5-5 have a single run on most tasks, so their median equals their best.
