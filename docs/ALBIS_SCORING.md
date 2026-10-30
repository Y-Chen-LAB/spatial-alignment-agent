# ALBIS scoring

All sa-301–312 tasks, including smoke variants, use `albis_alignment_v1`.
The scorer emits `scoring_rules` with metric directions, units, aggregation,
primary metric, and the exact inputs to the legacy `overall_mean`.

| Metric | Direction | Definition |
| --- | --- | --- |
| `mae` | Lower | Mean absolute x/y error against truth in the s00 frame, weighted by observation count; no fit. Primary score, in microns. |
| `mae_identity` | Lower | MAE of the input coordinates. |
| `mae_reduction` | Higher | `1 - mae / mae_identity`: 1 is exact, 0 is identity, negative is worse. Undefined for zero identity MAE; inspect `mae_reduction_comparable` before cross-task comparison. |
| `gene_pcc` | Higher | Existing SABench marker-gene grid Pearson correlation. |
| `gene_cosine` | Higher | Existing SABench cosine between per-grid expression averages for each marker, averaged across markers. |
| `gene_ssim` | Higher | Existing SABench grid structural similarity. |
| `gene_mi` | Higher | Existing SABench grid mutual information; unbounded, so its scale differs from the other terms. |
| `mnn_coverage` | Higher | `2 * number_of_mutual_pairs / (n_left + n_right)`. |
| `mnn_label_accuracy` | Higher | Fraction of mutual pairs with equal scorer-side Region labels. |
| `mnn_distance_um` | Lower | Mean Euclidean distance of mutual pairs, in microns. |
| `mnn_expression_cosine` | Higher | Mean cosine of raw marker-expression vectors for mutual pairs. A zero-norm vector contributes 0; no common genes or no pairs is undefined. |

MNN here means **mutual spatial 1-nearest neighbours** between submitted
coordinates of consecutive sections (or the task's declared star pairs).
It is a local diagnostic, not an ALBIS paper metric or an expression-space MNN
algorithm. Search uses exact cKDTree queries, with one neighbour chosen in ties;
duplicate coordinates can therefore affect coverage. There is no distance cutoff.
Even separated sections can have mutual pairs, and collapsed coordinates can
have tiny distances. Read coverage, label agreement, and MAE together.

All pair metrics average finite pair values with equal pair weights. Every
metric uses the requested train/eval/all observation mask; private labels stay
on the scorer side. Undefined values serialize to null, rather than perfect scores.

`overall_mean` remains available for existing reports and averages only PCC,
grid cosine, SSIM, and MI. It excludes MAE and all MNN diagnostics. It is not
the ALBIS primary score: an exact coordinate answer need not maximize these
expression metrics because sections and batch effects differ. Compare MAE
and identity-relative recovery, then inspect the individual metrics.

`scripts/build_albis_baselines_table.py` exports every gene and MNN component
and a corresponding `_n_runs` coverage column. Median rows report each
metric's own median; best rows report all metrics from the same lowest-MAE
run. Five-task component means require that component on all five tasks.
Old score JSON files lack MNN values; they remain blank with zero coverage.
Updating task profiles does not retroactively rescore saved submissions.

No dense all-pairs distance matrix or additional expression read is needed:
MNN cosine reuses the expression already loaded for gene grids, processing
matched rows in chunks of 4,096. Existing submissions can be rescored with
`scripts/score_agent_submission.py`; no new agent run is necessary.
