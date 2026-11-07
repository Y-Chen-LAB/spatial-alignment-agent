# Albis simulated alignment tasks (sa-301 to sa-305)

Five serial-alignment tasks built from the Albis simulator
(github.com/JanMatthias1/Albis), where the true frame of every slice is known exactly.
This file covers how they are built, how to run them, what we found while
validating them, and their known limitations. Status as of 2026-09-22.

## Why simulated

The real-data tasks in this suite score with SABench's gene-grid and region metrics,
and only a few (sa-019, sa-023, sa-025–027) have coordinate ground truth. Albis
gives ground truth for every observation, at three platform resolutions, with batch
effects of a known size.

## The five tasks

Each is one configuration from an Albis tutorial notebook, used unchanged: 6 domains,
8 cell types, 10 slabs along Z, a per-slice rotation of up to ±270° and a shift of up to
0.5 R.

| Task | Albis dataset | Notebook | Modality | Tissue | Composition | `batch_sigma` |
| --- | --- | --- | --- | --- | --- | --- |
| sa-301 | `strong_mix_cell` | strong_domain_mix, Part 1 | cell | 24,207 cells, R = 2050 µm | strong | 1.5 |
| sa-302 | `xenium_cell` | xenium_cell | cell | 600,000 cells, R = 6000 µm | baseline | 1.5 |
| sa-303 | `visium_spot` | visium_spot | spot, 55 µm / 100 µm pitch | 600,000 cells, R = 2050 µm | baseline | 0.3 |
| sa-304 | `visiumhd_bin16` | visiumHD_bin, 16 µm | bin | 600,000 cells, R = 2050 µm | baseline | 0.7 |
| sa-305 | `visiumhd_bin8` | visiumHD_bin, 8 µm | bin | 600,000 cells, R = 2050 µm | baseline | 0.8 |

Albis has two domain compositions. **Baseline** (the manuscript's Figure 2) enriches each
domain mildly (about 0.18 against 0.09–0.13), and its last domain is exactly uniform.
**Strong** gives each domain two signature cell types at 0.30. We left out the strong-mix
spot and bin16 variants because, for alignment, they differ from sa-303/sa-304 only in
composition.

Batch effects are included. Albis draws one log-fold-change per slice and gene, with
standard deviation `batch_sigma`, and resamples counts as Poisson. The agent sees
post-batch counts; the pre-batch layer is not exported. Each modality has one
`batch_sigma`. The manuscript's sweeps (Figure 3G: 9, 7 and 16 levels) live in the
separate `sim_paper` repository.

## Variants (sa-306 to sa-310)

Each variant changes one thing relative to a task above, so a score difference can be
attributed to that change.

| Task | Albis dataset | Compare with | What changes |
| --- | --- | --- | --- |
| sa-306 | `visium_spot_slide` | sa-303 | Spot lattice fixed to each slide: no lattice leak. Same per-slice transforms as sa-303. |
| sa-307 | `visiumhd_bin16_slide` | sa-304 | Bin grid fixed to each slide: no lattice leak. Same per-slice transforms as sa-304. |
| sa-308 | `strong_mix_cell_gap` | sa-301 | Block cut into 30 thin slabs (137 µm) and 9 kept at irregular spacing: slabs 1, 5, 8, 18, 19, 21, 22, 27, 28 (gaps 4, 3, 10, 1, 2, 1, 5, 1). The cell count is tripled so each section matches sa-301's density. Cells, so no lattice. |
| sa-309 | `visium_spot_sparse` | sa-306 | 120,000 cells instead of 600,000 in the same sphere, then QC: drop observations with 0 counts or fewer than 3 detected genes. |
| sa-310 | `visium_spot_sparse_gap` | sa-309 | sa-309 with the same irregular sections as sa-308. |

**Slide-frame grid (the leak fix).** `simulate_slide_frame` in `Albis/scripts/alignment_benchmark.py`
moves each slice's molecules by that slice's rigid transform first, then bins them on one grid
that is axis-aligned in every slide's frame, as a real Visium slide is. True bin positions come
from inverting the transform. On smoke data, every slice's input lattice now reads 0°, whatever
its true angle. Slices' true bin positions no longer share s00's lattice, so there is nothing
to snap to. ICP, which could lock onto the old lattice, got worse on the fixed bin16 smoke task
(MAE 120 → 184 µm).

`gap_slices` draws each gap log-uniformly from 1 to 10 slabs (seeded), and redraws until at least 5 sections remain and the largest gap is at least 4× the smallest. Counts per spot match the 10-slab version (sa-310 and sa-309 both have a median of 464 counts and 151 genes).
The agent prompt says that the spacing is irregular and not given, and says when QC was applied.

Smoke MAE (µm): the exact answer scores 0 on all five.

| Task | Identity | centroid_similarity | icp_similarity |
| --- | --- | --- | --- |
| sa-306 | 192 | 31 | 98 |
| sa-307 | 136 | 33 | 184 |
| sa-308 | 418 | 95 | 273 |
| sa-309 | 335 | 62 | 302 |
| sa-310 | 214 | 51 | 170 |

## How a task is built

1. `Albis/scripts/alignment_benchmark.py` runs the simulator and writes, per dataset,
   `albis_full.h5ad` (the simulator's own AnnData), `slices/sNN.h5ad`,
   `marker_genes.csv` and `catalog.json`.
2. `scripts/materialize_albis_tasks.py` (this repo) turns each `catalog.json` into
   `benchmarks/spatial_alignment/tasks/sa-30N-.../`.

Per-slice file contents:

| Field | Content | Agent sees it |
| --- | --- | --- |
| `obsm/spatial` | Albis `spatial_unaligned`: true coordinates after this slice's rigid offset | yes |
| `X` | post-batch counts, CSR float32 | yes |
| `var/_index` | gene names (`G…`), annotations dropped | yes |
| `obs/_index` | `sNN_0000123`, assigned after a seeded row shuffle | yes |
| `obs/x_GT`, `obs/y_GT` | true coordinates expressed in slice `s00`'s frame | no |
| `obs/Region` | `domain_true` | no |

**Reference frame.** The suite's MAE applies no rigid fit, so the answer frame has to be
fixed. We map every slice's true coordinates through `s00`'s own offset, so `s00`'s input
already is its answer. The prompt tells the agent to leave `s00` in place and bring
every other slice onto it.

**Index leakage.** Albis emits bins and spots in grid order and names them by position.
The export shuffles rows and renames observations. Empty bins and spots are dropped.

Scoring profile: `albis_alignment_v1`, which gives MAE (per slice and overall),
the four gene-grid metrics, and separate spatial MNN diagnostics. MAE is the
primary score. See [ALBIS_SCORING.md](ALBIS_SCORING.md) for all scoring rules.

## Running

```bash
# 1. Full-scale data (Albis repo).
cd ~/Albis
sbatch --array=0-4 scripts/alignment_benchmark.sbatch        # -> ~/p2b-data/paper2bench-data/albis_sim

# 2. Tasks (this repo)
cd ~/spatial-alignment-agent
./.venv/bin/python scripts/materialize_albis_tasks.py --data-root ~/p2b-data/paper2bench-data/albis_sim --force
./.venv/bin/p2b validate benchmarks/spatial_alignment/tasks/sa-30*/task.toml

# 3. Scripted baselines and scoring (see AGENT_RUNBOOK.md)
./.venv/bin/python scripts/run_task_baseline.py benchmarks/spatial_alignment/tasks/<task>/task.toml --method centroid_similarity --max-genes 100
./.venv/bin/python scripts/score_agent_submission.py benchmarks/spatial_alignment/tasks/<task>/task.toml <submission_dir>
```

### Agent runs on the cluster

`scripts/slurm/submit_alignment.sh` only draws rows from `run_matrix.csv`, which does
not list these tasks. `alignment_array.sbatch` accepts any rows file
(`agent,model,task,condition,rep,prompt_variant,prebootstrap`), so submit it directly:

```bash
source scripts/slurm/site_config.sh
ROWS=$P2B_ROWS_ROOT/albis_smoke_dryrun.csv
: > $ROWS
for a in "claude,claude-sonnet-4-6" "gemini,gemini-3.5-flash"; do
  for t in $(ls benchmarks/spatial_alignment/tasks | grep 'albis.*smoke'); do
    echo "$a,$t,bare,1,full-basic,never" >> $ROWS        # equipped: full-package-paper-prior,auto
  done
done
scripts/refresh_claude_login.sh                          # claude rows carry a copy of the login
sbatch --array=0-$(( $(wc -l < $ROWS) - 1 )) --partition=$P2B_PARTITION --cpus-per-task=$P2B_CPUS \
  --mem=16G --time=02:00:00 --exclude=$P2B_EXCLUDE_NODES \
  --export=ALL,P2B_ROWS=$ROWS,P2B_TIMEOUT_MINUTES=60 scripts/slurm/alignment_array.sbatch
```

Runs land in `agent_runs/cluster/<job>_<index>/`, logs in `logs/p2b-align-<job>_<index>.out`.
Codex rows cannot run on the cluster (see README).

## Findings from validation

### Scores on smoke data (5,000 cells, 4 slices)

MAE in µm. `overall_mean` averages the four gene metrics and excludes MAE.

| Task | Exact answer | Identity | centroid_similarity | pca_similarity | icp_similarity |
| --- | --- | --- | --- | --- | --- |
| sa-301 strong-mix cell | 0.0 | 605 | 118 | 270 | 341 |
| sa-302 xenium cell | 0.0 | 686 | 119 | 254 | 271 |
| sa-303 visium spot | 0.0 | 191 | 34 | 130 | 159 |
| sa-304 bin 16 µm | 0.0 | 136 | 34 | 211 | 120 |
| sa-305 bin 8 µm | 0.0 | 136 | 34 | 154 | 170 |

MAE ranks every method correctly. The gene metrics do not: on sa-305 the exact answer's
`overall_mean` (0.230) is below icp_similarity's (0.260).

### Why the gene-grid metrics are near zero, even for the exact answer

This is a property of the data, not a scorer bug. On full-scale sa-301 (24,207 cells), the
exact answer's gene PCC is 0.053. The metric averages each marker gene over a 10×10 grid
and correlates the grid means between adjacent slices. Its ceiling can be measured
directly by splitting **one** slice's cells in half at random and correlating the halves:

| Grid | Adjacent slices, exact answer | Two halves of the same slice (ceiling) |
| --- | --- | --- |
| 10×10 | 0.085 | 0.037 |
| 5×5 | 0.150 | 0.090 |

The ceiling is about as low as the observed value, so no alignment can do better. Single
genes carry little spatial signal at grid scale: 88% of marker-gene entries are zero
(`theta` = 0.4 is heavily overdispersed), and domain means differ by only about 1.2–2×.
The spatial structure is in joint cell-type composition across genes, which a per-gene
grid correlation cannot see. Batch effects play no part, since a per-gene scale factor
leaves a Pearson correlation unchanged.

### The region metrics can be beaten by a wrong answer

Scoring full-scale sa-301 with the region metrics (`sabench_light_alignment_v1`, `Region` = true domain):

| Submission | MAE (µm) | matching_accuracy | overlap_ratio_region |
| --- | --- | --- | --- |
| Exact answer | 0 | 0.815 | 0.637 |
| Identity | 1258 | 0.224 | 0.109 |
| centroid_similarity | 371 | **0.837** | **0.686** |

The region metrics separate "aligned" from "not aligned", but a submission 371 µm off
outscores the truth. Adjacent slabs really do differ (domain boundaries warp along Z,
and labels are fuzzed near boundaries), and a similarity transform may rescale to
maximise label overlap. On real data this cannot be checked. Here it can, and it argues
for **MAE as the primary score on these tasks**, with the region metrics as secondary
diagnostics at most.

### MAE reduction, and how failed runs are scored

Every task with coordinate ground truth also reports `mae_identity` (the MAE of submitting the
input coordinates unchanged) and `mae_reduction` = 1 − MAE / MAE_identity, overall and per slice.
1 is exact, 0 is no better than identity, and negative is worse than identity. Tasks whose
identity MAE is under 10 µm (only sa-023, at 1.1 µm) are flagged `mae_reduction_comparable = 0`
and left out of cross-task averages, because there the ratio blows up for any small error.

**Failed runs** (no scorable submission) are recorded with identity metrics, `mae = mae_identity`
and `mae_reduction = 0`, flagged `mae_imputed_from_identity = 1`. Previously they were recorded
with MAE 0, which reads as a perfect alignment. This is a placeholder: failed rows should be
rerun, and the placeholder is replaced once a rerun scores.

## Known limitations

1. **Bin/spot lattice leak (sa-303, sa-304, sa-305).** Albis builds the bin or spot
   grid in the true frame and then rotates and shifts the centres. The nearest-neighbour
   directions in a slice therefore reveal its rotation modulo 90° (bins) or 60° (spots).
   On smoke bin16 this read-out matched the true angle to within 0.01° on every slice.
   And since every slice's grid is the same true grid, lattices snap together at
   whole-bin offsets, which gives sub-bin registration for free.
   An agent does not have to *know* about this to benefit: ICP or any point-set
   registration can lock onto the lattice. Real Visium/Visium HD grids are fixed to the
   slide, so they carry no such information. The fix is in Albis: apply each slice's
   offset to the molecules and bin in that frame, keeping the true bin centres by
   inverting the transform. Until then, check agent traces on these three tasks for
   grid-angle or lattice-snapping logic before reading their scores. The cell tasks
   (sa-301, sa-302) are unaffected.
2. **Gene-grid metrics are uninformative here** (above). `overall_mean` must not be used to rank
   methods on these tasks.
3. **Region metrics are not faithful** (above).
4. **Slices are thick slabs.** Albis partitions the whole sphere into `n_slices` slabs
   of thickness 2R/n_slices and projects each to 2D. Adjacent slices are therefore more
   different than real serial sections 5–10 µm apart.
5. **One batch strength per modality.** No sweep yet. A family such as sa-301 at σ = 0,
   0.5, 1.5, 3 would be cheap to add (`batch_sigma` override in `alignment_benchmark.py`).
6. **Mostly baseline composition.** Four of the five tasks use the baseline mix, whose
   domains are only weakly distinguishable by composition. That is realistic, but it
   makes biology-driven alignment hard, and it is why sa-301 is marked `starter`.
7. **Smoke tasks live alongside the real ones** (`sa-30N-...-smoke`). They are for
   harness checks only. Leave them out of `run_matrix.csv` and campaign totals.

## Status

- Smoke tasks: built, validated, scored with every scripted baseline.
- Full-scale data: SLURM array 35869721, one job per dataset. sa-301 at full scale was
  also generated locally for the diagnostics above.
- Agent dry run: SLURM array 35869897. Claude Sonnet 4.6 and Gemini 3.5 Flash, bare
  condition, one run per smoke task. Results: pending.
