# Trimmed designs for the next stage

Written 2026-09-15. Evidence: `results/alignment_run_scores.csv` (531 real scored
alignment runs, 40 tasks, 5 prompt variants) and
`results/perturbation_variance/` (reconstructed from the two summary reports, see
`PERTURBATION_VARIANCE_ANALYSIS.md` for the caveat). Both benchmarks have the same
problem: almost all the budget goes into axes that do not move the outcome, and replicate
noise swamps the ones that do.

## What the numbers say about the current alignment suite

Variance decomposition on the 433 balanced runs (three prompt variants, three reps, 40
tasks, one model):

| Term | omega^2 | p | Note |
| --- | --- | --- | --- |
| task | 0.428 | 1e-30 | task means span 0.181 to 1.110 |
| prompt variant | 0.006 | 0.041 | means 0.414, 0.401, 0.379 |
| task x prompt | 0 (negative) | 0.50 | no evidence prompts reorder tasks |
| residual (replicate) | 0.51 | | pooled within-arm SD 0.172 |

Three consequences:

1. **Replicate noise (SD 0.172) is 13 times the prompt effect (spread 0.035).** With
   three reps the standard error per arm is 0.099. The suite as run cannot resolve the
   prompt question it was built to answer. Detecting a 0.035 difference at 80% power needs
   about 190 runs per arm; 0.10 needs 24; 0.15 needs 11.
2. **Tasks are not interchangeable but families largely are.** 62% of task-mean variance
   is within family, yet the 12 MOp sample subsets have a mean pairwise correlation of
   0.48 in their prompt-effect profiles, the highest of any group, against 0.08 for all
   task pairs. They are measuring the same thing 12 times.
3. **Fewer tasks cost less than expected.** A random 10-task subset reproduces the
   full-suite prompt means to within 0.055; 15 tasks to within 0.041. A purposively chosen
   14-task core (below) gets to 0.023, which is well inside the replicate noise floor.

Two tasks are broken as measuring instruments rather than informative: sa-006 and sa-007
have within-arm SD of 0.38 and 0.36, larger than the entire range of task means for most
families. They should be diagnosed before being kept.

## Proposal A: alignment, 14-task core

One task per dataset family, every metric profile represented, and all four tasks that have
published per-method numbers. Deviation from the full-suite prompt means: 0.023.

| Task | Why it is in | Profile | Within-arm SD |
| --- | --- | --- | --- |
| sa-001 DLPFC serial | paper Fig. 2a, full per-method table | light | 0.14 |
| sa-009 MHPR serial | paper Fig. 2b, full per-method table | light | 0.12 |
| sa-019 Flysta3D 3D | paper Fig. 5f, per-slice MAE | MAE | 0.05 |
| sa-027 Open-ST MLN full | paper Fig. 5g, per-slice MAE | MAE | 0.03 |
| sa-002 DLPFC robustness | the only perturbation-robustness task | robustness | 0.19 |
| sa-011 SCC P2 | NGS family, Fig. 4b group | gene-only | 0.14 |
| sa-006 MBSA / sa-007 MBSP | Fig. 4b group, but see noise warning | gene-only | 0.38 / 0.36 |
| sa-017 BaristaSeq / sa-018 STARmap | imaging family, Fig. 4b group | light | 0.17 / 0.10 |
| sa-020 MOp mouse 1 | large-scale serial, replaces all 12 subsets | light | 0.22 |
| sa-023 WMB 129-section | largest task in the suite, CCF ground truth | MAE | 0.22 |
| sa-010 AMBC cross-platform | the cross-platform mode | gene-only | 0.15 |
| sa-014 BCA Xenium | large-cell-count geometry task | geometry+gene | 0.20 |

Dropped: the 12 MOp subsets (sa-101 to sa-112), sa-021 (mouse 2 companion), sa-028 and
sa-029 (WMB starters superseded by sa-023), sa-003, sa-004, sa-005, sa-008, sa-012,
sa-013, sa-015, sa-016, sa-022, sa-025, sa-026. That is 26 of 40 tasks, and with them
about 65% of the run cost. They stay in the repo as an extended set, they just leave the
default design.

## Proposal B: alignment, what to spend the freed budget on

Reallocating the 26 dropped tasks' budget, at three reps that is roughly 234 runs:

| Spend | Runs | Buys |
| --- | --- | --- |
| Raise the core to 8 reps per arm | 14 tasks x 3 prompts x 5 extra reps = 210 | detects a 0.10 difference instead of 0.20; makes the prompt question answerable |
| Or: 2 model generations x 14 tasks x 1 prompt x 5 reps | 140 | the model-generation axis, which the perturbation data says is the largest lever, and which alignment has never tested (every run is gpt-5.3-codex) |
| Or: the 70 robustness sweep pairs at 2 reps | 140 | the richest published reference in the paper (MOESM8, 35 levels x 24 methods) |

My recommendation is the second and third together, dropping the prompt axis to a single
variant. Prompt explains 0.6% of variance; model generation is untested and, by the
perturbation evidence, worth about 6 times more. Concretely:

**Alignment design, next stage: 2 models x 14 tasks x 5 reps = 140 runs**, one prompt
variant (`full-package-paper-prior`, the one with method access), plus **11 paper methods
x 14 tasks = 154 method-baseline runs** for the reference scale. Total about 294 runs
versus 360 in the current balanced design, and it answers a question the current design
cannot.

## Proposal C: perturbation, trimmed

From `PERTURBATION_VARIANCE_ANALYSIS.md`: cell line omega^2 0.285, harness 0.058, model x
harness 0.049, model (same generation) 0.006, environment 0, profile 0, replicate noise 54
to 70% with pooled SD 0.085.

Cut:

- **Profile.** Two levels, no detectable effect on delta PCC (p = 0.50), half the runs.
  Keep it as a documented covariate on a small behavioural sub-study (does the agent fit a
  model at all, does it fetch external resources, cost and runtime), where it clearly does
  act.
- **Same-generation model siblings.** Claude 4.6 vs Codex 5.5 vs Gemini 3.6 span 0.027 and
  are not separable at n=3. Keep two, not three.
- **Cold vs warm as a full factor.** Main effect not significant; its interaction is
  explained by one known behaviour (cold Codex does not model without a profile). Replace
  with a 16-run confirmation arm.

Keep and strengthen:

- **All four cell lines**, blocked, since this is the largest term and it is a blocking
  variable rather than a treatment.
- **Harness crossed with model**, because the interaction (omega^2 0.049) is bigger than
  either main effect alone.
- **Model generation**, the largest effect observed anywhere in the benchmark (0.175 on
  Jurkat, GPT-6 Astra vs the older three), currently measured on one cell line only.

**Perturbation design, next stage: 2 generations (one older, one newer, same provider) x 2
harnesses (native, mini) x 4 cell lines x 8 reps = 128 runs.** That resolves generation,
harness, and their interaction at the 0.05 level for differences of about 0.06 delta PCC,
against the current 72-run design that resolves nothing below 0.20. Add a 16-run cold
confirmation arm and a profile behavioural sub-study of 24 runs if wanted; still under the
current total.

## Two prerequisites before either design runs

1. **Fix the MAE normalization check** on the perturbation side. The notebook documents a
   run at 25.9x its peer median caused by a CPM vs CP10K mismatch that passed the
   detector, and lists the round-number reconstruction as not yet implemented. Until that
   lands, MAE variance is partly measuring submission conventions.
2. **Diagnose sa-006 and sa-007** on the alignment side (within-arm SD 0.38 and 0.36,
   double any other task). Either they have a genuine bimodal solution space worth keeping
   as a finding, or the scorer is unstable on two-slice Visium; either way it should be
   known before they anchor the NGS half of the core.
