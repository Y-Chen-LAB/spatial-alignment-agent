# Spatial alignment case

92 agent tasks rebuilt from the SABench benchmark (Yan et al., *Nat. Comput. Sci.*
s43588-026-00977). They reference all 240 slices the paper used and score all 295 alignment
pairs it ran, 260 real and 35 simulated. The scorer reproduces the authors' own metric code:
see `METRIC_PARITY_AUDIT.md` for the comparison and for the four places we deliberately
differ.

## Before you run anything

```bash
cd ~/Downloads/genomics-agent
export HDF5_USE_FILE_LOCKING=FALSE        # required on network filesystems

./.venv/bin/p2b validate-suite --output results/validation.json
./.venv/bin/python scripts/build_tasks.py --check-coverage        # 240 of 240 slices
./.venv/bin/python scripts/build_tasks.py --check-pair-coverage   # 295 of 295 paper pairs
```

Slice data lives outside the repo and is reached through two symlinks, `agent_runs` and
`benchmarks/spatial_alignment/compiled_data`. If the data moves, re-point those two links;
nothing else refers to absolute data paths.

## The run matrix

`run_matrix.csv` holds one row per planned run. Regenerate it with:

```bash
./.venv/bin/python scripts/build_run_matrix.py
```

| Cell | Agent | Model | Tasks | Condition | Runs |
| --- | --- | --- | --- | --- | --- |
| main | claude (Claude Code CLI) | `claude-sonnet-4-6` | subset 45 | equipped, bare | 135 + 135 |
| main | codex (codex CLI on Azure) | `gpt-5.5` | subset 45 | equipped, bare | 135 + 135 |
| main | gemini (Gemini CLI) | `gemini-3.5-flash` | subset 45 | equipped, bare | 135 + 135 |
| gpt6_select | codex (codex CLI on Azure) | `gpt-6-astra` | 9 paper-numbered + sa-018 | equipped, bare | 30 + 30 |

One factorial: condition x model x 3 reps, 870 runs, every model running as a coding agent
through its native CLI inside the blind container. The GPT models reach Azure deployments
through the codex CLI (`docs/CREDENTIALS.md`, "codex on Azure"); there is no one-shot cell.
The harness question (the same model through mini-swe) is deferred, not folded in.

**Condition** collapses the old prompt-variant and environment axes into two levels, because
in 531 historical runs the prompt axis explained 0.6% of score variance. `equipped` is
`--prompt-variant full-package-paper-prior --prebootstrap-method-venv auto`; `bare` is
`--prompt-variant full-basic --prebootstrap-method-venv never`.

**Design subset** is the 45 tasks in `design_subset.txt`, spanning every metric profile,
family and difficulty and including all nine tasks with published per-method numbers, so
every arm can be placed against the paper somewhere.

**Checking the conditions were real.** `scripts/audit_run_conditions.py` reads three
independent witnesses per scored run (the prompt handed to the CLI, the workspace snapshot,
the bootstrap log) and requires all three to agree with the row's condition; a mismatch
voids the run. Run it before reading any equipped-versus-bare number.

**Neither cell IDs nor row order may encode position on a perturbed slice.** Two
index-level channels were found on 2026-09-20. STARmap names cells by position (`69x4486`),
and on a perturbed slice those names still hold the pre-perturbation frame, shared across
slices: gpt-5.5 parsed them and submitted that frame in six runs out of six (sa-018 only).
Separately, source files store cells sorted by position, and that order survives a rotation,
so the axis along which the row index increases is the original x-axis (sa-017, sa-018, the
sa-042 rotation sweep and sa-043 crop sweep, rank correlations 0.87 to 1.0). The perturbation
compiler now anonymizes coordinate IDs and shuffles rows with a seeded permutation on every
perturbed output; `scripts/shuffle_perturbed_rows.py` applied the same to files compiled
before that. All sa-018 runs made before the fix are void (`agent_runs/cluster_void/`).
`scripts/audit_scorer_inputs.py` checks all of this against every slice, derived from what the
scorer reads (label column, ground-truth frames, marker list) plus both index channels, and
exits non-zero on a finding; run it after recompiling any data. Column-level sanitisation is an
allowlist (`X`, `var`, `layers`, `obs/_index`, `obsm/spatial`); what that lets through was
enumerated across all 359 slices and is gene metadata and raw counts only. The remaining channel is the network: the container has
it because the CLIs need it, TASK.md names the dataset, and public copies of the source data
exist; the rule against fetching them is audited from traces, not enforced.

## Shards, for splitting the work

| Shard | Tasks | Runs | Agent-hours | Paper-numbered tasks | Contents |
| --- | --- | --- | --- | --- | --- |
| 1 | 22 | 420 | ~80 | 4 | half the design subset; every condition x model |
| 2 | 23 | 444 | ~85 | 5 | the other half; every condition x model |
| 3 | 47 | 0 | — | 0 | breadth tail; no rows unless the factorial is widened to all 92 tasks |

Shards 1 and 2 are balanced on measured median wall-clock, not slice count: across 526 timed
historical runs the rank correlation between slice count and runtime is only 0.33, because
agent time is reasoning turns rather than data volume. Either half alone is a usable
experiment. Task lists are in `shard_1_tasks.txt`, `shard_2_tasks.txt`, `shard_3_tasks.txt`.

## Cluster

```bash
# always dry-run first: prints the row count, the first rows, and the sbatch line
scripts/slurm/submit_alignment.sh --shard 1 --dry-run

scripts/slurm/submit_alignment.sh --shard 1          # 132 runs
scripts/slurm/submit_alignment.sh --shard 2
scripts/slurm/submit_alignment.sh --shard 3
scripts/slurm/submit_alignment.sh --cell D_condition --shard 1   # narrow to one cell
```

Site settings (partition, memory, walltime, concurrency, paths) are in
`scripts/slurm/site_config.sh` and are all environment-overridable:

```bash
P2B_PARTITION=gpu P2B_MAX_CONCURRENT=32 scripts/slurm/submit_alignment.sh --shard 1
```

The array script maps `$SLURM_ARRAY_TASK_ID` to a line of a filtered row file, so indices
stay dense when you submit part of the matrix.

### Containment

Blind solving on Linux runs inside `singularity exec --containall --cleanenv --no-home`, with
only the workspace, a private state directory, and the credential files bound. The repository,
the unsanitised slices, the split manifests and the scorer are absent. On macOS blind solving
uses the Seatbelt read-deny profile instead; `--blind-mode auto` picks the right one and
**fails rather than downgrading**, because a run labelled blind that has no boundary is worse
than one that does not start. See `containers/README.md` for the image.

```bash
export P2B_CONTAINER_MODULE=singularity/3.7.1        # if your site uses modules
export P2B_SINGULARITY_IMAGE=/path/to/python-3.11-slim.sif
```

If no runtime is available and you want to run anyway, `--blind-mode none` works but those
runs are workspace-isolated, not blind, and must not be pooled with blind runs in analysis.

**Codex cannot run on the cluster.** `--blind-solve` for Codex needs macOS `sandbox-exec`
for its read-deny profile, and the runner exits rather than silently downgrading isolation.
The submit script filters Codex rows out, and the array script refuses them if one slips
through. Run cell B on the Mac:

```bash
./.venv/bin/python scripts/run_codex_concurrent_queue.py --help
```

Claude's blind-solve has no Seatbelt dependency and is portable, but note it is therefore
weaker than Codex's on macOS: it relies on the copied workspace and the credential-stripped
environment, without the outer read-deny profile.

### mini-swe

Eleven flags separate a mini invocation that runs unattended from one that hangs or silently
truncates, and all of them are now set. The two that actually stop a batch run are
`MSWEA_CONFIGURED=1`, without which mini opens an interactive first-run prompt and dies with
`OSError: [Errno 22]` the moment stdin is not a terminal, and `environment.timeout`, whose
upstream default of 30 seconds kills any command that loads one of our slices. The upstream
cost limit of $3.00 is also disabled with `-l 0`, since it truncates a run that then looks
complete. Because `-c` replaces mini's default config list rather than adding to it, the
bundled `mini.yaml` is named explicitly.

mini reaches both providers through litellm, so it needs pay-per-token API keys rather than
the subscription tokens the native CLIs use; the Claude Code setup token is gated to that
client and returns masked 429s through litellm. Put the keys in
`$GEN2BENCH_CONFIG_DIR/anthropic-api-key` and `gemini-api-key`.

mini prints almost nothing to stdout, so the run record is the trajectory file. It is written
inside the workspace, which is the only writable path under containment, then copied to
`mini_trajectory.json` in the run directory and summarised into the manifest's `usage` block.

## Single run, locally

```bash
./.venv/bin/python scripts/run_local_coding_agent.py \
    sa-001-dlpfc-serial-align \
    --agent claude --model claude-sonnet-4-6 \
    --prompt-variant full-package-paper-prior \
    --prebootstrap-method-venv auto \
    --blind-solve --dangerous-bypass \
    --timeout-minutes 90

# see the prompt and argv without invoking the agent
./.venv/bin/python scripts/run_local_coding_agent.py sa-001-dlpfc-serial-align --dry-run
```

## Scoring

Runs score themselves. To score a submission by hand, or to re-score an old one after a
scorer change:

```bash
P2B_ALLOW_EVAL_SCORING=1 ./.venv/bin/python scripts/score_agent_submission.py \
    benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
    path/to/submission --score-split all

./.venv/bin/python scripts/metric_parity/rescore_all.py   # every retained submission
```

Re-scoring costs no agent time, because submissions are coordinate CSVs. It only works where
the task's slices have not been rebuilt; `results/dead_runs.csv` lists the runs that must be
re-run instead, with the reason.

## Comparing to the paper

`paper_pair_coverage.csv` maps each of the 295 published pairs to the task that scores it.
Per-method numbers exist for nine tasks only; `METRIC_PARITY_AUDIT.md` lists which, and
which source-data file to read them from.

## What each run records

`manifest.json` carries, beyond the task and agent settings:

| Field | Why |
| --- | --- |
| `cli_version` | a CLI upgrade mid-campaign changes behaviour and is otherwise invisible in the results |
| `elapsed_seconds` | duration, rather than leaving analysis to reconstruct it from the directory-name timestamp |
| `blind_mode`, `container_image` | which boundary actually applied, so a non-blind run cannot be mistaken for a blind one |
| `leakage_status`, `leakage_high_findings` | the audit verdict |
| `usage` | tokens and cost, below |

The `usage` block:

```json
{"model": "...", "input_tokens": 0, "cached_input_tokens": 0, "cache_creation_tokens": 0,
 "output_tokens": 0, "reasoning_output_tokens": 0, "api_calls": 0,
 "reported_cost_usd": null, "estimated_cost_usd": null, "cost_usd": null, "source": "..."}
```

Cache reads and cache writes are counted separately because Anthropic bills a write at about
1.25x input and a read at 10% of it, a twelvefold gap that makes a single "cached" number
uncostable. OpenAI does not bill writes, so `cache_creation_tokens` stays 0 for codex, which
is correct rather than missing.

Cost is split three ways. `reported_cost_usd` is what the vendor itself said and wins when
present, since it accounts for tiers we cannot see. `estimated_cost_usd` is tokens times
`config/model_prices.json`. `cost_usd` prefers the reported figure. An arm priced by estimate
should not be compared to the cent against one priced by report.

`config/model_prices.json` holds list prices checked on 2026-09-15 and is meant to be
re-checked: rates move, and a stale table produces confident wrong numbers. A model with no
entry still gets full token counts, with `estimated_cost_usd` left null rather than guessed.

History was backfilled from traces already on disk, with no agent time:

```bash
./.venv/bin/python scripts/backfill_run_usage.py --dry-run
./.venv/bin/python scripts/backfill_run_usage.py
```

That recovered 452 of 530 runs: 466,547,737 input tokens of which 445,620,736 were cached,
7,910,451 output tokens, and **$564.76** at current rates, about $1.25 per run. The 95.5%
cache hit rate is doing most of the work; the same tokens uncached would be roughly $1,800.
The 78 runs that died before their first turn completed have no usage to recover and get no
block rather than a fabricated one. `gpt-5.3-codex` is retired and has no list price, so it is
costed at the `gpt-5.5` rate; treat those dollars as an approximation.
