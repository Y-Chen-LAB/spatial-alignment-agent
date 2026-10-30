# Spatial Alignment Agent Runbook

## What Was Run

The current baseline results were produced with these commands from the repository root.

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --method identity --max-genes 25
```

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --method centroid_similarity --max-genes 25
```

```bash
uv run python scripts/run_dlpfc_rotation_robustness.py \
  --reference reference=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151673.h5ad \
  --moving moving=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151674.h5ad \
  --marker-genes /Users/yiqun/Downloads/paper2bench-data/DLPFC/DLPFC_gene_list.csv \
  --max-genes 25 \
  --angles 0 30 90 180 270 \
  --overlap-fractions 1.0 0.75 0.5
```

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-003-dlpfc-coarse-shuffled-pair/task.toml \
  --method centroid_similarity --max-genes 25
```

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-004-bca-downsampled-pair-align/task.toml \
  --method centroid_similarity --max-genes 25
```

```bash
uv run python scripts/run_bca_resolution_recovery.py \
  --raw slice_Rep1=/Users/yiqun/Downloads/paper2bench-data/BCA_raw/slice_Rep1.h5ad \
  --raw slice_Rep2=/Users/yiqun/Downloads/paper2bench-data/BCA_raw/slice_Rep2.h5ad \
  --downsampled slice_Rep1_DR=/Users/yiqun/Downloads/paper2bench-data/BCA_downsampled/slice_Rep1_DR.h5ad \
  --downsampled slice_Rep2_DR=/Users/yiqun/Downloads/paper2bench-data/BCA_downsampled/slice_Rep2_DR.h5ad \
  --aligned-downsampled STAligner_BCA_new_slices_Rep1=/Users/yiqun/Downloads/paper2bench-data/BCA_STAligner_DR/STAligner_BCA_new_slices_Rep1.h5ad \
  --aligned-downsampled STAligner_BCA_new_slices_Rep2=/Users/yiqun/Downloads/paper2bench-data/BCA_STAligner_DR/STAligner_BCA_new_slices_Rep2.h5ad \
  --label-key Cluster
```

```bash
uv run python scripts/summarize_alignment_runs.py
```

The suite summary is written to:

- `results/alignment_benchmark_summary.csv`
- `results/alignment_benchmark_summary.md`

## Agent-Runnable Interface

Prepare a task workspace:

```bash
uv run python scripts/prepare_agent_task.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --workspace-root agent_workspaces --force
```

The agent receives:

- `agent_workspaces/<task_id>/TASK.md`
- `agent_workspaces/<task_id>/data/*.h5ad`
- `agent_workspaces/<task_id>/submission/aligned_coordinates/`
- `agent_workspaces/<task_id>/score.sh`

The agent should write:

```text
agent_workspaces/<task_id>/submission/aligned_coordinates/<slice_id>.csv
```

Each CSV must have:

```text
obs_id,x,y
```

Score an agent submission:

```bash
agent_workspaces/<task_id>/score.sh
```

Equivalent explicit scorer:

```bash
uv run python scripts/score_agent_submission.py \
  benchmarks/spatial_alignment/tasks/<task_id>/task.toml \
  agent_workspaces/<task_id>/submission \
  --output agent_workspaces/<task_id>/scores.json
```

Plugin discovery metadata is in:

```text
benchmarks/spatial_alignment/agent_plugin.json
```

Runner plugin metadata is in:

```text
benchmarks/spatial_alignment/runner_plugin.json
```

## Task Versus Model

The benchmark task is just the task. Task files do not bind to a model or agent. They define data, prompt, output schema, and scoring.

Model and agent names appear only in runner commands. For example, `--model gpt-5-mini` means "run this model on the task"; it does not change the task definition or what the agent sees in `TASK.md`.

## Hosted LLM Codegen Runner

This runner calls the OpenAI Responses API, asks a model such as `gpt-5-mini` to write a Python solver, executes that solver locally in the prepared workspace, then runs the deterministic scorer.

Dry-run without calling the API:

```bash
uv run python scripts/run_openai_solver.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --model gpt-5-mini \
  --dry-run --force
```

Actual run:

```bash
OPENAI_API_KEY=... uv run python scripts/run_openai_solver.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --model gpt-5-mini \
  --reasoning-effort xhigh \
  --force
```

Artifacts are written under:

```text
agent_runs/openai_codegen/
```

Expected artifacts include `prompt.md`, `request_payload.json`, `response.json`, `solver_from_openai.py`, solver stdout/stderr, scorer stdout/stderr, `scores.json`, and `manifest.json`.

## Local Coding-Agent Runner

This runner prepares the same workspace, invokes a locally authenticated coding-agent CLI, then scores the resulting coordinate CSVs.

Codex dry-run:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex \
  --model gpt-5.3-codex \
  --dry-run --force
```

Codex run:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --force
```

Claude run:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent claude \
  --model claude-sonnet-4-6 \
  --force
```

Artifacts are written under:

```text
agent_runs/local_coding_agent/
```

For fully unattended local-agent runs, pass `--dangerous-bypass`. That mirrors the automated benchmark pattern in CompBioBench-style CLI runners, but it should only be used in disposable workspaces.

## Prompt Variants And Package-Backed Methods

There are now four active prompt modes for local CLI runners:

- `full-basic`: clean package-free control for the main rerun. The prepared workspace omits `bootstrap_method_venv.sh`, does not auto-use `.method-venv`, and does not expose package or method-family names in the prompt.
- `default`: legacy task prompt with the generic optional method helper but no paper-method context. Use only for comparability with older runs; it is not a clean no-package control.
- `full-package`: neutral package-backed prompt for the main rerun. The runner prebootstraps `paper-all`, exposes all supported package targets in a shuffled non-ranked order, points agents to public package documentation/API menus plus example API entry points, and asks agents to choose promising package(s) rather than satisfy a fixed package-count target.
- `full-package-paper-prior`: paper-informed package-backed prompt. It also prebootstraps `paper-all`, but adds high-level method priorities from local SABench source-data tables. Use this when you want to test whether giving the agent literature/benchmark priors improves package selection; keep it separate from neutral `full-package` results.

Legacy prompt modes from early experiments (`no-package`, `paper-method-list`,
`paper-method-package-first`, and `paper-method-package-balanced`) are no longer
active CLI choices. Historical run artifacts may still contain those labels.

Current recommendation for A/B comparisons:

- compare `full-basic` versus `full-package`
- compare neutral `full-package` versus paper-informed `full-package-paper-prior`
- keep model, workspace isolation, and task fixed when comparing prompts
- treat prior `default` versus package-hinted comparisons as confounded by package discovery/installability

Example A/B commands on `sa-001-dlpfc-serial-align`:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --workspace-root agent_workspaces_dlpfc_cli_no_package \
  --prompt-variant full-basic \
  --blind-solve \
  --force
```

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --workspace-root agent_workspaces_dlpfc_cli_neutral \
  --prompt-variant full-package \
  --blind-solve \
  --force
```

## Shared Method Venv

Prepared workspaces now include:

- `bootstrap_method_venv.sh`
- `spatial_method_envs.toml`
- `.method-venv/`
- `.method-cache/compat_h5ad/`

Purpose:

- `.method-venv` is a shared workspace-local Python 3.11 compatibility env; package methods use grouped or isolated `.method-venv*` envs.
- `spatial_method_envs.toml` is the declarative source of truth for method env names, Python versions, stack groups, aliases, package specs, and `paper-all` / `paper-common` target groups. The helper translates it into uv envs and copies it to `.method-cache/spatial_method_envs.toml`.
- `.method-cache/compat_h5ad/*.h5ad` are standard AnnData-compatible rewrites of the task slices for older third-party method stacks.

Typical usage:

```bash
./bootstrap_method_venv.sh paper-all
./bootstrap_method_venv.sh paper-common
./bootstrap_method_venv.sh ot
./bootstrap_method_venv.sh staligner
./bootstrap_method_venv.sh --recreate spateo
./bootstrap_method_venv.sh --recreate gpsa
./bootstrap_method_venv.sh --recreate cast
./bootstrap_method_venv.sh --recreate ot
```

Notes:

- If no method is passed, the helper only installs the shared compatibility stack and materializes compatible `.h5ad` files.
- Use `paper-all` to best-effort install all supported package targets into grouped or isolated method envs. PASTE and PASTE2 share `.method-venv-ot`; heavy/conflicting stacks such as Spateo, STAligner, CAST, SPACEL, scSLAT, SANTO, GPSA, STAIR, and STalign remain isolated.
- Use `paper-common` to best-effort install the smaller common paper-method package subset using the same grouped/isolated layout.
- For local CLI runners, package-backed prompt variants prebootstrap their configured target before the agent starts: `full-package` uses `paper-all`, and legacy package variants use `paper-common`.
- `full-package` prompts explicitly allow internet use for public package documentation and API lookup only. The Codex blind-solve wrapper denies private benchmark/repo/source paths through the outer macOS sandbox but does not add a network-deny rule, so Codex runs can still consult public docs when the host has network access.
- The `full-basic` and `no-package` variants disable method-helper exposure and prebootstrap. Use `full-basic`, not `default`, for clean no-package comparisons.
- Once `.method-venv/` exists, `agent_python.sh` automatically uses `.method-venv/bin/python` for the baseline stack. Method packages should be run with the Python executable listed in `.method-cache/METHOD_ENVS.md`.
- `uv` is the normal env builder. `--conda-fallback auto` only helps if `conda`, `mamba`, or `micromamba` is installed on `PATH`; it is a fallback for stubborn env creation, not the default execution path.
- The shared compatibility stack includes `scikit-learn` and `scikit-image`, which are commonly needed before method-specific installs.
- Switch method stacks with `--recreate`; do not assume heavy bio packages will coexist cleanly.
- For `paste-bio` and PASTE2, use the helper target `ot` instead of ad hoc installs. The merged OT env uses a compatible stack (`anndata==0.7.8`, `numpy<2`, `pot>=0.9.1`, `scikit-learn`) and writes compatible `.h5ad` files.
- Local CLI run dirs now preserve `prompt.md`, `command.json`, `raw_stdout.jsonl`, `raw_stderr.txt`, scorer output, and a `workspace_snapshot/` copy of the final workspace excluding `data/` and env/cache directories. That means repeated runs can safely reuse a workspace root without losing the per-run final code state.

## Operational Lessons

- For unattended Codex CLI benchmark runs, pass `--dangerous-bypass`. Without it, Codex may inherit a stricter sandbox and treat the prepared workspace as read-only, causing a false failure before any solver work happens.
- If `uv` fails inside a workspace due to local cache permission issues, set `UV_CACHE_DIR=/tmp/<cache-name>` before scoring or package installs.
- Some compiled task `.h5ad` files are not directly readable by older third-party AnnData stacks even though the repo-native readers can handle them. Prefer `.method-cache/compat_h5ad/*.h5ad` for package-backed methods.
- Package name order, import success order, and prebootstrap log order are not method rankings. The `full-package` prompt and `paper-all` helper use a shuffled supported-target order to reduce position bias, with PASTE/PASTE2 deliberately late in the smoke-test order.
- The `full-package` prompt intentionally has no required number of package attempts or families. Agents should pick method environments that fit the task, then invest effort in the strongest candidate rather than satisfying an artificial diversity quota.
- Current Codex batch wrappers default to `--model gpt-5.3-codex --reasoning-effort xhigh`; keep those fixed for package-doc smoke tests unless the experiment is explicitly about model or reasoning-effort effects.
- `PASTE` is runnable end-to-end in the shared method venv, but that does not mean it is strong on every task. If train score is weak, switch methods instead of anchoring on the first package-backed solver that runs.
- The package-doc prompt now includes discoverable package API examples for the compatibility H5AD cache, PASTE/PASTE2, Spateo, STalign, CAST, SPACEL, STAligner, scSLAT/SLAT, GPSA, STAIR, and SANTO. The examples are framed as starting points for package-backed submissions, not as a fixed probe checklist, package-count quota, or method ranking.

## Notifications

Use the repo's built-in notification path instead of custom mail watchers.

- Search the repo for `notify` and `SMTP` before improvising.
- Put `P2B_NOTIFY_*` values in `~/.config/paper2bench/env` or `~/.paper2bench.env`.
- The actual delivery path is [scripts/notify.py](/Users/yiqun/Desktop/chen-lab/genomics-agent/scripts/notify.py:1), and env loading lives in [scripts/lib_env.sh](/Users/yiqun/Desktop/chen-lab/genomics-agent/scripts/lib_env.sh:184).
- Prefer `--notify` on the runner or wrapper script when available.
- Only fall back to macOS notification tooling if the repo SMTP path is unavailable.

## CompBioBench-Style Registry

Export a CSV registry compatible with the CompBioBench task-list idea:

```bash
uv run python scripts/export_spatial_alignment_compbiobench_csv.py \
  --output benchmarks/spatial_alignment/benchmark.csv
```

The CSV contains `question_id`, `question`, `file_paths`, `difficulty`, `domain`, `task_file`, `workspace`, `prepare_command`, and `score_command`.

## All-Task Suite Wrapper

Run all five starter tasks through one runner. Model and agent modes default to dry-run; add `--execute` to actually call the model or local agent.

```bash
bash scripts/run_spatial_alignment_agent_suite.sh openai --model gpt-5-mini
```

```bash
OPENAI_API_KEY=... bash scripts/run_spatial_alignment_agent_suite.sh openai \
  --model gpt-5-mini --reasoning-effort xhigh --execute
```

```bash
bash scripts/run_spatial_alignment_agent_suite.sh codex \
  --model gpt-5.3-codex --reasoning-effort xhigh --execute
```

```bash
bash scripts/run_spatial_alignment_agent_suite.sh prepare
bash scripts/run_spatial_alignment_agent_suite.sh score
```

## Scoring Fields

All standard task submissions return:

- `scores.aggregate`: aggregate metric values across adjacent slice pairs.
- `scores.pairs`: per-pair metric values.
- `scores.n_slices`: number of scored slices.
- `scores.n_pairs`: number of adjacent scored pairs.
- `scores.n_marker_genes_requested`: number of marker genes requested for gene-grid scoring.

Current aggregate metric fields:

- `matching_accuracy`: SABench nearest-neighbor label agreement inside the overlapping bounding box.
- `overlap_ratio_region`: SABench alpha-shape overlap ratio across shared labels in the overlapping area.
- `gene_pcc`: SABench Pearson correlation on marker-gene grid averages after official overlap-by-grids filtering.
- `gene_cosine`: SABench cosine similarity on marker-gene grid averages after official overlap-by-grids filtering.
- `gene_ssim`: SABench SSIM on marker-gene grid averages after official overlap-by-grids filtering.
- `gene_mi`: SABench mutual information on marker-gene grid averages after official overlap-by-grids filtering.
- `overall_mean`: local convenience aggregate over the task profile's required metrics. The Nature paper's true overall score is a rank aggregation across methods/tasks, not a per-task raw mean.

Backward-compatible aliases remain in the JSON:

- `landmark_accuracy` = `matching_accuracy`
- `region_grid_iou` = `overlap_ratio_region`
- `bbox_overlap_fraction` is retained only as a diagnostic and is not a SABench paper metric.

Baseline runners additionally write `run_manifest.json` with:

- `task_id`
- `method`
- `runtime_seconds`
- `peak_memory_mb`
- input slice paths
- nested `scores`

The robustness runner writes:

- `robustness_summary.csv`: one row per rotation/crop condition.
- `summary.json`: `mean`, `std`, and `min` for each score field over perturbation conditions.

## Hidden Perturbation Grid (paper Fig 3 protocol)

For tasks whose `data/manifest.json` declares a `hidden_perturbation_grid_file`, the runner now re-executes the agent's solver across the full grid and aggregates `mean / min / max / std` per paper metric. This implements the paper's robustness evaluation (paper lines 478-487, Fig 3).

Manual invocation:

```bash
uv run python scripts/run_hidden_perturbation_grid.py \
  --task benchmarks/spatial_alignment/tasks/sa-002-dlpfc-robust-perturb/task.toml \
  --solver path/to/agent_solver.py \
  --reference-slice "reference=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151673.h5ad" \
  --moving-source-slice "moving=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151674.h5ad" \
  --output-dir agent_runs/hidden_grid/<run_name>
```

Per cell the harness regenerates the moving slice via `scripts/compile_perturbed_h5ad.py`, materializes an ephemeral workspace, runs the solver with `cwd=workspace` and `PAPER2BENCH_WORKSPACE=workspace`, then scores under the task's `metric_profile`. Solvers are expected to use relative paths (`Path("data/...")`).

`scripts/run_openai_solver.py` automatically triggers the harness once the public score completes for a robustness task; the run's `manifest.json` records a `hidden_grid` summary block. `scripts/run_local_coding_agent.py` does not auto-trigger because Codex, Claude, and Gemini lack a fixed solver-file convention; invoke the harness manually with `--solver` pointing at the relevant submitted script.
