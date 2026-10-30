# paper2bench

`paper2bench` is a benchmark compiler for turning published biological analyses into executable, model-free benchmark suites. The current proof of concept is a spatial-genomics alignment suite derived from SABench tutorial data, with deterministic scoring and separate runner adapters.

## What Is Here

- General `paper2bench` design: `PAPER2BENCH.md`
- Task specs: `benchmarks/spatial_alignment/tasks/*/task.toml`
- Agent workspaces: `agent_workspaces/<task_id>/`
- Scorers and runners: `scripts/`
- Detailed runbook: `docs/RUNBOOK.md`
- Plugin metadata: `benchmarks/spatial_alignment/agent_plugin.json`
- Runner metadata: `benchmarks/spatial_alignment/runner_plugin.json`

SABench tutorial h5ad files are expected at `/Users/yiqun/Downloads/paper2bench-data/` (layout mirrors the Zenodo deposit at `10.5281/zenodo.18605715`; symlinking into an extracted `SABench.zip` tree is fine).

For the full SABench whole-mouse-brain family (`Data99-227`), the paper-faithful source is Allen ABC atlas `Zhuang-ABCA-1`, not the imputation mirrors. The repo helper script mirrors the official tutorial logic:

```bash
uv run python scripts/fetch_wmb_abc_atlas.py --plan-only
```

On a machine with outbound network access and the ABC client installed, run:

```bash
uv pip install "abc_atlas_access[notebooks] @ git+https://github.com/alleninstitute/abc_atlas_access.git"
uv run python scripts/fetch_wmb_abc_atlas.py \
  --download-base ~/Downloads/paper2bench-data/allen_abc_atlas \
  --matrix-kind raw
```

That fetches the `Zhuang-ABCA-1` metadata, CCF coordinates, and raw expression matrix and writes a local section-count summary to `results/wmb_abc_atlas_download_summary.json`.

## Task Versus Runner

Benchmark tasks are model-free. A `task.toml` defines only the biological task: data, instructions, output contract, metric profile, and scoring metadata. It should not name `gpt-5-mini`, Codex, Claude, or any other model.

Models and agents are runner choices. The `--model` and `--agent` flags below are only for the evaluation harness that runs against a task; they are not part of the benchmark task definition. The long-term product boundary is `paper2bench compile`; the runner scripts in this repo are adapters for validation and agent development.

## Setup

```bash
uv sync
uv run python -m compileall src scripts tests
uv run python -m unittest discover -s tests
```

`uv.toml` sets the cache inside the repo, so plain `uv run ...` should work without manually exporting `UV_CACHE_DIR`.
Hosted-model scripts silently load local env files from `.env`, `.env.local`, `~/.config/paper2bench/env`, or `~/.paper2bench.env` before execution.

## Compiler CLI

The reusable implementation starts with the `p2b` CLI. It inspects task specs, validates that task packages are model-free and executable, hashes task definitions, and compiles reviewed task cards into task packages.

Inspect one task as normalized paper2bench IR:

```bash
uv run p2b inspect benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml
```

Validate the current spatial-alignment suite:

```bash
uv run p2b validate-suite --output results/paper2bench_validation_report.json
```

Compile a reviewed task card into a runnable task package:

```bash
uv run p2b compile-card examples/task_cards/spatial_alignment_dlpfc_demo.toml \
  --out-dir /tmp/sa-card-demo-dlpfc-serial-align \
  --force --validate
```

The compiler accepts dependency-free TOML/JSON task cards today. YAML task cards are supported only when `pyyaml` is installed.

## Start Agent Benchmark Runs

Use this wrapper when you want to start producing benchmark results without remembering every lower-level script:

```bash
bash scripts/start_agent_benchmark.sh validate
bash scripts/start_agent_benchmark.sh baseline
bash scripts/start_agent_benchmark.sh openai-smoke --model gpt-5-mini --execute
bash scripts/start_agent_benchmark.sh collect
```

Recommended first pass:

```bash
bash scripts/start_agent_benchmark.sh openai-smoke --model gpt-5-mini
```

That dry-runs the hosted-model setup on `sa-001` and writes the prompt/request artifacts without calling the API. Add `--execute` to actually call the model.

Pass `--notify` to any mode to get an email (or macOS notification as fallback) when the run finishes, with elapsed time, exit status, and the latest scored aggregate. Email is sent via SMTP if the following env is configured (for Gmail, generate an App Password at <https://myaccount.google.com/apppasswords>):

```bash
export P2B_NOTIFY_TO=you@example.com
export P2B_NOTIFY_SMTP_HOST=smtp.gmail.com
export P2B_NOTIFY_SMTP_PORT=587
export P2B_NOTIFY_SMTP_USER=you@gmail.com
export P2B_NOTIFY_SMTP_PASS=your-16-char-app-password
```

Put those in `~/.config/paper2bench/env` or `~/.paper2bench.env` and `lib_env.sh` will pick them up automatically. Without SMTP config the runner falls back to a macOS `display notification`.

Run all five starter tasks:

```bash
bash scripts/start_agent_benchmark.sh openai-all --model gpt-5-mini --execute
```

Run a local coding agent instead:

```bash
bash scripts/start_agent_benchmark.sh codex-smoke --model gpt-5.3-codex --execute
```

If `mini-swe-agent` is installed and the `mini` executable is on `PATH`, run the same task with mini-SWE-agent:

```bash
bash scripts/start_agent_benchmark.sh mini-swe-smoke --model gpt-5-mini --dangerous-bypass --execute
```

Results are summarized to:

```text
results/alignment_benchmark_summary.csv
results/alignment_benchmark_summary.md
agent_runs/agent_run_summary.csv
agent_runs/agent_run_summary.md
```

Agent/model runners always score as a deterministic post-step after the agent process exits. The agent may run `./score.sh` for debugging, but the benchmark result is produced by the runner. If the API call, generated solver, local agent, or scorer fails, the run is marked `status = failed`, `assigned_zero = true`, and `scores.json` records zero aggregate metrics plus the failure stage.

Tasks score all observations. The solve-time workspace contains no ground-truth coordinates; the runner performs the deterministic post-run score.

## Run Baselines

Run a single baseline:

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --method centroid_similarity --max-genes 25
```

Summarize all current baseline outputs:

```bash
uv run python scripts/summarize_alignment_runs.py
```

Summary files are written to:

```text
results/alignment_benchmark_summary.csv
results/alignment_benchmark_summary.md
```

## Prepare And Score A Task

Prepare a SWE/Terminal-Bench-style workspace:

```bash
uv run python scripts/prepare_agent_task.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --workspace-root agent_workspaces --force
```

The agent writes coordinate CSVs to:

```text
agent_workspaces/<task_id>/submission/aligned_coordinates/<slice_id>.csv
```

Each CSV must contain:

```text
obs_id,x,y
```

Score a submission:

```bash
agent_workspaces/sa-001-dlpfc-serial-align/score.sh
```

## Run A Hosted Model

This path calls the OpenAI Responses API, asks a model such as `gpt-5-mini` to write a Python solver, executes that solver locally, and scores the resulting files.

Dry-run without API calls:

```bash
uv run python scripts/run_openai_solver.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --model gpt-5-mini --dry-run --force
```

Actual run:

```bash
uv run python scripts/run_openai_solver.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --model gpt-5-mini --reasoning-effort high --force
```

Artifacts are written under `agent_runs/openai_codegen/`.

## Run A Local Coding Agent

This path invokes a locally authenticated coding-agent CLI in the prepared workspace, then scores its output.

Codex:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex --model gpt-5.3-codex --reasoning-effort high --force
```

Claude:

```bash
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent claude --model claude-sonnet-4-6 --force
```

Artifacts are written under `agent_runs/local_coding_agent/`.

## Run All Tasks

The all-task wrapper defaults to dry-run for model/agent modes. Add `--execute` to actually call the model or local agent.

Hosted `gpt-5-mini` dry-run:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh openai --model gpt-5-mini
```

Hosted `gpt-5-mini` execution:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh openai \
  --model gpt-5-mini --reasoning-effort high --execute
```

Local Codex execution:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh codex \
  --model gpt-5.3-codex --reasoning-effort high --execute
```

Prepare all task workspaces:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh prepare
```

Score existing submissions for all tasks:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh score
```

Restrict to one task:

```bash
bash scripts/run_spatial_alignment_agent_suite.sh openai \
  --task benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml
```

## Scoring

Standard submissions produce `scores.json` with aggregate and per-pair fields. Current metrics include:

- `matching_accuracy`
- `overlap_ratio_region`
- `gene_pcc`
- `gene_cosine`
- `gene_ssim`
- `gene_mi`
- `overall_mean`

Notes:

- `matching_accuracy` and `overlap_ratio_region` follow the SABench paper / official repo semantics.
- `overall_mean` is a local within-task convenience summary; the paper's true overall score is a rank aggregation across methods and tasks.
- If a task declares a paper metric profile but is missing required labels or marker genes, scoring now fails instead of silently dropping those metrics.

The scorer is deterministic and lives in `scripts/score_agent_submission.py`.

## Current Limitations

The current agent-facing scorer evaluates fixed public task inputs. The `sa-002` robustness harness already supports hidden-style perturbations, but it is not yet wrapped as a full hidden-test workflow where the submitted solver is rerun on unseen generated data.

The current alignment method is a weak reference baseline for validating the benchmark harness. Real method adapters for PASTE2, STAligner, SPACEL, and related tools are the next benchmark-quality step.
