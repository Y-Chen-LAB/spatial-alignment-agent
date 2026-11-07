# Running the benchmark

Run commands from the repository root. Configure the input data as described in
[DATA_ACCESS.md](DATA_ACCESS.md), then install the Python environment:

```bash
uv sync
export PYTHONPATH=src
export HDF5_USE_FILE_LOCKING=FALSE
uv run python -m unittest discover tests
```

## Run a coding agent

Install and authenticate the chosen agent CLI; see [CREDENTIALS.md](CREDENTIALS.md).
Set `AGENT_MODEL` to a model available through that CLI. On Linux, configure a pinned
container image as described in [containers/README.md](../containers/README.md).

```bash
export AGENT_MODEL=your-model-name
uv run python scripts/run_local_coding_agent.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent claude --model "$AGENT_MODEL" \
  --prompt-variant full-basic --prebootstrap-method-venv never \
  --blind-solve --blind-mode auto --timeout-minutes 90
```

For the equipped condition, use `--prompt-variant full-package-paper-prior` and
`--prebootstrap-method-venv auto`. The runner prepares the workspace, invokes the agent and
scores its submission. `--dry-run` prepares the prompt and command without invoking the
agent. Use `--help` to inspect runtime, container, output-directory and resource options.

## Prepare a workspace separately

```bash
uv run python scripts/prepare_agent_task.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --workspace-root agent_workspaces --blind-solve
```

The workspace contains `TASK.md`, sanitized input slices, and a submission directory.
Workspace preparation alone does not establish a container or filesystem boundary;
use the runner's blind mode for an isolated agent execution.

Submissions contain one CSV per slice:

```
submission/aligned_coordinates/<slice_id>.csv
```

Each CSV has columns `obs_id,x,y`; task instructions specify the coordinate frame.

## Score a submission

Scoring takes place on the host, with access to reference data:

```bash
uv run python scripts/score_agent_submission.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  /path/to/submission --output /path/to/scores.json
```

Use the score split specified by the task. Explicit evaluation-split scoring is controlled
by the scorer's evaluation guard; see `--help` before overriding a split.

## Baselines and campaigns

A simple scripted baseline can verify the input/output pipeline:

```bash
uv run python scripts/run_task_baseline.py \
  benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --method identity --max-genes 25
```

Published-method implementations and analysis utilities are indexed in
[scripts/README.md](../scripts/README.md). Campaign definitions are in
`benchmarks/spatial_alignment/campaigns/`; cluster configuration and submission helpers
are under `scripts/slurm/`. Inspect the selected campaign and site configuration before
submitting jobs. The [benchmark design](BENCHMARK.md) explains conditions, isolation and scoring.
