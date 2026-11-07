# Benchmark design

Spatial Alignment Agent evaluates coding agents on 121 spatial transcriptomics tasks.
The suite combines tasks derived from SABench, MOp MERFISH, cross-technology mouse-brain
pairs, Albis simulations, and GEASO/Open-ST/whole-brain reconstruction datasets.
See the [task cards](TASK_CARDS.md) for individual specifications and [data sources](DATA_SOURCES.md)
for dataset provenance and preprocessing limitations.

## Task definition

Each task package defines its input slices, instructions, output contract and scoring
profile independently of the agent or model used to solve it. The main task families are
serial-section alignment, cross-technology alignment, rotation and crop robustness,
resolution recovery, and three-dimensional reconstruction.

Agents submit one CSV per slice under `submission/aligned_coordinates/`, with columns
`obs_id,x,y`. Observation identifiers must match the corresponding input slice. Reference
frames and any fixed-section requirements are specified in each task's instructions.

## Agent conditions

The runner supports two principal conditions:

| Condition | Prompt variant | Method environment preparation |
|---|---|---|
| Bare | `full-basic` | `never` |
| Equipped | `full-package-paper-prior` | `auto` |

The equipped condition provides published-method guidance and package environments.
Model, condition, repetition and task selections are recorded in campaign matrices under
`benchmarks/spatial_alignment/campaigns/`. These matrices describe particular experiments;
they need not include every task in the suite. The trace explorer provides the run-level
results associated with its displayed figures.

## Isolation and evaluation

Agent-visible data are sanitized copies. Reference annotations, ground-truth coordinates
and the host-side scorer are withheld during blind solving. Linux runs use Apptainer or
Singularity containment; macOS runs use the runner's Seatbelt boundary. A run made with
`--blind-mode none` does not establish the same isolation and should be identified separately.
See [container setup](../containers/README.md) and the [runbook](RUNBOOK.md).

Sanitization covers both metadata and index channels: coordinate-bearing observation names
are anonymized and perturbed inputs are shuffled to prevent recovery from row order.
Network access required by the agent CLIs remains a limitation; source-data retrieval must
be assessed from traces. The [sanitization audit](audits/SANITIZATION_AUDIT.md) and
[data-isolation audit](audits/PAPER_DATA_ISOLATION_AUDIT.md) document these checks.

Scores follow the task's metric profile and available reference information. They include
coordinate error, region agreement and gene-expression agreement; not every metric is
available for every task. Albis uses coordinate MAE as its primary score. Metric parity and
deliberate departures from source implementations are documented in the
[metric audit](audits/METRIC_PARITY_AUDIT.md) and [Albis scoring specification](ALBIS_SCORING.md).

## Reproducibility

Run records retain the prompt, command, model/CLI identifiers, submitted code, coordinates,
scoring output and available timing/token usage. Check run conditions with
`scripts/audit_run_conditions.py` and scorer inputs with `scripts/audit_scorer_inputs.py`
when reproducing an analysis. Prices in `config/model_prices.json` are historical inputs to
cost estimates; they do not establish current provider pricing.

See [RUNBOOK.md](RUNBOOK.md) for commands and the [Trace Explorer](https://ericzhy0815.github.io/agent-trace-explorer/)
for interactive inspection of runs and alignments.
