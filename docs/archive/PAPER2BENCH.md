# paper2bench

`paper2bench` is a benchmark compiler for turning published biological analyses into executable, model-free benchmark suites.

The core idea is simple: a paper describes tasks, data, expected outputs and evaluation logic. `paper2bench` makes those pieces explicit, serializes them into agent-runnable benchmark packages, validates them with reference solutions and weak baselines, and emits runner metadata so any model or coding agent can be evaluated against the same tasks.

## North Star

`paper2bench` should be a benchmark compiler, not a benchmark runner. Its core job is to turn scientific artifacts into stable, model-free task packages. Model selection, agent scaffolding, prompting, tool access and hosted/local execution belong to a separate runner layer.

Product thesis:

```text
paper2bench turns biological papers and associated data/code into validated,
executable, model-free benchmark suites with reproducible scoring, hidden-test
support and pluggable runner adapters.
```

## Product Boundary

`paper2bench` is not the agent. It is the benchmark compiler and packaging layer.

The compiler produces:

- Task specs
- Data manifests
- LLM-readable prompts
- Output contracts
- Deterministic scorers
- Hidden-test harnesses
- Reference/baseline solutions
- Runner/plugin metadata

It should not bind tasks to a model. A task is model-free; model choice belongs to the runner.

Runner adapters are supported for compatibility and validation, but they are not the core product boundary.

## Inputs

Minimal input:

```yaml
paper:
  title: Benchmarking alignment methods for spatial transcriptomics data
  doi: 10.1038/s43588-026-00977-z
data:
  - id: data001
    path: /path/to/151673.h5ad
    type: anndata_h5ad
task:
  goal: Align adjacent spatial transcriptomics slices.
  output_contract: aligned_coordinate_csv_v1
scoring:
  profile: sabench_gene_landmark_efficiency_v1
```

Full input can include:

- Paper PDF, DOI or URL
- Dataset directories, URLs, checksums and licenses
- Code repository and notebooks
- Researcher-authored task spec
- Scoring rules or scoring code
- Expected output schema
- Hidden perturbation rules
- Reference solution
- Difficulty and resource constraints

## Outputs

Canonical task package:

```text
benchmarks/<domain>/<suite>/tasks/<task_id>/
  task.toml
  instruction.md
  data_manifest.json
  scoring_manifest.json
  hidden_tests.toml
  solution/
    solve.py
  tests/
    test_schema.py
```

Prepared agent workspace:

```text
agent_workspaces/<task_id>/
  TASK.md
  data/
  submission/
  score.sh
```

Runner artifacts:

```text
agent_runs/<runner>/<task_id>_<model>_<timestamp>/
  prompt.md
  command.json or request_payload.json
  raw_stdout.jsonl
  raw_stderr.txt
  scores.json
  manifest.json
```

## Compiler Stages

1. Ingest source materials.
2. Extract candidate task units from figures, tables, methods, code and datasets.
3. Normalize data into typed contracts.
4. Infer or accept output contracts.
5. Attach deterministic scoring profiles.
6. Generate LLM-readable prompts without leaking hidden tests.
7. Materialize task packages and agent workspaces.
8. Validate with schema checks, weak baselines and reference solutions.
9. Rank tasks by discriminativeness, scientific value and execution cost.
10. Emit runner/plugin metadata for hosted models, local coding agents and script baselines.

LLM calls can help at stages 2, 4, 6 and 9. Deterministic checks remain authoritative.

## Seven-Layer Architecture

| Layer | Purpose | Output |
|---|---|---|
| Source ingestion | Ingest paper, DOI, PDF, datasets, manifests, notebooks, repositories, tutorials and scoring text. | `source_bundle/`, provenance graph |
| Extraction and task synthesis | Propose candidate task units, data slices, output contracts and metrics. LLM-assisted but not trusted. | candidate task cards |
| Task IR | Canonical, typed, model-free representation of a task. | `task.toml`, `data_manifest.json`, `scoring_manifest.json` |
| Domain modules | Biological plugins that normalize data, generate tasks, validate outputs and score. | compiled task suite |
| Workspace materializer | Emits SWE/Terminal-Bench-style executable workspaces. | `TASK.md`, data links, `score.sh`, submission folder |
| Validation harness | Proves tasks are runnable, deterministic, non-leaky and biologically meaningful enough to accept. | validation report |
| Runner adapters | Execute scripts, hosted codegen, local CLI agents and future workflow agents. | run artifacts and scorecards |

The core split is:

```text
paper2bench compile:
  scientific artifacts -> benchmark artifacts

runner layer:
  benchmark artifacts + runner adapter -> scored run artifacts
```

The runner layer can live in this repo for validation, but task compilation must not depend on any particular model, prompt or agent.

## Core Contracts

The internal representation should be explicit, typed, versioned, hashable and model-free.

Conceptual model:

```text
BenchmarkSuite
  Task
    provenance
    biological_context
    inputs
    instructions
    output_contract
    scoring
    hidden_tests
    baselines
    resource_policy
    workspace_layout
    validation_status
```

Minimal task spec:

```toml
schema_version = "paper2bench.task.v0.1"
id = "sabench-real-data001-data002"
domain = "spatial_genomics"
modality = "spatial_transcriptomics"
task_type = "alignment"
input_contract = "anndata_h5ad_spatial_v1"
output_contract = "aligned_coordinate_csv_v1"
metric_profile = "sabench_gene_landmark_efficiency_v1"
```

Richer task spec:

```toml
schema_version = "paper2bench.task.v0.1"
suite_id = "sabench_spatial_alignment_v0"
task_id = "sabench.bca_xenium_visium.large_scale.pair_001"
module = "spatial_alignment"
task_type = "pairwise_spatial_alignment"
version = "0.1.0"

title = "Align BCA Xenium and Visium slices"
summary = "Given two spatial transcriptomics slices, produce aligned coordinates for the moving slice in the fixed slice coordinate frame."

[provenance]
paper_doi = "10.1038/s43588-026-00977-z"
paper_title = "Benchmarking alignment methods for spatial transcriptomics data"
source_repo = "Yunzhi-Yan/SABench"
compiler_version = "paper2bench-0.1.0"

[biological_context]
organism = "human"
tissue = "breast cancer"
modalities = ["spatial_transcriptomics"]
platforms = ["Xenium", "Visium"]
task_family = "cross_platform_large_scale"

[workspace]
task_md = "TASK.md"
input_dir = "data/"
submission_dir = "submission/"
score_entrypoint = "score.sh"

[output_contract]
kind = "file_collection"
root = "submission/aligned_coordinates"
required_pattern = "{slice_id}.csv"

[scoring]
profile = "spatial_alignment_v0.cross_platform"
aggregation = "weighted_mean"
score_min = 0.0
score_max = 1.0

[resources]
timeout_minutes = 60
max_memory_gb = 32
network = "disabled"
gpu = "optional"
```

Output contract:

```text
submission/aligned_coordinates/<slice_id>.csv
```

Required columns:

```text
obs_id,x,y
```

Runner contract:

```bash
prepare(task) -> workspace
run(workspace, runner_config) -> submission
score(task, submission) -> scores.json
```

The IR must deliberately avoid fields like `model`, `temperature`, `agent_name`, `provider` or runner prompt settings. Those belong in runner metadata and run manifests.

## Generic Input-To-Benchmark Spec

For a future researcher-facing interface, the ideal minimal spec is:

```yaml
id: my-paper/my-task
domain: spatial_genomics
data:
  - id: sample_a
    path: data/sample_a.h5ad
    role: reference_slice
  - id: sample_b
    path: data/sample_b.h5ad
    role: moving_slice
task:
  prompt: Align the two slices into a shared coordinate frame.
  expected_outputs:
    - type: aligned_coordinate_csv_v1
      slice_ids: [sample_a, sample_b]
scoring:
  metrics:
    - matching_accuracy
    - overlap_ratio_region
    - gene_pcc
  label_key: Region
  marker_genes: marker_genes.csv
hidden_tests:
  - type: rotation
    values: hidden
  - type: crop_overlap
    values: hidden
```

`paper2bench` compiles this into task files, prompts, workspaces and scorers.

## Source Bundle

The ingestion pipeline should create a source bundle with provenance, not silently invent tasks.

```text
source_bundle/
  paper/
    paper.pdf
    metadata.json
    extracted_sections.json
    citations.json
  data/
    raw_manifest.json
    normalized_manifest.json
    checksums.json
  code/
    repo_manifest.json
    notebooks_manifest.json
    environment_candidates.json
  provenance_graph.json
```

Source ingestion components:

- Paper ingester: extracts title, DOI, abstract, methods, data availability, code availability, figures, tables, supplements and metric definitions.
- Dataset ingester: normalizes local or remote data into assets with checksums, format, modality, platform, tissue, organism, split, visibility and license.
- Code/repo ingester: fingerprints notebooks, scripts, environment files, expected outputs and example scoring code.
- Task-card builder: converts paper/data/code evidence into human-reviewable candidate task cards with citation pointers back to source evidence.

## LLM Use Policy

Use LLMs as proposal generators and documentation assistants, not trusted sources of truth.

Good uses:

- Extract candidate task units from papers, tutorials, repos and manifests.
- Draft `instruction.md` and `TASK.md`.
- Suggest output schemas from examples.
- Map paper metrics to existing scorer plugins.
- Identify ambiguity, missing data or unclear gold standards.
- Generate weak-baseline code skeletons.
- Draft rubrics for human review.
- Summarize biological context and provenance.

Bad uses:

- Create hidden gold answers without deterministic derivation or human review.
- Decide final scores for numeric or data-analysis tasks.
- Rewrite scorer code without tests.
- Infer private labels not present in data.
- Resolve ambiguous biological claims as facts.
- Mix runner-specific assumptions into task definitions.

Invariant:

```text
A task can be compiled with LLM assistance, but once accepted it is scored by
deterministic code, calibrated rubrics or explicitly versioned judge protocols.
```

## Hidden Tests

Hidden tests are first-class IR objects, not ad hoc private folders.

| Hidden-test type | Example | Use |
|---|---|---|
| Held-out asset | private slice pair, labels or landmarks | prevent overfitting |
| Seeded perturbation | rotation, crop, noise, subsampling | robustness and generalization |
| Private scoring reference | landmarks, 3D coordinates, annotations | prevent answer leakage |

Example:

```toml
schema_version = "paper2bench.hidden_tests.v0.1"

[[hidden_tests]]
id = "rot90_low_overlap_seeded"
module = "spatial_alignment"
type = "seeded_perturbation"
visibility_to_agent = "description_only"
source_assets = ["moving_slice", "fixed_slice"]
generator = "paper2bench.modules.spatial_alignment.hidden_tests:make_overlap_rotation_case"
seed_policy = "sealed"
num_instances = 5

[hidden_tests.params]
rotation_degrees = [90, 180]
overlap_fraction = [0.25, 0.40]
coordinate_noise_sd = 0.0

[hidden_tests.scoring]
profile = "spatial_alignment_v0.robustness"
aggregate_into_public_score = true
post_run_disclosure = "aggregate_only"
```

Execution protocol:

```text
public development:
  runner receives public task workspace
  runner produces solver.py or solve.sh

hidden execution:
  fresh hidden workspace is created
  solver.py is copied in
  no conversation history or interactive agent access
  solver runs under fixed budget
  scorer evaluates hidden outputs

reporting:
  public metrics plus hidden aggregate metrics
  no hidden labels or hidden seed disclosure
```

Hidden data, gold references, private seeds and scorer internals should never be mounted into the agent workspace.

## Scoring Model

Use a three-tier hierarchy.

### Tier 1: deterministic scoring

Default for `paper2bench`.

Examples:

- Spatial alignment: matching accuracy, alpha-shape overlap ratio by region, gene PCC/cosine/SSIM/MI and MAE when ground truth coordinates exist. `bbox_overlap_fraction` can still be logged as a diagnostic, but it is not a SABench paper metric.
- Single-cell annotation: ARI/NMI, marker recovery and annotation F1.
- Differential expression: overlap with reference genes, rank correlation, directionality agreement and FDR calibration.
- Image segmentation: Dice, IoU, boundary F1 and object-level precision/recall.

### Tier 2: rubric scoring

Use when outputs are semi-structured but still objectively checkable, such as reports that must include QC, normalization choices, ranked tables and interpretation grounded in submitted artifacts.

### Tier 3: LLM judge scoring

Use only when interpretation quality matters and deterministic scoring is insufficient. Judge prompts must be versioned, blinded to runner/model identity, calibrated against human-labeled examples and stored separately from deterministic scores.

For the first serious SABench release, scoring should stay 100% deterministic.

## Domain Module Interface

Biological domains should be plugins. The core compiler knows about tasks, assets, contracts, manifests and validation. Modules know biology-specific formats, task synthesis and scoring.

```python
class BioModule:
    name: str
    version: str

    def inspect_source_bundle(self, source_bundle): ...
    def propose_task_units(self, evidence): ...
    def normalize_assets(self, task_card, data_manifest): ...
    def build_task_ir(self, task_card, assets): ...
    def materialize_workspace(self, task_ir, out_dir): ...
    def validate_submission(self, task_ir, submission_dir): ...
    def score(self, task_ir, submission_dir, private_refs=None): ...
    def make_weak_baseline(self, task_ir): ...
    def make_hidden_tests(self, task_ir, policy): ...
```

Good next modules after spatial alignment:

- Single-cell annotation
- Differential expression
- Spatial domain detection
- Image segmentation
- Metagenomics profiling
- Perturb-seq or CRISPR screens
- Multi-omics integration

Avoid starting with tasks whose success depends mainly on free-form biological interpretation unless a calibrated rubric or judge protocol exists.

## Biologist Authoring Interface

The authoring surface should be a task card, not raw TOML.

```yaml
schema_version: paper2bench.task_card.v0.1
title: Align two spatial transcriptomics slices
module: spatial_alignment
source:
  paper_doi: 10.1038/s43588-026-00977-z
  dataset: BCA_Xenium_Visium
question: >
  Given fixed and moving spatial transcriptomics slices, align the moving
  slice into the coordinate frame of the fixed slice.
inputs:
  fixed_slice: data/fixed.h5ad
  moving_slice: data/moving.h5ad
submission:
  folder: submission/aligned_coordinates
  files:
    - name: moving_slice.csv
      columns: [obs_id, x, y]
metrics:
  profile: spatial_alignment_v0.cross_platform
hidden_tests:
  - type: rotate
    degrees: [90, 180]
  - type: low_overlap
    overlap_fraction: [0.25, 0.40]
baseline:
  type: identity_or_procrustes
```

The compiler turns this into task IR, manifests, workspace, scorer and validation artifacts.

## Validation Gates

Task acceptance should be as important as task generation.

Acceptance states:

```text
draft -> executable -> validated -> calibrated -> sealed -> published
```

Required gates:

| Gate | Question | Check |
|---|---|---|
| Schema validity | Does the task conform to IR? | JSON/TOML schema validation |
| Data integrity | Are assets present and immutable? | checksums and manifest paths |
| License/provenance | Can assets be redistributed or referenced? | license and citation report |
| Workspace reproducibility | Can a clean environment run `score.sh`? | fresh workspace smoke test |
| Submission validation | Are malformed outputs caught? | negative fixtures |
| Weak baseline | Does a trivial solver run? | baseline score generated |
| Oracle/reference | Can a reference solution score high? | reference threshold |
| Null failure | Does empty/random output fail? | null score below threshold |
| Metric determinism | Are repeated scores identical? | repeated scorer runs |
| Metric sensitivity | Do perturbations change scores as expected? | mutation tests |
| Hidden leakage | Are gold files absent from agent workspace? | file scan and isolated mounts |
| Scorer tampering | Can agent modify scorer or harness? | separate scoring workspace/container |
| Resource boundedness | Does scoring finish under budget? | timeout and memory tests |
| Difficulty calibration | Does the task separate baselines? | baseline spread |

For SABench, require at least null/random, identity, centroid/geometric and known-method or reference baselines when practical.

## SABench As First Backend

The SABench backend currently demonstrates the pattern:

- Five executable starter tasks from local tutorial data.
- A 295-unit paper-scale catalog.
- Top-25 shortlist by scientific value and likely discrimination.
- Deterministic spatial-alignment scorer.
- Hosted OpenAI codegen runner.
- Local Codex/Claude coding-agent runner.
- CompBioBench-style CSV registry.

Key SABench outputs:

- `benchmarks/spatial_alignment/tasks/`
- `benchmarks/spatial_alignment/catalog/sabench_all_tasks.csv`
- `benchmarks/spatial_alignment/catalog/sabench_top25_tasks.csv`
- `benchmarks/spatial_alignment/SABENCH_EXPANSION_PLAN.md`

## Why This Generalizes

Most computational biology papers have the same latent structure:

- Data inputs
- Biological question
- Analysis procedure
- Expected artifact or claim
- Quantitative or qualitative validation
- Failure modes and assumptions

`paper2bench` turns that latent structure into explicit benchmark objects. Spatial alignment is just the first case; the same pattern should work for differential expression, cell-type annotation, variant interpretation, survival modeling, ligand-receptor analysis, spatial deconvolution and multi-omics integration.

## Design Rules

- Tasks are model-free.
- Submissions are files, not conversations.
- Scoring is deterministic whenever possible.
- Hidden tests are separated from public task prompts.
- Raw metric fields are preserved; aggregates are secondary.
- Every run emits traceable artifacts.
- Data provenance and licenses are first-class.
- Weak baselines are required before adding more tasks.
- A task is not accepted until it can be prepared, run and scored in a clean workspace.
- LLMs draft; validators decide.
- Releases are immutable.

## Near-Term Roadmap

1. Stabilize the spatial-alignment task IR and schema.
2. Add validation reports, task/scorer hashes and malformed-submission fixtures.
3. Add hidden solver-script execution for perturbation tasks.
4. Materialize the SABench top-25 datasets into a named `sabench_spatial_alignment_v0.1` release.
5. Add compiler-grade ingestion for DOI/PDF, repo/notebook and dataset manifests.
6. Add one non-alignment biology module, preferably single-cell annotation or differential expression.
7. Add a notebook/CLI/web task-authoring flow for biologists.
