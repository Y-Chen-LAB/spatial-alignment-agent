#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_env.sh
source "${script_dir}/lib_env.sh"
p2b_source_local_env

usage() {
  cat <<'EOF'
Start here for paper2bench spatial-alignment agent benchmarks.

Usage:
  bash scripts/start_agent_benchmark.sh MODE [options]

Recommended sequence:
  bash scripts/start_agent_benchmark.sh validate
  bash scripts/start_agent_benchmark.sh baseline
  bash scripts/start_agent_benchmark.sh openai-smoke --execute
  bash scripts/start_agent_benchmark.sh collect

Modes:
  validate      Validate model-free task packages.
  prepare       Prepare all agent workspaces.
  baseline      Run deterministic weak baselines and summarize results.
  openai-smoke  Run hosted OpenAI codegen on the smallest starter task.
  openai-all    Run hosted OpenAI codegen on all five starter tasks.
  codex-smoke   Run local Codex CLI on the smallest starter task.
  codex-all     Run local Codex CLI on all five starter tasks.
  mini-swe-smoke Run local mini-SWE-agent CLI on the smallest starter task.
  mini-swe-all  Run local mini-SWE-agent CLI on all five starter tasks.
  claude-smoke  Run local Claude Code CLI on the smallest starter task.
  claude-all    Run local Claude Code CLI on all five starter tasks.
  gemini-smoke  Run local Gemini CLI on the smallest starter task.
  gemini-all    Run local Gemini CLI on all five starter tasks.
  azure-smoke   Run Azure OpenAI codegen on the smallest starter task.
  azure-all     Run Azure OpenAI codegen on all five starter tasks.
  score         Score existing submissions in agent_workspaces/.
  collect       Collect baseline and agent run summaries.

Options:
  --execute                  Actually call model/agent runners. Without this, agent modes dry-run.
  --model MODEL              Override runner model.
  --reasoning-effort LEVEL   Reasoning effort for OpenAI/Codex; default high.
  --prompt-variant VARIANT   Optional prompt enrichment variant for local CLI runners.
  --prebootstrap-method-venv MODE
                              For local CLI runners: auto, never, or always. Default auto.
  --method-bootstrap-timeout-seconds N
                              Timeout for prebuilding package-backed method venvs; 0 disables it.
  --dangerous-bypass         Pass unattended bypass flags to local coding agents.
  --workspace-root DIR       Workspace root; default agent_workspaces.
  --notify                   Send an email/notification when the mode finishes. See scripts/notify.py for env setup.
  --help                     Show this help.

Outputs:
  results/alignment_benchmark_summary.{csv,md}
  agent_runs/agent_run_summary.{csv,md}
  agent_runs/openai_codegen/*/scores.json
  agent_runs/local_coding_agent/*/scores.json
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mode="$1"
shift

starter_task="benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml"
all_tasks=(
  "benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-002-dlpfc-robust-perturb/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-003-dlpfc-coarse-shuffled-pair/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-004-bca-downsampled-pair-align/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-005-bca-resolution-recovery/task.toml"
)

model=""
reasoning_effort="xhigh"
prompt_variant=""
prebootstrap_method_venv=""
method_bootstrap_timeout_seconds=""
execute=0
dangerous_bypass=0
notify=0
workspace_root="agent_workspaces"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      execute=1
      shift
      ;;
    --model)
      model="$2"
      shift 2
      ;;
    --reasoning-effort)
      reasoning_effort="$2"
      shift 2
      ;;
    --prompt-variant)
      prompt_variant="$2"
      shift 2
      ;;
    --prebootstrap-method-venv)
      prebootstrap_method_venv="$2"
      shift 2
      ;;
    --method-bootstrap-timeout-seconds)
      method_bootstrap_timeout_seconds="$2"
      shift 2
      ;;
    --dangerous-bypass)
      dangerous_bypass=1
      shift
      ;;
    --notify)
      notify=1
      shift
      ;;
    --workspace-root)
      workspace_root="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

agent_args=("--workspace-root" "$workspace_root" "--reasoning-effort" "$reasoning_effort")
if [[ -n "$prompt_variant" ]]; then
  agent_args+=("--prompt-variant" "$prompt_variant")
fi
if [[ -n "$prebootstrap_method_venv" ]]; then
  agent_args+=("--prebootstrap-method-venv" "$prebootstrap_method_venv")
fi
if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
  agent_args+=("--method-bootstrap-timeout-seconds" "$method_bootstrap_timeout_seconds")
fi
if [[ -n "$model" ]]; then
  agent_args+=("--model" "$model")
fi
if [[ "$execute" -eq 1 ]]; then
  agent_args+=("--execute")
fi
if [[ "$dangerous_bypass" -eq 1 ]]; then
  agent_args+=("--dangerous-bypass")
fi

run_agent_suite() {
  local runner="$1"
  local task_mode="$2"
  local cmd=(bash scripts/run_spatial_alignment_agent_suite.sh "$runner" "${agent_args[@]}")
  if [[ "$task_mode" == "smoke" ]]; then
    cmd+=("--task" "$starter_task")
  fi
  "${cmd[@]}"
}

run_baseline() {
  uv run python scripts/run_task_baseline.py \
    benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
    --method centroid_similarity --max-genes 25
  uv run python scripts/run_task_baseline.py \
    benchmarks/spatial_alignment/tasks/sa-003-dlpfc-coarse-shuffled-pair/task.toml \
    --method centroid_similarity --max-genes 25
  uv run python scripts/run_task_baseline.py \
    benchmarks/spatial_alignment/tasks/sa-004-bca-downsampled-pair-align/task.toml \
    --method centroid_similarity --max-genes 25
  uv run python scripts/run_dlpfc_rotation_robustness.py \
    --reference "reference=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151673.h5ad" \
    --moving "moving=/Users/yiqun/Downloads/paper2bench-data/DLPFC/151674.h5ad" \
    --marker-genes "/Users/yiqun/Downloads/paper2bench-data/DLPFC/DLPFC_gene_list.csv" \
    --max-genes 25
  uv run python scripts/run_bca_resolution_recovery.py \
    --raw "slice_Rep1=/Users/yiqun/Downloads/paper2bench-data/BCA_raw/slice_Rep1.h5ad" \
    --raw "slice_Rep2=/Users/yiqun/Downloads/paper2bench-data/BCA_raw/slice_Rep2.h5ad" \
    --downsampled "slice_Rep1_DR=/Users/yiqun/Downloads/paper2bench-data/BCA_downsampled/slice_Rep1_DR.h5ad" \
    --downsampled "slice_Rep2_DR=/Users/yiqun/Downloads/paper2bench-data/BCA_downsampled/slice_Rep2_DR.h5ad" \
    --aligned-downsampled "STAligner_BCA_new_slices_Rep1=/Users/yiqun/Downloads/paper2bench-data/BCA_STAligner_DR/STAligner_BCA_new_slices_Rep1.h5ad" \
    --aligned-downsampled "STAligner_BCA_new_slices_Rep2=/Users/yiqun/Downloads/paper2bench-data/BCA_STAligner_DR/STAligner_BCA_new_slices_Rep2.h5ad"
  uv run python scripts/summarize_alignment_runs.py
}

dispatch_mode() {
  case "$mode" in
    validate)
      uv run p2b validate-suite --output results/paper2bench_validation_report.json
      ;;
    prepare)
      bash scripts/run_spatial_alignment_agent_suite.sh prepare --workspace-root "$workspace_root"
      ;;
    baseline)
      run_baseline
      ;;
    openai-smoke)      run_agent_suite openai smoke ;;
    openai-all)        run_agent_suite openai all ;;
    codex-smoke)       run_agent_suite codex smoke ;;
    codex-all)         run_agent_suite codex all ;;
    mini-swe-smoke)    run_agent_suite mini-swe smoke ;;
    mini-swe-all)      run_agent_suite mini-swe all ;;
    claude-smoke)      run_agent_suite claude smoke ;;
    claude-all)        run_agent_suite claude all ;;
    gemini-smoke)      run_agent_suite gemini smoke ;;
    gemini-all)        run_agent_suite gemini all ;;
    azure-smoke)       run_agent_suite azure-openai smoke ;;
    azure-all)         run_agent_suite azure-openai all ;;
    score)
      bash scripts/run_spatial_alignment_agent_suite.sh score --workspace-root "$workspace_root"
      ;;
    collect)
      uv run python scripts/summarize_alignment_runs.py
      uv run python scripts/summarize_agent_runs.py
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      usage >&2
      exit 2
      ;;
  esac
}

if [[ "$notify" -eq 1 ]]; then
  p2b_run_with_notify "$mode" dispatch_mode
else
  dispatch_mode
fi
