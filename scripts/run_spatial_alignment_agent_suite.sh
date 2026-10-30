#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib_env.sh
source "${script_dir}/lib_env.sh"
p2b_source_local_env

usage() {
  cat <<'EOF'
Run spatial-alignment benchmark tasks through one runner.
Task specs are model-free; --model and --agent options only configure the runner.

Usage:
  bash scripts/run_spatial_alignment_agent_suite.sh MODE [options]

Modes:
  openai       Hosted OpenAI codegen runner; default model gpt-5-mini.
  azure-openai Azure OpenAI codegen runner; default deployment gpt-6-astra.
  codex        Local Codex CLI coding-agent runner; default model gpt-5.3-codex.
  mini-swe     Local mini-SWE-agent CLI runner; default model gpt-5-mini.
  claude       Local Claude Code CLI coding-agent runner; default model claude-sonnet-4-6.
  gemini       Local Gemini CLI coding-agent runner; default model gemini-3.5-flash.
  prepare      Prepare all task workspaces only.
  score        Score existing submissions for all tasks.

Options:
  --model MODEL              Override the default model.
  --reasoning-effort LEVEL   Reasoning effort for OpenAI/Codex runners; default high.
  --prompt-variant VARIANT   Optional prompt enrichment variant for local CLI runners.
  --prebootstrap-method-venv MODE
                              For local CLI runners: auto, never, or always. Default auto.
  --method-bootstrap-timeout-seconds N
                              Timeout for prebuilding package-backed method venvs; 0 disables it.
  --execute                  Actually call the model/agent. Without this, model runners dry-run.
  --dangerous-bypass         Pass unattended bypass flags to local coding agents.
  --task TASK_TOML           Restrict to one task. Can be repeated.
  --workspace-root DIR       Workspace root; default agent_workspaces.
  --help                     Show this help.

Examples:
  bash scripts/run_spatial_alignment_agent_suite.sh openai --model gpt-5-mini
  bash scripts/run_spatial_alignment_agent_suite.sh openai --model gpt-5-mini --execute
  bash scripts/run_spatial_alignment_agent_suite.sh codex --model gpt-5.3-codex --execute
  bash scripts/run_spatial_alignment_agent_suite.sh mini-swe --model gpt-5-mini --dangerous-bypass --execute
  bash scripts/run_spatial_alignment_agent_suite.sh prepare
  bash scripts/run_spatial_alignment_agent_suite.sh score
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mode="$1"
shift

all_tasks=(
  "benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-002-dlpfc-robust-perturb/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-003-dlpfc-coarse-shuffled-pair/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-004-bca-downsampled-pair-align/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-005-bca-resolution-recovery/task.toml"
)

custom_tasks=()
model=""
reasoning_effort="xhigh"
prompt_variant=""
prebootstrap_method_venv=""
method_bootstrap_timeout_seconds=""
execute=0
dangerous_bypass=0
workspace_root="agent_workspaces"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --execute)
      execute=1
      shift
      ;;
    --dangerous-bypass)
      dangerous_bypass=1
      shift
      ;;
    --task)
      custom_tasks+=("$2")
      shift 2
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

if [[ ${#custom_tasks[@]} -gt 0 ]]; then
  tasks=("${custom_tasks[@]}")
else
  tasks=("${all_tasks[@]}")
fi

task_id_from_toml() {
  local task_id
  task_id="$(sed -nE 's/^id[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/p' "$1" | head -n 1)"
  if [[ -z "$task_id" ]]; then
    echo "Could not find task id in $1" >&2
    return 1
  fi
  printf '%s\n' "$task_id"
}

run_prepare() {
  local task="$1"
  uv run python scripts/prepare_agent_task.py "$task" --workspace-root "$workspace_root" --force
}

run_score() {
  local task="$1"
  local task_id
  task_id="$(task_id_from_toml "$task")"
  uv run python scripts/score_agent_submission.py \
    "$task" "$workspace_root/$task_id/submission" \
    --output "$workspace_root/$task_id/scores.json"
}

case "$mode" in
  openai)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> openai ${model}: ${task}"
      cmd=(uv run python scripts/run_openai_solver.py "$task" --model "$model" --reasoning-effort "$reasoning_effort" --workspace-root "$workspace_root" --force)
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      "${cmd[@]}"
    done
    ;;
  azure-openai)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> azure-openai ${model}: ${task}"
      cmd=(uv run python scripts/run_openai_solver.py "$task" --provider azure --model "$model" --reasoning-effort "$reasoning_effort" --workspace-root "$workspace_root" --force)
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      "${cmd[@]}"
    done
    ;;
  codex)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> codex ${model}: ${task}"
      cmd=(uv run python scripts/run_local_coding_agent.py "$task" --agent codex --model "$model" --reasoning-effort "$reasoning_effort" --workspace-root "$workspace_root" --force)
      if [[ -n "$prompt_variant" ]]; then
        cmd+=(--prompt-variant "$prompt_variant")
      fi
      if [[ -n "$prebootstrap_method_venv" ]]; then
        cmd+=(--prebootstrap-method-venv "$prebootstrap_method_venv")
      fi
      if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
        cmd+=(--method-bootstrap-timeout-seconds "$method_bootstrap_timeout_seconds")
      fi
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      if [[ "$dangerous_bypass" -eq 1 ]]; then
        cmd+=(--dangerous-bypass)
      fi
      "${cmd[@]}"
    done
    ;;
  mini-swe)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> mini-swe ${model}: ${task}"
      cmd=(uv run python scripts/run_local_coding_agent.py "$task" --agent mini-swe --model "$model" --workspace-root "$workspace_root" --force)
      if [[ -n "$prompt_variant" ]]; then
        cmd+=(--prompt-variant "$prompt_variant")
      fi
      if [[ -n "$prebootstrap_method_venv" ]]; then
        cmd+=(--prebootstrap-method-venv "$prebootstrap_method_venv")
      fi
      if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
        cmd+=(--method-bootstrap-timeout-seconds "$method_bootstrap_timeout_seconds")
      fi
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      if [[ "$dangerous_bypass" -eq 1 ]]; then
        cmd+=(--dangerous-bypass)
      fi
      "${cmd[@]}"
    done
    ;;
  claude)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> claude ${model}: ${task}"
      cmd=(uv run python scripts/run_local_coding_agent.py "$task" --agent claude --model "$model" --workspace-root "$workspace_root" --force)
      if [[ -n "$prompt_variant" ]]; then
        cmd+=(--prompt-variant "$prompt_variant")
      fi
      if [[ -n "$prebootstrap_method_venv" ]]; then
        cmd+=(--prebootstrap-method-venv "$prebootstrap_method_venv")
      fi
      if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
        cmd+=(--method-bootstrap-timeout-seconds "$method_bootstrap_timeout_seconds")
      fi
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      if [[ "$dangerous_bypass" -eq 1 ]]; then
        cmd+=(--dangerous-bypass)
      fi
      "${cmd[@]}"
    done
    ;;
  gemini)
    [ -n "$model" ] || { echo "--model is required for this agent; no default." >&2; exit 2; }
    for task in "${tasks[@]}"; do
      echo "==> gemini ${model}: ${task}"
      cmd=(uv run python scripts/run_local_coding_agent.py "$task" --agent gemini --model "$model" --workspace-root "$workspace_root" --force)
      if [[ -n "$prompt_variant" ]]; then
        cmd+=(--prompt-variant "$prompt_variant")
      fi
      if [[ -n "$prebootstrap_method_venv" ]]; then
        cmd+=(--prebootstrap-method-venv "$prebootstrap_method_venv")
      fi
      if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
        cmd+=(--method-bootstrap-timeout-seconds "$method_bootstrap_timeout_seconds")
      fi
      if [[ "$execute" -eq 0 ]]; then
        cmd+=(--dry-run)
      fi
      if [[ "$dangerous_bypass" -eq 1 ]]; then
        cmd+=(--dangerous-bypass)
      fi
      "${cmd[@]}"
    done
    ;;
  prepare)
    for task in "${tasks[@]}"; do
      echo "==> prepare: ${task}"
      run_prepare "$task"
    done
    ;;
  score)
    for task in "${tasks[@]}"; do
      echo "==> score: ${task}"
      run_score "$task"
    done
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    usage >&2
    exit 2
    ;;
esac
