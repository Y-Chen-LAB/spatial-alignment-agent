#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

count=1
model="${1:?--model is required; there is no default}"
reasoning_effort="xhigh"
score_timeout_seconds=600
method_bootstrap_timeout_seconds=""
workspace_root="/tmp/p2b_spatial_full_prompt_paper_smoke"
notify=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count) count="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --score-timeout-seconds) score_timeout_seconds="$2"; shift 2 ;;
    --method-bootstrap-timeout-seconds) method_bootstrap_timeout_seconds="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --notify) notify=1; shift ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      echo "Unexpected argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "${repo_root}"

tasks=(
  "benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml"
  "benchmarks/spatial_alignment/tasks/sa-009-mhpr-merfish-serial-align/task.toml"
)

variants=(
  "full-basic"
  "full-package"
)

failures=0
for variant in "${variants[@]}"; do
  for task in "${tasks[@]}"; do
    task_id="$(basename "$(dirname "${task}")")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ${variant} ${task_id} x${count}"
    args=(
      "${task}"
      --count "${count}"
      --model "${model}"
      --reasoning-effort "${reasoning_effort}"
      --prompt-variant "${variant}"
      --workspace-root "${workspace_root}/${variant}"
      --score-timeout-seconds "${score_timeout_seconds}"
      --blind-solve
      --label "paper-clean ${variant} ${task_id} x${count}"
    )
    if [[ -n "${method_bootstrap_timeout_seconds}" ]]; then
      args+=(--method-bootstrap-timeout-seconds "${method_bootstrap_timeout_seconds}")
    fi
    if (( notify )); then
      args+=(--notify)
    fi
    if ! bash scripts/run_codex_repeat_batch.sh "${args[@]}"; then
      failures=$((failures + 1))
    fi
  done
done

if (( failures > 0 )); then
  exit 1
fi
