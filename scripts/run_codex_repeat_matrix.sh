#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=scripts/lib_env.sh
source "${script_dir}/lib_env.sh"
p2b_source_local_env

count=5
model=""   # no default; --model is required
reasoning_effort="xhigh"
prompt_variant="default"
workspace_root="/tmp/p2b_spatial_codex_repeats"
delay_seconds=0
notify=0
label=""
custom_tasks=()
blind_solve=1
dangerous_bypass=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count) count="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --prompt-variant) prompt_variant="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --delay-seconds) delay_seconds="$2"; shift 2 ;;
    --notify) notify=1; shift ;;
    --label) label="$2"; shift 2 ;;
    --task) custom_tasks+=("$2"); shift 2 ;;
    --blind-solve) blind_solve=1; shift ;;
    --non-blind) blind_solve=0; shift ;;
    --dangerous-bypass) dangerous_bypass=1; shift ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      custom_tasks+=("$1")
      shift
      ;;
  esac
done

if [ -z "$model" ]; then
  echo "--model is required; there is no default, because a stale one would launch the whole run on a model that no longer exists." >&2
  exit 2
fi

cd "${repo_root}"

if (( blind_solve && dangerous_bypass )); then
  echo "--dangerous-bypass cannot be combined with blind solve mode." >&2
  exit 2
fi

if (( delay_seconds > 0 )); then
  sleep "${delay_seconds}"
fi

tasks=()
if [[ ${#custom_tasks[@]} -gt 0 ]]; then
  tasks=("${custom_tasks[@]}")
else
  while IFS= read -r task; do
    tasks+=("$task")
  done < <(find benchmarks/spatial_alignment/tasks -mindepth 2 -maxdepth 2 -name task.toml | sort)
fi

if [[ ${#tasks[@]} -eq 0 ]]; then
  echo "No tasks found." >&2
  exit 1
fi

start_epoch="$(date +%s)"
if [[ -z "$label" ]]; then
  label="all sa-* codex ${prompt_variant} x${count}"
fi

task_failures=0
for task in "${tasks[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting ${task}"
  batch_args=(
    "$task"
    --count "$count"
    --model "$model"
    --reasoning-effort "$reasoning_effort"
    --prompt-variant "$prompt_variant"
    --workspace-root "$workspace_root"
  )
  if (( blind_solve )); then
    batch_args+=(--blind-solve)
  else
    batch_args+=(--non-blind)
  fi
  if (( dangerous_bypass )); then
    batch_args+=(--dangerous-bypass)
  fi
  if ! bash scripts/run_codex_repeat_batch.sh "${batch_args[@]}"; then
    task_failures=$((task_failures + 1))
  fi
done

if (( notify )); then
  summary="$(p2b_score_summary_since "$start_epoch" 2>/dev/null || true)"
  status_word="OK"
  if (( task_failures > 0 )); then
    status_word="FAIL(tasks=${task_failures}/${#tasks[@]})"
  fi
  p2b_notify "paper2bench [${label}] ${status_word}" "tasks: ${#tasks[@]}
repeats_per_task: ${count}
failed_tasks: ${task_failures}

${summary}"
fi

if (( task_failures > 0 )); then
  exit 1
fi
