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
batch_size=5
delay_seconds=0
notify=0
basic_workspace_root="/tmp/p2b_spatial_codex_basic_repeats"
package_workspace_root="/tmp/p2b_spatial_codex_pkg_repeats"
label=""
custom_tasks=()
blind_solve=1
dangerous_bypass=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count) count="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --delay-seconds) delay_seconds="$2"; shift 2 ;;
    --notify) notify=1; shift ;;
    --basic-workspace-root) basic_workspace_root="$2"; shift 2 ;;
    --package-workspace-root) package_workspace_root="$2"; shift 2 ;;
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

if (( delay_seconds > 0 )); then
  sleep "${delay_seconds}"
fi

if (( batch_size <= 0 )); then
  echo "--batch-size must be positive." >&2
  exit 2
fi

if (( blind_solve && dangerous_bypass )); then
  echo "--dangerous-bypass cannot be combined with blind solve mode." >&2
  exit 2
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
  label="all sa-* codex batched x${count}"
fi

total_tasks="${#tasks[@]}"
batch_count=$(( (total_tasks + batch_size - 1) / batch_size ))
pipeline_failures=0

run_variant() {
  local prompt_variant="$1"
  local workspace_root="$2"
  local variant_label="$3"
  local start_idx end_idx batch_idx i

  for ((start_idx = 0; start_idx < total_tasks; start_idx += batch_size)); do
    end_idx=$((start_idx + batch_size))
    if (( end_idx > total_tasks )); then
      end_idx="${total_tasks}"
    fi
    batch_idx=$((start_idx / batch_size + 1))

    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] starting ${variant_label} batch ${batch_idx}/${batch_count} tasks $((start_idx + 1))-${end_idx}/${total_tasks}"

    cmd=(
      bash scripts/run_codex_repeat_matrix.sh
      --count "$count"
      --model "$model"
      --reasoning-effort "$reasoning_effort"
      --prompt-variant "$prompt_variant"
      --workspace-root "$workspace_root"
      --label "${variant_label} batch ${batch_idx}/${batch_count} x${count}"
    )
    if (( blind_solve )); then
      cmd+=(--blind-solve)
    else
      cmd+=(--non-blind)
    fi
    if (( dangerous_bypass )); then
      cmd+=(--dangerous-bypass)
    fi

    for ((i = start_idx; i < end_idx; i++)); do
      cmd+=(--task "${tasks[i]}")
    done

    if ! "${cmd[@]}"; then
      pipeline_failures=$((pipeline_failures + 1))
    fi
  done
}

run_variant "full-basic" "${basic_workspace_root}" "all sa-* codex full-basic"
run_variant "full-package" "${package_workspace_root}" "all sa-* codex full-package"

if (( notify )); then
  summary="$(p2b_score_summary_since "$start_epoch" 2>/dev/null || true)"
  status_word="OK"
  if (( pipeline_failures > 0 )); then
    status_word="FAIL(batches=${pipeline_failures})"
  fi
  p2b_notify "paper2bench [${label}] ${status_word}" "tasks: ${total_tasks}
repeats_per_task: ${count}
batch_size: ${batch_size}
prompt_variants: 2
failed_batches: ${pipeline_failures}

${summary}"
fi

if (( pipeline_failures > 0 )); then
  exit 1
fi
