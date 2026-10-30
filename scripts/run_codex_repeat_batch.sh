#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
# shellcheck source=scripts/lib_env.sh
source "${script_dir}/lib_env.sh"
p2b_source_local_env

task=""
count=5
model=""   # no default; --model is required
reasoning_effort="xhigh"
prompt_variant="default"
workspace_root="/tmp/p2b_spatial_codex_repeats"
delay_seconds=0
notify=0
label=""
blind_solve=1
dangerous_bypass=0
score_timeout_seconds=300
method_bootstrap_timeout_seconds=""
timeout_minutes=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --count) count="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --reasoning-effort) reasoning_effort="$2"; shift 2 ;;
    --prompt-variant) prompt_variant="$2"; shift 2 ;;
    --workspace-root) workspace_root="$2"; shift 2 ;;
    --delay-seconds) delay_seconds="$2"; shift 2 ;;
    --score-timeout-seconds) score_timeout_seconds="$2"; shift 2 ;;
    --method-bootstrap-timeout-seconds) method_bootstrap_timeout_seconds="$2"; shift 2 ;;
    --timeout-minutes) timeout_minutes="$2"; shift 2 ;;
    --notify) notify=1; shift ;;
    --label) label="$2"; shift 2 ;;
    --blind-solve) blind_solve=1; shift ;;
    --non-blind) blind_solve=0; shift ;;
    --dangerous-bypass) dangerous_bypass=1; shift ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ -z "$task" ]]; then
        task="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [ -z "$model" ]; then
  echo "--model is required; there is no default, because a stale one would launch the whole batch on a model that no longer exists." >&2
  exit 2
fi

if [[ -z "$task" ]]; then
  echo "Usage: bash scripts/run_codex_repeat_batch.sh TASK [--count N] [--prompt-variant VARIANT] [--workspace-root DIR] [--delay-seconds N] [--timeout-minutes N] [--score-timeout-seconds N] [--method-bootstrap-timeout-seconds N] [--notify] [--non-blind]" >&2
  exit 2
fi

if (( blind_solve && dangerous_bypass )); then
  echo "--dangerous-bypass cannot be combined with blind solve mode." >&2
  exit 2
fi

cd "${repo_root}"

if (( delay_seconds > 0 )); then
  sleep "${delay_seconds}"
fi

start_epoch="$(date +%s)"
task_name="$(basename "$(dirname "$task")")"
if [[ -z "$label" ]]; then
  label="${task_name} codex ${prompt_variant} x${count}"
fi

failures=0
for ((i = 1; i <= count; i++)); do
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] ${label} run ${i}/${count}"
  run_args=(
    "$task"
    --agent codex
    --model "$model"
    --reasoning-effort "$reasoning_effort"
    --workspace-root "$workspace_root"
    --prompt-variant "$prompt_variant"
    --score-timeout-seconds "$score_timeout_seconds"
    --force
  )
  if [[ -n "$method_bootstrap_timeout_seconds" ]]; then
    run_args+=(--method-bootstrap-timeout-seconds "$method_bootstrap_timeout_seconds")
  fi
  if [[ -n "$timeout_minutes" ]]; then
    run_args+=(--timeout-minutes "$timeout_minutes")
  fi
  if (( blind_solve )); then
    run_args+=(--blind-solve)
  elif (( dangerous_bypass )); then
    run_args+=(--dangerous-bypass)
  fi
  if ! UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}" uv run python scripts/run_local_coding_agent.py "${run_args[@]}"; then
    failures=$((failures + 1))
  fi
done

if (( notify )); then
  summary="$(p2b_score_summary_since "$start_epoch" 2>/dev/null || true)"
  status_word="OK"
  if (( failures > 0 )); then
    status_word="FAIL(${failures}/${count})"
  fi
  p2b_notify "paper2bench [${label}] ${status_word}" "elapsed_since: ${start_epoch}
failures: ${failures}/${count}

${summary}"
fi

if (( failures > 0 )); then
  exit 1
fi
