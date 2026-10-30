#!/usr/bin/env bash

p2b_notify() {
  # p2b_notify "subject" "body"
  local script_dir repo_root
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"
  python3 "${repo_root}/scripts/notify.py" --subject "$1" --body "$2" 2>&1 || true
}

p2b_score_summary_since() {
  # p2b_score_summary_since START_EPOCH_SECONDS
  python3 - "$1" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

start = float(sys.argv[1])
metric_order = [
    "overall_mean",
    "mae",
    "bbox_overlap_fraction",
    "landmark_accuracy",
    "region_grid_iou",
    "matching_accuracy",
    "overlap_ratio_region",
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def task_id_from(manifest: dict, score: dict, run_dir: Path) -> str:
    value = score.get("task_id")
    if value:
        return str(value)
    workspace = str(manifest.get("workspace", ""))
    if workspace:
        return Path(workspace).name
    task = str(manifest.get("task", ""))
    if task:
        return Path(task).parent.name
    return run_dir.name


def aggregate_from(score: dict) -> dict:
    scores = score.get("scores", score)
    if isinstance(scores, dict) and isinstance(scores.get("aggregate"), dict):
        return scores["aggregate"]
    return {}


def status_from(manifest: dict, score: dict, aggregate: dict) -> str:
    if score.get("assigned_zero") or score.get("status") == "failed" or manifest.get("status") == "failed":
        return "failed"
    if manifest.get("dry_run"):
        return "dry_run"
    if aggregate:
        return str(manifest.get("status") or score.get("status") or "scored")
    return str(manifest.get("status") or score.get("status") or "missing_score")


runs: list[tuple[float, Path, dict, dict]] = []
seen: set[Path] = set()

for manifest_path in Path("agent_runs").glob("*/*/manifest.json"):
    run_dir = manifest_path.parent
    try:
        mtime = max(manifest_path.stat().st_mtime, (run_dir / "scores.json").stat().st_mtime if (run_dir / "scores.json").exists() else 0)
    except OSError:
        continue
    if mtime < start:
        continue
    manifest = read_json(manifest_path)
    score = read_json(run_dir / "scores.json") if (run_dir / "scores.json").exists() else {}
    runs.append((mtime, run_dir, manifest, score))
    seen.add((run_dir / "scores.json").resolve())

for score_path in list(Path("results").glob("*/*/scores.json")) + list(Path("results").glob("*/*/*/scores.json")):
    try:
        mtime = score_path.stat().st_mtime
    except OSError:
        continue
    if mtime < start or score_path.resolve() in seen:
        continue
    runs.append((mtime, score_path.parent, {}, read_json(score_path)))

runs.sort(key=lambda item: item[0])
if not runs:
    sys.exit(0)

lines: list[str] = ["run summaries:"]
for _, run_dir, manifest, score in runs[-12:]:
    aggregate = aggregate_from(score)
    task_id = task_id_from(manifest, score, run_dir)
    runner = manifest.get("runner") or manifest.get("agent") or run_dir.parent.name
    model = manifest.get("model")
    status = status_from(manifest, score, aggregate)
    assigned_zero = bool(score.get("assigned_zero"))
    failure_stage = score.get("failure_stage") or manifest.get("failure_stage")
    lines.append(f"- {task_id}")
    lines.append(f"  runner: {runner}" + (f" model: {model}" if model else ""))
    lines.append(f"  status: {status}" + (" assigned_zero: true" if assigned_zero else ""))
    if failure_stage:
        lines.append(f"  failure_stage: {failure_stage}")
    if aggregate:
        metrics = [f"{key}={aggregate[key]}" for key in metric_order if key in aggregate]
        if metrics:
            lines.append("  metrics: " + ", ".join(metrics))
    lines.append(f"  scores: {run_dir / 'scores.json'}")

print("\n".join(lines))
PY
}

p2b_run_with_notify() {
  # p2b_run_with_notify LABEL CMD [ARGS...]
  local label="$1"; shift
  local start end rc status_word body latest_scores recent_scores score_failure
  start=$(date +%s)
  "$@"
  rc=$?
  end=$(date +%s)
  local dur=$((end - start))
  recent_scores="$(p2b_score_summary_since "$start" 2>/dev/null || true)"
  score_failure=0
  if [[ "${recent_scores}" == *"status: failed"* || "${recent_scores}" == *"assigned_zero: true"* ]]; then
    score_failure=1
  fi
  if [[ $rc -eq 0 && "$score_failure" -eq 0 ]]; then
    status_word="OK"
  elif [[ $rc -eq 0 && "$score_failure" -eq 1 ]]; then
    status_word="FAIL(scores)"
  else
    status_word="FAIL(rc=$rc)"
  fi

  body="paper2bench [${label}] ${status_word}
elapsed: ${dur}s
cmd: $*
"
  if [[ -n "${recent_scores}" ]]; then
    body+="
${recent_scores}"
  fi

  # Pick the latest scored run: prefer agent_runs/ (agent output), fall back to results/ (baselines).
  # Within each tree, sort descending so the timestamped run dir comes first. Skip assigned_zero failures.
  _p2b_latest_scored() {
    local tree="$1"
    ls -d ${tree}/*/*/ 2>/dev/null | sort -r | while read -r dir; do
      local f="${dir}scores.json"
      [[ -f "$f" ]] || continue
      if python3 -c "import json,sys; s=json.load(open('$f')); sys.exit(0 if not s.get('assigned_zero') else 1)" 2>/dev/null; then
        echo "$f"; return
      fi
    done
  }
  latest_scores="$(_p2b_latest_scored agent_runs)"
  [[ -z "$latest_scores" ]] && latest_scores="$(_p2b_latest_scored results)"
  if [[ -z "${recent_scores}" && -n "${latest_scores}" ]]; then
    body+="
latest scores: ${latest_scores}
$(python3 -c "import json; s=json.load(open('${latest_scores}')); a=s.get('scores',{}).get('aggregate') or s.get('aggregate',{}); print('\\n'.join(f'  {k}: {v}' for k,v in a.items()))" 2>/dev/null || echo '  (unable to parse)')"
  fi

  p2b_notify "paper2bench [${label}] ${status_word} (${dur}s)" "${body}"
  return $rc
}

p2b_source_local_env() {
  local script_dir repo_root env_file
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${script_dir}/.." && pwd)"

  local env_files=(
    "${repo_root}/.env"
    "${repo_root}/.env.local"
    "${HOME}/.config/paper2bench/env"
    "${HOME}/.paper2bench.env"
  )

  if [[ -n "${PAPER2BENCH_ENV_FILE:-}" ]]; then
    env_files=("${PAPER2BENCH_ENV_FILE}" "${env_files[@]}")
  fi

  for env_file in "${env_files[@]}"; do
    if [[ -f "${env_file}" ]]; then
      local nounset_was_enabled=0
      if [[ "$-" == *u* ]]; then
        nounset_was_enabled=1
        set +u
      fi
      set -a
      # shellcheck source=/dev/null
      source "${env_file}"
      set +a
      if [[ "${nounset_was_enabled}" -eq 1 ]]; then
        set -u
      fi
    fi
  done
}
