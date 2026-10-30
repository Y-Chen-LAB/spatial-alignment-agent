#!/usr/bin/env bash
# One task, start to finish, with the checks that matter afterwards.
#
# Run this before committing a shard to a queue. It exercises the whole chain on the cheapest
# task that still carries both metric families: workspace preparation, ground-truth
# sanitisation, the isolation boundary, the agent, scoring, the leakage audit and usage
# accounting. sa-018 is three small slices and takes about ten minutes.
set -euo pipefail

TASK="${TASK:-sa-018-starmap-mmpc-serial-align}"
AGENT="${AGENT:-claude}"
MODEL="${MODEL:?set MODEL, for example claude-opus-5 or gpt-5.5; there is no default}"
RUNS="${RUNS:-agent_runs/e2e}"
WS="${WS:-${TMPDIR:-/tmp}/p2b-e2e-ws}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${P2B_PYTHON:-$REPO/.venv/bin/python}"
export HDF5_USE_FILE_LOCKING=FALSE

echo "== 1. the boundary =="
"$PY" scripts/verify_containment.py

echo
echo "== 2. credentials =="
"$PY" scripts/check_access.py | grep -E "^\[|$AGENT" || true

echo
echo "== 3. run $TASK with $AGENT/$MODEL =="
rm -rf "$WS"
"$PY" scripts/run_local_coding_agent.py \
    "benchmarks/spatial_alignment/tasks/$TASK/task.toml" \
    --agent "$AGENT" --model "$MODEL" \
    --prompt-variant full-package-paper-prior \
    --prebootstrap-method-venv never \
    --blind-solve \
    --workspace-root "$WS" --runs-root "$RUNS" \
    --timeout-minutes "${TIMEOUT_MINUTES:-30}"

RUN_DIR="$(ls -dt "$RUNS"/${TASK}_* | head -1)"

echo
echo "== 4. what came back =="
"$PY" - "$RUN_DIR" <<'PYCHECK'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
manifest = json.loads((run / "manifest.json").read_text())
usage = manifest.get("usage") or {}
ok = True

def check(label, value, passed):
    global ok
    ok = ok and passed
    print(f"  [{'ok  ' if passed else 'FAIL'}] {label:26s} {value}")

# The runner writes "scored" on success and "failed" otherwise; there is no "completed".
check("status", manifest.get("status"), manifest.get("status") == "scored")
check("isolation", manifest.get("blind_mode"), manifest.get("blind_mode") in {"seatbelt", "container"})
check("leakage", manifest.get("leakage_status"), manifest.get("leakage_status") == "clean")
check("cli version recorded", manifest.get("cli_version"), bool(manifest.get("cli_version")))
check("elapsed recorded", manifest.get("elapsed_seconds"), bool(manifest.get("elapsed_seconds")))
check("tokens recorded", usage.get("input_tokens"), bool(usage.get("input_tokens")))
check("cost recorded", usage.get("cost_usd"), usage.get("cost_usd") is not None)

scores_path = run / "scores.json"
if scores_path.is_file():
    agg = json.loads(scores_path.read_text())["scores"]["aggregate"]
    # A submission that parses but aligns nothing still scores, so zero on every metric means
    # the agent produced geometry, not that the harness worked.
    nonzero = any(v for k, v in agg.items() if isinstance(v, (int, float)) and v)
    check("scored", {k: round(v, 3) for k, v in list(agg.items())[:4]}, nonzero)
else:
    check("scored", "no scores.json", False)

print("\n  run:", run)
sys.exit(0 if ok else 1)
PYCHECK
