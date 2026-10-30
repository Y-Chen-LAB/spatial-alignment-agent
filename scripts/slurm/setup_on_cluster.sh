#!/usr/bin/env bash
# Run once on the cluster, after `git clone` and after the data bundle has arrived.
#
# Two things a clone cannot carry: the task files name slices by a path under the machine that
# built them, and benchmarks/spatial_alignment/compiled_data is a symlink to data kept outside
# the repository and excluded from git. This fixes both.
set -euo pipefail

DATA_ROOT="${1:-$HOME/p2b-data}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

[ -d "$DATA_ROOT/paper2bench-data" ] || { echo "missing $DATA_ROOT/paper2bench-data; transfer the bundle first" >&2; exit 2; }
[ -d "$DATA_ROOT/compiled_data" ]    || { echo "missing $DATA_ROOT/compiled_data; transfer the bundle first" >&2; exit 2; }

PY="${P2B_PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY; create the venv first, or set P2B_PYTHON" >&2; exit 2; }

echo "1. linking compiled_data, which git does not carry"
rm -rf benchmarks/spatial_alignment/compiled_data
ln -s "$DATA_ROOT/compiled_data" benchmarks/spatial_alignment/compiled_data

echo "2. rebasing the absolute slice paths in the task files"
"$PY" scripts/check_portability.py --data-root "$DATA_ROOT/paper2bench-data" --apply

echo "3. checking every slice now resolves"
export HDF5_USE_FILE_LOCKING=FALSE
"$PY" scripts/build_tasks.py --check-coverage
"$PY" -m paper2bench.cli validate-suite --output results/validation.json >/dev/null 2>&1 \
  || "$PY" -c "import sys; sys.path.insert(0,'src'); from paper2bench.cli.main import main; sys.argv=['p2b','validate-suite','--output','results/validation.json']; main()"
"$PY" -c "import json; d=json.load(open('results/validation.json')); print(f\"   validate: {d['n_ok']} / {d['n_tasks']} ok\")"

echo "4. credentials"
"$PY" scripts/check_access.py || echo "   (fix the above before submitting)"

cat <<'NEXT'

Remaining, in order:
  singularity build gen2bench.sif containers/gen2bench.def
  sha256sum gen2bench.sif                          # record it
  export P2B_SINGULARITY_IMAGE=$PWD/gen2bench.sif
  ./.venv/bin/python scripts/check_azure_openai_access.py --deployment gpt-6-astra
  export P2B_MAX_CONCURRENT=1
  scripts/slurm/submit_alignment.sh --agent azure-openai --task sa-001-dlpfc-serial-align  # three serialized smoke runs
  scripts/slurm/submit_alignment.sh --shard 1 --dry-run
NEXT
