#!/usr/bin/env bash
# Build the Python environment on a cluster, from a fresh clone.
#
# Kept separate from setup_on_cluster.sh because this is the part that depends on the site:
# which Python the modules provide, and whether wheels are available. The other script only
# needs an interpreter to already exist.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY_BIN="${P2B_PYTHON_BIN:-}"
if [ -z "$PY_BIN" ]; then
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null; then
            ver=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]:02d}")')
            [ "$ver" -ge 311 ] && PY_BIN="$candidate" && break
        fi
    done
fi
[ -n "$PY_BIN" ] || {
    echo "no Python 3.11 or newer on PATH. Try: module avail python; module load python/3.11" >&2
    echo "or set P2B_PYTHON_BIN to an interpreter." >&2
    exit 2
}
echo "using $PY_BIN ($("$PY_BIN" --version))"

"$PY_BIN" -m venv .venv
# h5py, numpy, scipy and scikit-learn all ship manylinux wheels, so nothing here compiles and
# no HDF5 module needs loading. If pip starts building from source, the wheel index is being
# bypassed rather than the library being absent.
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e .

echo
echo "checking the install"
export HDF5_USE_FILE_LOCKING=FALSE
.venv/bin/python - <<'PYCHECK'
import importlib
for name in ("h5py", "numpy", "pandas", "scipy", "sklearn", "alpha_shapes"):
    importlib.import_module(name)
    print(f"  ok  {name}")
import genomics_agent_bench, paper2bench
print("  ok  genomics_agent_bench, paper2bench")
PYCHECK

echo
echo "next: scripts/slurm/setup_on_cluster.sh <data root>"
