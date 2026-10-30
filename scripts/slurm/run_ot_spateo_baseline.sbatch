#!/usr/bin/env bash
#SBATCH --job-name=p2b-otsp
#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
# opteron nodes (053-070) lack instructions torch 2.14 needs (Illegal instruction); the rest are P2B_EXCLUDE_NODES
#SBATCH --exclude=compute-[053-054,057-058,062,065,068-070,113-116,143,145,171-173]
# PASTE2 / Spateo scripted baselines on one task.
#   sbatch -o logs/baselines_ot_spateo/%x-%j.out scripts/slurm/run_ot_spateo_baseline.sbatch \
#       paste2|spateo <task.toml> [extra args for scripts/run_<method>_baseline.py]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
METHOD=$1; TASK=$2; shift 2
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE
case $METHOD in
  paste2) PY=.method-venv-ot/bin/python ;;
  spateo) PY=.method-venv-spateo/bin/python ;;
  *) echo "unknown method $METHOD"; exit 2 ;;
esac
echo "[$(date -Is)] $(hostname) $METHOD $TASK $*"
"$PY" "scripts/run_${METHOD}_baseline.py" "$TASK" "$@"
echo "[$(date -Is)] done"
