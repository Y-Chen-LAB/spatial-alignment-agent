#!/usr/bin/env bash
# Site settings for the spatial-alignment campaign. Sourced by submit and array scripts so
# nothing is duplicated between them. Override any of these in the environment.

# The repo is wherever this file lives, so the same config works on the Mac and the cluster.
: "${P2B_REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${P2B_PYTHON:=$P2B_REPO/.venv/bin/python}"
: "${P2B_MATRIX:=$P2B_REPO/benchmarks/spatial_alignment/campaigns/run_matrix.csv}"
: "${P2B_RUNS_ROOT:=$P2B_REPO/agent_runs/cluster}"
: "${P2B_ROWS_ROOT:=$P2B_RUNS_ROOT/rows}"
# Workspaces are node-local: a shared scratch (fastscratch) could not answer ls within 20 s
# with sixty runs installing packages on it. Node /tmp ran out of space instead, so the
# package cache is shared per node (P2B_UV_CACHE) and each run's thirteen method
# environments hard-link into it rather than carrying their own copy of torch.
: "${P2B_WORKSPACE_ROOT:=${TMPDIR:-/tmp}/p2b_workspaces}"
: "${P2B_UV_CACHE:=${TMPDIR:-/tmp}/p2b-uv-cache-$USER}"

# SLURM resources. Alignment tasks are CPU and memory bound, not GPU bound; the agent may
# still import torch, so a GPU partition is optional rather than required.
: "${P2B_PARTITION:=shared}"
: "${P2B_EXCLUDE_NODES:=compute-053,compute-054,compute-091,compute-113,compute-114,compute-115,compute-116,compute-143,compute-145,compute-171,compute-172,compute-173}"  # 053-070: old Opteron nodes, scoring there runs 6x+ slower and timed out; 172: no network; 145: NFS at 18 MB/s; the others: local disk too small for the package cache
: "${P2B_CPUS:=4}"
: "${P2B_MEM:=48G}"                 # gemini-equipped OOMs at 32G on Open-ST and Slide-seq tasks; at 15/model in flight the 2.5T per-user cap is not binding
: "${P2B_TIME:=07:00:00}"            # 6 h for the agent plus an hour to score
: "${P2B_MAX_CONCURRENT:=0}"        # array throttle; 0 = none, the scheduler decides (jobs are CPU-only)

# Per-run limits handed to the runner.
: "${P2B_TIMEOUT_MINUTES:=360}"      # 90 was not enough on the larger tasks
: "${P2B_SCORE_TIMEOUT:=3600}"       # 30 min was not enough for the largest submissions
: "${P2B_MAX_OUTPUT_TOKENS:=64000}"   # Azure output cap, reasoning included; 0 = model maximum
: "${P2B_API_TIMEOUT:=3600}"          # seconds to wait for the model, polling in background mode

# Containment. Blind solving on Linux needs a container: the Seatbelt read-deny profile is
# macOS only, so without this the agent can read the repo, the unsanitised slices and the
# scorer. The runner refuses to run blind without one rather than downgrading silently.
: "${P2B_SINGULARITY_BIN:=singularity}"
# The image built by scripts/slurm/build_image.sh, if it is where that puts it.
if [ -z "${P2B_SINGULARITY_IMAGE:-}" ] && [ -f "$P2B_REPO/gen2bench.sif" ]; then
    P2B_SINGULARITY_IMAGE="$P2B_REPO/gen2bench.sif"
fi
: "${P2B_SINGULARITY_IMAGE:=docker://python:3.11-slim}"
: "${P2B_CONTAINER_MODULE:=}"        # e.g. singularity/3.11.4; empty = try singularity, then apptainer
: "${P2B_GPU:=0}"                   # set by submit_alignment.sh --gpu; exposes the device in the container
: "${P2B_GRES:=}"                   # GPU type for --gpu, e.g. gpu:tesa100:1; empty means any one GPU

# Credentials. Files, never values. One per line in $GEN2BENCH_CONFIG_DIR.
: "${GEN2BENCH_CONFIG_DIR:=${XDG_CONFIG_HOME:-$HOME/.config}/gen2bench}"
export GEN2BENCH_CONFIG_DIR

# HDF5 file locking breaks on most network filesystems.
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/p2b-mpl-$USER"; mkdir -p "$MPLCONFIGDIR"

# Export everything so the array job, which sbatch --export=ALL starts from the submit
# shell's environment, sees the same values the submit script resolved.
export P2B_REPO P2B_PYTHON P2B_MATRIX P2B_RUNS_ROOT P2B_ROWS_ROOT P2B_WORKSPACE_ROOT P2B_UV_CACHE
export P2B_PARTITION P2B_EXCLUDE_NODES P2B_CPUS P2B_MEM P2B_TIME P2B_MAX_CONCURRENT
export P2B_TIMEOUT_MINUTES P2B_SCORE_TIMEOUT P2B_MAX_OUTPUT_TOKENS P2B_API_TIMEOUT
export P2B_SINGULARITY_BIN P2B_SINGULARITY_IMAGE P2B_CONTAINER_MODULE P2B_GPU P2B_GRES
