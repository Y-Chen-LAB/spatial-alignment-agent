#!/usr/bin/env bash
# Submit the all-tasks method-baseline campaign: one throttled SLURM array per resource class.
# Rows come from scripts/build_baseline_campaign_rows.py ($CAMPAIGN/rows/<class>.tsv).
#   bash scripts/slurm/submit_baseline_campaign.sh [class ...]     (default: all five classes)
# Throttles keep the campaign at <= 60 CPU jobs and <= 12 GPU jobs running at once.
set -euo pipefail
REPO=${SAA_REPO:-$HOME/spatial-alignment-agent}
CAMPAIGN=/dcs07/chen/data/alignment-agent/baselines_campaign
cd "$REPO"
# shellcheck disable=SC1091
source scripts/slurm/site_config.sh >/dev/null
EXC=$P2B_EXCLUDE_NODES
# Some method wheels (STAligner: annoy built with native flags) die with SIGILL on the haswell/broadwell nodes; keep CPU jobs on
# AVX-512 Intel nodes.
CPU_CONSTRAINT=${CPU_CONSTRAINT:-"icelake-server|sapphirerapids"}
# GPU jobs: partitions gpu,bstgpu,bstgpu2,bstgpu3 (L40S nodes 171/173 allowed for GPU baselines; 172 has no network).
GPU_PARTS=gpu,bstgpu,bstgpu2,bstgpu3
GPU_EXC=compute-053,compute-054,compute-057,compute-058,compute-062,compute-065,compute-068,compute-069,compute-070,compute-113,compute-114,compute-115,compute-116,compute-143,compute-145,compute-172
GPU_EXC_SMALL="$GPU_EXC,compute-117,compute-123"                          # 117/123: V100 32 GB, GEASO runs out of GPU memory
GPU_EXC_BIG="$GPU_EXC,compute-117,compute-123,compute-171,compute-173"    # big tasks on A100 80 GB / H100 only
mkdir -p "$CAMPAIGN/logs"
CLASSES=${*:-cpu_small cpu_big cpu_stack gpu_small gpu_big}
for k in $CLASSES; do
  rows=$CAMPAIGN/rows/$k.tsv
  [ -s "$rows" ] || { echo "skip $k (no rows)"; continue; }
  n=$(wc -l < "$rows")
  case $k in
    cpu_small) opts=(-p shared,scavenge --constraint="$CPU_CONSTRAINT" --exclude="$EXC" -c 8 --mem=64G  -t 2-00:00:00); thr=36 ;;
    cpu_big)   opts=(-p shared,scavenge --constraint="$CPU_CONSTRAINT" --exclude="$EXC" -c 8 --mem=180G -t 3-00:00:00); thr=14 ;;
    cpu_stack) opts=(-p shared,scavenge --constraint="$CPU_CONSTRAINT" --exclude="$EXC" -c 8 --mem=200G -t 3-00:00:00); thr=10 ;;
    gpu_small) opts=(-p $GPU_PARTS --gres=gpu:1 --exclude="$GPU_EXC_SMALL" -c 8 --mem=96G  -t 1-00:00:00); thr=8 ;;
    gpu_big)   opts=(-p $GPU_PARTS --gres=gpu:1 --exclude="$GPU_EXC_BIG"   -c 8 --mem=240G -t 3-00:00:00); thr=4 ;;
    *) echo "unknown class $k"; exit 2 ;;
  esac
  [ -n "${THROTTLE:-}" ] && thr=$THROTTLE
  jid=$(sbatch --parsable -J "p2b-bl-$k" "${opts[@]}" --array=0-$((n - 1))%$thr \
        --export=ALL,ROWS="$rows" scripts/slurm/run_baseline_cell.sbatch)
  cp "$rows" "$CAMPAIGN/rows/submitted_${k}_${jid}.tsv"   # index -> row mapping for this array
  echo "$(date -Is) $k array $jid rows $n throttle $thr" | tee -a "$CAMPAIGN/submissions.log"
done
