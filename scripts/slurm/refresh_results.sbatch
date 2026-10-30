#!/usr/bin/env bash
#SBATCH --job-name=p2b-refresh
#SBATCH --output=/dcs07/chen/data/alignment-agent/logs/refresh-%j.out
#SBATCH --error=/dcs07/chen/data/alignment-agent/logs/refresh-%j.out
# Rebuild every downstream table, figure and bundle from the results present now (baselines campaign and agent
# runs). Earlier bundles are kept: new archives carry today's date. Each step is logged and a failing step does
# not stop the later ones. Submit from the repo root:
#   sbatch -p bstgpu,scavenge,shared -c 8 --mem=96G -t 1-00:00:00 scripts/slurm/refresh_results.sbatch
set -uo pipefail
REPO=${SAA_REPO:-$HOME/spatial-alignment-agent}
A=/dcs07/chen/data/alignment-agent
PY=$REPO/.venv/bin/python
EXPY=$A/method-venvs/export/bin/python   # pandas + pyarrow for the parquet export
TAG=$(date +%Y%m%d)
cd "$REPO"
export PYTHONUNBUFFERED=1 HDF5_USE_FILE_LOCKING=FALSE MPLCONFIGDIR=/fastscratch/myscratch/ychen5/mpl
mkdir -p "$MPLCONFIGDIR"
NT=${SLURM_CPUS_PER_TASK:-8}
FAILED=()
FROM=${REFRESH_FROM:-}   # resume: skip every step before this one (e.g. REFRESH_FROM=coords_full)
step() {  # step NAME CMD...
    local name=$1; shift
    if [ -n "$FROM" ]; then
        [ "$name" = "$FROM" ] || { echo "=== skip $name"; return 0; }
        FROM=
    fi
    echo; echo "=== [$(date -Is)] $name"
    "$@"; local rc=$?
    echo "=== [$(date -Is)] $name exit $rc"
    [ $rc -eq 0 ] || FAILED+=("$name")
}

# 1. agent run index (new runs since the last build), traces for new runs, score exports
step index        $PY scripts/agent_strategies/build_index.py --force
step traces       $PY scripts/agent_strategies/extract_traces.py --workers "$NT"
step scores       $PY scripts/agent_strategies/export_scores.py
# 2. baseline tables
step coverage     $PY scripts/baseline_campaign_report.py
step failures     $PY scripts/document_baseline_failures.py
step albis_table  $PY scripts/build_albis_baselines_table.py
step paper_prep   $PY scripts/build_baselines_vs_paper.py prepare
for D in dlpfc mhpr mob dlpfc_rot dlpfc_crop mhpr_rot mhpr_crop; do
    step "paper_score_$D" $PY scripts/score_sabench_paper_metrics.py --dataset "$D" \
        --runs-csv $A/baselines_campaign/paper_check/runs.csv --out $A/baselines_campaign/paper_check/scores
done
step paper_build  $PY scripts/build_baselines_vs_paper.py build
# 3. figures (redrawn with the current baselines), sampled figure points, full coordinates
step figures      $PY scripts/plot_aligned_results.py --force --workers 6
step points       $PY scripts/export_aligned_points.py --figures $A/aligned_figures --out $A/aligned_points_$TAG
step coords_full  $EXPY scripts/export_all_coordinates.py --out $A/coordinates_full_$TAG --force
# 4. archives
step zip_figures  bash -c "cd $A && rm -f aligned_figures_$TAG.zip && zip -qr aligned_figures_$TAG.zip aligned_figures"
step tgz_points   tar czf $A/aligned_points_$TAG.tgz -C $A aligned_points_$TAG
step tar_coords   tar cf $A/coordinates_full_$TAG.tar -C $A coordinates_full_$TAG   # parquet is already compressed
# MNN add-on tables (scripts/run_mnn_addon.py) and the Albis MNN tables travel with the bundle's scores/
step mnn_tables   $PY -c "
import json,shutil,pathlib
b=pathlib.Path('$A/agent_strategies_bundle/scores'); b.mkdir(exist_ok=True); mf=b/'MANIFEST.json'
names=json.loads(mf.read_text()) if mf.exists() else ['agent_scores.csv','baseline_scores.csv','paper_reported.csv','PAPER_SOURCE_DATA.md']
for n in ['mnn_metrics_all.csv','mnn_metrics_summary.csv','albis_mnn_metrics.csv','albis_mnn_summary.csv']:
    src=pathlib.Path('$REPO/results')/n
    if src.exists():
        shutil.copy2(src,b/n); names+=[n] if n not in names else []
mf.write_text(json.dumps(names,indent=1))
print(names)"
step bundle       $PY scripts/agent_strategies/package_bundle.py
step raw_bundle   $PY $A/raw_bundle.py
step raw_dated    cp $A/agent_strategies_bundle/agent_runs_raw.zip $A/agent_strategies_bundle/agent_runs_raw_$TAG.zip
# 5. generalization (strategy synthesis): cells, policies, synthesis traces, applied scores and coordinates
step gen_bundle   $PY scripts/generalization/package_bundle.py --tag $TAG

echo; echo "[$(date -Is)] done; failed steps: ${FAILED[*]:-none}"
ls -la $A/*_$TAG.* $A/agent_strategies_bundle/ $A/generalization/bundle/ 2>/dev/null
[ ${#FAILED[@]} -eq 0 ]
