"""Re-score retained agent submissions under the current scorer and diff against stored scores."""
import json, os, sys, glob
from pathlib import Path
sys.path.insert(0, "src")
os.environ["P2B_ALLOW_EVAL_SCORING"] = "1"
import numpy as np, pandas as pd
from genomics_agent_bench.metrics import AlignmentScoreConfig, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices
from genomics_agent_bench.task_spec import load_task_spec

METRICS = ["gene_pcc","gene_cosine","gene_ssim","gene_mi","matching_accuracy","overlap_ratio_region","mae","overall_mean"]

def rescore(task_id, limit=None):
    task = load_task_spec(Path(f"benchmarks/spatial_alignment/tasks/{task_id}/task.toml"))
    man = load_split_manifest(task.split_manifest_path) if task.split_manifest_path else None
    masks = split_masks_for_slices(task.runnable_slices, man, split_name=task.score_split) if man else None
    runs = sorted(glob.glob(f"agent_runs/local_coding_agent/{task_id}_*/"))
    rows = []
    for run in runs[:limit]:
        sub = Path(run)/"workspace_snapshot/submission"
        old_f = Path(run)/"scores.json"
        if not (sub/"aligned_coordinates").is_dir() or not old_f.exists():
            continue
        try:
            old = json.loads(old_f.read_text())["scores"]["aggregate"]
        except Exception:
            continue
        try:
            new = score_alignment_task(
                task.runnable_slices, sub/"aligned_coordinates",
                marker_genes=task.marker_genes,
                config=AlignmentScoreConfig(label_key=task.label_key, regions=task.scored_regions),
                split_masks=masks, score_split=task.score_split,
                metric_profile=task.metric_profile, pair_mode=task.pair_mode)["aggregate"]
        except Exception as exc:
            rows.append({"run": Path(run).name, "error": str(exc)[:70]}); continue
        r = {"run": Path(run).name}
        for m in METRICS:
            if m in old and m in new:
                r[f"old_{m}"], r[f"new_{m}"] = float(old[m]), float(new[m])
        rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    task_id = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    d = rescore(task_id, limit)
    out = Path(f"/private/tmp/claude-501/-Users-yiqun-Desktop-chen-lab-genomics-agent/0323f10f-03ad-496c-a534-0fab5c29be1f/scratchpad/rescore_{task_id}.csv")
    d.to_csv(out, index=False)
    print(f"{task_id}: {len(d)} runs re-scored")
    if "error" in d.columns:
        print("  errors:", d.error.notna().sum())
    for m in METRICS:
        if f"old_{m}" in d.columns:
            o, n = d[f"old_{m}"], d[f"new_{m}"]
            print(f"  {m:22s} old {o.mean():7.4f} -> new {n.mean():7.4f}  "
                  f"mean delta {(n-o).mean():+8.4f}  max |delta| {(n-o).abs().max():7.4f}  "
                  f"rank corr {o.corr(n, method='spearman'):.3f}")
