"""Re-score every retained submission whose task data did not change, under the current scorer.

Submissions are coordinate CSVs, so re-scoring costs no agent time. A task whose slices were
rebuilt cannot be re-scored, because the agent saw different input coordinates; those runs are
listed as dead and have to be re-run. The dead list is explicit rather than inferred, because
the MLN centring keeps every obs_id, so scoring would silently succeed with wrong numbers.
"""
from __future__ import annotations
import glob, json, os, sys, time
from pathlib import Path

sys.path.insert(0, "src")
os.environ["P2B_ALLOW_EVAL_SCORING"] = "1"
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import pandas as pd
from genomics_agent_bench.metrics import AlignmentScoreConfig, score_alignment_task
from genomics_agent_bench.splits import load_split_manifest, split_masks_for_slices
from genomics_agent_bench.task_spec import load_task_spec

DEAD = {
    "sa-006-mbsa-serial-align": "Space Ranger 1.0.0 matrices (2026-09-14)",
    "sa-007-mbsp-serial-align": "Space Ranger 1.0.0 matrices (2026-09-14)",
    "sa-009-mhpr-merfish-serial-align": "bregma rebuild (2026-09-14) + BASS region labels (2026-09-15)",
    "sa-019-flysta3d-e14-16-3d-serial-align": "input frame rebuilt to paper raw_x/raw_y (2026-09-15)",
    "sa-022-openst-mln-serial-align": "MLN per-slice centring (2026-09-15)",
    "sa-023-wmb-zhuang-abca1-129-section-3d-mae": "WMB cellxgene cell set (2026-09-14)",
    "sa-025-openst-mln-adjacent-02-03-3d-align": "MLN per-slice centring (2026-09-15)",
    "sa-026-openst-mln-early-window-02-06-3d-align": "MLN per-slice centring (2026-09-15)",
    "sa-027-openst-mln-full-19-slice-3d-align": "MLN per-slice centring (2026-09-15)",
}
for t in ["sa-020-mop-mouse1-atlas-serial-align", "sa-021-mop-mouse2-atlas-serial-align"] + \
         [p.name for p in sorted(Path("benchmarks/spatial_alignment/tasks").glob("sa-1*"))]:
    DEAD.setdefault(t, "MOp slice membership from cell_labels.csv (2026-09-14)")

METRICS = ["gene_pcc","gene_cosine","gene_ssim","gene_mi","matching_accuracy",
           "overlap_ratio_region","mae","overall_mean"]


def main() -> None:
    rows, dead_rows = [], []
    tasks = sorted(p.parent.name for p in Path("benchmarks/spatial_alignment/tasks").glob("*/task.toml"))
    for task_id in tasks:
        runs = sorted(glob.glob(f"agent_runs/local_coding_agent/{task_id}_*/"))
        runs = [r for r in runs if (Path(r)/"workspace_snapshot/submission/aligned_coordinates").is_dir()]
        if not runs:
            continue
        if task_id in DEAD:
            for r in runs:
                dead_rows.append({"task": task_id, "run": Path(r).name, "reason": DEAD[task_id]})
            print(f"{task_id}: {len(runs)} runs DEAD ({DEAD[task_id]})", flush=True)
            continue
        task = load_task_spec(Path(f"benchmarks/spatial_alignment/tasks/{task_id}/task.toml"))
        man = load_split_manifest(task.split_manifest_path) if task.split_manifest_path else None
        masks = split_masks_for_slices(task.runnable_slices, man, split_name=task.score_split) if man else None
        t0 = time.time(); ok = 0
        for run in runs:
            sub = Path(run)/"workspace_snapshot/submission/aligned_coordinates"
            row = {"task": task_id, "run": Path(run).name}
            try:
                old = json.loads((Path(run)/"scores.json").read_text())["scores"]["aggregate"]
            except Exception:
                old = {}
            try:
                new = score_alignment_task(
                    task.runnable_slices, sub.parent/"aligned_coordinates",
                    marker_genes=task.marker_genes,
                    config=AlignmentScoreConfig(label_key=task.label_key, regions=task.scored_regions),
                    split_masks=masks, score_split=task.score_split,
                    metric_profile=task.metric_profile, pair_mode=task.pair_mode)["aggregate"]
                ok += 1
            except Exception as exc:
                row["error"] = str(exc)[:90]; rows.append(row); continue
            for m in METRICS:
                if m in new: row[m] = float(new[m])
                if m in old: row[f"old_{m}"] = float(old[m])
            rows.append(row)
        print(f"{task_id}: {ok}/{len(runs)} re-scored in {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("results/rescored_run_scores.csv", index=False)
        pd.DataFrame(dead_rows).to_csv("results/dead_runs.csv", index=False)

    pd.DataFrame(rows).to_csv("results/rescored_run_scores.csv", index=False)
    pd.DataFrame(dead_rows).to_csv("results/dead_runs.csv", index=False)
    print(f"\nre-scored {len(rows)} runs across {len({r['task'] for r in rows})} tasks")
    print(f"dead {len(dead_rows)} runs across {len({r['task'] for r in dead_rows})} tasks")


if __name__ == "__main__":
    main()
