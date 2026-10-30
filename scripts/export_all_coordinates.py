"""Every cell's predicted aligned coordinates from every baseline method and every scored agent run, per task.

    uv run --with pyarrow --with pandas --with numpy --with h5py \
        python scripts/export_all_coordinates.py --out /dcs07/chen/data/alignment-agent/coordinates_full

Writes into --out:
  <task>.parquet   one row per cell (all cells, no sampling), wide:
                   slice, obs_id, label, x_in, y_in, x_gt, y_gt, z_gt, then <source>__x, <source>__y per source
                   (float32; a source column is missing-free: a source is only kept when it covers every slice)
  sources.csv      one row per source: source, task, kind (baseline / agent), method or model, condition,
                   prompt_variant, job, run_dir, the scorer's aggregate (mae, mae_reduction, overall_mean,
                   matching_accuracy), and the path its coordinates came from
  skipped.csv      sources left out and why (missing slice file, wrong row count, obs order differs, non-finite)
  README.md

Sources:
  baselines  results/baselines/<task>/<variant>/aligned_coordinates/ with a scores.json, every variant except
             the generalization experiment's synth_* policies
  agents     agent_traces/index.json rows with status "scored" (bare and equipped; superseded runs are never
             indexed), coordinates from <run>/workspace_snapshot/submission/aligned_coordinates/
Coordinates must follow the task's obs order exactly (the scorer's own rule), so they are joined by position after
checking obs_id. Reads only the listed files; no recursive scans.
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_aligned_points import read_slice, slices_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "benchmarks" / "spatial_alignment" / "tasks"
BASELINES = REPO / "results" / "baselines"
INDEX = Path("/dcs07/chen/data/alignment-agent/agent_traces/index.json")
AGG_KEYS = ("mae", "mae_reduction", "overall_mean", "matching_accuracy")
SHORT = {"claude-opus-5-5": "opus55", "claude-sonnet-5-5": "sonnet55", "claude-sonnet-4-6": "sonnet46",
         "gpt-6-astra": "gpt6astra", "gpt-5.5": "gpt55", "gemini-3.5-flash": "gemini35flash"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/dcs07/chen/data/alignment-agent/coordinates_full"))
    p.add_argument("--only", nargs="*", default=[], help="task id prefixes, e.g. sa-001 sa-411")
    p.add_argument("--force", action="store_true", help="rewrite tasks whose parquet exists")
    return p.parse_args()


def aggregate(scores_json):
    try:
        d = json.loads(Path(scores_json).read_text())
    except (OSError, ValueError):
        return {}
    a = d.get("scores", d).get("aggregate", {}) or {}
    return {k: a.get(k) for k in AGG_KEYS}


def read_source(coord_dir, slice_ids, ids_by_slice):
    """(n, 2) float32 per slice in task order, or (None, reason)."""
    out = []
    for sid in slice_ids:
        f = Path(coord_dir) / f"{sid}.csv"
        if not f.exists():
            return None, f"missing {sid}.csv"
        df = pd.read_csv(f, usecols=lambda c: c in ("obs_id", "x", "y"), dtype={"obs_id": str})
        if not {"obs_id", "x", "y"} <= set(df.columns):
            return None, f"{sid}.csv lacks obs_id/x/y"
        if len(df) != len(ids_by_slice[sid]):
            return None, f"{sid}.csv has {len(df)} rows, expected {len(ids_by_slice[sid])}"
        if not np.array_equal(df["obs_id"].to_numpy(str), ids_by_slice[sid]):
            return None, f"{sid}.csv obs order differs from the task"
        xy = df[["x", "y"]].to_numpy(np.float64)
        if not np.isfinite(xy).all():
            return None, f"{sid}.csv has non-finite coordinates"
        out.append(xy.astype(np.float32))
    return np.concatenate(out), None


def main():
    A = args()
    A.out.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX.read_text())
    runs_by_task = {}
    for r in index:
        if r.get("status") == "scored":
            runs_by_task.setdefault(r["task"], []).append(r)
    tasks = sorted(d.name for d in TASKS.iterdir() if (d / "task.toml").exists() and not d.name.endswith("-smoke"))
    if A.only:
        tasks = [t for t in tasks if any(t.startswith(o) for o in A.only)]
    src_rows, skip_rows = [], []
    src_csv, skip_csv = A.out / "sources.csv", A.out / "skipped.csv"
    for t in tasks:
        target = A.out / f"{t}.parquet"
        slices, label_key = slices_of(TASKS / t / "task.toml")
        if not slices:
            skip_rows.append({"task": t, "source": "", "reason": "task defines no runnable slices"})
            continue
        sids = list(slices)
        cols = {k: [] for k in ("slice", "obs_id", "label")}
        num = {k: [] for k in ("x_in", "y_in", "x_gt", "y_gt", "z_gt")}
        ids_by_slice = {}
        for sid in sids:
            path = Path(slices[sid])
            path = path if path.is_absolute() else REPO / path
            ids, xy, lab, gt = read_slice(path, label_key)
            ids_by_slice[sid] = ids
            cols["slice"].append(np.full(len(ids), sid, dtype=object)); cols["obs_id"].append(ids)
            cols["label"].append(lab)
            num["x_in"].append(xy[:, 0]); num["y_in"].append(xy[:, 1])
            for j, k in enumerate(("x_gt", "y_gt", "z_gt")):
                num[k].append(gt[:, j])
        frame = {k: np.concatenate(v) for k, v in cols.items()}
        frame.update({k: np.concatenate(v).astype(np.float32) for k, v in num.items()})

        sources = []  # (name, coord_dir, meta)
        bdir = BASELINES / t
        if bdir.is_dir():
            for v in sorted(p.name for p in bdir.iterdir()):
                if v.startswith("synth_") or not (bdir / v / "scores.json").exists():
                    continue
                sources.append((f"bl_{v}", bdir / v / "aligned_coordinates",
                                {"kind": "baseline", "method": v, "scores": bdir / v / "scores.json"}))
        for r in sorted(runs_by_task.get(t, []), key=lambda r: (r["model"], r["condition"], r["job"])):
            name = f"{SHORT.get(r['model'], r['model'])}_{r['condition']}_{r['job']}"
            rd = Path(r["path"])
            sources.append((name, rd / "workspace_snapshot" / "submission" / "aligned_coordinates",
                            {"kind": "agent", "method": r["model"], "condition": r["condition"],
                             "prompt_variant": r.get("prompt_variant"), "job": r["job"], "run_dir": str(rd),
                             "scores": rd / "scores.json"}))
        kept = 0
        for name, cdir, meta in sources:
            xy, why = read_source(cdir, sids, ids_by_slice)
            if xy is None:
                skip_rows.append({"task": t, "source": name, "reason": why, "path": str(cdir)})
                continue
            frame[f"{name}__x"], frame[f"{name}__y"] = xy[:, 0], xy[:, 1]
            src_rows.append({"source": name, "task": t, "kind": meta["kind"], "method_or_model": meta["method"],
                             "condition": meta.get("condition", ""), "prompt_variant": meta.get("prompt_variant", ""),
                             "job": meta.get("job", ""), "run_dir": meta.get("run_dir", ""),
                             **aggregate(meta["scores"]), "coordinates": str(cdir)})
            kept += 1
        if target.exists() and not A.force:
            print(f"{t}: exists, kept (use --force); {kept} sources listed", flush=True)
            continue
        pd.DataFrame(frame).to_parquet(target, index=False, compression="zstd")
        print(f"{t}: {len(frame['obs_id'])} cells, {kept} sources kept, "
              f"{len(sources) - kept} skipped -> {target.name}", flush=True)
        # keep the tables current while the export runs
        pd.DataFrame(src_rows).to_csv(src_csv, index=False)
        pd.DataFrame(skip_rows).to_csv(skip_csv, index=False)
    pd.DataFrame(src_rows).to_csv(src_csv, index=False)
    pd.DataFrame(skip_rows).to_csv(skip_csv, index=False)
    (A.out / "README.md").write_text(f"""# Full predicted coordinates, every method and agent run

Exported {date.today().isoformat()} by scripts/export_all_coordinates.py.

- `<task>.parquet`: one row per cell, every cell of the task (no sampling). Columns: `slice, obs_id, label,
  x_in, y_in` (the task input), `x_gt, y_gt, z_gt` (truth where the task has it, else NaN), then
  `<source>__x, <source>__y` for each method or run (float32, the submitted aligned coordinates).
- `sources.csv`: what each `<source>` is. `bl_<variant>` = a baseline method variant from
  results/baselines; `<model>_<bare|equipped>_<job>` = one scored agent run. Includes the scorer's aggregate
  (mae, mae_reduction, overall_mean, matching_accuracy) and the source path.
- `skipped.csv`: sources left out and why (missing file, row count or obs order differs from the task,
  non-finite coordinates). A source is kept only when it covers every slice.

Agent runs: agent_traces/index.json, status scored, bare and equipped (superseded runs are not indexed).
Baselines: every variant with a scores.json except the generalization experiment's synth_* policies.

Read in Python: `pd.read_parquet("sa-001-dlpfc-serial-align.parquet")`; in R: `arrow::read_parquet(...)`.
""")
    print(f"done: {len(src_rows)} sources, {len(skip_rows)} skipped -> {A.out}")


if __name__ == "__main__":
    main()
