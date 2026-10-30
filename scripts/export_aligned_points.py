"""Export the cells behind the aligned-result figures, so each panel can be redrawn on its own.

Run on the cluster from the genomics-agent repo root (the folder with benchmarks/, results/, agent_runs/):

    python scripts/export_aligned_points.py --figures results/aligned_figures --out results/aligned_points
    tar czf aligned_points.tgz -C results aligned_points        # copy this file back

For every task folder in --figures that has a summary.json (written by plot_aligned_results.py) it writes
    <out>/<task>/cells.csv.gz    slice, obs_id, label, x_in, y_in, x_gt, y_gt, z_gt   (sampled cells, one row each)
    <out>/<task>/points.csv.gz   panel, slice, obs_id, x, y[, z]                      (each drawn panel's coordinates)
    <out>/<task>/panels.json     panel key, title, score, row, run folder, source files found
Only the sections that appear in the task's figures are exported (the pairNN_<ref>__<moving> files), at most
--cap cells per section (fixed seed), plus --cap3d cells from every section when the task has a stack3d figure.
Nothing is written outside --out. Missing inputs are reported and skipped, never guessed.
"""
import argparse, csv, glob, gzip, json, os, re, sys
from pathlib import Path

import numpy as np

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib
import h5py


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--figures", default="results/aligned_figures")
    p.add_argument("--out", default="results/aligned_points")
    p.add_argument("--tasks", default="benchmarks/spatial_alignment/tasks")
    p.add_argument("--baselines", default="results/baselines")
    p.add_argument("--cap", type=int, default=12000)
    p.add_argument("--cap3d", type=int, default=900)
    p.add_argument("--only", nargs="*", default=[])
    return p.parse_args()


def slices_of(task_toml):
    spec = tomllib.loads(Path(task_toml).read_text())
    raw = spec.get("slices") or spec.get("data", {}).get("slices") or []
    out = {}
    for s in raw:
        k, v = s.split("=", 1); out[k] = v
    return out, spec.get("label_key")


def _col(g, name):
    """One obs column from an h5ad obs group: plain dataset (strings or numbers) or categorical (codes + categories)."""
    if name not in g: return None
    d = g[name]
    if isinstance(d, h5py.Group):
        if "codes" in d and "categories" in d:
            cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in d["categories"][:]])
            codes = d["codes"][:]; out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "")
            return out
        return None
    v = d[:]
    if v.dtype.kind in ("S", "O"): return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
    cats_attr = d.attrs.get("categories")
    if cats_attr is not None:   # old anndata: codes in the dataset, categories referenced from uns/obs
        cats = g[cats_attr] if not isinstance(cats_attr, (str, bytes)) else None
        if cats is not None:
            names = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats[:]])
            return np.where(v >= 0, names[np.clip(v, 0, None)], "")
    return v


def read_slice(path, label_key):
    """Read obs ids, input coordinates, labels and true coordinates with h5py (works for old and new h5ad encodings)."""
    with h5py.File(path, "r") as h:
        obs = h["obs"]
        idx_name = obs.attrs.get("_index", "_index")
        idx_name = idx_name.decode() if isinstance(idx_name, bytes) else idx_name
        ids = _col(obs, idx_name if idx_name in obs else "_index")
        xy = np.asarray(h["obsm"]["spatial"][:])[:, :2].astype(float)
        lab = _col(obs, label_key) if label_key else None
        lab = np.array([str(x) for x in lab]) if lab is not None else np.array([""] * len(ids))
        gt = np.full((len(ids), 3), np.nan)
        for j, c in enumerate(("x_GT", "y_GT", "z_GT")):
            v = _col(obs, c)
            if v is not None and np.asarray(v).dtype.kind in "fiu": gt[:, j] = np.asarray(v, float)
        if np.isnan(gt[:, 0]).all():
            for key in ("spatial_3d_aligned", "X_CCF"):
                if key in h["obsm"]:
                    m = np.asarray(h["obsm"][key][:], float); gt[:, : min(3, m.shape[1])] = m[:, :3]; break
    return np.asarray(ids).astype(str), xy, lab, gt


def read_coords(csv_path):
    rows = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            oid = row.get("obs_id") or row.get("cell_id") or next(iter(row.values()))
            try:
                rows[str(oid)] = (float(row["x"]), float(row["y"]), float(row["z"]) if row.get("z") not in (None, "") else np.nan)
            except (KeyError, ValueError):
                continue
    return rows


def find_submission(run_dir, slice_id):
    for c in (f"{run_dir}/submission/aligned_coordinates/{slice_id}.csv", f"{run_dir}/workspace_snapshot/submission/aligned_coordinates/{slice_id}.csv",
              f"{run_dir}/workspace/submission/aligned_coordinates/{slice_id}.csv"):
        if os.path.exists(c): return c
    hits = glob.glob(f"{run_dir}/**/aligned_coordinates/{slice_id}.csv", recursive=True)
    return hits[0] if hits else None


def baseline_variant(title, baselines, metric):
    """Panel title -> baseline variant name, as plot_aligned_results.py labels them."""
    m = re.match(r"Best (published|scripted): (\S+)", title)
    if m: return m.group(2)
    fam = title.strip().lower()
    cands = {k: v for k, v in baselines.items() if k.lower() == fam or k.lower().startswith(fam + "_")}
    if not cands: return None
    better = min if metric == "mae" else max
    return better(cands, key=lambda k: cands[k])


def find_baseline(base_root, task, variant, slice_id):
    root = Path(base_root) / task / variant
    for pat in (f"**/aligned_coordinates/{slice_id}.csv", f"**/{slice_id}.csv", f"**/*{slice_id}*.csv"):
        hits = sorted(root.glob(pat))
        if hits: return str(hits[0])
    return None


def main():
    A = args(); out_root = Path(A.out); out_root.mkdir(parents=True, exist_ok=True)
    report = []
    for summ in sorted(Path(A.figures).glob("*/summary.json")):
        task = summ.parent.name
        if A.only and task not in A.only: continue
        S = json.loads(summ.read_text())
        toml = Path(A.tasks) / task / "task.toml"
        if not toml.exists(): report.append((task, "no task.toml")); continue
        slices, label_key = slices_of(toml)
        pairs = [re.match(r"(pair|agree)\d+_(.+?)__(.+)\.jpg$", f) for f in S.get("figures", [])]
        need = sorted({s for m in pairs if m for s in (m.group(2), m.group(3)) if s in slices})
        has3d = "stack3d.jpg" in S.get("figures", [])
        rng = np.random.default_rng(0)
        cells_rows, keep = [], {}
        for sid, path in slices.items():
            cap = A.cap if sid in need else (A.cap3d if has3d else 0)
            if not cap: continue
            if not os.path.isabs(path): path = str(Path(path))
            if not os.path.exists(path): report.append((task, f"missing h5ad {sid}")); continue
            ids, xy, lab, gt = read_slice(path, label_key)
            idx = np.arange(len(ids)) if len(ids) <= cap else np.sort(rng.choice(len(ids), cap, replace=False))
            keep[sid] = set(ids[idx])
            for i in idx: cells_rows.append((sid, ids[i], lab[i], xy[i, 0], xy[i, 1], gt[i, 0], gt[i, 1], gt[i, 2]))
        tdir = out_root / task; tdir.mkdir(exist_ok=True)
        with gzip.open(tdir / "cells.csv.gz", "wt", newline="") as f:
            w = csv.writer(f); w.writerow(["slice", "obs_id", "label", "x_in", "y_in", "x_gt", "y_gt", "z_gt"]); w.writerows(cells_rows)
        panels_meta, n_pts = [], 0
        with gzip.open(tdir / "points.csv.gz", "wt", newline="") as f:
            w = csv.writer(f); w.writerow(["panel", "slice", "obs_id", "x", "y", "z"])
            for ri, row in enumerate(S.get("panels", [])):
                for pi, p in enumerate(row):
                    title = p.get("title", ""); key = f"r{ri}p{pi}"
                    meta = dict(key=key, row=ri, title=title, score=p.get("score", ""), run=p.get("run"), sources={})
                    if title.startswith(("Input", "Ground truth")):
                        meta["kind"] = "input" if title.startswith("Input") else "truth"; panels_meta.append(meta); continue
                    variant = None
                    if not p.get("run"):
                        variant = baseline_variant(title, S.get("baselines", {}), S.get("metric")); meta["variant"] = variant
                    for sid in keep:
                        src = find_submission(p["run"], sid) if p.get("run") else (find_baseline(A.baselines, task, variant, sid) if variant else None)
                        if not src: continue
                        meta["sources"][sid] = src
                        coords = read_coords(src)
                        for oid in keep[sid]:
                            c = coords.get(oid)
                            if c: w.writerow([key, sid, oid, c[0], c[1], "" if np.isnan(c[2]) else c[2]]); n_pts += 1
                    meta["kind"] = "run" if p.get("run") else "baseline"
                    panels_meta.append(meta)
        (tdir / "panels.json").write_text(json.dumps(dict(task=task, metric=S.get("metric"), label_key=label_key, pairs=[[m.group(2), m.group(3)] for m in pairs if m and m.group(1) == "pair"],
                                                          stack3d=has3d, slice_order=list(slices), panels=panels_meta), indent=1))
        found = sum(bool(m["sources"]) for m in panels_meta if m["kind"] in ("run", "baseline"))
        total = sum(m["kind"] in ("run", "baseline") for m in panels_meta)
        report.append((task, f"{len(cells_rows)} cells, {found}/{total} panels with coordinates, {n_pts} points"))
        print(task, report[-1][1], flush=True)
    (out_root / "export_report.txt").write_text("\n".join(f"{t}\t{m}" for t, m in report) + "\n")
    print(f"done: {len(report)} tasks; see {out_root}/export_report.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
