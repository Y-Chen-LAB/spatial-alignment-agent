#!/usr/bin/env python3
"""Score submissions with the SABench-exact path and write them in the paper's table layout.

Separate from score_agent_submission.py and from every default metric profile: nothing here
writes to a run's scores.json. Outputs go under --out (default
/dcs07/chen/data/alignment-agent/sabench_comparison/scores/<dataset>/).

Datasets with published per-method tables (references/paper_source_data/s43588-026-00977):

  dlpfc        sa-001  MOESM7 DLPFC_GeneBased_{PCC,Cos_Sim,SSIM,MI}, DLPFC_LandmarkBased_Heatmap
  mhpr         sa-009  MOESM7 MHPR_GeneBased_*, MHPR_LandmarkBased_Heatmap
  mob          sa-030  MOESM10 MOB_GeneBased_*
  dlpfc_rot    sa-040  MOESM8 DLPFC_rotation (matching accuracy per angle)
  dlpfc_crop   sa-041  MOESM8 DLPFC_crop
  mhpr_rot     sa-042  MOESM8 MHPR_rotation
  mhpr_crop    sa-043  MOESM8 MHPR_crop

Two input kinds:
  --runs-csv   agent runs (columns: dir, task); coordinates from
               workspace_snapshot/submission/aligned_coordinates/<slice_id>.csv, expression and
               labels from the task's slices.
  --h5ad       aligned AnnData files (a method's own output, e.g. our PASTE re-run), in
               slice order; X, var, obs/Region and obsm/spatial are read from them.

Frames (see sabench_exact.canonicalize_to_reference_pose):
  submitted  coordinates exactly as returned
  canonical  whole stack moved rigidly so slice 1 sits in the pose the paper's methods returned
             it in: for DLPFC the SABench shuffled 151673 pose; elsewhere the task's input pose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from genomics_agent_bench import sabench_exact as SX  # noqa: E402
from genomics_agent_bench.h5ad import H5ADSlice, _read_obs_column  # noqa: E402
from genomics_agent_bench.io import read_coordinate_outputs  # noqa: E402
from genomics_agent_bench.task_spec import load_task_spec  # noqa: E402

PAPER = REPO / "references/paper_source_data/s43588-026-00977"
SHUFFLED_151673 = Path(
    "/dcs07/chen/data/alignment-agent/sabench_ref/zip/SABench/Tutorial/coarse_align_interactive_data/151673_shuffled.h5ad"
)
TASKS = REPO / "benchmarks/spatial_alignment/tasks"

DATASETS = {
    "dlpfc": dict(task="sa-001-dlpfc-serial-align", kind="serial", gene_table=PAPER / "43588_2026_977_MOESM7_ESM/DLPFC_GeneBased_PCC.csv",
                  regions=["Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "WM"]),
    "mhpr": dict(task="sa-009-mhpr-merfish-serial-align", kind="serial", gene_table=PAPER / "43588_2026_977_MOESM7_ESM/MHPR_GeneBased_PCC.csv",
                 regions=["BST", "MPA", "MPN", "PV", "PVH", "PVT", "V3", "fx"]),
    "mob": dict(task="sa-030-mob-stereoseq-slideseqv2-cross-platform", kind="serial", gene_table=PAPER / "43588_2026_977_MOESM10_ESM/MOB_GeneBased_PCC.csv",
                regions=[]),
    "dlpfc_rot": dict(task="sa-040-dlpfc-rotation-sweep", kind="sweep"),
    "dlpfc_crop": dict(task="sa-041-dlpfc-crop-sweep", kind="sweep"),
    "mhpr_rot": dict(task="sa-042-mhpr-rotation-sweep", kind="sweep"),
    "mhpr_crop": dict(task="sa-043-mhpr-crop-sweep", kind="sweep"),
}
TASK_TO_DATASET = {v["task"]: k for k, v in DATASETS.items()}

# sa-042 / sa-043 were compiled before the MHPR BASS-region rebuild: their perturbed Data6
# slices carry Cell_class in obs/Region while the Data5 reference carries the BASS anatomical
# regions, so matching accuracy is identically 0 under the default scorer. The perturbed
# slices hold the same 5557 cells as MHPR_animal1_n009 (same obs ids, reordered), so the
# SABench path takes their Region from there by obs id. Reported as a task-data bug.
MHPR_DATA6 = Path("/users/ychen5/p2b-data/paper2bench-data/MHPR_paper/MHPR_animal1_n009.h5ad")
REGION_OVERRIDE = {"mhpr_rot": MHPR_DATA6, "mhpr_crop": MHPR_DATA6}


def region_from(source: Path, obs_names) -> np.ndarray:
    with h5py.File(source, "r") as h:
        names = _read_obs_column(h, "_index") if "obs/_index" in h else None
        lab = _read_obs_column(h, "Region")
    lookup = dict(zip(map(str, names), lab))
    return np.array([lookup[str(n)] for n in obs_names])


def paper_gene_list(dataset: str) -> list[str]:
    cols = pd.read_csv(DATASETS[dataset]["gene_table"], index_col=0, nrows=0).columns
    # pandas renames repeated headers ETV1 -> ETV1.1; the paper table repeats the gene
    return [c.rsplit(".", 1)[0] if c.rsplit(".", 1)[-1].isdigit() else c for c in cols]


def _read_genes(path: Path, genes: list[str]) -> tuple[list[str], np.ndarray]:
    present, values = H5ADSlice(path).expression_for_genes(genes)
    return present, values


def _read_region(path: Path) -> np.ndarray | None:
    with h5py.File(path, "r") as h:
        return _read_obs_column(h, "Region") if "obs/Region" in h else None


class TaskTemplate:
    """Expression, labels and input poses for one task, loaded once and shared by all runs."""

    def __init__(self, dataset: str):
        self.dataset = dataset
        spec = DATASETS[dataset]
        self.kind = spec["kind"]
        self.task = load_task_spec(TASKS / spec["task"] / "task.toml")
        self.slices = self.task.runnable_slices
        self.genes = paper_gene_list(dataset) if self.kind == "serial" else []
        self.regions = spec.get("regions", [])
        unique = list(dict.fromkeys(self.genes))
        self.exact = []
        for s in self.slices:
            present, expr = _read_genes(s.path, unique) if unique else ([], None)
            missing = sorted(set(unique) - set(present))
            if missing:
                raise ValueError(f"{dataset}: {s.id} lacks paper genes {missing[:10]}")
            region = _read_region(s.path)
            if dataset in REGION_OVERRIDE and len(self.exact) > 0:
                region = region_from(REGION_OVERRIDE[dataset], s.obs_names)
            self.exact.append(SX.ExactSlice(s.id, s.spatial, expr, present, region))
        self.input_pose0 = self.slices[0].spatial
        if dataset == "dlpfc":
            with h5py.File(SHUFFLED_151673, "r") as h:
                self.canonical_pose0 = np.asarray(h["obsm/spatial"][()], dtype=float)
                names = _read_obs_column(h, "_index") if "obs/_index" in h else None
            assert names is None or list(names) == list(self.slices[0].obs_names), "151673 obs order differs"
        else:
            self.canonical_pose0 = self.input_pose0

    def with_coords(self, coords: list[np.ndarray]) -> list[SX.ExactSlice]:
        return [SX.ExactSlice(e.slice_id, c, e.expression, e.genes, e.region) for e, c in zip(self.exact, coords)]


def score_stack(tpl: TaskTemplate, coords: list[np.ndarray], label: dict, mi_seed: int) -> dict[str, list[pd.DataFrame]]:
    out: dict[str, list[pd.DataFrame]] = {"per_gene": [], "per_pair_gene": [], "landmark": [], "landmark_pair": [], "sweep": []}
    frames = {"submitted": coords}
    canon, info = SX.canonicalize_to_reference_pose(coords, tpl.canonical_pose0)
    frames["canonical"] = canon
    for frame, cs in frames.items():
        slices = tpl.with_coords(cs)
        meta = {**label, "frame": frame, "canon_angle_deg": info["angle_deg"], "canon_slice1_rms": info["slice1_rms_residual"]}
        if tpl.kind == "serial":
            long, mean = SX.score_gene_tables(slices, tpl.genes, mi_random_state=mi_seed)
            out["per_gene"].append(mean.assign(**meta))
            out["per_pair_gene"].append(long.assign(**meta))
            if tpl.regions:
                pair_frame, summary = SX.score_landmark_table(slices, tpl.regions)
                out["landmark"].append(pd.DataFrame([{**meta, **summary}]))
                out["landmark_pair"].append(pair_frame.assign(**meta))
        else:
            ref = slices[0]
            rows = [{**meta, "level": s.slice_id.split("_", 1)[1], "Average_Accuracy": SX.overlap_accuracy(ref, s)} for s in slices[1:]]
            out["sweep"].append(pd.DataFrame(rows))
    return out


def load_h5ad_stack(paths: list[Path], genes: list[str]) -> tuple[list[np.ndarray], list[SX.ExactSlice]]:
    unique = list(dict.fromkeys(genes))
    ex = []
    for p in paths:
        s = H5ADSlice(p)
        present, expr = s.expression_for_genes(unique) if unique else ([], None)
        ex.append(SX.ExactSlice(p.stem, s.spatial, expr, present, _read_region(p)))
    return [e.coords for e in ex], ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--runs-csv", type=Path, help="CSV with columns dir, task (agent runs)")
    ap.add_argument("--h5ad", type=Path, nargs="*", help="aligned AnnData files in slice order")
    ap.add_argument("--label", default="method", help="row label for --h5ad input")
    ap.add_argument("--use-task-expression", action="store_true",
                    help="with --h5ad: take only obsm/spatial from the files (obs order must match the task) "
                         "and X / Region from the task slices, e.g. when the method output re-encoded X")
    ap.add_argument("--mi-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/dcs07/chen/data/alignment-agent/sabench_comparison/scores"))
    args = ap.parse_args()

    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl = TaskTemplate(args.dataset)
    results: dict[str, list[pd.DataFrame]] = {}
    status_rows = []

    def merge(part):
        for k, v in part.items():
            results.setdefault(k, []).extend(v)

    if args.h5ad:
        coords, ex = load_h5ad_stack(args.h5ad, tpl.genes)
        if args.use_task_expression:
            for e, s in zip(ex, tpl.slices):
                if list(H5ADSlice(Path(args.h5ad[ex.index(e)])).obs_names) != list(s.obs_names):
                    raise ValueError(f"obs order of {e.slice_id} differs from task slice {s.id}")
        else:
            tpl.exact = ex  # score the method's own X / labels, as SABench does
        merge(score_stack(tpl, coords, {"run_id": args.label, "source": "h5ad"}, args.mi_seed))
        stem = f"h5ad_{args.label}"
    else:
        runs = pd.read_csv(args.runs_csv)
        runs = runs[runs["task"] == DATASETS[args.dataset]["task"]]
        for _, r in runs.iterrows():
            run_dir = Path(r["dir"])
            sub = run_dir / "workspace_snapshot/submission/aligned_coordinates"
            run_id = f"{run_dir.parent.name}/{run_dir.name}"  # same run name recurs across array replicates
            row = {"run_id": run_id, "dir": str(run_dir)}
            try:
                got = read_coordinate_outputs(tpl.slices, sub)
                coords = [got[s.id] for s in tpl.slices]
                merge(score_stack(tpl, coords, {"run_id": run_id, "source": "agent"}, args.mi_seed))
                row["sabench_status"] = "scored"
            except Exception as exc:  # recorded, imputed later by the comparison builder
                row["sabench_status"] = "no_valid_submission"
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                if not isinstance(exc, (FileNotFoundError, ValueError)):
                    row["traceback"] = traceback.format_exc()[-800:]
            status_rows.append(row)
            print(row["run_id"], row["sabench_status"], row.get("error", ""), flush=True)
        stem = "agents"
        pd.DataFrame(status_rows).to_csv(out_dir / f"{stem}_status.csv", index=False)

    for key, frames in results.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"{stem}_{key}.csv", index=False)
    (out_dir / f"{stem}_meta.json").write_text(json.dumps({"dataset": args.dataset, "task": DATASETS[args.dataset]["task"],
        "n_genes": len(tpl.genes), "regions": tpl.regions, "mi_seed": args.mi_seed}, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
