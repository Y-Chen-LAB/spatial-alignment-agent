"""Rows for the all-tasks method-baseline campaign (scripts/slurm/run_baseline_cell.sbatch).

Every non-agent method on every non-smoke task outside sa-3xx (Albis, already complete), with the Albis settings
(results/albis_baselines_comparison_README.md and the runner docstrings). A cell is skipped when
results/baselines/<task>/<variant>/scores.json already exists, or when it is not applicable (written to
not_applicable.tsv with the reason).

Size classes (from the task inventory: slices, observations, gene panels):
  stack  -- >= 15 slices or > 1M observations (3D stacks up to 129 sections, million-cell OpenST): every runner that
            supports it fits on <= 5,000 observations per slice and applies the fitted map to all of them.
  big    -- > 400k observations in total or one slice > 150k (Visium HD / Xenium pairs): more memory; Spateo uses
            morpho_align_ref on 5,000 per slice (as on Albis sa-302/305); CAST_MARK on 20,000 per slice when the
            dense expression matrix would not fit.
  small  -- everything else; runner defaults.

Usage: .venv/bin/python scripts/build_baseline_campaign_rows.py  (writes $CAMPAIGN/rows/*.tsv)
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = Path("/dcs07/chen/data/alignment-agent/baselines_campaign")
RESULTS = REPO / "results" / "baselines"
TASKS = REPO / "benchmarks" / "spatial_alignment" / "tasks"

SCRIPTED = ["identity", "centroid_similarity", "pca_similarity", "icp_similarity"]
# variant -> (runner, extra args, device)
PUBLISHED = {
    "paste": ("paste", "", "cpu"),
    "paste2": ("paste2", "", "cpu"),
    "spateo": ("spateo", "", "cpu"),
    "spacel": ("spacel", "", "cpu"),
    "geaso": ("geaso", "--write-alt --resume", "gpu"),
    "cast": ("cast", "", "gpu"),
    "cast_rescale": ("cast", "--rescale --method-name cast_rescale", "gpu"),
    "staligner": ("staligner", "", "cpu"),
    "slat": ("slat", "", "cpu"),           # also writes slat_lsq (and slat_allowflip)
    "stalign": ("stalign", "", "cpu"),
    "stalign_expr": ("stalign", "--signal expr", "cpu"),
    "stair": ("stair", "", "cpu"),
    "santo": ("santo", "", "cpu"),
    "gpsa": ("gpsa", "", "cpu"),
    "moscot": ("moscot", "", "cpu"),
}
# runner -> option that caps observations per slice used for fitting
SUBSAMPLE_OPT = {"paste": "--max-obs-per-slice", "paste2": "--max-obs-per-slice", "spateo": "--max-obs-per-slice",
                 "spacel": "--max-obs-per-slice", "staligner": "--max-obs-per-slice", "slat": "--max-obs-per-slice",
                 "stair": "--max-obs-per-slice", "santo": "--max-obs-per-slice", "moscot": "--max-obs-per-slice",
                 "cast": "--max-obs-mark"}
NO_SHARED_FEATURES = {"sa-406-p22-mouse-rna-h3k4me3-cross-modality-pair"}


def size_class(r: dict) -> str:
    n, tot, mx = int(r["n_slices"]), int(r["tot_obs"]), int(r["max_obs"])
    if n >= 15 or tot > 1_000_000:
        return "stack"
    if tot > 400_000 or mx > 150_000:
        return "big"
    return "small"


def main() -> None:
    inv = {r["id"]: r for r in csv.DictReader(open(CAMPAIGN / "inventory.tsv"), delimiter="\t")}
    rows: dict[str, list] = {}
    na = []
    for tid, r in sorted(inv.items()):
        if tid.startswith("sa-3"):
            continue
        toml = TASKS / tid / "task.toml"
        cls = size_class(r)
        for v in SCRIPTED:
            if not (RESULTS / tid / v / "scores.json").exists():
                rows.setdefault("cpu_small", []).append((v, "task_baseline", str(toml), ""))
        for v, (runner, extra, dev) in PUBLISHED.items():
            if (RESULTS / tid / v / "scores.json").exists():
                continue
            if tid in NO_SHARED_FEATURES and not (v == "stalign"):
                na.append((tid, v, "RNA genes vs H3K4me3 genomic peaks: the two slices share no feature, and this "
                           "method needs a shared expression space (STalign's density image is the only "
                           "expression-free method; the scripted methods are geometric)"))
                continue
            args = extra.split() if extra else []
            if cls == "stack" and runner in SUBSAMPLE_OPT:
                args += [SUBSAMPLE_OPT[runner], "5000"]
            elif cls == "stack" and runner == "gpsa":
                pass  # default 3,000 per slice already
            elif runner == "spateo" and int(r["max_obs"]) > 20_000:
                args += ["--max-obs-per-slice", "5000"]
            elif runner == "cast" and int(r["max_obs"]) * int(r["min_genes"]) * 4 > 3e9:
                args += ["--max-obs-mark", "20000"]
            if runner == "spacel" and tid in ("sa-406-p22-mouse-rna-h3k4me3-cross-modality-pair",):
                pass
            klass = f"{dev}_{'big' if cls in ('big', 'stack') else 'small'}"
            if cls == "stack" and dev == "cpu":
                klass = "cpu_stack"
            rows.setdefault(klass, []).append((v, runner, str(toml), " ".join(args), int(r["tot_obs"]) * int(r["n_slices"])))
    (CAMPAIGN / "rows").mkdir(parents=True, exist_ok=True)
    for k, rs in rows.items():
        # longest-expected first so the slow cells start early (Albis runtimes: GPSA hours, STalign/PASTE2/SANTO ~1 h)
        slow = {"gpsa": 0, "stalign": 1, "stalign_expr": 1, "paste2": 2, "santo": 3, "spacel": 4, "geaso": 0}
        rs = sorted(rs, key=lambda x: (slow.get(x[0], 9), -x[4]) if len(x) == 5 else (9, 0))
        with open(CAMPAIGN / "rows" / f"{k}.tsv", "w") as f:
            for row in rs:
                f.write("\t".join(row[:4]) + "\n")
        print(k, len(rs))
    with open(CAMPAIGN / "not_applicable.tsv", "w") as f:
        f.write("task\tvariant\treason\n")
        for t in na:
            f.write("\t".join(t) + "\n")
    print("not applicable", len(na))


if __name__ == "__main__":
    main()
