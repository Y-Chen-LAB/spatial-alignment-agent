#!/usr/bin/env python3
"""Our method-baseline runs (results/baselines/<task>/<variant>/) against the numbers the papers report for the same
methods, each on the paper's own metric. Writes results/baselines_vs_paper.csv.

    prepare: symlink farm + runs CSV so scripts/score_sabench_paper_metrics.py can score baseline outputs
             (it expects <dir>/workspace_snapshot/submission/aligned_coordinates)
             .venv/bin/python scripts/build_baselines_vs_paper.py prepare
             then, per dataset (dlpfc mhpr mob dlpfc_rot dlpfc_crop mhpr_rot mhpr_crop), as a SLURM job:
             .venv/bin/python scripts/score_sabench_paper_metrics.py --dataset D --runs-csv $W/runs.csv --out $W/scores
    build:   .venv/bin/python scripts/build_baselines_vs_paper.py build

Sources and metrics
  SABench (Yan et al. 2026, Nat Comput Sci; references/paper_source_data/s43588-026-00977), SABench-exact scorer
  (src/genomics_agent_bench/sabench_exact.py), canonical frame as in results/sabench_comparison:
    sa-001 DLPFC, sa-009 MHPR (MOESM7): mean over the paper's gene columns of PCC / Cos_Sim / SSIM / MI, and
      landmark Average_Accuracy and the mean ROI overlap over the paper's regions;
    sa-030 MOB (MOESM10): gene metrics only;
    sa-040..043 robustness sweeps (MOESM8): matching accuracy, mean over levels;
    sa-019 Flysta3D, sa-027 MLN (MOESM10): per-slice MAE, mean over slices (our scorer's mae_by_slice uses the
      same definition: the reference slice scores 102.3 on Flysta3D and 0.25 on MLN in both).
  GEASO (Wang, Liu & Ma 2026, Nat Commun; results/paper_baselines/geaso_2026.csv), our scorer's paper_metrics
  (geaso_v2):
    sa-407 PCC / CLC / AAS per rotation; sa-408..410 stitching MAE (paper pixels converted to um, as our task);
    sa-411 per-pair MAE (paper rows keyed sa-023: the same 129 Zhuang ABCA-1 sections); sa-412 per-pair CLC
    (paper rows keyed sa-027: the same OpenST sections). Mean over the conditions both sides have.

Verdict (per task x method x metric, on the summary value):
  bounded metrics (0..1 scale: PCC, Cos, SSIM, MI, accuracy, overlap, CLC): |diff| <= 0.02 reproduced,
  <= 0.06 close, else not reproduced; AAS (degrees, 360 - error): |diff| <= 5 reproduced, <= 20 close;
  MAE: |diff| / paper <= 15 % reproduced, <= 40 % close. "not comparable" overrides when the inputs differ
  (sa-001: the paper shuffled all four DLPFC slices, our task gives the native pose).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "references/paper_source_data/s43588-026-00977"
RESULTS = REPO / "results" / "baselines"
W = Path("/dcs07/chen/data/alignment-agent/baselines_campaign/paper_check")
OUT = REPO / "results" / "baselines_vs_paper.csv"

SAB = {"dlpfc": "sa-001-dlpfc-serial-align", "mhpr": "sa-009-mhpr-merfish-serial-align",
       "mob": "sa-030-mob-stereoseq-slideseqv2-cross-platform", "dlpfc_rot": "sa-040-dlpfc-rotation-sweep",
       "dlpfc_crop": "sa-041-dlpfc-crop-sweep", "mhpr_rot": "sa-042-mhpr-rotation-sweep",
       "mhpr_crop": "sa-043-mhpr-crop-sweep"}
GENE_FILES = {"dlpfc": ("MOESM7", "DLPFC", "_DLPFCS"), "mhpr": ("MOESM7", "MHPR", "_MHPR"),
              "mob": ("MOESM10", "MOB", "_MOB")}
SWEEPS = {"dlpfc_rot": "DLPFC_rotation", "dlpfc_crop": "DLPFC_crop", "mhpr_rot": "MHPR_rotation",
          "mhpr_crop": "MHPR_crop"}
# our variant -> SABench row/column name
SAB_NAME = {"paste": "PASTE", "paste2": "PASTE2", "staligner": "STAligner", "gpsa": "GPSA", "slat": "SLAT",
            "cast": "CAST", "stalign": "STalign", "stair": "STAIR", "spacel": "SPACEL",
            "spateo_rigidpart": "Spateo_rigid", "spateo": "Spateo_nonrigid", "santo": "SANTO"}
GEASO_NAME = {"CAST": "cast", "GEASO": "geaso", "Moscot": "moscot", "PASTE": "paste", "PASTE2": "paste2",
              "SLAT": "slat", "SPACEL": "spacel", "STAlign": "stalign", "Spateo": "spateo"}
GENE_METRICS = ["PCC", "Cos_Sim", "SSIM", "MI"]
FRAME = "canonical"


def scores(task: str, variant: str) -> dict | None:
    p = RESULTS / task / variant / "scores.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("scores", d)


def verdict(metric: str, ours: float, paper: float) -> str:
    if ours is None or paper is None or not np.isfinite(ours) or not np.isfinite(paper):
        return "not comparable"
    d = abs(ours - paper)
    if metric.startswith("MAE"):
        r = d / max(abs(paper), 1e-9)
        return "reproduced" if r <= 0.15 else "close" if r <= 0.40 else "not reproduced"
    if metric == "AAS":
        return "reproduced" if d <= 5 else "close" if d <= 20 else "not reproduced"
    return "reproduced" if d <= 0.02 else "close" if d <= 0.06 else "not reproduced"


# ------------------------------------------------------------------------------------------- prepare
def prepare() -> None:
    rows = []
    for ds, task in SAB.items():
        for v in SAB_NAME:
            src = RESULTS / task / v / "aligned_coordinates"
            if not src.is_dir() or not (RESULTS / task / v / "scores.json").exists():
                continue
            d = W / "runs" / task / v
            link = d / "workspace_snapshot" / "submission" / "aligned_coordinates"
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink(src.resolve(), link)
            rows.append({"dir": str(d), "task": task})
    pd.DataFrame(rows).to_csv(W / "runs.csv", index=False)
    print(len(rows), "baseline runs ->", W / "runs.csv")


# ------------------------------------------------------------------------------------------- build
def sab_rows() -> list[dict]:
    out = []
    S = W / "scores"
    for ds in ("dlpfc", "mhpr", "mob"):
        task = SAB[ds]
        moesm, name, suf = GENE_FILES[ds]
        f = S / ds / "agents_per_gene.csv"
        if not f.exists():
            continue
        pg = pd.read_csv(f)
        pg = pg[pg.frame == FRAME]
        lm = pd.read_csv(S / ds / "agents_landmark.csv") if (S / ds / "agents_landmark.csv").exists() else None
        if lm is not None:
            lm = lm[lm.frame == FRAME]
        plm_p = PAPER / f"43588_2026_977_{moesm}_ESM/{name}_LandmarkBased_Heatmap.csv"
        plm = pd.read_csv(plm_p, index_col=0) if plm_p.exists() else None
        for v, pname in SAB_NAME.items():
            rid = f"{task}/{v}"
            g = pg[pg.run_id == rid]
            for m in GENE_METRICS:
                pt = pd.read_csv(PAPER / f"43588_2026_977_{moesm}_ESM/{name}_GeneBased_{m}.csv", index_col=0)
                cand = [i for i in pt.index if i == f"{pname}{suf}" or (pname == "STAligner" and i.startswith(f"STAligner{suf}"))]
                if not cand:
                    continue
                pv = pt.loc[cand].mean(axis=1)  # per row: mean over the paper's gene columns
                paper_val = float(pv.max()) if pname == "STAligner" else float(pv.iloc[0])
                ours = float(g[m].mean()) if len(g) else None
                out.append(dict(task=task, source="SABench", paper_table=f"{moesm} {name}_GeneBased_{m}", method=pname,
                                variant=v, metric=f"{m} (mean over genes)", ours=ours, paper=paper_val,
                                paper_row=("STAligner_best (max over landmark variants; range "
                                           f"{pv.min():.3f}-{pv.max():.3f})") if pname == "STAligner" else cand[0]))
            if plm is not None and lm is not None:
                cand = [i for i in plm.index if i == f"{pname}{suf}" or (pname == "STAligner" and i.startswith(f"STAligner{suf}"))]
                if not cand:
                    continue
                roi = [c for c in plm.columns if c.startswith("Average_ROI_Overlap")]
                l = lm[lm.run_id == rid]
                for metric, ours, pser in (
                        ("Average_Accuracy", float(l["Average_Accuracy"].mean()) if len(l) else None, plm.loc[cand, "Average_Accuracy"]),
                        ("ROI overlap (mean over regions)", float(l[roi].mean(axis=1).mean()) if len(l) else None,
                         plm.loc[cand, roi].mean(axis=1))):
                    out.append(dict(task=task, source="SABench", paper_table=f"{moesm} {name}_LandmarkBased_Heatmap",
                                    method=pname, variant=v, metric=metric, ours=ours,
                                    paper=float(pser.max()) if pname == "STAligner" else float(pser.iloc[0]),
                                    paper_row="STAligner_best" if pname == "STAligner" else cand[0]))
    for ds, sheet in SWEEPS.items():
        task = SAB[ds]
        f = S / ds / "agents_sweep.csv"
        if not f.exists():
            continue
        sw = pd.read_csv(f)
        sw = sw[sw.frame == FRAME]
        paper = pd.read_csv(PAPER / f"43588_2026_977_MOESM8_ESM/{sheet}.csv")
        levels = sorted(sw.level.unique())
        paper.index = levels[: len(paper)]
        for v, pname in SAB_NAME.items():
            col = "STAligner_best" if pname == "STAligner" else pname
            if col not in paper.columns:
                continue
            s = sw[sw.run_id == f"{task}/{v}"].set_index("level")["Average_Accuracy"].reindex(paper.index)
            ok = s.notna() & paper[col].notna()
            out.append(dict(task=task, source="SABench", paper_table=f"MOESM8 {sheet}", method=pname, variant=v,
                            metric=f"matching accuracy (mean over {int(ok.sum())} levels)",
                            ours=float(s[ok].mean()) if ok.any() else None,
                            paper=float(paper.loc[ok, col].mean()) if ok.any() else float(paper[col].mean()),
                            paper_row=col))
    # Flysta3D / MLN per-slice MAE
    for task, fname in (("sa-019-flysta3d-e14-16-3d-serial-align", "Flysta3D_MAE"),
                        ("sa-027-openst-mln-full-19-slice-3d-align", "MLN_MAE")):
        paper = pd.read_csv(PAPER / f"43588_2026_977_MOESM10_ESM/{fname}.csv")
        for v, pname in SAB_NAME.items():
            if pname not in paper.columns:
                continue
            sc = scores(task, v)
            ours = None
            n = 0
            if sc and sc.get("mae_by_slice"):
                o = np.array(list(sc["mae_by_slice"].values()), dtype=float)
                p = paper[pname].to_numpy(dtype=float)
                k = min(len(o), len(p))
                ok = np.isfinite(o[:k]) & np.isfinite(p[:k])
                ours, n = float(o[:k][ok].mean()), int(ok.sum())
                pv = float(p[:k][ok].mean())
            else:
                pv = float(paper[pname].mean())
            out.append(dict(task=task, source="SABench", paper_table=f"MOESM10 {fname}", method=pname, variant=v,
                            metric=f"MAE per slice (mean over {n or len(paper)} slices)", ours=ours, paper=pv,
                            paper_row=pname))
    return out


def geaso_rows() -> list[dict]:
    g = pd.read_csv(REPO / "results/paper_baselines/geaso_2026.csv")
    task_of = {"sa-023-wmb-zhuang-abca1-129-section-3d-mae": "sa-411-merfish-hemibrain-129-section-3d-reconstruction",
               "sa-027-openst-mln-full-19-slice-3d-align": "sa-412-openst-mln-19-section-3d-reconstruction-paper"}
    out = []
    for (ptask, method, metric), grp in g.groupby(["task", "method", "metric"]):
        task = task_of.get(ptask, ptask)
        v = GEASO_NAME.get(method)
        if v is None:
            continue
        sc = scores(task, v)
        pairs = (sc or {}).get("paper_metrics", {}).get("pairs", []) if sc else []
        key = {"PCC": "pcc", "CLC": "clc", "AAS": "aas", "MAE": "mae_paper"}[metric]
        ours_by_cond = {}
        for k, p in enumerate(pairs, start=1):
            if key not in p:
                continue
            name = p["pair"]
            if "rot" in name:  # target__source_rot030 -> rotation 30°
                ang = int(name.rsplit("rot", 1)[1])
                cond = f"rotation {ang}°"
            else:
                cond = f"pair {k}"
            ours_by_cond[cond] = float(p[key])
        vals = []
        for r in grp.itertuples():
            cond = r.condition.split(", ")[-1] if task.startswith(("sa-408", "sa-409", "sa-410")) else r.condition
            pv = r.value_um if (metric == "MAE" and task.startswith(("sa-408", "sa-409", "sa-410"))) else r.value
            if pd.notna(pv):
                vals.append((cond, float(pv), ours_by_cond.get(cond)))
        both = [(c, p, o) for c, p, o in vals if o is not None and np.isfinite(o)]
        paper_mean = float(np.mean([p for _, p, _ in both])) if both else (float(np.mean([p for _, p, _ in vals])) if vals else None)
        ours_mean = float(np.mean([o for _, _, o in both])) if both else None
        unit = " (um)" if metric == "MAE" and task.startswith(("sa-408", "sa-409", "sa-410")) else \
            " (task units; paper: units of the released data)" if metric == "MAE" else ""
        out.append(dict(task=task, source="GEASO", paper_table=f"{grp.paper_figure.iloc[0]} ({ptask[:6]} rows)",
                        method=method, variant=v, metric=f"{'MAE' if metric == 'MAE' else metric}{unit} (mean over {len(both) or len(vals)} conditions)",
                        ours=ours_mean, paper=paper_mean, paper_row=method,
                        per_condition=json.dumps({c: [round(p, 4), None if o is None else round(o, 4)] for c, p, o in vals})))
    return out


REASONS = {
    "sa-001-dlpfc-serial-align": ("not comparable", "the paper randomly rotated/translated all four DLPFC slices "
                                  "before alignment; our task gives the native, roughly registered pose "
                                  "(identity already scores accuracy 0.759), and SABench released only two shuffled slices"),
}


def build() -> None:
    rows = sab_rows() + geaso_rows()
    df = pd.DataFrame(rows)
    df["diff"] = df.ours - df.paper
    df["rel_diff"] = np.where(df.metric.str.startswith("MAE"), df["diff"] / df.paper.abs(), np.nan)
    v, why = [], []
    for r in df.itertuples():
        if pd.isna(r.ours):
            v.append("not comparable")
            why.append("no successful run of ours for this cell (see results/baselines_coverage.csv)")
            continue
        if r.task in REASONS:
            v.append(REASONS[r.task][0])
            why.append(REASONS[r.task][1] + f"; metric-level verdict would be '{verdict(r.metric, r.ours, r.paper)}'")
            continue
        vv = verdict(r.metric, r.ours, r.paper)
        notes = []
        if r.task.startswith(("sa-042", "sa-043")):
            notes.append("MHPR sweep labels fixed by obs id in the SABench-exact path (task h5ads carry Cell_class)")
        if r.task.startswith("sa-030"):
            notes.append("our Slide-seqV2 slice keeps 21,724 beads vs the paper's 20,835; input pose of the paper's runs unknown")
        if r.task.startswith("sa-411") and "MAE" in r.metric:
            notes.append("paper MAE is in the released data's coordinate units; ours in task units")
        if r.method == "STAligner":
            notes.append("paper row is STAligner_best over hand-picked landmark domains; ours uses all cells as the landmark")
        if r.method in ("Spateo", "Spateo_nonrigid") and r.source == "GEASO":
            notes.append("GEASO paper does not say rigid or non-rigid Spateo; ours is the non-rigid output")
        why.append("; ".join(notes) if notes else f"|diff| = {abs(r.diff):.3g}")
        v.append(vv)
    df["verdict"] = v
    df["reason"] = why
    cols = ["task", "source", "paper_table", "method", "variant", "metric", "ours", "paper", "diff", "rel_diff",
            "verdict", "reason", "paper_row"] + (["per_condition"] if "per_condition" in df else [])
    df[cols].to_csv(OUT, index=False)
    print(df.groupby(["source", "verdict"]).size())
    print("wrote", OUT, len(df))


if __name__ == "__main__":
    {"prepare": prepare, "build": build}[sys.argv[1]]()
