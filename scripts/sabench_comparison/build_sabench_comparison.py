#!/usr/bin/env python3
"""Put agent runs next to the SABench paper's per-method tables, in the paper's layout.

Reads the SABench-exact scores written by scripts/score_sabench_paper_metrics.py and the
paper source data, and writes results/sabench_comparison/. See that directory's README.

Agent rows: one per model x prompt arm (bare = full-basic, equipped =
full-package-paper-prior) x statistic (median over runs = primary, best run = secondary).
Runs without a usable submission count as failures and are scored 0 on every metric, the
paper's own convention for methods that failed; runs that never reached the agent
(provider rate limit, method-environment bootstrap) are infrastructure failures and dropped.

Submissions flagged non_spatial by geometry_check.py (coordinates that are no similarity
image of the input slice, e.g. an expression embedding written into x, y) are scored 0 like
failures in the primary output; --nonspatial keep writes the as-scored sensitivity view to
results/sabench_comparison/sensitivity_nonspatial_kept/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from score_sabench_paper_metrics import DATASETS, paper_gene_list  # noqa: E402

PAPER = REPO / "references/paper_source_data/s43588-026-00977"
SCORES = Path("/dcs07/chen/data/alignment-agent/sabench_comparison/scores")
VALID = Path("/dcs07/chen/data/alignment-agent/sabench_comparison/validation")
OUT = REPO / "results/sabench_comparison"
NONSPATIAL = "zero"
FRAME = "canonical"
GENE_METRICS = ["PCC", "Cos_Sim", "SSIM", "MI"]
ARM = {"full-basic": "bare", "full-package-paper-prior": "equipped"}
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-sonnet-4-6", "gpt-5.5", "gpt-6-astra", "gemini-3.5-flash"]
GENE_FILES = {
    "dlpfc": ("MOESM7", "DLPFC", "_DLPFCS"),
    "mhpr": ("MOESM7", "MHPR", "_MHPR"),
    "mob": ("MOESM10", "MOB", "_MOB"),
}
SWEEP_FILES = {"dlpfc_rot": "DLPFC_rotation", "dlpfc_crop": "DLPFC_crop", "mhpr_rot": "MHPR_rotation", "mhpr_crop": "MHPR_crop"}


# ------------------------------------------------------------------ run bookkeeping
def run_table() -> pd.DataFrame:
    inv = pd.read_csv(SCORES.parent / "runs_target.csv")
    inv["run_id"] = inv["dir"].map(lambda d: f"{Path(d).parent.name}/{Path(d).name}")
    inv["arm"] = inv["pv"].map(ARM)
    rows = []
    for ds in DATASETS:
        f = SCORES / ds / "agents_status.csv"
        if not f.exists():
            continue
        st = pd.read_csv(f).assign(dataset=ds)
        rows.append(st)
    st = pd.concat(rows, ignore_index=True)
    t = st.merge(inv[["run_id", "model", "arm", "status", "stage"]], on="run_id", how="left")

    def classify(r):
        if r.sabench_status == "scored":
            return "scored"
        m = json.loads((Path(REPO / r.dir) / "manifest.json").read_text()) if (REPO / r.dir / "manifest.json").exists() else {}
        msg = str(m.get("failure_message", ""))
        if ("rate_limit" in msg and '"status":"rejected"' in msg) or m.get("failure_stage") == "method_env_prebootstrap":
            return "infra_excluded"
        return "failed_zero"

    t["outcome"] = t.apply(classify, axis=1)
    g = pd.read_csv(SCORES.parent / "geometry_check.csv")[["dataset", "run_id", "max_resid_ratio", "non_spatial"]]
    t = t.merge(g, on=["dataset", "run_id"], how="left")
    t.loc[(t.outcome == "scored") & (t.non_spatial == True), "outcome"] = "non_spatial"  # noqa: E712
    return t


def counts_as_zero(outcome: str) -> bool:
    return outcome == "failed_zero" or (outcome == "non_spatial" and NONSPATIAL == "zero")


# ------------------------------------------------------------------ paper tables
def paper_gene_table(ds: str, metric: str) -> pd.DataFrame:
    moesm, name, _ = GENE_FILES[ds]
    return pd.read_csv(PAPER / f"43588_2026_977_{moesm}_ESM/{name}_GeneBased_{metric}.csv", index_col=0)


def paper_landmark(ds: str) -> pd.DataFrame | None:
    _, name, _ = GENE_FILES[ds]
    p = PAPER / f"43588_2026_977_MOESM7_ESM/{name}_LandmarkBased_Heatmap.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


# ------------------------------------------------------------------ agent aggregation
def agent_gene_rows(ds: str, runs: pd.DataFrame, frame: str = FRAME):
    """Per run per gene values (failures as zeros), then median / best per model x arm."""
    genes = paper_gene_list(ds)
    cols = paper_gene_table(ds, "PCC").columns
    f = SCORES / ds / "agents_per_gene.csv"
    pg = pd.read_csv(f) if f.exists() else pd.DataFrame()
    pg = pg[pg["frame"] == frame] if len(pg) else pg
    r = runs[(runs.dataset == ds) & (runs.outcome != "infra_excluded")]
    per_run = {}  # run_id -> DataFrame(gene_position x metric)
    for _, row in r.iterrows():
        if not counts_as_zero(row.outcome):
            x = pg[pg.run_id == row.run_id].sort_values("gene_position")
            per_run[row.run_id] = x.set_index("gene_position")[GENE_METRICS]
        else:
            per_run[row.run_id] = pd.DataFrame(0.0, index=range(len(genes)), columns=GENE_METRICS)
    lm = None
    if (SCORES / ds / "agents_landmark.csv").exists():
        lm = pd.read_csv(SCORES / ds / "agents_landmark.csv")
        lm = lm[lm.frame == frame].set_index("run_id")
    out_gene = {m: {} for m in GENE_METRICS}
    out_lm = {}
    run_level = []
    for (model, arm), g in r.groupby(["model", "arm"]):
        ids = list(g.run_id)
        stack = {m: pd.DataFrame({i: per_run[i][m].to_numpy() for i in ids}) for m in GENE_METRICS}
        run_means = pd.DataFrame({m: stack[m].mean(axis=0) for m in GENE_METRICS})
        lmr = None
        if lm is not None:
            lmc = [c for c in lm.columns if c.startswith("Average_")]
            outc = g.set_index("run_id")["outcome"]
            lmr = pd.DataFrame([pd.Series(0.0, index=lmc) if counts_as_zero(outc[i]) else lm.loc[i, lmc] for i in ids], index=ids).astype(float)
            run_means = run_means.join(lmr["Average_Accuracy"]).join(lmr.filter(like="ROI").mean(axis=1).rename("ROI_mean"))
        best = run_means.mean(axis=1).idxmax()  # one best run per model x arm, on all its paper metrics
        for i in ids:
            run_level.append({"dataset": ds, "model": model, "arm": arm, "run_id": i,
                              "outcome": g.set_index("run_id").loc[i, "outcome"], "is_best": i == best, **run_means.loc[i].to_dict()})
        for stat in ("median", "best"):
            label = f"AGENT:{model}:{arm}:{stat}"
            for m in GENE_METRICS:
                vals = stack[m].median(axis=1) if stat == "median" else stack[m][best]
                out_gene[m][label] = pd.Series(vals.to_numpy(), index=cols)
            if lmr is not None:
                out_lm[label] = lmr.median(axis=0) if stat == "median" else lmr.loc[best]
    meta = r.groupby(["model", "arm"]).outcome.value_counts().unstack(fill_value=0)
    return out_gene, out_lm, pd.DataFrame(run_level), meta


def validation_rows(ds: str, frame: str = FRAME):
    """Our re-runs of paper methods (h5ad_* files), scored by the same path."""
    gene, lm = {m: {} for m in GENE_METRICS}, {}
    for f in sorted((SCORES / ds).glob("h5ad_*_per_gene.csv")):
        name = f.name[len("h5ad_"):-len("_per_gene.csv")]
        label = ("BASELINE:" if name.startswith("identity") else "RERUN:") + name
        x = pd.read_csv(f)
        x = x[x.frame == frame].sort_values("gene_position")
        cols = paper_gene_table(ds, "PCC").columns
        for m in GENE_METRICS:
            gene[m][label] = pd.Series(x[m].to_numpy(), index=cols)
        lf = f.with_name(f.name.replace("_per_gene", "_landmark"))
        if lf.exists():
            l = pd.read_csv(lf)
            lm[label] = l[l.frame == frame].iloc[0][[c for c in l.columns if c.startswith("Average_")]].astype(float)
    return gene, lm


# ------------------------------------------------------------------ ranks
def collapse_staligner(f: pd.DataFrame, suffix: str) -> pd.DataFrame:
    f = f.copy()
    f.index = [i.replace(suffix, "") for i in f.index]
    st = [i for i in f.index if i.startswith("STAligner")]
    if st:
        best, worst = f.loc[st].max(), f.loc[st].min()
        f = f.drop(st)
        f.loc["STAligner_best"], f.loc["STAligner_worst"] = best, worst
    return f


def rank_with_insertion(paper: pd.DataFrame, agents: pd.DataFrame, gene_cols, lm_cols) -> pd.DataFrame:
    """Rank paper methods among themselves; rank each agent row against the paper methods only.

    Per metric: higher is better, ties share the lowest rank (the MOESM9 tables show shared
    ranks). Gene_based_overall / Landmark_based_overall: rank of the mean of the metric ranks,
    which reproduces how MOESM9's overall columns follow from its per-metric columns.
    """
    rows = []

    def ranks(frame):
        r = pd.DataFrame(index=frame.index)
        for c in gene_cols + lm_cols:
            r[c] = frame[c].rank(ascending=False, method="min")
        if gene_cols:
            r["Gene_based_overall"] = r[gene_cols].mean(axis=1).rank(method="min")
        if lm_cols:
            r["Landmark_based_overall"] = r[lm_cols].mean(axis=1).rank(method="min")
        return r

    base = ranks(paper)
    for i in base.index:
        rows.append({"row": i, "kind": "paper", **base.loc[i].to_dict()})
    for i in agents.index:
        rr = ranks(pd.concat([paper, agents.loc[[i]]]))
        rows.append({"row": i, "kind": "baseline" if str(i).startswith("BASELINE") else "agent", **rr.loc[i].to_dict()})
    return pd.DataFrame(rows)


def write_frame_sensitivity() -> None:
    """Same submissions scored as submitted versus moved into the canonical slice-1 pose."""
    rows = []
    for ds in ("dlpfc", "mhpr", "mob"):
        f = SCORES / ds / "agents_per_gene.csv"
        if not f.exists():
            continue
        g = pd.read_csv(f).groupby(["run_id", "frame"])[GENE_METRICS + ["canon_angle_deg"]].mean().unstack("frame")
        for m in GENE_METRICS:
            rows.append({"dataset": ds, "metric": m, "n_runs": len(g),
                         "mean_submitted": g[(m, "submitted")].mean(), "mean_canonical": g[(m, "canonical")].mean(),
                         "mean_abs_shift": (g[(m, "canonical")] - g[(m, "submitted")]).abs().mean(),
                         "max_abs_shift": (g[(m, "canonical")] - g[(m, "submitted")]).abs().max(),
                         "median_abs_canon_angle_deg": g[("canon_angle_deg", "submitted")].abs().median()})
    pd.DataFrame(rows).to_csv(OUT / "frame_sensitivity.csv", index=False)


def write_validation() -> None:
    """Our PASTE / PASTE_p0 re-runs against the paper's rows, gene by gene."""
    d = OUT / "validation"
    d.mkdir(exist_ok=True)
    pairs = [("dlpfc", "rerun_PASTE_shuffled", "PASTE_DLPFCS"), ("dlpfc", "rerun_PASTE_native", "PASTE_DLPFCS"),
             ("dlpfc", "rerun_PASTE2_shuffled", "PASTE2_DLPFCS"), ("dlpfc", "rerun_PASTE_p0_shuffled", "PASTE_p0_DLPFCS"),
             ("dlpfc", "rerun_PASTE_p0_native", "PASTE_p0_DLPFCS"), ("dlpfc", "identity_native", "PASTE_DLPFCS"),
             ("mhpr", "rerun_PASTE", "PASTE_MHPR"), ("mhpr", "rerun_PASTE_float64", "PASTE_MHPR")]
    gene_rows, summ = [], []
    for ds, lab, row in pairs:
        f = SCORES / ds / f"h5ad_{lab}_per_gene.csv"
        if not f.exists():
            continue
        lm = paper_landmark(ds)
        g = pd.read_csv(f)
        l = pd.read_csv(f.with_name(f"h5ad_{lab}_landmark.csv"))
        for fr in ("submitted", "canonical"):
            x = g[g.frame == fr].sort_values("gene_position").reset_index(drop=True)
            for m in GENE_METRICS:
                p = paper_gene_table(ds, m).loc[row].to_numpy()
                o = x[m].to_numpy()
                gene_rows.append(pd.DataFrame({"dataset": ds, "rerun": lab, "frame": fr, "paper_row": row, "metric": m,
                                               "gene_column": paper_gene_table(ds, m).columns, "ours": o, "paper": p, "diff": o - p}))
                summ.append({"dataset": ds, "rerun": lab, "frame": fr, "paper_row": row, "metric": m, "ours_mean": o.mean(), "paper_mean": p.mean(),
                             "mean_abs_diff": np.abs(o - p).mean(), "median_abs_diff": np.median(np.abs(o - p)),
                             "max_abs_diff": np.abs(o - p).max(), "frac_genes_within_0.01": float(np.mean(np.abs(o - p) < 0.01)),
                             "pearson_r": np.corrcoef(o, p)[0, 1]})
            ll = l[l.frame == fr].iloc[0]
            for c in lm.columns:
                summ.append({"dataset": ds, "rerun": lab, "frame": fr, "paper_row": row, "metric": c, "ours_mean": ll[c], "paper_mean": lm.loc[row, c],
                             "mean_abs_diff": abs(ll[c] - lm.loc[row, c]), "max_abs_diff": abs(ll[c] - lm.loc[row, c])})
    if gene_rows:
        pd.concat(gene_rows).to_csv(d / "rerun_vs_paper_per_gene.csv", index=False)
        pd.DataFrame(summ).to_csv(d / "rerun_vs_paper_summary.csv", index=False)
    sw = []
    for ds, sheet in SWEEP_FILES.items():
        vf = VALID / "sweeps" / f"{DATASETS[ds]['task']}_PASTE.csv"
        if vf.exists():
            v = pd.read_csv(vf)
            paper = pd.read_csv(PAPER / f"43588_2026_977_MOESM8_ESM/{sheet}.csv")["PASTE"].to_numpy()
            v = v.assign(table=sheet, paper_PASTE=paper[: len(v)], diff=v["Average_Accuracy"].to_numpy() - paper[: len(v)])
            sw.append(v)
    if sw:
        pd.concat(sw).to_csv(d / "robustness_PASTE_rerun_vs_paper.csv", index=False)


# ------------------------------------------------------------------ main
def main() -> None:
    global OUT, NONSPATIAL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonspatial", choices=["zero", "keep"], default="zero")
    NONSPATIAL = ap.parse_args().nonspatial
    if NONSPATIAL == "keep":
        OUT = OUT / "sensitivity_nonspatial_kept"
    OUT.mkdir(parents=True, exist_ok=True)
    runs = run_table()
    runs.to_csv(OUT / "agent_run_outcomes.csv", index=False)
    run_level_all, rank_tables, concord = [], [], []
    for ds in ("dlpfc", "mhpr", "mob"):
        if not (SCORES / ds / "agents_status.csv").exists():
            continue
        d = OUT / ds
        d.mkdir(exist_ok=True)
        _, name, suffix = GENE_FILES[ds]
        ag_gene, ag_lm, run_level, meta = agent_gene_rows(ds, runs)
        rv_gene, rv_lm = validation_rows(ds)
        run_level_all.append(run_level)
        meta.to_csv(d / f"{name}_agent_run_counts.csv")
        summary = {}
        for m in GENE_METRICS:
            paper = paper_gene_table(ds, m)
            rows = pd.concat([paper, pd.DataFrame(rv_gene[m]).T, pd.DataFrame(ag_gene[m]).T])
            rows.to_csv(d / f"{name}_GeneBased_{m}_with_agents.csv")
            summary[m] = rows.mean(axis=1)
        summ = pd.DataFrame(summary)
        plm = paper_landmark(ds)
        lm_cols = []
        if plm is not None:
            lrows = pd.concat([plm, pd.DataFrame(rv_lm).T, pd.DataFrame(ag_lm).T])[plm.columns]
            lrows.to_csv(d / f"{name}_LandmarkBased_Heatmap_with_agents.csv")
            summ["Average_Accuracy"] = lrows["Average_Accuracy"]
            summ["ROI_Overlap_Overall"] = lrows.filter(like="ROI").mean(axis=1)
            lm_cols = ["Average_Accuracy", "ROI_Overlap_Overall"]
        summ.index.name = "row"
        summ.to_csv(d / f"{name}_summary_means.csv")
        # ranks: paper methods (STAligner collapsed to best/worst as in MOESM9) + each agent (median rows)
        paper_part = collapse_staligner(summ.loc[[i for i in summ.index if not str(i).startswith(("AGENT", "RERUN", "BASELINE"))]], suffix)
        agent_part = summ.loc[[i for i in summ.index if str(i).startswith(("AGENT", "BASELINE"))]]
        rk = rank_with_insertion(paper_part, agent_part, GENE_METRICS, lm_cols)
        rk.insert(0, "dataset", name)
        rk.to_csv(d / f"{name}_sabench_style_rank.csv", index=False)
        rank_tables.append(rk)
        # headline: paper methods (STAligner collapsed) + identity baseline + agent medians/best, means and ranks
        means = pd.concat([paper_part, summ.loc[[i for i in summ.index if str(i).startswith(("AGENT", "BASELINE", "RERUN"))]]])
        overall = rk.set_index("row")[[c for c in ("Gene_based_overall", "Landmark_based_overall") if c in rk.columns]]
        head = means.join(overall)
        n = runs[(runs.dataset == ds) & (runs.outcome != "infra_excluded")].groupby(["model", "arm"]).outcome.agg(
            n_runs="size", n_zeroed=lambda o: int(sum(counts_as_zero(x) for x in o)))
        for stat in ("median", "best"):
            for (model, arm), v in n.iterrows():
                lab = f"AGENT:{model}:{arm}:{stat}"
                if lab in head.index:
                    head.loc[lab, "n_runs"], head.loc[lab, "n_zeroed"] = v.n_runs, v.n_zeroed
        head.insert(0, "dataset", name)
        head.rename_axis("row").to_csv(d / f"{name}_headline.csv")
        # concordance of single-dataset paper ranks with MOESM9 group ranks
        group = {"dlpfc": "NGS_datasets_rank", "mhpr": "Image_datasets_rank"}.get(ds)
        if group:
            R = pd.read_csv(PAPER / f"43588_2026_977_MOESM9_ESM/{group}.csv", index_col=0)
            p = rk[rk.kind == "paper"].set_index("row")
            common = [i for i in R.index if i in p.index]
            for c in [c for c in R.columns if c in p.columns]:
                concord.append({"dataset": name, "moesm9_table": group, "metric": c, "n_methods": len(common),
                                "kendall_tau": kendalltau(p.loc[common, c], R.loc[common, c])[0]})
    if run_level_all:
        pd.concat(run_level_all).to_csv(OUT / "agent_run_level_scores.csv", index=False)
    if rank_tables:
        pd.concat(rank_tables).to_csv(OUT / "sabench_style_rank_by_dataset.csv", index=False)
    pd.DataFrame(concord).to_csv(OUT / "rank_concordance_single_dataset_vs_moesm9.csv", index=False)
    write_validation()
    write_frame_sensitivity()

    # robustness sweeps (MOESM8): matching accuracy per level
    rob_summary = []
    for ds, sheet in SWEEP_FILES.items():
        f = SCORES / ds / "agents_sweep.csv"
        if not f.exists():
            continue
        paper = pd.read_csv(PAPER / f"43588_2026_977_MOESM8_ESM/{sheet}.csv")
        sw = pd.read_csv(f)
        sw = sw[sw.frame == FRAME]
        levels = sorted(sw.level.unique())
        paper.index = levels[: len(paper)]
        r = runs[(runs.dataset == ds) & (runs.outcome != "infra_excluded")]
        per_run = {}
        for _, row in r.iterrows():
            if not counts_as_zero(row.outcome):
                per_run[row.run_id] = sw[sw.run_id == row.run_id].set_index("level")["Average_Accuracy"].reindex(levels)
            else:
                per_run[row.run_id] = pd.Series(0.0, index=levels)
        cols = {}
        for (model, arm), g in r.groupby(["model", "arm"]):
            st = pd.DataFrame({i: per_run[i] for i in g.run_id})
            best = st.mean().idxmax()
            cols[f"AGENT:{model}:{arm}:median"] = st.median(axis=1)
            cols[f"AGENT:{model}:{arm}:best"] = st[best]
        vf = VALID / "sweeps" / f"{DATASETS[ds]['task']}_PASTE.csv"
        if vf.exists():
            v = pd.read_csv(vf).set_index("level")["Average_Accuracy"]
            cols["RERUN:PASTE"] = v.reindex(levels)
        table = pd.concat([paper, pd.DataFrame(cols)], axis=1)
        table.index.name = "level"
        table.to_csv(OUT / f"robustness_{sheet}_with_agents.csv")
        s = pd.DataFrame({"mean_accuracy": table.mean(), "sd_accuracy": table.std(), "n_levels": table.notna().sum()})
        s.insert(0, "table", sheet)
        rob_summary.append(s)
    if rob_summary:
        pd.concat(rob_summary).rename_axis("row").to_csv(OUT / "robustness_summary.csv")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
