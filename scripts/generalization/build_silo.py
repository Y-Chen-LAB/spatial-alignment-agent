#!/usr/bin/env python3
"""Build the siloed workspace a synthesis agent sees for one cell of the generalization experiment.

A cell is (split, scheme, fold, visibility):
  split       random | family                       (folds.json)
  scheme      train4: train on the other four folds, test on this one
              train1: train on this fold, test on the other four
  visibility  blind: runs without any score; scored: runs plus every run's and reference aligner's score

The silo is the agent's whole world (bound alone into the container, see run_synthesis.py):

  SYNTHESIS.md               the instructions
  training_tasks.txt
  runs/<task>/<run>/         bare runs of claude-opus-5-5 and gpt-6-astra on training tasks only:
                             trace.md, meta.json, solution/ (final code + SOLUTION.md)
  reference_aligners/        identity, centroid, PCA and ICP similarity aligners (code only)
  tasks/<task>/              empty mount points; the training tasks' blind inputs are bound read-only here
  scores/                    scored cells only
  output/                    where the agent writes policy/, PLAYBOOK.md, NOTES.md

In blind cells every score is stripped: score_* fields from meta.json, the score row of trace.md and the
score line of SOLUTION.md. Held-out tasks are absent altogether: no runs, no inputs, no names.

records     full (default): trace.md + meta.json + solution/ per run
            solution: solution/ plus a slim meta.json (who, which task, the final entry point; scores only in
            scored cells). No trace and no process stats: reading 315 traces is what made synthesis expensive.

    python3 scripts/generalization/build_silo.py --split family --scheme train4 --fold 0 --visibility blind
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FOLDS = REPO / "benchmarks/spatial_alignment/generalization/folds.json"
GEN = Path("/dcs07/chen/data/alignment-agent/generalization")
TRACES = Path("/dcs07/chen/data/alignment-agent/agent_traces")
INPUTS = GEN / "task_inputs"
MODELS = ("claude-opus-5-5", "gpt-6-astra")
SLIM_META = ("run_id", "model", "task", "family", "dataset", "status", "final_solution")
REFERENCE = ("identity", "centroid_similarity", "pca_similarity", "icp_similarity")
PRIMARY = "primary_score"


def cell_name(split: str, scheme: str, fold: int, visibility: str) -> str:
    return f"{split}-{scheme}-f{fold}-{visibility}"


def train_test(split: str, scheme: str, fold: int) -> tuple[list[str], list[str]]:
    folds = [f["tasks"] for f in json.loads(FOLDS.read_text())["splits"][split]["folds"]]
    this = folds[fold]
    rest = sorted(t for i, f in enumerate(folds) if i != fold for t in f)
    return (rest, this) if scheme == "train4" else (this, rest)


def _aggregate(path: Path) -> dict:
    try:
        sc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    agg = ((sc.get("scores") or {}).get("aggregate")) or sc.get("aggregate") or {}
    return {k: v for k, v in agg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def primary(agg: dict, identity_mae: float | None) -> float | None:
    """The score our tables report: MAE reduction against the identity alignment when the task has a
    coordinate truth (identity MAE >= 10), else the task's overall mean (label/expression agreement)."""
    if agg.get("mae") is not None and identity_mae and identity_mae >= 10:
        return 1 - agg["mae"] / identity_mae
    if agg.get("mae") is not None:
        return None
    return agg.get("overall_mean", agg.get("matching_accuracy"))


def strip_scores_text(text: str) -> str:
    keep = [ln for ln in text.splitlines()
            if not ln.startswith("| score |") and not ln.startswith("- overall score")]
    return "\n".join(keep) + "\n"


def build(split: str, scheme: str, fold: int, visibility: str, force: bool = False, records: str = "full",
          suffix: str = "") -> Path:
    name = cell_name(split, scheme, fold, visibility) + suffix
    cell = GEN / "synthesis" / name
    silo = cell / "silo"
    if silo.exists():
        if not force:
            raise SystemExit(f"{silo} exists; pass --force to rebuild")
        shutil.rmtree(silo)
    train, test = train_test(split, scheme, fold)
    train_set = set(train)
    scored = visibility == "scored"
    (silo / "output").mkdir(parents=True)
    (silo / "training_tasks.txt").write_text("\n".join(train) + "\n")

    # Runs: bare, scored (i.e. produced a submission), the two models, training tasks only.
    index = json.loads((TRACES / "index.json").read_text())
    run_rows = []
    for r in index:
        if r["model"] not in MODELS or r["condition"] != "bare" or r["status"] != "scored" or r["task"] not in train_set:
            continue
        src = TRACES / "runs" / r["run_id"]
        if not (src / "trace.md").exists():
            continue
        short = f"{r['model']}__{r['run_id'].split('__')[-1]}"
        dst = silo / "runs" / r["task"] / short
        (dst / "solution").mkdir(parents=True)
        meta = json.loads((src / "meta.json").read_text())
        if not scored:
            meta = {k: v for k, v in meta.items() if not k.startswith("score_") and k != "assigned_zero"}
        meta.pop("run_dir", None)
        if records == "solution":
            meta = {k: v for k, v in meta.items() if k in SLIM_META or k.startswith("score_")}
        (dst / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        if records == "full":
            trace = (src / "trace.md").read_text(errors="replace")
            (dst / "trace.md").write_text(trace if scored else strip_scores_text(trace))
        sol = src / "final_solution"
        if sol.is_dir():
            for f in sol.rglob("*"):
                if f.is_file():
                    out = dst / "solution" / f.relative_to(sol)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    text = f.read_text(errors="replace")
                    out.write_text(text if scored or f.name != "SOLUTION.md" else strip_scores_text(text))
        run_rows.append({"task": r["task"], "run": short, "model": r["model"],
                         **({PRIMARY: r.get("score_mae_reduction") if r.get("score_mae") is not None
                             else r.get("score_overall_mean", r.get("score_matching_accuracy"))} if scored else {}),
                         **({k: v for k, v in r.items() if k.startswith("score_") and k != "score_status"} if scored else {})})

    # Reference aligners: code only (no outputs, no scorer).
    ref = silo / "reference_aligners"
    (ref / "refalign").mkdir(parents=True)
    for mod in ("alignment.py", "h5ad.py", "io.py"):
        shutil.copy2(REPO / "src/genomics_agent_bench" / mod, ref / "refalign" / mod)
    (ref / "refalign" / "__init__.py").write_text("")
    shutil.copy2(Path(__file__).with_name("reference_aligners_cli.py"), ref / "run_reference.py")

    # Read-only mount points for the training tasks' blind inputs.
    for t in train:
        (silo / "tasks" / t).mkdir(parents=True)

    if scored:
        rows = []
        for t in train:
            ident = _aggregate(REPO / "results/baselines" / t / "identity" / "scores.json").get("mae")
            for m in REFERENCE:
                agg = _aggregate(REPO / "results/baselines" / t / m / "scores.json")
                if agg:
                    rows.append({"task": t, "aligner": m, PRIMARY: primary(agg, ident),
                                 **{f"score_{k}": v for k, v in agg.items()}})
        _write_csv(silo / "scores" / "reference_aligner_scores.csv", rows)
        _write_csv(silo / "scores" / "run_scores.csv", run_rows)
        (silo / "scores" / "SCORES.md").write_text(SCORES_MD)

    (silo / "SYNTHESIS.md").write_text(synthesis_md(scored, len(train), len(run_rows), records))
    binds = [f"{INPUTS / t}:{silo / 'tasks' / t}:ro" for t in train]
    (cell / "binds.json").write_text(json.dumps(binds, indent=1) + "\n")
    (cell / "cell.json").write_text(json.dumps({"cell": name, "split": split, "scheme": scheme, "fold": fold,
                                                 "visibility": visibility, "records": records, "train": train, "test": test,
                                                 "n_runs": len(run_rows)}, indent=1) + "\n")
    return cell


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys or ["task"])
        w.writeheader()
        w.writerows(rows)


SCORES_MD = """# Scores on the training tasks

These are the benchmark's scores for each prior run and each reference aligner, computed afterwards against
hidden ground truth that the runs never saw (and that you cannot see either). Higher `primary_score` is better.

- `primary_score`: for tasks with a coordinate ground truth, the reduction in mean absolute coordinate error
  relative to leaving the slices where they are (1 = perfect, 0 = no better than not aligning, negative =
  worse). For tasks without one, the mean of the label- and expression-agreement metrics (0 to 1).
- `score_mae`: mean absolute error of aligned coordinates against the truth (lower is better).
- `score_overall_mean`, `score_gene_*`, `score_label_*`, `score_matching_accuracy`: agreement of matched cells
  across aligned slices in labels and marker-gene expression (higher is better).

Files: `run_scores.csv` (one row per run in runs/), `reference_aligner_scores.csv` (one row per task and
reference aligner).
"""


TESTING_MD = """
How to test, so that the work stays quick:
- Test end to end on small and medium training tasks only, each run capped at 10 minutes of wall time
  (`timeout 600 bash output/policy/run.sh <workspace>`). Do not run the policy in full on the largest tasks
  (many sections or over a million cells): check that it scales by timing it on a subset of their sections
  and extrapolating, and rely on the fallback for the rest.
- Run each test as one blocking command in the foreground and read its result when it returns. Do not start
  it in the background and poll it with repeated `sleep` and check commands: every check is another full turn.

"""


def synthesis_md(scored: bool, n_tasks: int, n_runs: int, records: str = "full") -> str:
    evidence = (
        "`scores/` holds the benchmark score of every run and of every reference aligner on its training task "
        "(definitions in `scores/SCORES.md`). Use them to judge which strategies worked where."
        if scored else
        "No scores are given: you are not told how well any run or reference aligner did. Judge strategies from " +
        ("the traces themselves" if records == "full" else "the solutions themselves")
        + " and from your own diagnostics on the training inputs."
    )
    run_files = ("`trace.md` (the agent's steps, cleaned),\n  `meta.json` (run metadata), `solution/` (the code that "
                 "produced its final submission, with `SOLUTION.md`)." if records == "full" else
                 "`solution/` (the code that produced\n  its final submission, with `SOLUTION.md`) and `meta.json` "
                 "(model, task, entry point). The runs' step-by-step\n  traces are not included.")
    testing = TESTING_MD if records == "solution" else ""
    return f"""# Synthesize a general alignment strategy

## The job

Earlier AI agents (claude-opus-5-5 and gpt-6-astra) each solved spatial-transcriptomics slice-alignment tasks
on their own. You have their records for {n_tasks} tasks ({n_runs} runs). Your job is to **synthesize what
they did into one general strategy** for aligning slices, and to package it so it can be applied to **new
tasks you have not seen**. Those new tasks may come from other tissues, platforms and task designs.

Do not try to beat any one run on any one task. The goal is the strategy that generalizes best: which
approaches the runs converged on, when each one is the right choice, how to recognize that from the inputs,
and how to combine or choose among them (an ensemble or decision rule).

## What you have (and nothing else)

- `runs/<task>/<model>__<job>/`: one directory per prior run: {run_files}
- `tasks/<task>/`: each training task's inputs exactly as the agents saw them (`TASK.md`, `data/*.h5ad`,
  `agent_python.sh`). Read-only: to try something on a task, copy the task directory into `scratch/` first.
- `reference_aligners/`: four simple aligners (identity, centroid similarity, PCA similarity, ICP
  similarity). `python3 reference_aligners/run_reference.py --help` shows how to run them on a workspace.
- {evidence}

There is no ground truth and no scorer here. The network reaches only the model API and the Python package
index (PyPI); nothing else is reachable, so do not try to look things up online.

## What to deliver, under `output/`

1. `output/policy/run.sh`: the strategy as a program. It is run as `bash output/policy/run.sh <workspace>`,
   where `<workspace>` has the same layout as `tasks/<task>/` (a `TASK.md`, `data/*.h5ad`, `agent_python.sh`
   and an empty `submission/aligned_coordinates/`). It must write the coordinate CSVs that `TASK.md` asks
   for into `<workspace>/submission/aligned_coordinates/`.
   - Decide everything from `TASK.md` and the data. Never look up a task by its name or ID, and never
     hard-code a per-task answer: the tasks it will meet are not in your training set.
   - Keep the policy self-contained under `output/policy/`: copy in whatever code you reuse from the runs or
     from `reference_aligners/`.
   - Python: `<workspace>/agent_python.sh` gives numpy, scipy, pandas, scikit-learn and h5py. If you need other
     public packages, list them in `output/policy/requirements.txt`: before `run.sh` starts they are installed
     from PyPI into the environment that `<workspace>/agent_python.sh` runs. The policy directory itself is
     read-only when it runs; write intermediates under `<workspace>/` (for example `<workspace>/work/`).
   - Budget per task: 8 CPU cores, 64 GB of memory, and the wall time in seconds in the environment variable
     `POLICY_TIME_BUDGET_S` (3 hours for most tasks). Plan for the larger inputs: some tasks have more than
     100 sections or more than a million cells in total.
   - Always leave a complete, valid submission: if a step fails or runs out of time, fall back to a cheaper
     method rather than leaving files missing.
2. `output/PLAYBOOK.md`: the same strategy in prose, written for another agent who will solve a new task by
   hand: what the runs learned, the decision rules (which approach when, and how to tell from the inputs),
   checks that catch a bad alignment without ground truth, and the mistakes to avoid.
3. `output/NOTES.md`: how you got there: which runs and tasks support each rule, what you tried, and how you
   tested the policy (it should run end to end on several training tasks of different kinds).

Work in `scratch/` for experiments, and keep it under 20 GB (delete copied task data you no longer need).
{testing}When you finish, give a short summary of the strategy.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "family"], required=True)
    ap.add_argument("--scheme", choices=["train4", "train1"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--visibility", choices=["blind", "scored"], required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--records", choices=["full", "solution"], default="full")
    ap.add_argument("--suffix", default="", help="cell directory suffix, e.g. -sol, to keep a cell's earlier run")
    a = ap.parse_args()
    cell = build(a.split, a.scheme, a.fold, a.visibility, a.force, a.records, a.suffix)
    info = json.loads((cell / "cell.json").read_text())
    print(f"{info['cell']}: {len(info['train'])} training tasks, {info['n_runs']} runs, {len(info['test'])} test tasks -> {cell}")


if __name__ == "__main__":
    main()
