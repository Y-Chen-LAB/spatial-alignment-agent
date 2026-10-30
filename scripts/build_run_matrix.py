"""Define the design subset and emit one row per planned agent run.

Two task sets. The breadth set is all 92 tasks and exists to buy statistical power: every
configuration runs on every task, so the contrast that matters is paired within task, and
task count is the cheapest way to shrink the detectable effect. The design subset of 45 is
for the arms too expensive to run everywhere, and is chosen to span every metric profile,
family and difficulty while including all nine tasks with published per-method numbers, so
at least part of every arm can be placed against the paper.
"""
from __future__ import annotations

import argparse, glob, json, tomllib
from datetime import datetime
from pathlib import Path
import pandas as pd

TASK_DIR = Path("benchmarks/spatial_alignment/tasks")

# Tasks SABench published per-method numbers for. Always in the design subset.
PAPER_NUMBERS = {
    "sa-001-dlpfc-serial-align", "sa-009-mhpr-merfish-serial-align",
    "sa-019-flysta3d-e14-16-3d-serial-align", "sa-027-openst-mln-full-19-slice-3d-align",
    "sa-030-mob-stereoseq-slideseqv2-cross-platform", "sa-040-dlpfc-rotation-sweep",
    "sa-041-dlpfc-crop-sweep", "sa-042-mhpr-rotation-sweep", "sa-043-mhpr-crop-sweep",
}

# Condition expands to the flags the runner already takes. Part 2 of the plan replaces this
# with a --condition flag; until then the driver passes these through.
CONDITIONS = {
    "equipped": {"prompt_variant": "full-package-paper-prior", "prebootstrap": "auto"},
    "bare":     {"prompt_variant": "full-basic",               "prebootstrap": "never"},
}


def load_tasks() -> pd.DataFrame:
    rows = []
    for tp in sorted(glob.glob(str(TASK_DIR / "*/task.toml"))):
        d = tomllib.load(open(tp, "rb"))
        rows.append({
            "task": d["id"], "profile": d["metric_profile"],
            "difficulty": d.get("difficulty", "medium"),
            "family": d["id"].split("-")[2] if len(d["id"].split("-")) > 2 else "other",
            "n_slices": len(d.get("slices", [])),
            "paper_numbers": d["id"] in PAPER_NUMBERS,
        })
    return pd.DataFrame(rows)


def measured_runtimes() -> tuple[dict[str, float], float]:
    """Median wall-clock minutes per task from the historical runs, and the overall median.

    Duration is not recorded in the manifest, so it is reconstructed from the run directory's
    start timestamp and the manifest's `finished_at`.
    """
    rows = []
    for manifest in glob.glob("agent_runs/local_coding_agent/*/manifest.json"):
        name = Path(manifest).parent.name
        try:
            finished = json.load(open(manifest)).get("finished_at")
            start = datetime.strptime("_".join(name.split("_")[-2:]), "%Y%m%d_%H%M%S")
            end = datetime.fromisoformat(finished.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        task = name.rsplit("_codex_", 1)[0].rsplit("_claude_", 1)[0].rsplit("_mini-swe_", 1)[0]
        minutes = (end - start).total_seconds() / 60
        if 0 < minutes < 360:
            rows.append({"task": task, "minutes": minutes})
    if not rows:
        return {}, 12.0
    frame = pd.DataFrame(rows).groupby("task").minutes.median()
    return frame.to_dict(), float(frame.median())


def shard_tasks(tasks: pd.DataFrame, subset: list[str]) -> dict[str, int]:
    """Split the work three ways so it can be spread across machines or people.

    Shards 1 and 2 are halves of the design subset, so either half alone is still a usable
    experiment. Shard 3 is everything outside the subset: the breadth-only tail that adds
    statistical power but carries no arm of its own.

    The weight is measured median wall-clock per task, not slice count. Across 526 timed
    historical runs the rank correlation between slice count and runtime is only 0.33: the
    129-slice WMB task finishes in about 11 minutes while a 6-slice MOp task takes 21, because
    agent time is reasoning turns rather than data volume. Balancing on slices produced a
    5-against-40 task split for no real gain. Tasks that have never run take the global median.

    Assignment is greedy longest-processing-time, with the nine paper-numbered tasks dealt
    first so neither half is left without something that can be placed against the paper.
    """
    runtimes, fallback = measured_runtimes()
    shard: dict[str, int] = {t: 3 for t in tasks.task}
    load = {1: 0.0, 2: 0.0}
    paper = [t for t in subset if t in PAPER_NUMBERS]
    rest = [t for t in subset if t not in PAPER_NUMBERS]

    for group in (paper, rest):
        for task in sorted(group, key=lambda t: -runtimes.get(t, fallback)):
            target = 1 if load[1] <= load[2] else 2
            shard[task] = target
            load[target] += runtimes.get(task, fallback)
    return shard


def design_subset(tasks: pd.DataFrame, n: int = 45) -> list[str]:
    """The nine paper-numbered tasks, then round-robin across (profile, family) strata.

    Round-robin rather than proportional, because the AMBC family is 50 of the 92 tasks and
    a proportional draw would spend most of the budget on one 10-slice dataset. Within a
    stratum the fewest-slices task is taken first, so the subset stays cheap to run.
    """
    chosen = [t for t in tasks.task if t in PAPER_NUMBERS]
    pool = tasks[~tasks.task.isin(chosen)].copy()
    strata = {k: list(v.sort_values("n_slices").task) for k, v in pool.groupby(["profile", "family"])}
    while len(chosen) < n and any(strata.values()):
        for key in sorted(strata):
            if len(chosen) >= n:
                break
            if strata[key]:
                chosen.append(strata[key].pop(0))
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--subset-size", type=int, default=45)
    ap.add_argument("--output", type=Path, default=Path("benchmarks/spatial_alignment/campaigns/run_matrix.csv"))
    args = ap.parse_args()

    tasks = load_tasks()
    subset = design_subset(tasks, args.subset_size)
    shard = shard_tasks(tasks, subset)
    all_tasks = sorted(tasks.task)

    # The active study compares the selected solver configurations under the equipped condition.
    # The Azure model name is a deployment name, so verify it with
    # scripts/check_azure_openai_access.py before launching that cell.
    # One factorial: condition x model x 3 reps on the design subset, every model an agent
    # through its native CLI. gpt-6-astra runs under both conditions on the nine tasks with
    # published per-method numbers. No harness cell and no one-shot cell; the harness
    # question (mini-swe) is deferred, not folded in.
    MAIN = [("claude", "claude-sonnet-4-6"), ("codex", "gpt-5.5"), ("gemini", "gemini-3.5-flash")]
    # gpt-6-astra: the nine paper-numbered tasks, the trial task, and every task in the
    # paired-priority list, so the presentation subset has all four models on every task.
    priority = set(Path("benchmarks/spatial_alignment/campaigns/paired_priority_tasks.txt").read_text().split())
    paper_tasks = [t for t in subset if t in PAPER_NUMBERS or t == "sa-018-starmap-mmpc-serial-align" or t in priority]
    cells = []
    for condition in ("equipped", "bare"):
        for agent, model in MAIN:
            cells.append(("main", agent, model, subset, condition, True))
        cells.append(("gpt6_select", "codex", "gpt-6-astra", paper_tasks, condition, True))

    rows = []
    for tp in sorted(glob.glob(str(TASK_DIR / "*/task.toml"))):
        d = tomllib.load(open(tp, "rb"))
        rows.append({
            "task": d["id"], "profile": d["metric_profile"],
            "difficulty": d.get("difficulty", "medium"),
            "family": d["id"].split("-")[2] if len(d["id"].split("-")) > 2 else "other",
            "n_slices": len(d.get("slices", [])),
            "paper_numbers": d["id"] in PAPER_NUMBERS,
        })
    return pd.DataFrame(rows)


def measured_runtimes() -> tuple[dict[str, float], float]:
    """Median wall-clock minutes per task from the historical runs, and the overall median.

    Duration is not recorded in the manifest, so it is reconstructed from the run directory's
    start timestamp and the manifest's `finished_at`.
    """
    rows = []
    for manifest in glob.glob("agent_runs/local_coding_agent/*/manifest.json"):
        name = Path(manifest).parent.name
        try:
            finished = json.load(open(manifest)).get("finished_at")
            start = datetime.strptime("_".join(name.split("_")[-2:]), "%Y%m%d_%H%M%S")
            end = datetime.fromisoformat(finished.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
        task = name.rsplit("_codex_", 1)[0].rsplit("_claude_", 1)[0].rsplit("_mini-swe_", 1)[0]
        minutes = (end - start).total_seconds() / 60
        if 0 < minutes < 360:
            rows.append({"task": task, "minutes": minutes})
    if not rows:
        return {}, 12.0
    frame = pd.DataFrame(rows).groupby("task").minutes.median()
    return frame.to_dict(), float(frame.median())


def shard_tasks(tasks: pd.DataFrame, subset: list[str]) -> dict[str, int]:
    """Split the work three ways so it can be spread across machines or people.

    Shards 1 and 2 are halves of the design subset, so either half alone is still a usable
    experiment. Shard 3 is everything outside the subset: the breadth-only tail that adds
    statistical power but carries no arm of its own.

    The weight is measured median wall-clock per task, not slice count. Across 526 timed
    historical runs the rank correlation between slice count and runtime is only 0.33: the
    129-slice WMB task finishes in about 11 minutes while a 6-slice MOp task takes 21, because
    agent time is reasoning turns rather than data volume. Balancing on slices produced a
    5-against-40 task split for no real gain. Tasks that have never run take the global median.

    Assignment is greedy longest-processing-time, with the nine paper-numbered tasks dealt
    first so neither half is left without something that can be placed against the paper.
    """
    runtimes, fallback = measured_runtimes()
    shard: dict[str, int] = {t: 3 for t in tasks.task}
    load = {1: 0.0, 2: 0.0}
    paper = [t for t in subset if t in PAPER_NUMBERS]
    rest = [t for t in subset if t not in PAPER_NUMBERS]

    for group in (paper, rest):
        for task in sorted(group, key=lambda t: -runtimes.get(t, fallback)):
            target = 1 if load[1] <= load[2] else 2
            shard[task] = target
            load[target] += runtimes.get(task, fallback)
    return shard


def design_subset(tasks: pd.DataFrame, n: int = 45) -> list[str]:
    """The nine paper-numbered tasks, then round-robin across (profile, family) strata.

    Round-robin rather than proportional, because the AMBC family is 50 of the 92 tasks and
    a proportional draw would spend most of the budget on one 10-slice dataset. Within a
    stratum the fewest-slices task is taken first, so the subset stays cheap to run.
    """
    chosen = [t for t in tasks.task if t in PAPER_NUMBERS]
    pool = tasks[~tasks.task.isin(chosen)].copy()
    strata = {k: list(v.sort_values("n_slices").task) for k, v in pool.groupby(["profile", "family"])}
    while len(chosen) < n and any(strata.values()):
        for key in sorted(strata):
            if len(chosen) >= n:
                break
            if strata[key]:
                chosen.append(strata[key].pop(0))
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--subset-size", type=int, default=45)
    ap.add_argument("--output", type=Path, default=Path("benchmarks/spatial_alignment/campaigns/run_matrix.csv"))
    args = ap.parse_args()

    tasks = load_tasks()
    subset = design_subset(tasks, args.subset_size)
    shard = shard_tasks(tasks, subset)
    all_tasks = sorted(tasks.task)

    # The active study compares the selected solver configurations under the equipped condition.
    # The Azure model name is a deployment name, so verify it with
    # scripts/check_azure_openai_access.py before launching that cell.
    cells = [
        ("A_breadth",       "claude",      "claude-sonnet-4-6", all_tasks, "equipped", True),
        ("A_breadth",       "gemini",      "gemini-3.5-flash",  all_tasks, "equipped", True),
        # The GPT arms run through the codex CLI pointed at the Azure deployments (no ChatGPT
        # login; the runner stages a provider config and the key), so they are agents in the
        # same sense as the claude and gemini arms, and B_replication uses the harness the
        # 531 historical runs used. gpt-5.3-codex is gone from the CLI's model cache but is
        # still deployed on Azure, which is the only reason that arm remains runnable.
        ("B_replication",   "codex",       "gpt-5.3-codex",     subset,    "equipped", True),
        ("B_generation",    "codex",       "gpt-5.5",           subset,    "equipped", True),
        ("B_generation",    "codex",       "gpt-6-astra",       subset,    "equipped", True),
        # The bare condition: no method hints in the prompt and nothing pre-installed, so the
        # agent chooses and installs its own tools. Two arms, one per vendor, on the subset:
        # the equipped-vs-bare contrast is paired within task and agent.
        ("C_bare",          "claude",      "claude-sonnet-4-6", subset,    "bare",     True),
        ("C_bare",          "codex",       "gpt-6-astra",       subset,    "bare",     True),
    ]

    rows = []
    for cell, agent, model, task_list, condition, implemented in cells:
        for task in task_list:
            for rep in range(1, args.reps + 1):
                rows.append({
                    "cell": cell, "agent": agent, "model": model, "task": task,
                    "condition": condition, "rep": rep,
                    "prompt_variant": CONDITIONS[condition]["prompt_variant"],
                    "prebootstrap": CONDITIONS[condition]["prebootstrap"],
                    "in_design_subset": task in subset,
                    "shard": shard[task],
                    "runnable_today": implemented,
                })
    m = pd.DataFrame(rows)
    m.insert(0, "run_index", range(len(m)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.output, index=False)

    Path("benchmarks/spatial_alignment/campaigns/design_subset.txt").write_text("\n".join(subset) + "\n")
    for s_id in (1, 2, 3):
        members = sorted(t for t, v in shard.items() if v == s_id)
        Path(f"benchmarks/spatial_alignment/shard_{s_id}_tasks.txt").write_text("\n".join(members) + "\n")

    print(f"design subset: {len(subset)} tasks ({sum(t in PAPER_NUMBERS for t in subset)} with paper numbers)")
    print(f"wrote {len(m)} runs to {args.output}\n")
    print(m.groupby(["cell", "agent", "model", "condition"]).size().to_string())
    runtimes, fallback = measured_runtimes()
    print("\nshards:")
    info = tasks.assign(shard=tasks.task.map(shard))
    for s_id, g in info.groupby("shard"):
        runs = m[m.task.isin(g.task)]
        hours = sum(runtimes.get(t, fallback) for t in runs.task) / 60
        print(f"  shard {s_id}: {len(g):2d} tasks  {len(runs):4d} runs  "
              f"~{hours:5.1f} agent-hours  {int(g.paper_numbers.sum())} with paper numbers")


if __name__ == "__main__":
    main()
