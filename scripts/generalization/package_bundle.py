#!/usr/bin/env python3
"""Zip the generalization (strategy synthesis) experiment: every cell's inputs, deliverables, traces and scores.

    python scripts/generalization/package_bundle.py [--tag YYYYMMDD]

Writes to /dcs07/chen/data/alignment-agent/generalization/bundle/:
  generalization_bundle_<tag>.zip       README, scores/, cells/<cell>/ (small, text)
  generalization_coordinates_<tag>.zip  the policies' aligned coordinates on the held-out tasks (large)

Per cell: cell.json, SYNTHESIS.md, training_tasks.txt, visible_runs.json (meta.json of every run the synthesizer
was shown), output/ (policy code, PLAYBOOK.md, NOTES.md), run/ (manifest, command, raw_stdout.jsonl trace,
raw_stderr, egress log), scratch_code/ (the synthesizer's own .py/.sh/.md/.json/.txt under 1 MB), and
attempts/<name>/ (earlier attempts: their run/ and output/). The silo's task inputs and data files are left out.
Text is scanned for credential patterns and redacted, as in raw_bundle.py.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import statistics as st
import zipfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEN = Path("/dcs07/chen/data/alignment-agent/generalization")
AGENT_RUNS = Path("/dcs07/chen/data/alignment-agent/agent_runs/cluster")
BASELINES = REPO / "results" / "baselines"
OUT = GEN / "bundle"
TEXT_EXT = {".py", ".sh", ".R", ".r", ".md", ".json", ".jsonl", ".txt", ".toml", ".yaml", ".yml", ".cfg"}
RUN_FILES = ["manifest.json", "command.json", "raw_stdout.jsonl", "raw_stderr.txt", "egress.jsonl"]
SECRET = re.compile(r"(sk-(?:proj-)?[A-Za-z0-9_\-]{20,}|AIza[0-9A-Za-z_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|"
                    r"ya29\.[0-9A-Za-z_\-]{20,}|(?i:(?:api[_-]?key|access[_-]?token|refresh[_-]?token|oauth[_-]?token)"
                    r"[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9_\-\.]{16,})")
EXCLUDED = ("sa-005", "sa-023")
SWEEPS = ("sa-040", "sa-041", "sa-042", "sa-043", "sa-403", "sa-407", "sa-408", "sa-409", "sa-410")
n_red = 0


def redact(b: bytes) -> bytes:
    global n_red
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return b
    s2, n = SECRET.subn("[REDACTED]", s)
    n_red += n
    return s2.encode("utf-8") if n else b


def add_file(z, src: Path, arc: str, limit=50_000_000):
    if src.is_file() and src.stat().st_size <= limit:
        z.writestr(arc, redact(src.read_bytes()))


def add_tree(z, root: Path, arc: str, text_only=True, limit=1_000_000):
    if not root.is_dir():
        return
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = Path(dp) / f
            if text_only and p.suffix not in TEXT_EXT:
                continue
            add_file(z, p, f"{arc}/{p.relative_to(root)}", limit)


def score(p: Path):
    try:
        return json.load(open(p))["scores"]["aggregate"]["overall_mean"]
    except Exception:
        return None


def agent_bare_means() -> dict:
    """Per (model, task): mean overall_mean over the scored bare replicates (done markers only)."""
    done = set(os.listdir(AGENT_RUNS / "done"))
    vals: dict = {}
    for j in os.listdir(AGENT_RUNS):
        if not j[0].isdigit():
            continue
        try:
            a, m, t, c, r = (AGENT_RUNS / j / "row.csv").read_text().strip().split(",")[:5]
        except Exception:
            continue
        if c != "bare" or f"{a}__{m}__{t}__{c}__rep{r}" not in done:
            continue
        best = None
        for s in os.listdir(AGENT_RUNS / j):
            d = AGENT_RUNS / j / s
            try:
                if json.load(open(d / "manifest.json")).get("status") == "scored":
                    best = score(d / "scores.json") if score(d / "scores.json") is not None else best
            except Exception:
                continue
        if best is not None:
            vals.setdefault((m, t), {})[r] = best
    return {k: st.mean(v.values()) for k, v in vals.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=date.today().strftime("%Y%m%d"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cells = sorted(d for d in os.listdir(GEN / "synthesis") if (GEN / "synthesis" / d / "cell.json").exists())

    # scores: one row per (held-out task, synth variant), with the source agents for reference
    agents = agent_bare_means()
    rows = []
    for t in sorted(os.listdir(BASELINES)):
        if not t.startswith("sa-") or t.startswith(EXCLUDED):
            continue
        for v in sorted(os.listdir(BASELINES / t)):
            if not v.startswith("synth_"):
                continue
            d = BASELINES / t / v
            try:
                man = json.load(open(d / "run_manifest.json"))
            except Exception:
                man = {}
            o, a = agents.get(("claude-opus-5-5", t)), agents.get(("gpt-6-astra", t))
            rows.append({"task": t, "variant": v, "cell": man.get("cell"), "split": man.get("split"),
                         "scheme": man.get("scheme"), "fold": man.get("fold"), "visibility": man.get("visibility"),
                         "status": man.get("status"), "runtime_seconds": man.get("runtime_seconds"),
                         "overall_mean": score(d / "scores.json"), "opus_bare": o, "astra_bare": a,
                         "pooled_bare": st.mean([o, a]) if o is not None and a is not None else None,
                         "sweep_task": t.startswith(SWEEPS)})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    summary = []
    for v in sorted({r["variant"] for r in rows}):
        R = [r for r in rows if r["variant"] == v and r["overall_mean"] is not None and r["pooled_bare"] is not None]
        if R:
            summary.append((v, len(R), st.mean(r["overall_mean"] for r in R), st.mean(r["pooled_bare"] for r in R)))

    readme = README.format(tag=args.tag, n_cells=len(cells), n_rows=len(rows), table="\n".join(
        f"| `{v}` | {n} | {p:.3f} | {q:.3f} | {p - q:+.3f} |" for v, n, p, q in summary))
    zp = OUT / f"generalization_bundle_{args.tag}.zip"
    with zipfile.ZipFile(f"{zp}.part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.md", readme)
        z.writestr("scores/synth_scores.csv", buf.getvalue())
        for f in ("submissions.log",):
            add_file(z, GEN / f, f"logs/{f}")
        add_tree(z, GEN / "rows", "logs/rows")
        for c in cells:
            cd, arc = GEN / "synthesis" / c, f"cells/{c}"
            add_file(z, cd / "cell.json", f"{arc}/cell.json")
            add_file(z, cd / "silo" / "SYNTHESIS.md", f"{arc}/SYNTHESIS.md")
            add_file(z, cd / "silo" / "training_tasks.txt", f"{arc}/training_tasks.txt")
            visible = []
            runs = cd / "silo" / "runs"
            if runs.is_dir():
                for m in sorted(runs.glob("*/*/meta.json")):
                    try:
                        visible.append(json.load(open(m)))
                    except Exception:
                        pass
            z.writestr(f"{arc}/visible_runs.json", json.dumps(visible, indent=1))
            add_tree(z, cd / "silo" / "output", f"{arc}/output")
            for f in RUN_FILES:
                add_file(z, cd / "run" / f, f"{arc}/run/{f}")
            add_tree(z, cd / "silo" / "scratch", f"{arc}/scratch_code")
            att = cd / "attempts"
            if att.is_dir():
                for a in sorted(att.iterdir()):
                    for f in RUN_FILES:
                        add_file(z, a / "run" / f, f"{arc}/attempts/{a.name}/run/{f}")
                    add_tree(z, a / "output", f"{arc}/attempts/{a.name}/output")
            print(c, flush=True)
    os.replace(f"{zp}.part", zp)
    print(f"wrote {zp} ({zp.stat().st_size / 1e6:.0f} MB), {n_red} redactions", flush=True)

    zc = OUT / f"generalization_coordinates_{args.tag}.zip"
    with zipfile.ZipFile(f"{zc}.part", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("README.md", "Aligned coordinates written by each synthesized policy on its held-out tasks:\n"
                   "`<task>/<variant>/aligned_coordinates/*.csv`, plus run_manifest.json, scores.json and policy_log.txt.\n"
                   f"Scores and cell names: generalization_bundle_{args.tag}.zip, scores/synth_scores.csv.\n")
        for r in rows:
            d = BASELINES / r["task"] / r["variant"]
            for f in ("run_manifest.json", "scores.json", "policy_log.txt"):
                add_file(z, d / f, f"{r['task']}/{r['variant']}/{f}")
            ac = d / "aligned_coordinates"
            if ac.is_dir():
                for p in sorted(ac.iterdir()):
                    z.write(p, f"{r['task']}/{r['variant']}/aligned_coordinates/{p.name}")
    os.replace(f"{zc}.part", zc)
    print(f"wrote {zc} ({zc.stat().st_size / 1e6:.0f} MB)", flush=True)
    (OUT / "README.md").write_text(readme)


README = """# Generalization experiment bundle (strategy synthesis)

Built {tag} by `scripts/generalization/package_bundle.py` (repo genomics-agent, branch `spatial-alignment`).

**Question.** Can the strategies of the bare Opus 5.5 and GPT-6-astra runs, pooled, be synthesized into one
program that works on tasks it never saw?

**Design.**
- 119 tasks (sa-005 and sa-023 excluded), split into 5 folds two ways: at random, and by task family.
- `train4`: train on 4 folds, test on the 5th. `train1`: train on 1, test on 4. train1 was on hold when this
  was built, so its cells are incomplete.
- `blind`: the synthesizer sees the training runs' solutions only. `scored`: it also sees their scores.
- The synthesizer is Claude Code (Opus 5.5) sealed in a silo: a container with only the silo mounted, a
  network jail whose one way out is an allowlist proxy (model API and PyPI, logged in `run/egress.jsonl`), and
  web, subagent, remote and scheduling tools disabled.
- It delivers `policy/run.sh`, `PLAYBOOK.md` and `NOTES.md`. The policy then runs on each held-out task
  directly, with no model, and is scored by the benchmark scorer.
- Cells named `*-sol` (and every train4 cell built after 2026-10-13) were synthesized from solutions and scores
  only, at effort medium. The other 8 family-train4 cells were built earlier from full traces at effort high,
  and are kept for comparison. Their results are the variants without `_sol`.

{n_cells} cells. {n_rows} applied (task, variant) rows in `scores/synth_scores.csv`.

## Results (held-out tasks; pooled = mean of the Opus and astra bare scores, each averaged over replicates)

| Variant | n | Policy | Pooled | Difference |
|---|---|---|---|---|
{table}

Random split: the policies roughly match Opus and sit about 0.02 below the pooled agents. Family split: they
match pooled on 4 of 5 families, and fall short on the 50 cross-platform pair tasks (family fold f0), where
no training example shows how to match across gene panels.

**Copy detection.** Several policies detect sections that share identical expression rows and solve
their relative transform exactly. This fires only on the 9 sweep and stitch tasks (`sweep_task` column), whose sections
are transformed copies of one section. The trick comes from the bare agents themselves, and it does not
inflate the results: dropping those tasks moves every policy-vs-pooled gap by 0.005 or less.

## Files

- `scores/synth_scores.csv`: task, variant, cell, split, scheme, fold, visibility, apply status, runtime,
  overall_mean, opus_bare, astra_bare, pooled_bare, sweep_task.
- `cells/<cell>/`: `cell.json` (folds, train and test tasks), `SYNTHESIS.md` (the synthesizer's
  instructions), `training_tasks.txt`, `visible_runs.json` (metadata of every run it was shown), `output/` (its
  deliverables), `run/` (manifest, command line, `raw_stdout.jsonl` = full trace, stderr, egress log),
  `scratch_code/` (its own scripts under 1 MB), and `attempts/<name>/` (earlier attempts, kept for the
  record; `authfail_*` = killed by an expired login, rerun from scratch).
- `logs/`: `submissions.log` (every array, resubmission and incident) and the row files each array ran.
- Coordinates: `generalization_coordinates_{tag}.zip` holds each policy's aligned coordinates, manifest, scores
  and policy log per held-out task.
"""

if __name__ == "__main__":
    main()
