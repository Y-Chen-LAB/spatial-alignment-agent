#!/usr/bin/env python3
"""Write the bundle README and zip the cleaned traces, solutions and scores.

Bundle: /dcs07/chen/data/alignment-agent/agent_strategies_bundle/
  README.md, scores/, tables/ (if built)          loose, small
  agent_strategies_bundle_<date>.zip              README + scores + tables + runs/<run_id>/...
  tables.zip (with --tables-only)                 tables/ + README only

Index driven: the files of each run are known (trace.md, meta.json, steps.json, and the
final_solution files listed in meta.json), so nothing is scanned. While zipping, every text
file is checked again for credential patterns and redacted if one slipped through.
One process, Python zipfile. Splits by task family if the zip would pass ~2 GB.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import TRACES_ROOT, build as build_index  # noqa: E402
from extract_traces import SECRET_PATTERNS, redact  # noqa: E402

BUNDLE = Path("/dcs07/chen/data/alignment-agent/agent_strategies_bundle")
RUNS = TRACES_ROOT / "runs"
SPLIT_BYTES = 1_900_000_000


def run_files(run_id: str) -> list[Path]:
    d = RUNS / run_id
    out = [d / "trace.md", d / "meta.json", d / "steps.json", d / "final_solution" / "SOLUTION.md"]
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        return [p for p in out if p.exists()]
    for f in (meta.get("final_solution") or {}).get("copied") or []:
        out.append(d / "final_solution" / f.split("  (")[0])
    return [p for p in out if p.exists()]


def readme(rows, stats) -> str:
    n = Counter(r["status"] for r in rows)
    by = Counter((r["agent"], r["status"]) for r in rows)
    sc = Counter((r["model"], r["condition"]) for r in rows if r["status"] == "scored")
    models = sorted({m for m, _ in sc})
    tab = "\n".join(f"| {m} | {sc.get((m, 'bare'), 0)} | {sc.get((m, 'equipped'), 0)} |" for m in models)
    hows = stats["how"]
    return f"""# Agent strategies bundle (spatial alignment)

Built {date.today().isoformat()} from the genomics-agent repo, branch `spatial-alignment`,
by `scripts/agent_strategies/` (build_index.py, extract_traces.py, export_scores.py,
package_bundle.py; tag_strategies.py and build_tables.py for tables/).

## Which runs

All runs under `agent_runs/cluster/<array_task>/<run_dir>/`. `agent_runs/superseded/` and
`agent_runs/cluster_void/` are excluded entirely. Of {len(rows)} run directories:
{n.get('scored', 0)} scored, {n.get('failed', 0)} failed, {n.get('no_manifest', 0)} never finished
(no manifest; listed in scores/agent_scores.csv with status `no_manifest`, no trace extracted),
{n.get(None, 0)} with an empty status. Every run with a manifest has a `runs/<run_id>/` folder,
failed ones included.

By agent CLI and status: {', '.join(f'{a} {s}: {v}' for (a, s), v in sorted(by.items(), key=str))}.

Scored runs by model and condition (bare = prompt variant `full-basic`; equipped =
`full-package-paper-prior`, which exposes the published-method environments):

| model | bare | equipped |
|---|---|---|
{tab}

`run_id` = `<run_dir>__<array_task>`. The run_dir name alone is not unique: replicate runs launched
in the same second share it ({stats['dup_names']} names occur more than once).

## Files

### runs/<run_id>/
- `trace.md`: cleaned transcript. Header: model, condition, task, family, status, score
  (overall_mean, mae, mae_reduction, matching_accuracy), wall time, token totals from the CLI,
  cost, step and submission-write counts, final entry point. Body: numbered steps, each with a
  UTC timestamp and minutes since the first event (Claude and Gemini; Codex logs no timestamps),
  an estimated count of generated tokens (characters / 4; the CLIs do not report per-step
  counts reliably), the agent's own messages in full, tool calls trimmed to 40 lines and tool
  results trimmed to 15 lines. Steps that wrote `submission/aligned_coordinates` are flagged.
  The task prompt is not repeated (it is the same for all runs of a task and condition).
- `final_solution/`: the code that wrote the submitted coordinates. `SOLUTION.md` names the
  entry point, the last command that wrote the submission and why each other file is included
  (local helper it imports, or an upstream stage script that writes an intermediate the entry
  loads). Code only (.py/.R/.sh, each < 1 MB); never .h5ad/.npz/.pkl/.csv. When the final write
  was inline Python run from the shell, the code is saved as `stepNNN_inline.py`.
  For failed runs it holds whatever code was found by the same rules (often nothing).
- `meta.json`: run_id, run_dir, array task, agent, model, reasoning_effort, condition,
  prompt_variant, task, family, dataset, status, failure_stage, agent_returncode,
  failure_message (short), started_at, finished_at, elapsed_seconds, leakage_status, all
  `score_*` aggregates, `exploration` (below), `final_solution` (entry, how, step, files, note),
  parse_ok (the parser found at least one tool call), snapshot_code_files, scripts_written;
  after tagging also `tags`/`tags_final` (strategy keywords hit anywhere / in the final code).
- `steps.json`: the normalized step list (kind text/thinking/tool, ts, tool name, full command or
  written file content up to 200k chars, output trimmed to 60 lines, writes_submission,
  ran_scripts). Machine-readable input for the strategy tagger.

`exploration` fields: n_steps, n_tool_calls, n_tool_errors, n_text_msgs, n_scripts_written (code
files written through the trace or newer than the run start in the snapshot), n_script_runs,
n_inline_code_runs, n_submission_writes, n_submission_rewrites, t_first_submission_s (+ method:
`timestamp`, or `step-fraction-of-wall` for Codex = wall time x step index / tool calls),
wall_s, input/cached/output/reasoning tokens, cost_usd, n_web_search, n_subagent_steps.

### scores/
- `agent_scores.csv`: one row per indexed agent run (all statuses), with every aggregate metric
  as `score_<metric>` (overall_mean, mae, mae_reduction, matching_accuracy, gene_pcc, ...),
  tokens, cost and elapsed_seconds. Filter `status == "scored"` for the analysed set.
- `baseline_scores.csv`: every non-agent method result present on {date.today().isoformat()} under
  `results/baselines/<task>/<variant>/scores.json` (method = published method family, variant =
  the run setting), with `snapshot_date`. The baselines campaign is still running; rerun
  `python scripts/agent_strategies/export_scores.py` then `package_bundle.py` to refresh.
- `paper_reported.csv`: copy of `results/paper_baselines/geaso_2026.csv` (GEASO paper source data
  mapped to our task ids). `PAPER_SOURCE_DATA.md` points to the SABench source-data CSVs in
  `references/paper_source_data/s43588-026-00977/` (not copied).

### tables/ (if present)
Strategy tables (exploration space, final approach, keyword definitions) as long CSV and LaTeX;
see tables/README.md.

## How final_solution was identified

For each tool step the parser records which local scripts it ran (`python X.py`,
`agent_python.sh X.py`, `uv run python X.py`, ...) and whether it wrote the submission: the
script (or a helper function it calls) contains a CSV write (`to_csv`, `savetxt`, `csv.writer`,
...) together with `aligned_coordinates`/`submission`; or inline Python in the command does; or
the command copies files into `submission/aligned_coordinates`. The last successful such step
gives the entry point. Copy steps are traced back to the script producing the copied files.
Scored runs by method: {', '.join(f'{a} {h}: {v}' for (a, h), v in sorted(hows.items(), key=str))}.
`fallback-mtime` = no writing command matched, so the newest snapshot script that writes
coordinates was taken. {stats['recovered']} solutions include a file recovered from the trace
(written outside the workspace, e.g. /tmp, so missing from the snapshot).

## Cleaning applied to traces
- ANSI escape codes removed; progress-bar redraws (`\\r`) reduced to the last frame.
- Only exact duplicate consecutive lines and runs of progress/training-log lines (tqdm `%|`,
  `it/s`, `Epoch k/N`, `loss=`) are collapsed; everything else is kept and long outputs are
  trimmed by line count (head + tail).
- Tokens of 200+ base64-like characters are dropped (`[N-char blob dropped]`); lines over 400
  characters are cut.
- Workspace paths `/tmp/p2b_workspaces/<job>/<task>` are shown as `$WS`.
- Credentials: OpenAI-style `sk-...`, Google `AIza...`, OAuth `ya29.`/`1//0` tokens, GitHub/Slack
  tokens, AWS keys, JWTs, `Bearer ...` and `api_key=`/`token=`/`password=` assignments are
  redacted in traces, steps.json and copied code; the packager re-checks every file
  ({stats['secret_hits']} hits at packaging time).
- Not included: the system/init events (tool lists, session ids, sockets), rate-limit events,
  the task prompt, data files.

## Caveats
- Codex `file_change` events do not carry patch contents, so intermediate versions of Codex
  scripts are lost; only the final snapshot version exists. Codex has no per-event timestamps.
- Gemini assistant text arrives as streamed deltas and is concatenated between tool calls.
- {stats['claude_no_tool']} failed Claude runs have no tool calls at all (the agent stopped at
  an API/agent error before acting); their traces are short.
- The final-solution finder is heuristic (regex on commands and code). Pipelines passing data
  between scripts through files are followed by file-name matching only, so an upstream stage
  can be missed or an unrelated one included. SOLUTION.md says what was matched.
- Per-step token counts in trace.md are estimates; use meta.json/agent_scores.csv totals.
- `workspace_snapshot/scores.json` is written by the harness after the run; agents never saw it.
"""


def zip_runs(zpath: Path, members: list[tuple[Path, str]], stats):
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in members:
            if isinstance(src, str):  # literal content
                z.writestr(arc, src)
                continue
            data = src.read_text(errors="replace")
            if any(p.search(data) for p in SECRET_PATTERNS):
                red = redact(data)
                if red != data:
                    stats["secret_hits"] += 1
                    data = red
            z.writestr(arc, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-only", action="store_true")
    a = ap.parse_args()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    rows = build_index()
    if a.tables_only:
        tdir = BUNDLE / "tables"
        names = json.loads((tdir / "MANIFEST.json").read_text())
        zip_runs(BUNDLE / "tables.zip", [(tdir / f, f"tables/{f}") for f in names], {"secret_hits": 0})
        print(BUNDLE / "tables.zip", (BUNDLE / "tables.zip").stat().st_size)
        return
    stats = {"secret_hits": 0, "how": Counter(), "recovered": 0, "claude_no_tool": 0,
             "dup_names": sum(1 for v in Counter(r["run_dir"] for r in rows).values() if v > 1)}
    members = []
    fam_size = Counter()
    for r in rows:
        if r["status"] == "no_manifest":
            continue
        try:
            meta = json.loads((RUNS / r["run_id"] / "meta.json").read_text())
        except Exception:
            continue
        f = meta.get("final_solution") or {}
        if r["status"] == "scored":
            stats["how"][(r["agent"], f.get("how") if (f.get("entry") or f.get("inline_file")) else "none")] += 1
            stats["recovered"] += any("recovered" in x for x in f.get("copied") or [])
        if r["status"] == "failed" and r["agent"] == "claude" and not meta.get("parse_ok"):
            stats["claude_no_tool"] += 1
        for p in run_files(r["run_id"]):
            rel = p.relative_to(RUNS)
            members.append((p, f"runs/{rel}", r.get("family") or "other"))
            fam_size[r.get("family") or "other"] += p.stat().st_size
    # scan first so the README reports the hit count, then write README and zip
    for p, _, _ in members:
        t = p.read_text(errors="replace")
        if any(pt.search(t) for pt in SECRET_PATTERNS) and redact(t) != t:
            stats["secret_hits"] += 1
    text = readme(rows, stats)
    (BUNDLE / "README.md").write_text(text)
    common = [(BUNDLE / "README.md", "README.md")]
    for sub in ("scores", "tables"):
        mf = BUNDLE / sub / "MANIFEST.json"
        names = json.loads(mf.read_text()) if mf.exists() else (
            ["agent_scores.csv", "baseline_scores.csv", "paper_reported.csv", "PAPER_SOURCE_DATA.md"] if sub == "scores" else [])
        common += [(BUNDLE / sub / n, f"{sub}/{n}") for n in names if (BUNDLE / sub / n).exists()]
    total = sum(fam_size.values())
    tag = date.today().strftime("%Y%m%d")
    zstats = {"secret_hits": 0}
    if total < SPLIT_BYTES * 4:  # text compresses ~5x; one zip stays well under 2 GB
        zp = BUNDLE / f"agent_strategies_bundle_{tag}.zip"
        zip_runs(zp, common + [(p, arc) for p, arc, _ in members], zstats)
        out = [zp]
    else:
        out = []
        for fam in sorted(fam_size):
            zp = BUNDLE / f"agent_strategies_bundle_{tag}_{fam}.zip"
            zip_runs(zp, common + [(p, arc) for p, arc, fm in members if fm == fam], zstats)
            out.append(zp)
    print(f"{len(members)} run files ({total / 1e6:.0f} MB raw); secret hits redacted: {zstats['secret_hits']}")
    for zp in out:
        print(zp, f"{zp.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
