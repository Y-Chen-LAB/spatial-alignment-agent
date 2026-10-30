#!/usr/bin/env python3
"""Create observable trace-vignette table and timeline figure for spatial runs."""

from __future__ import annotations

import json
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_DIR = ROOT / "agent_runs" / "analysis" / "spatial_post_run_structure"
FIG_DIR = STRUCTURE_DIR / "figures"
MANUSCRIPT_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark"
MANUSCRIPT_FIG_DIR = MANUSCRIPT_DIR / "figures"
MANUSCRIPT_TABLE_DIR = MANUSCRIPT_DIR / "tables"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
INVENTORY = STRUCTURE_DIR / "structural_run_inventory.csv"
PROMPT_TRACE_SUMMARY = STRUCTURE_DIR / "prompt_trace_summary.csv"

PACKAGE_PATTERNS = [
    "paste2",
    "paste",
    "spateo",
    "spacel",
    "staligner",
    "stalign",
    "cast",
    "scslat",
    "gpsa",
    ".method-venv",
]

STAGE_ORDER = [
    "data_inspection",
    "baseline_write",
    "package_probe",
    "failure_or_blocker",
    "custom_alignment",
    "self_evaluation",
]
STAGE_LABELS = {
    "data_inspection": "Inspect data",
    "baseline_write": "Valid baseline",
    "package_probe": "Package/API probe",
    "failure_or_blocker": "Failure/stall",
    "custom_alignment": "Custom align",
    "self_evaluation": "Self-check",
}

TRACE_TABLE_SIGNALS = {
    "baseline-insurance": "Valid baseline first; risky refinement failed; lighter custom fallback validated.",
    "slow-package-fallback": "Package attempt stalled; process was killed; validated fallback preserved the run.",
    "geometry-expression-check": "Geometry search was checked against expression consistency before submission.",
    "large-multislice": "Long multi-slice custom alignment completed with all outputs validated.",
    "package-backed-selection": "Several packages were probed; Spateo was selected by input-only diagnostics.",
    "package-error-recovery": "Package API/data issues were patched, then a slow rerun was stopped.",
    "provider-disconnect-zero": "Substantial package-backed work completed, but provider failure zeroed the endpoint.",
}
STAGE_COLORS = {
    "data_inspection": "#1F5AA6",
    "baseline_write": "#E6AB02",
    "package_probe": "#1B9E77",
    "failure_or_blocker": "#B55D4C",
    "custom_alignment": "#7570B3",
    "self_evaluation": "#D95F02",
}

CASE_SPECS = [
    {
        "case_id": "baseline-insurance",
        "family": "Baseline insurance",
        "run_name": "sa-001-dlpfc-serial-align_codex_gpt-5.3-codex_20260427_113325",
        "decision": (
            "Wrote a complete valid raw-coordinate submission before trying refinements; "
            "after an unstable expression-guided trial, reran a lighter rigid-ICP path and validated outputs."
        ),
        "evidence": (
            "Visible trace: baseline-on-disk status, raw/centroid/ICP/expression comparisons, "
            "runtime failure recovery, final row-order and finite-coordinate checks."
        ),
        "message_terms": ["valid baseline", "baseline on disk", "expression-guided", "lighter", "validated"],
        "status_event": 27,
        "command_event": 29,
        "command_terms": ["transform", "wrote", "valid"],
    },
    {
        "case_id": "slow-package-fallback",
        "family": "Slow package fallback",
        "run_name": "sa-001-dlpfc-serial-align_codex_gpt-5.3-codex_20260426_162015",
        "decision": (
            "Started with PASTE as a stronger method, observed long-running OT jobs, then killed stalled "
            "experiments and kept the validated submission rather than losing the run."
        ),
        "evidence": (
            "Visible trace: PASTE bootstrap and API introspection, full pairwise runs killed after stalling, "
            "subsampled transport attempt, final output validation."
        ),
        "message_terms": ["paste", "stronger alignment", "still", "long-running", "terminated", "validated"],
        "status_event": 63,
        "command_event": 61,
        "command_terms": ["numItermax", "optimality", "paste"],
    },
    {
        "case_id": "geometry-expression-check",
        "family": "Geometry + expression check",
        "run_name": "sa-003-dlpfc-coarse-shuffled-pair_codex_gpt-5.3-codex_20260426_160506",
        "decision": (
            "Searched rotations/reflections with ICP, then used expression-space consistency to reject "
            "plausible-but-wrong orientations before writing the final pair alignment."
        ),
        "evidence": (
            "Visible trace: geometry-only registration search, near-180-degree solution, expression "
            "validation confirming the same transform, exact output-order validation."
        ),
        "message_terms": ["geometry-only", "180", "expression-space", "validation confirms", "obs_id"],
        "status_event": 19,
        "command_event": 15,
        "command_terms": ["best chamfer", "init 180", "scale"],
    },
    {
        "case_id": "large-multislice",
        "family": "Large multi-slice execution",
        "run_name": "sa-020-mop-mouse1-atlas-serial-align_codex_gpt-5.3-codex_20260426_224822",
        "decision": (
            "For a 33-slice task, reused an in-workspace serial aligner, monitored the long run, "
            "and validated all required outputs after a large internal consistency gain."
        ),
        "evidence": (
            "Visible trace: complete baseline instruction, 33 input slices, expression-cluster centroids "
            "plus ICP, adjacent median-NN distance 4740.3 to 12.87, 33 output checks."
        ),
        "message_terms": ["complete valid baseline", "stronger one", "33", "4740.3", "12.87", "validated"],
        "status_event": 21,
        "command_event": 19,
        "command_terms": ["Loaded", "Raw adjacent", "Aligned adjacent", "Wrote"],
    },
    {
        "case_id": "package-backed-selection",
        "family": "Package-backed selection",
        "run_name": "sa-008-ambc-xenium-multisection-serial-align_codex_gpt-5.3-codex_pv-full-package-paper-prior_20260430_031411",
        "decision": (
            "Compared identity, custom rigid, PASTE, PASTE2 subset, and Spateo candidates; selected "
            "Spateo because its allowed-input diagnostic was best and all full-slice outputs validated."
        ),
        "evidence": (
            "Visible trace: multiple method-env probes, PASTE worse than identity, PASTE2 subset marked "
            "costly, Spateo affine selected in the method-attempt report."
        ),
        "message_terms": ["valid baseline", "spateo", "paste", "PASTE2", "completed", "report"],
        "status_event": 69,
        "command_event": 83,
        "command_terms": ["identity", "spateo_affine", "selected_by_internal_metric"],
    },
    {
        "case_id": "package-error-recovery",
        "family": "Package error recovery",
        "run_name": "sa-006-mbsa-serial-align_codex_gpt-5.3-codex_pv-full-package-paper-prior_20260430_030053",
        "decision": (
            "Tried PASTE/PASTE2 in the OT environment, patched shared-gene and duplicate-name issues, "
            "then stopped a too-slow OT rerun and kept a valid fallback submission."
        ),
        "evidence": (
            "Visible trace: incompatible gene dimensions, duplicate var-name errors, long OT optimization, "
            "process cleanup, method-attempt notes, final output validation."
        ),
        "message_terms": ["different gene sets", "duplicate", "taking longer", "stop", "valid baseline", "report"],
        "status_event": 33,
        "command_event": 57,
        "command_terms": ["paste_pairwise", "paste2_partial", "selected"],
    },
    {
        "case_id": "provider-disconnect-zero",
        "family": "Provider disconnect after work",
        "run_name": "sa-105-mop-mouse1sample5-serial-align_codex_gpt-5.3-codex_pv-full-package-paper-prior_20260502_001321",
        "decision": (
            "Created baseline outputs, probed Spateo/SPACEL/PASTE2, completed a long Spateo run and "
            "PASTE2 subset/full-pair probes, but the run was zeroed after a provider/websocket failure."
        ),
        "evidence": (
            "Visible trace: method-env API probes, Spateo finished after 2592 seconds, PASTE2 improved "
            "subset/full-pair NN distances, reconnect/DNS errors before final completion."
        ),
        "message_terms": ["Baseline", "spateo", "PASTE2", "2592", "Reconnecting", "full pair"],
        "status_event": 62,
        "command_event": 48,
        "command_terms": ["finished [2592", "wrote", "saved"],
        "stderr_terms": ["failed to connect to websocket"],
    },
]


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    font_chain = [
        "Avenir Next LT Pro",
        "Avenir",
        "Avenir Next",
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 16,
            "font.weight": "bold",
            "axes.labelsize": 18,
            "axes.titlesize": 19,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.linestyle": "-",
            "grid.alpha": 0.12,
            "grid.color": "#4D4D4D",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.labelsize": 13,
            "ytick.labelsize": 12,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def apply_bold_font(fig: plt.Figure) -> None:
    if not AVENIR_BOLD_PATH.exists():
        return
    prop_cache: dict[float, font_manager.FontProperties] = {}
    for text in fig.findobj(match=mtext.Text):
        size = float(text.get_fontsize())
        if size not in prop_cache:
            prop_cache[size] = font_manager.FontProperties(fname=str(AVENIR_BOLD_PATH), size=size)
        text.set_fontproperties(prop_cache[size])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def command_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        records.append(
            {
                "event_index": index,
                "command": item.get("command") or "",
                "output": item.get("aggregated_output") or "",
                "exit_code": item.get("exit_code"),
            }
        )
    return records


def agent_messages(events: list[dict[str, Any]]) -> list[str]:
    messages: list[str] = []
    for event in events:
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                messages.append(text)
    return messages


def agent_message_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = str(item.get("text") or "").strip()
            if text:
                records.append({"event_index": index, "text": text})
    return records


def trace_tags(command: str, output: str, exit_code: Any) -> set[str]:
    text = f"{command}\n{output}".lower()
    tags: set[str] = set()
    if any(token in text for token in ["task.md", "data/", ".h5ad", "h5py", "obsm", "ls -la data"]):
        tags.add("data_inspection")
    if "submission/aligned_coordinates" in text and any(
        token in text for token in ["to_csv", "write", "wrote", "baseline", "identity"]
    ):
        tags.add("baseline_write")
    if ".method-venv" in text or any(pkg in text for pkg in PACKAGE_PATTERNS):
        tags.add("package_probe")
    if exit_code not in (0, None) or any(
        token in text
        for token in [
            "traceback",
            "modulenotfounderror",
            "importerror",
            "killed",
            "timeout",
            "timed out",
            "operation not permitted",
            "error:",
            "failed",
        ]
    ):
        tags.add("failure_or_blocker")
    if any(
        token in text
        for token in [
            "ckdtree",
            "nearestneighbors",
            "procrustes",
            "affine",
            "pca",
            "truncatedsvd",
            "svd",
            "icp",
            "phasecorr",
        ]
    ):
        tags.add("custom_alignment")
    if any(token in text for token in ["diagnostic", "evaluate", "score", "metric", "report.json", "compare"]):
        tags.add("self_evaluation")
    return tags


def inline_excerpt(text: str, max_chars: int = 220) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def raw_lines_matching(text: str, terms: list[str], max_chars: int = 300) -> str:
    normalized = text.replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if terms:
        lowered_terms = [term.lower() for term in terms]
        hits = [line for line in lines if any(term in line.lower() for term in lowered_terms)]
    else:
        hits = lines[:3]
    if not hits:
        hits = lines[:3]
    return inline_excerpt(" | ".join(hits), max_chars=max_chars)


def stderr_excerpt(run_dir: Path, terms: list[str], max_chars: int = 260) -> str:
    if not terms:
        return ""
    path = run_dir / "raw_stderr.txt"
    if not path.exists():
        return ""
    return raw_lines_matching(path.read_text(encoding="utf-8", errors="replace"), terms, max_chars=max_chars)


def clean_message(text: str, max_chars: int = 180) -> str:
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def selected_messages(messages: list[str], terms: list[str], max_items: int = 3) -> list[str]:
    hits: list[str] = []
    lowered = [term.lower() for term in terms]
    for message in messages:
        msg_l = message.lower()
        if any(term in msg_l for term in lowered):
            short = clean_message(message)
            if short not in hits:
                hits.append(short)
        if len(hits) >= max_items:
            break
    if not hits:
        hits = [clean_message(m) for m in messages[:max_items]]
    return hits


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "`": r"\textasciigrave{}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def short_task(task_id: str) -> str:
    match = re.match(r"sa-(\d+)", task_id)
    if match:
        return f"SA-{match.group(1)}"
    return task_id.replace("sa-", "SA-")


def short_prompt_label(prompt_label: str) -> str:
    if prompt_label == "Full package + paper prior":
        return "Full + prior"
    return prompt_label


def build_case_table(inventory: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    inv = inventory.set_index("run_name", drop=False)
    rows: list[dict[str, Any]] = []
    timeline: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []

    for spec in CASE_SPECS:
        run_name = spec["run_name"]
        if run_name not in inv.index:
            missing.append(run_name)
            continue
        row = inv.loc[run_name]
        run_dir = ROOT / str(row["run_dir"]) if not str(row["run_dir"]).startswith("/") else Path(str(row["run_dir"]))
        events = read_jsonl(run_dir / "raw_stdout.jsonl")
        commands = command_records(events)
        messages = agent_messages(events)
        messages_by_event = {record["event_index"]: record["text"] for record in agent_message_records(events)}
        commands_by_event = {record["event_index"]: record for record in commands}
        command_counter: Counter[str] = Counter()
        command_points: list[dict[str, Any]] = []
        for cmd_idx, record in enumerate(commands):
            tags = trace_tags(record["command"], record["output"], record["exit_code"])
            for tag in sorted(tags):
                command_counter[tag] += 1
                command_points.append(
                    {
                        "case_id": spec["case_id"],
                        "family": spec["family"],
                        "command_index": cmd_idx,
                        "event_index": record["event_index"],
                        "stage": tag,
                        "exit_code": record["exit_code"],
                    }
                )
        timeline[spec["case_id"]] = command_points
        rows.append(
            {
                "case_id": spec["case_id"],
                "trace_family": spec["family"],
                "task_id": row["task_id"],
                "task_short": short_task(str(row["task_id"])),
                "prompt_label": row["prompt_label"],
                "score": float(row["overall_mean"]),
                "submitted_approach_family": row["submitted_approach_family"],
                "observable_search_family": row["observable_search_family"],
                "package_probe_count": int(row["n_package_probe"]),
                "method_env_success": int(row["n_method_env_success"]),
                "method_env_failure": int(row["n_method_env_failed"]),
                "command_failure_count": int(row["command_failure_count"]),
                "trace_events": int(row["trace_event_count"]),
                "commands": int(row["command_count"]),
                "observable_decision": spec["decision"],
                "evidence_summary": spec["evidence"],
                "selected_status_messages": " | ".join(selected_messages(messages, spec["message_terms"])),
                "exact_status_event": spec.get("status_event", ""),
                "exact_status_excerpt": inline_excerpt(
                    messages_by_event.get(int(spec["status_event"]), "") if "status_event" in spec else ""
                ),
                "exact_command_event": spec.get("command_event", ""),
                "exact_command_exit_code": commands_by_event.get(int(spec["command_event"]), {}).get("exit_code", "")
                if "command_event" in spec
                else "",
                "exact_command_excerpt": raw_lines_matching(
                    commands_by_event.get(int(spec["command_event"]), {}).get("output", "")
                    if "command_event" in spec
                    else "",
                    list(spec.get("command_terms", [])),
                ),
                "exact_stderr_excerpt": stderr_excerpt(run_dir, list(spec.get("stderr_terms", []))),
                "stage_counts": "|".join(f"{stage}:{command_counter[stage]}" for stage in STAGE_ORDER),
                "run_name": run_name,
                "run_dir": str(run_dir),
            }
        )

    if missing:
        raise RuntimeError("Missing case-study runs in inventory: " + ", ".join(missing))
    return pd.DataFrame(rows), timeline


def write_latex_table(cases: pd.DataFrame) -> None:
    MANUSCRIPT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for _, row in cases.iterrows():
        cells = [
            row["trace_family"],
            f"{row['task_short']} ({short_prompt_label(str(row['prompt_label']))}); score {row['score']:.3f}",
            f"event {row['exact_status_event']}: {row['exact_status_excerpt']}",
            TRACE_TABLE_SIGNALS.get(str(row["case_id"]), str(row["observable_decision"])),
        ]
        rows.append(" & ".join(latex_escape(str(cell)) for cell in cells) + r" \\")

    table = r"""\begin{table*}[t]
  \centering
    \caption{Concrete observable trace vignettes. These examples summarize visible artifacts only:
  status messages, commands, command outputs, final artifacts, and run metadata. Exact status excerpts
  are whitespace-normalized copies from the raw stream and are not hidden chain-of-thought transcripts.}
  \label{tab:trace-vignettes}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.12}
  \begin{tabular}{@{}p{0.16\textwidth}p{0.16\textwidth}p{0.34\textwidth}p{0.26\textwidth}@{}}
    \toprule
    Trace family & Example & Exact visible status excerpt & Observable process signal \\
    \midrule
"""
    table += "\n".join("    " + line for line in rows)
    table += r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (MANUSCRIPT_TABLE_DIR / "trace_case_studies_table.tex").write_text(table, encoding="utf-8")


def write_exact_excerpt_table(cases: pd.DataFrame) -> None:
    rows: list[str] = []
    for _, row in cases.iterrows():
        source = f"raw_stdout.jsonl event {row['exact_status_event']} / agent_message"
        rows.append(
            " & ".join(
                latex_escape(str(cell))
                for cell in [
                    row["trace_family"],
                    source,
                    row["exact_status_excerpt"],
                ]
            )
            + r" \\"
        )
        command_source = (
            f"raw_stdout.jsonl event {row['exact_command_event']} / command_output"
            f" (exit {row['exact_command_exit_code']})"
        )
        command_excerpt = row["exact_command_excerpt"]
        if str(row.get("exact_stderr_excerpt", "")).strip():
            command_source += "; raw_stderr.txt"
            command_excerpt = f"{command_excerpt} | stderr: {row['exact_stderr_excerpt']}"
        rows.append(
            " & ".join(
                latex_escape(str(cell))
                for cell in [
                    "",
                    command_source,
                    command_excerpt,
                ]
            )
            + r" \\"
        )

    table = r"""\begin{table*}[t]
  \centering
  \caption{Exact observable trace excerpts for the prototype vignettes in
  \cref{tab:trace-vignettes}. Excerpts are copied from agent-visible status
  messages, command outputs, or runner stderr with whitespace normalized for
  typesetting. They are not private reasoning transcripts.}
  \label{tab:trace-excerpts}
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.10}
  \begin{tabular}{@{}p{0.18\textwidth}p{0.28\textwidth}p{0.48\textwidth}@{}}
    \toprule
    Vignette & Source & Exact observable excerpt \\
    \midrule
"""
    table += "\n".join("    " + line for line in rows)
    table += r"""
    \bottomrule
  \end{tabular}
\end{table*}
"""
    (MANUSCRIPT_TABLE_DIR / "trace_exact_excerpt_table.tex").write_text(table, encoding="utf-8")


def plot_timeline(cases: pd.DataFrame, timeline: dict[str, list[dict[str, Any]]]) -> plt.Figure:
    setup_theme()
    fig = plt.figure(figsize=(14.0, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[0.36, 0.64], wspace=0.02)
    label_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])
    rng = np.random.default_rng(5)

    y_positions = np.arange(len(cases))[::-1]
    stage_offset = {stage: (i - (len(STAGE_ORDER) - 1) / 2) * 0.055 for i, stage in enumerate(STAGE_ORDER)}

    for y, (_, row) in zip(y_positions, cases.iterrows()):
        points = timeline[row["case_id"]]
        max_cmd = max([int(p["command_index"]) for p in points] + [1])
        ax.hlines(y, 0, 1, color="#D9D9D9", linewidth=1.6, zorder=0)
        label_ax.text(
            0.98,
            y,
            f"{row['trace_family']}\n{row['task_short']}   score={row['score']:.2f}",
            ha="right",
            va="center",
            fontsize=11.8,
            fontweight="bold",
            color="#222222",
        )
        for stage in STAGE_ORDER:
            stage_points = [p for p in points if p["stage"] == stage]
            if not stage_points:
                continue
            x = np.array([p["command_index"] / max_cmd for p in stage_points], dtype=float)
            x = np.clip(x + rng.normal(0, 0.004, size=x.shape), 0, 1)
            yy = np.full_like(x, y + stage_offset[stage], dtype=float)
            ax.scatter(
                x,
                yy,
                s=72,
                color=STAGE_COLORS[stage],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.90,
                zorder=3,
            )

    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(-0.75, len(cases) - 0.25)
    label_ax.axis("off")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.75, len(cases) - 0.25)
    ax.set_yticks([])
    ax.set_xlabel("Command order (normalized)", fontsize=18)
    ax.tick_params(axis="x", labelsize=15)
    fig.suptitle("Observable Agent Trace Vignettes", x=0.02, y=0.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(
        0.38,
        0.075,
        "Dots are visible command-stage tags from public run artifacts; no hidden reasoning text is shown.",
        ha="left",
        va="center",
        fontsize=11.2,
        color="#333333",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=STAGE_COLORS[stage],
            markeredgecolor="white",
            markersize=9,
            label=STAGE_LABELS[stage],
        )
        for stage in STAGE_ORDER
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        prop={"family": "Avenir", "weight": "bold", "size": 12.0},
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        frameon=False,
    )
    apply_bold_font(fig)
    fig.savefig(FIG_DIR / "trace_case_study_timelines.pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(FIG_DIR / "trace_case_study_timelines.png", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(FIG_DIR / "trace_case_study_timelines.svg", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIG_DIR / "trace_case_study_timelines.pdf", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIG_DIR / "trace_case_study_timelines.png", bbox_extra_artists=(legend,), bbox_inches="tight")
    fig.savefig(MANUSCRIPT_FIG_DIR / "trace_case_study_timelines.svg", bbox_extra_artists=(legend,), bbox_inches="tight")
    return fig


def plot_process_excerpts(cases: pd.DataFrame, inventory: pd.DataFrame) -> plt.Figure:
    setup_theme()
    mpl.rcParams.update(
        {
            "font.size": 7.5,
            "font.weight": "normal",
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "legend.fontsize": 6.8,
        }
    )
    prompt = pd.read_csv(PROMPT_TRACE_SUMMARY)
    prompt_order = ["Basic", "Rich/package", "Full package + paper prior"]
    prompt_labels = {
        "Basic": "Basic",
        "Rich/package": "Package-aware",
        "Full package + paper prior": "Full + prior",
    }
    prompt = prompt.set_index("prompt_label").loc[prompt_order].reset_index()
    zero_rates = (
        inventory.assign(is_zero=lambda d: pd.to_numeric(d["overall_mean"], errors="coerce").fillna(0).eq(0))
        .groupby("prompt_label")["is_zero"]
        .mean()
        .reindex(prompt_order)
        .fillna(0)
        .to_numpy()
    )
    env_called = (
        inventory.assign(n_method_env_any=lambda d: pd.to_numeric(d["n_method_env_any"], errors="coerce").fillna(0))
        .groupby("prompt_label")["n_method_env_any"]
        .sum()
        .reindex(prompt_order)
        .fillna(0)
        .gt(0)
        .to_numpy()
    )

    fig = plt.figure(figsize=(7.35, 4.7), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.42, 0.58],
        left=0.065,
        right=0.985,
        bottom=0.13,
        top=0.845,
        wspace=0.13,
    )
    ax = fig.add_subplot(grid[0, 0])
    card_ax = fig.add_subplot(grid[0, 1])
    card_ax.axis("off")

    x = np.arange(len(prompt))
    width = 0.17
    metrics = [
        ("package_probe_rate", "Package\nprobe", "#7884B4"),
        ("method_env_success_rate", "Env\nsuccess", "#AADCA9"),
        ("method_env_failure_rate", "Env\nfailure", "#E9A6A1"),
        ("zero_rate", "Endpoint\nzero", "#606060"),
    ]
    prompt["zero_rate"] = zero_rates
    prompt["method_env_called"] = env_called
    for offset, (col, label, color) in enumerate(metrics):
        vals = prompt[col].to_numpy(dtype=float)
        bars = ax.bar(
            x + (offset - 1.5) * width,
            vals,
            width=width,
            color=color,
            edgecolor="#4D4D4D",
            linewidth=0.6,
            label=label.replace("\n", " "),
            alpha=0.92,
        )
        for j, (rect, val) in enumerate(zip(bars, vals)):
            no_env_call = col.startswith("method_env_") and not bool(prompt.loc[j, "method_env_called"])
            if no_env_call:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    0.038,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=9.0,
                    fontweight="bold",
                    color="#333333",
                )
                continue
            label_y = rect.get_height() + 0.025 if val > 0 else 0.030
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                label_y,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="bold",
                color="#333333",
            )
    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels([prompt_labels[p] for p in prompt["prompt_label"]], fontsize=7.8, fontweight="bold")
    ax.set_ylabel("Fraction of runs", fontsize=8.2, fontweight="bold")
    ax.set_title("a  Process rates by affordance", loc="left", fontsize=9.4, fontweight="bold", pad=3)
    ax.grid(axis="y", alpha=0.16, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            color="#333333",
            markerfacecolor="#333333",
            markersize=7.0,
        )
    )
    labels.append("No env call")
    ax.legend(
        handles,
        labels,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.015),
        fontsize=6.6,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.text(0.0, 1.02, "b  Prototype observable traces", fontsize=8.4, fontweight="bold", va="bottom")
    card_ax.text(
        0.0,
        0.967,
        "Exact stream excerpts, paired with command evidence; hidden reasoning text is not shown.",
        fontsize=6.4,
        color="#555555",
        va="top",
    )

    card_cases = [
        ("baseline-insurance", "#7884B4"),
        ("package-backed-selection", "#2E9E7E"),
        ("package-error-recovery", "#B64342"),
        ("provider-disconnect-zero", "#606060"),
    ]
    y_top = 0.90
    card_h = 0.172
    gap = 0.045
    for i, (case_id, color) in enumerate(card_cases):
        row = cases.loc[cases["case_id"] == case_id].iloc[0]
        y = y_top - i * (card_h + gap) - card_h
        card_ax.add_patch(
            Rectangle(
                (0.0, y),
                0.985,
                card_h,
                facecolor="#FFFFFF",
                edgecolor="#D8D8D8",
                linewidth=0.8,
            )
        )
        card_ax.add_patch(Rectangle((0.0, y), 0.014, card_h, facecolor=color, edgecolor=color, linewidth=0))
        header = (
            f"{row['trace_family']} · {row['task_short']} · "
            f"{short_prompt_label(str(row['prompt_label']))} · score {row['score']:.3f}"
        )
        card_ax.text(0.025, y + card_h - 0.020, header, fontsize=6.6, fontweight="bold", va="top")
        status = textwrap.fill(
            f"status e{row['exact_status_event']}: \"{inline_excerpt(str(row['exact_status_excerpt']), 138)}\"",
            width=67,
            break_long_words=False,
        )
        command_override = {
            "baseline-insurance": "transform 151673 -> 151674 scale 0.99976; wrote final CSVs",
            "package-backed-selection": "identity 0.412; spateo_affine 0.529; selected_by_internal_metric = spateo_affine",
            "package-error-recovery": "paste_pairwise and paste2_partial logged; selected fallback remains valid",
            "provider-disconnect-zero": "Spateo finished [2592.0746s] and wrote CSVs; stderr shows websocket lookup failure",
        }
        command = textwrap.fill(
            f"command e{row['exact_command_event']}: {command_override.get(case_id, inline_excerpt(str(row['exact_command_excerpt']), 100))}",
            width=67,
            break_long_words=False,
        )
        card_ax.text(0.025, y + card_h - 0.046, status, fontsize=5.1, va="top", color="#333333", linespacing=1.03)
        card_ax.text(0.025, y + 0.013, command, fontsize=4.95, va="bottom", color="#555555", linespacing=1.02)
    for out_dir in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
        fig.savefig(out_dir / "trace_process_excerpts.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / "trace_process_excerpts.png", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(out_dir / "trace_process_excerpts.svg", bbox_inches="tight", pad_inches=0.02)
    return fig


def write_markdown_summary(cases: pd.DataFrame) -> None:
    lines = [
        "# Observable Trace Case Studies",
        "",
        "These vignettes summarize visible artifacts only: status messages, commands, command outputs, final files, and run metadata.",
        "They are not hidden chain-of-thought transcripts.",
        "",
    ]
    for _, row in cases.iterrows():
        wrapped_decision = textwrap.fill(str(row["observable_decision"]), width=92)
        wrapped_evidence = textwrap.fill(str(row["evidence_summary"]), width=92)
        lines.extend(
            [
                f"## {row['trace_family']}",
                "",
                f"- Run: `{row['run_name']}`",
                f"- Task/prompt/score: `{row['task_id']}`, {row['prompt_label']}, {row['score']:.3f}",
                f"- Env success/failure: {row['method_env_success']}/{row['method_env_failure']}",
                f"- Package probes / command failures: {row['package_probe_count']} / {row['command_failure_count']}",
                f"- Observable decision: {wrapped_decision}",
                f"- Evidence: {wrapped_evidence}",
                f"- Example visible status lines: {row['selected_status_messages']}",
                f"- Exact status excerpt source: `raw_stdout.jsonl` event {row['exact_status_event']}",
                f"> {row['exact_status_excerpt']}",
                f"- Exact command excerpt source: `raw_stdout.jsonl` event {row['exact_command_event']} "
                f"(exit {row['exact_command_exit_code']})",
                f"> {row['exact_command_excerpt']}",
                *(
                    [
                        "- Exact stderr excerpt source: `raw_stderr.txt`",
                        f"> {row['exact_stderr_excerpt']}",
                    ]
                    if str(row.get("exact_stderr_excerpt", "")).strip()
                    else []
                ),
                "",
            ]
        )
    (STRUCTURE_DIR / "trace_case_studies.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    inventory = pd.read_csv(INVENTORY)
    cases, timeline = build_case_table(inventory)
    cases.to_csv(STRUCTURE_DIR / "trace_case_studies.csv", index=False)
    write_latex_table(cases)
    write_exact_excerpt_table(cases)
    write_markdown_summary(cases)
    plot_timeline(cases, timeline)
    plot_process_excerpts(cases, inventory)
    print(f"Wrote {STRUCTURE_DIR / 'trace_case_studies.csv'}")
    print(f"Wrote {MANUSCRIPT_TABLE_DIR / 'trace_case_studies_table.tex'}")
    print(f"Wrote {MANUSCRIPT_TABLE_DIR / 'trace_exact_excerpt_table.tex'}")
    print(f"Wrote {MANUSCRIPT_FIG_DIR / 'trace_case_study_timelines.png'}")
    print(f"Wrote {MANUSCRIPT_FIG_DIR / 'trace_process_excerpts.png'}")


if __name__ == "__main__":
    main()
