#!/usr/bin/env python3
"""Compare common-task spatial prompt runs, including the latest 30x2 round."""

from __future__ import annotations

import json
import math
import os
import re
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD_SELECTED = ROOT / "agent_runs" / "orchestrated" / "spatial_xhigh_analysis_ready_selected_240_latest.csv"
FULL_PACKAGE_QUEUE_STATES = [
    {
        "path": Path("/tmp/p2b_fpp_localstalign_30x2_20260429-232112.json"),
        "source": "main_30x2",
        "kind": "main",
    },
    {
        "path": Path("/tmp/p2b_fpp_localstalign_failed5_rerun_fixed2_20260430-125524.json"),
        "source": "failed5_fixed_rerun",
        "kind": "rerun",
        "rep_override": 2,
    },
    {
        "path": Path("/tmp/p2b_fpp_sa103_112_2x_20260430-2130.json"),
        "source": "mop103_112_2x",
        "kind": "main",
    },
    {
        "path": Path("/tmp/p2b_fpp_failed3_rerun_20260501-1010.json"),
        "source": "mop103_112_failed3_rerun",
        "kind": "rerun",
    },
    {
        "path": Path("/tmp/p2b_fpp_remaining3_rerun_20260501-1115.json"),
        "source": "remaining_failure3_rerun",
        "kind": "rerun",
    },
    {
        "path": Path("/tmp/p2b_fpp_rep3_40x1_20260501-180156.json"),
        "source": "balanced_rep3_40x1",
        "kind": "main",
        "require_finished": True,
    },
    {
        "path": Path("/tmp/p2b_fpp_rep3_sa104_cleanup_20260502-1413.json"),
        "source": "balanced_rep3_sa104_cleanup",
        "kind": "rerun",
        "require_finished": True,
        "rep_override": 3,
    },
]
FALLBACK_QUEUE_WORKSPACE_TOKENS = {
    "main_30x2": "p2b_spatial_fpp_localstalign_30x2_20260429-232112",
    "failed5_fixed_rerun": "p2b_spatial_fpp_localstalign_failed5_rerun_fixed2_20260430-125524",
    "mop103_112_2x": "p2b_fpp_sa103_112_2x_20260430-2130",
    "mop103_112_failed3_rerun": "p2b_fpp_failed3_rerun_20260501-1010",
    "remaining_failure3_rerun": "p2b_fpp_remaining3_rerun_20260501-1115",
    "balanced_rep3_40x1": "p2b_fpp_rep3_40x1_20260501-180156",
    "balanced_rep3_sa104_cleanup": "p2b_fpp_rep3_sa104_cleanup_20260502-1413",
}
RUNS_ROOT = ROOT / "agent_runs" / "local_coding_agent"
OUT_DIR = ROOT / "agent_runs" / "analysis" / "spatial_common_prompt_latest"
MANUSCRIPT_FIG_DIR = ROOT / "manuscripts" / "icml2026-agent-benchmark" / "figures"
PAPER_SOURCE_ROOT = ROOT / "references" / "paper_source_data" / "s43588-026-00977"
AVENIR_BOLD_PATH = ROOT / "assets" / "fonts" / "Avenir-Bold.otf"
AVENIR_BOLD_FAMILY = "Avenir Next LT Pro"
MANUSCRIPT_ALIASES = {
    "01_task_mean_overall_distribution": "task_mean_overall_distribution",
    "04_package_signal_by_prompt": "package_signal_by_prompt",
    "20_endpoint_package_summary": "endpoint_package_summary",
}

PROMPT_ORDER = ["default", "paper-method-package-first", "full-package-paper-prior"]
PROMPT_LABELS = {
    "default": "Basic",
    "paper-method-package-first": "Package-aware",
    "full-package-paper-prior": "Full + prior",
}
PACKAGE_DISPLAY_LABELS = {
    "paste": "PASTE",
    "paste2": "PASTE2",
    "spateo": "Spateo",
    "spacel": "SPACEL",
    "staligner": "STAligner",
    "scslat": "scSLAT",
    "gpsa": "GPSA",
    "cast": "CAST",
}
PROMPT_COLORS = {
    "default": "#1F5AA6",
    "paper-method-package-first": "#D95F02",
    "full-package-paper-prior": "#1B9E77",
}
METRICS = [
    "overall_mean",
    "gene_pcc",
    "gene_cosine",
    "gene_ssim",
    "gene_mi",
    "matching_accuracy",
    "overlap_ratio_region",
    "bbox_overlap_fraction",
    "mae",
]
METRIC_LABELS = {
    "overall_mean": "Overall",
    "gene_pcc": "Gene PCC",
    "gene_cosine": "Gene cosine",
    "gene_ssim": "Gene SSIM",
    "gene_mi": "Gene MI",
    "matching_accuracy": "Match acc.",
    "overlap_ratio_region": "Overlap ratio",
    "bbox_overlap_fraction": "BBox overlap",
    "mae": "MAE",
}
PACKAGE_PATTERNS = {
    "paste": [r"\bimport\s+paste\b", r"\bpaste\.pairwise", r"\bpaste\.stack", r"\bPASTE\b", r"\.method-venv-ot\b"],
    "paste2": [r"\bpaste2\b", r"\bPASTE2\b", r"\.method-venv-ot\b"],
    "spateo": [r"\bspateo\b", r"\bst\.align\b", r"\.method-venv-spateo\b"],
    "spacel": [r"\bSPACEL\b", r"\bScube\b", r"\bspacel\b", r"\.method-venv-spacel\b"],
    "staligner": [r"\bSTAligner\b", r"\bstaligner\b", r"\.method-venv-staligner\b"],
    "stalign": [r"\bSTalign\b", r"\bstalign\b", r"\.method-venv-stalign\b"],
    "scslat": [r"\bscSLAT\b", r"\bSLAT\b", r"\bslat\b", r"\.method-venv-scslat\b"],
    "gpsa": [r"\bGPSA\b", r"\bgpsa\b", r"\.method-venv-gpsa\b"],
    "cast": [r"\bCAST\b", r"\bCAST_MARK\b", r"\.method-venv-cast\b"],
    "stair": [r"\bSTAIR\b", r"\bstair\b", r"\.method-venv-stair\b"],
    "santo": [r"\bSANTO\b", r"\bsanto\b", r"\.method-venv-santo\b"],
}
ENV_TO_PACKAGES = {
    "ot": ["paste", "paste2"],
    "spateo": ["spateo"],
    "spacel": ["spacel"],
    "staligner": ["staligner"],
    "stalign": ["stalign"],
    "scslat": ["scslat"],
    "gpsa": ["gpsa"],
    "cast": ["cast"],
    "stair": ["stair"],
    "santo": ["santo"],
}
TEXT_SUFFIXES = {".py", ".r", ".R", ".sh", ".jl", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
SKIP_NAMES = {
    "TASK.md",
    "scores.json",
    "manifest.json",
    "leakage_audit.json",
    "command.json",
    "agent_python.sh",
    "bootstrap_method_venv.sh",
    "spatial_method_envs.toml",
}


def setup_theme() -> None:
    if AVENIR_BOLD_PATH.exists():
        font_manager.fontManager.addfont(str(AVENIR_BOLD_PATH))
    font_chain = [AVENIR_BOLD_FAMILY, "Avenir", "Avenir Next", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_chain,
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.linestyle": "-",
            "grid.alpha": 0.12,
            "grid.color": "#4D4D4D",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.labelsize": 16,
            "ytick.labelsize": 17,
            "legend.frameon": False,
            "legend.fontsize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_task_toml(task_id: str) -> dict[str, Any]:
    import tomllib

    path = ROOT / "benchmarks" / "spatial_alignment" / "tasks" / task_id / "task.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def normalize_path(value: str | Path) -> str:
    return str(value).replace("/private/tmp/", "/tmp/")


def save_fig(fig: plt.Figure, name: str) -> None:
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    avenir_bold = AVENIR_BOLD_PATH if AVENIR_BOLD_PATH.exists() else None
    for text in fig.findobj(match=mtext.Text):
        if avenir_bold is not None:
            text.set_fontproperties(font_manager.FontProperties(fname=str(avenir_bold), size=text.get_fontsize()))
        else:
            text.set_fontfamily("Avenir")
            text.set_fontweight("bold")
    outputs = [
        fig_dir / f"{name}.png",
        fig_dir / f"{name}.pdf",
        fig_dir / f"{name}.svg",
    ]
    alias = MANUSCRIPT_ALIASES.get(name)
    if alias is not None:
        outputs.extend(
            [
                MANUSCRIPT_FIG_DIR / f"{alias}.png",
                MANUSCRIPT_FIG_DIR / f"{alias}.pdf",
                MANUSCRIPT_FIG_DIR / f"{alias}.svg",
            ]
        )
    for output in outputs:
        fig.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def run_dir_by_workspace() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for manifest_path in RUNS_ROOT.glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        workspace = manifest.get("workspace")
        if workspace:
            out[normalize_path(workspace)] = manifest_path.parent
    return out


def score_metrics(run_dir: Path) -> tuple[dict[str, float], dict[str, Any]]:
    scores_path = run_dir / "scores.json"
    if not scores_path.exists():
        return {}, {}
    payload = read_json(scores_path)
    scores = payload.get("scores") or {}
    agg = scores.get("aggregate") or payload.get("mean") or {}
    diagnostics = scores.get("diagnostics") or {}
    metrics = {metric: agg.get(metric) for metric in METRICS if metric in agg}
    if "bbox_overlap_fraction" not in metrics and "bbox_overlap_fraction" in diagnostics:
        metrics["bbox_overlap_fraction"] = diagnostics.get("bbox_overlap_fraction")
    return metrics, payload


def parse_method_env_commands(run_dir: Path) -> tuple[Counter[str], Counter[str]]:
    success: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    raw_path = run_dir / "raw_stdout.jsonl"
    if not raw_path.exists():
        return success, failed
    env_re = re.compile(r"\.method-venv-([a-z0-9]+)\b")
    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "command_execution":
            continue
        command = item.get("command") or ""
        envs = set(env_re.findall(command))
        if not envs:
            continue
        exit_code = item.get("exit_code")
        for env in envs:
            if exit_code == 0:
                success[env] += 1
            else:
                failed[env] += 1
    return success, failed


def workspace_text(run_dir: Path) -> str:
    snapshot = run_dir / "workspace_snapshot"
    if not snapshot.exists():
        return ""
    chunks: list[str] = []
    for path in snapshot.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES or path.name in SKIP_NAMES:
            continue
        if any(part in {".method-cache"} or part.startswith(".method-venv") for part in path.parts):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(chunks)


def package_rows_for_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cache: dict[str, tuple[str, Counter[str], Counter[str]]] = {}
    compiled = {
        package: [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        for package, patterns in PACKAGE_PATTERNS.items()
    }
    for _, row in runs.iterrows():
        run_dir = Path(str(row["run_dir"]))
        key = str(run_dir)
        if key not in cache:
            cache[key] = (*parse_method_env_commands(run_dir), workspace_text(run_dir))  # type: ignore[arg-type]
        env_success, env_failed, text = cache[key]
        for package, patterns in compiled.items():
            text_signal = any(pattern.search(text) for pattern in patterns)
            env_any = False
            env_success_any = False
            for env, packages in ENV_TO_PACKAGES.items():
                if package in packages:
                    env_any = env_any or env_success.get(env, 0) > 0 or env_failed.get(env, 0) > 0
                    env_success_any = env_success_any or env_success.get(env, 0) > 0
            rows.append(
                {
                    "task_id": row["task_id"],
                    "prompt_variant": row["prompt_variant"],
                    "prompt_label": row["prompt_label"],
                    "run_name": row["run_name"],
                    "run_dir": row["run_dir"],
                    "package": package,
                    "text_signal": bool(text_signal),
                    "method_env_command_any": bool(env_any),
                    "method_env_command_success": bool(env_success_any),
                    "package_signal": bool(text_signal or env_any),
                    "package_success_signal": bool(text_signal or env_success_any),
                }
            )
    return pd.DataFrame(rows)


def load_latest_round() -> pd.DataFrame:
    workspace_map = run_dir_by_workspace()
    rows: list[dict[str, Any]] = []
    failed_rep_by_task: dict[str, Any] = {}
    entry_by_source = {entry["source"]: entry for entry in FULL_PACKAGE_QUEUE_STATES}

    for entry in FULL_PACKAGE_QUEUE_STATES:
        if entry["kind"] != "main" or not entry["path"].exists():
            continue
        state = read_json(entry["path"])
        if entry.get("require_finished") and not state.get("finished_at"):
            continue
        for job in state.get("jobs", []):
            if job.get("status") == "failed":
                failed_rep_by_task[job["task_id"]] = job.get("rep")

    for entry in FULL_PACKAGE_QUEUE_STATES:
        state_path = entry["path"]
        if not state_path.exists():
            continue
        state = read_json(state_path)
        if entry.get("require_finished") and not state.get("finished_at"):
            continue
        source = entry["source"]
        is_rerun = entry["kind"] == "rerun"
        for job in state.get("jobs", []):
            if job.get("status") != "completed":
                continue
            task_id = job["task_id"]
            workspace = Path(job["workspace_root"]) / task_id
            run_dir = workspace_map.get(normalize_path(workspace))
            if run_dir is None:
                raise RuntimeError(f"Could not find manifest for workspace {workspace}")
            manifest = read_json(run_dir / "manifest.json")
            metrics, score_payload = score_metrics(run_dir)
            score_status = score_payload.get("status") or manifest.get("status")
            assigned_zero = bool(score_payload.get("assigned_zero", False))
            success_envs, failed_envs = parse_method_env_commands(run_dir)
            if "rep_override" in entry:
                analysis_rep = entry["rep_override"]
            elif is_rerun:
                analysis_rep = failed_rep_by_task.get(task_id, job.get("rep"))
            else:
                analysis_rep = job.get("rep")
            score_success = bool(manifest.get("status") == "scored" and not assigned_zero)
            row: dict[str, Any] = {
                "task_id": task_id,
                "prompt_variant": "full-package-paper-prior",
                "prompt_label": PROMPT_LABELS["full-package-paper-prior"],
                "analysis_rep": analysis_rep,
                "queue_source": source,
                "queue_job_id": job.get("id"),
                "queue_status": job.get("status"),
                "runner_status": manifest.get("status"),
                "score_status": score_status,
                "assigned_zero": assigned_zero,
                "score_success": score_success,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "workspace": manifest.get("workspace"),
                "leak_findings": manifest.get("leakage_findings", np.nan),
                "high_leak_findings": manifest.get("leakage_high_findings", np.nan),
                "successful_method_env_commands": ";".join(f"{k}:{v}" for k, v in sorted(success_envs.items())),
                "failed_method_env_commands": ";".join(f"{k}:{v}" for k, v in sorted(failed_envs.items())),
                "used_method_env_success": bool(success_envs),
                "failure_stage": score_payload.get("failure_stage"),
                "failure_message": score_payload.get("failure_message"),
                "replacement_priority": int(is_rerun),
            }
            row.update(metrics)
            rows.append(row)
            if not is_rerun and not score_success:
                failed_rep_by_task[task_id] = analysis_rep

    # The long-running batch queue JSON files are temporary, but the run
    # manifests are durable. Fall back to manifest workspaces so figure
    # regeneration does not silently drop completed Full + prior rows when a
    # /tmp queue-state file has been cleaned up.
    for manifest_path in RUNS_ROOT.glob("*/manifest.json"):
        try:
            manifest = read_json(manifest_path)
        except Exception:
            continue
        if (manifest.get("prompt_variant") or "default") != "full-package-paper-prior":
            continue
        workspace = normalize_path(str(manifest.get("workspace") or ""))
        source = None
        for candidate_source, token in FALLBACK_QUEUE_WORKSPACE_TOKENS.items():
            if token in workspace:
                source = candidate_source
                break
        if source is None:
            continue
        entry = entry_by_source[source]
        rep_match = re.search(r"_rep(\d+)", workspace)
        if "rep_override" in entry:
            analysis_rep = entry["rep_override"]
        elif rep_match:
            analysis_rep = int(rep_match.group(1))
        else:
            continue
        task_path = manifest.get("task")
        task_id = Path(str(task_path)).parent.name if task_path else manifest_path.parent.name.split("_codex_")[0]
        run_dir = manifest_path.parent
        metrics, score_payload = score_metrics(run_dir)
        score_status = score_payload.get("status") or manifest.get("status")
        assigned_zero = bool(score_payload.get("assigned_zero", False))
        success_envs, failed_envs = parse_method_env_commands(run_dir)
        score_success = bool(manifest.get("status") == "scored" and not assigned_zero)
        row = {
            "task_id": task_id,
            "prompt_variant": "full-package-paper-prior",
            "prompt_label": PROMPT_LABELS["full-package-paper-prior"],
            "analysis_rep": analysis_rep,
            "queue_source": source,
            "queue_job_id": "",
            "queue_status": manifest.get("status"),
            "runner_status": manifest.get("status"),
            "score_status": score_status,
            "assigned_zero": assigned_zero,
            "score_success": score_success,
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "workspace": manifest.get("workspace"),
            "leak_findings": manifest.get("leakage_findings", np.nan),
            "high_leak_findings": manifest.get("leakage_high_findings", np.nan),
            "successful_method_env_commands": ";".join(f"{k}:{v}" for k, v in sorted(success_envs.items())),
            "failed_method_env_commands": ";".join(f"{k}:{v}" for k, v in sorted(failed_envs.items())),
            "used_method_env_success": bool(success_envs),
            "failure_stage": score_payload.get("failure_stage"),
            "failure_message": score_payload.get("failure_message"),
            "replacement_priority": int(entry["kind"] == "rerun"),
        }
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    for metric in METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if not df.empty:
        df = df.sort_values(
            ["task_id", "analysis_rep", "score_success", "replacement_priority", "run_name"],
            ascending=[True, True, False, False, False],
        )
        df = df.drop_duplicates(["task_id", "analysis_rep"], keep="first")
        df = df.sort_values(["task_id", "analysis_rep"]).drop(columns=["replacement_priority"])
    return df


def paper_base_method(row_name: str, dataset_suffix: str) -> str:
    name = str(row_name)
    name = re.sub(r"_landmark\d+$", "", name)
    suffix = f"_{dataset_suffix}"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    return name.replace("_best", "").replace("_worst", "")


def load_paper_direct_scores() -> pd.DataFrame:
    folder = PAPER_SOURCE_ROOT / "43588_2026_977_MOESM7_ESM"
    metric_files = {
        "gene_pcc": "GeneBased_PCC",
        "gene_cosine": "GeneBased_Cos_Sim",
        "gene_ssim": "GeneBased_SSIM",
        "gene_mi": "GeneBased_MI",
    }
    dataset_suffix = {"DLPFC": "DLPFCS", "MHPR": "MHPR"}
    rows: list[pd.DataFrame] = []
    for dataset, suffix in dataset_suffix.items():
        pieces = []
        for metric, file_mid in metric_files.items():
            path = folder / f"{dataset}_{file_mid}.csv"
            if path.exists():
                values = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce").mean(axis=1)
                pieces.append(values.rename(metric))
        landmark_path = folder / f"{dataset}_LandmarkBased_Heatmap.csv"
        if landmark_path.exists():
            landmarks = pd.read_csv(landmark_path, index_col=0).apply(pd.to_numeric, errors="coerce")
            if "Average_Accuracy" in landmarks.columns:
                pieces.append(landmarks["Average_Accuracy"].rename("matching_accuracy"))
            roi_cols = [col for col in landmarks.columns if str(col).startswith("Average_ROI_Overlap")]
            if roi_cols:
                pieces.append(landmarks[roi_cols].mean(axis=1).rename("overlap_ratio_region"))
        if pieces:
            data = pd.concat(pieces, axis=1)
            data.insert(0, "paper_row", data.index)
            data["dataset"] = dataset
            data["method"] = data["paper_row"].map(lambda value: paper_base_method(value, suffix))
            available = [m for m in ["gene_pcc", "gene_cosine", "gene_ssim", "gene_mi", "matching_accuracy", "overlap_ratio_region"] if m in data]
            data["paper_overall_like_agent"] = data[available].mean(axis=1)
            rows.append(data.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paper_robust_best(dataset: str) -> dict[str, Any]:
    folder = PAPER_SOURCE_ROOT / "43588_2026_977_MOESM8_ESM"
    pieces = []
    for kind in ["rotation", "crop"]:
        path = folder / f"{dataset}_{kind}.csv"
        if path.exists():
            pieces.append(pd.read_csv(path).apply(pd.to_numeric, errors="coerce").mean().rename(kind))
    if not pieces:
        return {}
    combo = pd.concat(pieces, axis=1)
    combo["robust_mean"] = combo.mean(axis=1)
    best_method = str(combo["robust_mean"].idxmax())
    return {
        "paper_context": f"{dataset} robustness",
        "paper_match_type": "family_robustness_mean",
        "paper_metric": "mean robustness accuracy",
        "paper_best_method": best_method,
        "paper_best_value": float(combo.loc[best_method, "robust_mean"]),
        "paper_lower_is_better": False,
        "paper_caveat": "Mean of paper rotation/crop robustness tables; local task composite is not an identical paper metric.",
    }


def paper_mae_best(table_name: str, context: str, caveat: str) -> dict[str, Any]:
    path = PAPER_SOURCE_ROOT / "43588_2026_977_MOESM10_ESM" / f"{table_name}.csv"
    if not path.exists():
        return {}
    means = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").mean().sort_values()
    best_method = str(means.index[0])
    return {
        "paper_context": context,
        "paper_match_type": "direct_mae_table",
        "paper_metric": "mean MAE",
        "paper_best_method": best_method,
        "paper_best_value": float(means.iloc[0]),
        "paper_lower_is_better": True,
        "paper_caveat": caveat,
    }


def paper_rank_best(table_name: str, metric: str, context: str, caveat: str) -> dict[str, Any]:
    path = PAPER_SOURCE_ROOT / "43588_2026_977_MOESM9_ESM" / f"{table_name}.csv"
    if not path.exists():
        return {}
    ranks = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")
    if metric not in ranks:
        return {}
    vals = ranks[metric].dropna().sort_values()
    best_method = str(vals.index[0])
    return {
        "paper_context": context,
        "paper_match_type": "aggregate_rank_only",
        "paper_metric": f"{metric} rank",
        "paper_best_method": best_method,
        "paper_best_value": float(vals.iloc[0]),
        "paper_lower_is_better": True,
        "paper_caveat": caveat,
    }


def direct_best(dataset: str, match_type: str, caveat: str) -> dict[str, Any]:
    paper_scores = load_paper_direct_scores()
    sub = paper_scores[paper_scores["dataset"] == dataset]
    if sub.empty:
        return {}
    best = sub.sort_values("paper_overall_like_agent", ascending=False).iloc[0]
    return {
        "paper_context": f"{dataset} direct gene+landmark",
        "paper_match_type": match_type,
        "paper_metric": "paper_overall_like_agent",
        "paper_best_method": best["method"],
        "paper_best_value": float(best["paper_overall_like_agent"]),
        "paper_lower_is_better": False,
        "paper_caveat": caveat,
    }


def paper_match_for_task(task_id: str) -> dict[str, Any]:
    toml = read_task_toml(task_id)
    metric_profile = toml.get("metric_profile", "")
    if task_id == "sa-001-dlpfc-serial-align":
        return direct_best("DLPFC", "direct_raw_metric_table", "Closest apples-to-apples paper table for this local task.")
    if task_id == "sa-002-dlpfc-robust-perturb":
        return paper_robust_best("DLPFC")
    if task_id == "sa-003-dlpfc-coarse-shuffled-pair":
        return direct_best("DLPFC", "same_dataset_family_modified_task", "DLPFC paper table is direct for the dataset family, but this local task adds coarse shuffled initialization.")
    if task_id == "sa-009-mhpr-merfish-serial-align":
        return direct_best("MHPR", "direct_raw_metric_table", "Closest apples-to-apples paper table for this local task.")
    if task_id == "sa-019-flysta3d-e14-16-3d-serial-align":
        return paper_mae_best("Flysta3D_MAE", "Flysta3D 3D MAE", "Direct paper MAE table, but compare local MAE cautiously because coordinate scales can differ.")
    if task_id == "sa-022-openst-mln-serial-align":
        match = paper_mae_best(
            "MLN_MAE",
            "Open-ST MLN MAE",
            "Paper MLN MAE table is same dataset family, but this local task uses gene+label alignment metrics rather than local MAE.",
        )
        if match:
            match["paper_match_type"] = "paper_family_mae_no_local_mae"
        return match
    if task_id in {
        "sa-025-openst-mln-adjacent-02-03-3d-align",
        "sa-026-openst-mln-early-window-02-06-3d-align",
        "sa-027-openst-mln-full-19-slice-3d-align",
    }:
        return paper_mae_best("MLN_MAE", "Open-ST MLN MAE", "Direct MLN paper MAE table; local windowed tasks are subsets of the full paper MLN stack.")
    image_like = any(token in task_id for token in ["mop", "wmb", "xenium", "starmap", "baristaseq", "mhpr"])
    if image_like:
        metric = "Landmark_based_overall" if "light_alignment" in metric_profile or "geometry" in metric_profile else "Gene_based_overall"
        return paper_rank_best(
            "Image_datasets_rank",
            metric,
            "aggregate image-dataset rank",
            "Aggregate paper rank only; no task-specific raw paper score is available in the local workbooks.",
        )
    metric = "Gene_based_overall" if "gene_only" in metric_profile else "Landmark_based_overall"
    return paper_rank_best(
        "NGS_datasets_rank",
        metric,
        "aggregate NGS-dataset rank",
        "Aggregate paper rank only; no task-specific raw paper score is available in the local workbooks.",
    )


def gold_standard_for_task(task_id: str) -> dict[str, Any]:
    toml = read_task_toml(task_id)
    metric_profile = str(toml.get("metric_profile", ""))
    paper_data_ids = toml.get("paper_data_ids", [])
    label_key = str(toml.get("label_key", "Region"))
    marker_gene_file = toml.get("marker_gene_file")
    split_manifest = toml.get("split_manifest")

    if metric_profile == "albis_alignment_v1":
        gold_type = "hidden_simulated_coordinates"
        source = "ALBIS simulated coordinates in the s00 input frame"
        comparability = "simulated recovery; no direct SABench paper comparison"
        hidden_fields = f"obs/x_GT;obs/y_GT;obs/{label_key}"
        scorer_uses = "MAE against hidden x/y GT without fitting, gene-grid metrics, and spatial MNN diagnostics"
    elif metric_profile == "sabench_mae_gene_efficiency_v1":
        if task_id == "sa-023-wmb-zhuang-abca1-129-section-3d-mae":
            source = "Allen CCF atlas coordinates in obs/x_GT,y_GT,z_GT"
            comparability = "paper-faithful local gold for WMB CCF MAE; no direct paper per-method table extracted locally"
        elif "flysta3d" in task_id:
            source = "Flysta3D published 3D atlas coordinates in obs/x_GT,y_GT,z_GT"
            comparability = "paper-family gold; absolute MAE scale differs from extracted paper table notes"
        elif "openst-mln" in task_id:
            source = "Open-ST MLN published 3D-aligned frame in obs/x_GT,y_GT,z_GT and obsm/spatial_3d_aligned"
            comparability = "paper-family gold; local window/full tasks share MLN gold but absolute MAE scale differs from paper table notes"
        else:
            source = "Hidden 3D ground-truth coordinates obs/x_GT,y_GT,z_GT"
            comparability = "paper-family MAE gold"
        gold_type = "hidden_3d_or_ccf_coordinates"
        hidden_fields = "obs/x_GT;obs/y_GT;obs/z_GT"
        scorer_uses = "Procrustes-aligned MAE against hidden x/y GT plus gene-grid metrics"
    elif metric_profile in {"sabench_light_alignment_v1", "sabench_light_robustness_v1", "sabench_geometry_gene_alignment_v1"}:
        gold_type = "hidden_region_labels_plus_marker_expression"
        source = f"Scorer-only obs/{label_key} labels plus held-out marker-gene split"
        hidden_fields = f"obs/{label_key}; split_manifest eval mask"
        scorer_uses = "Gene-grid PCC/cosine/SSIM/MI plus nearest-label accuracy and alpha-shape region overlap"
        if task_id in {"sa-001-dlpfc-serial-align", "sa-009-mhpr-merfish-serial-align"}:
            comparability = "best direct local match to paper source-data metric tables"
        elif task_id == "sa-003-dlpfc-coarse-shuffled-pair":
            comparability = "same DLPFC dataset family, but local perturbation differs from paper direct table"
        else:
            comparability = "paper-style metric gold; exact task-specific paper table not extracted locally"
    elif metric_profile == "resolution_recovery_alignment_v1":
        gold_type = "hidden_high_resolution_alignment_reference"
        source = "Held-out high-resolution/downsampled aligned reference coordinates and labels"
        hidden_fields = f"obs/{label_key}; aligned downsampled/reference coordinate metadata"
        scorer_uses = "Similarity-lifted coordinate recovery scored by label/region overlap"
        comparability = "local paper-inspired improvement task, not a direct extracted paper score table"
    elif metric_profile == "sabench_gene_only_alignment_v1":
        gold_type = "held_out_marker_expression_only"
        source = "Held-out marker-gene split from source H5AD expression"
        hidden_fields = "split_manifest eval mask"
        scorer_uses = "Gene-grid PCC/cosine/SSIM/MI on held-out marker genes"
        comparability = "paper-style gene metric; no direct task-specific paper score table extracted locally"
    else:
        gold_type = "unknown_or_diagnostic"
        source = "Unknown"
        hidden_fields = ""
        scorer_uses = ""
        comparability = "not checked"

    return {
        "task_id": task_id,
        "metric_profile": metric_profile,
        "paper_data_ids": ";".join(map(str, paper_data_ids)) if isinstance(paper_data_ids, list) else str(paper_data_ids),
        "gold_standard_type": gold_type,
        "gold_standard_source": source,
        "hidden_from_agent": hidden_fields,
        "scorer_uses": scorer_uses,
        "marker_gene_file": str(marker_gene_file or ""),
        "split_manifest": str(split_manifest or ""),
        "paper_comparability": comparability,
    }


def build_paper_pairing(task_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for task_id, sub in task_means.groupby("task_id", observed=True):
        match = paper_match_for_task(str(task_id))
        wide = sub.pivot(index="task_id", columns="prompt_variant", values="overall_mean")
        base: dict[str, Any] = {
            "task_id": task_id,
            "basic_overall": wide.get("default", pd.Series(dtype=float)).iloc[0] if "default" in wide else np.nan,
            "rich_package_overall": wide.get("paper-method-package-first", pd.Series(dtype=float)).iloc[0]
            if "paper-method-package-first" in wide
            else np.nan,
            "full_package_paper_prior_overall": wide.get("full-package-paper-prior", pd.Series(dtype=float)).iloc[0]
            if "full-package-paper-prior" in wide
            else np.nan,
        }
        base["full_minus_basic"] = base["full_package_paper_prior_overall"] - base["basic_overall"]
        base["full_minus_rich_package"] = base["full_package_paper_prior_overall"] - base["rich_package_overall"]
        rows.append({**base, **match})
    paired = pd.DataFrame(rows)

    direct_metrics: list[dict[str, Any]] = []
    paper_scores = load_paper_direct_scores()
    direct_map = {"DLPFC": ["sa-001-dlpfc-serial-align", "sa-003-dlpfc-coarse-shuffled-pair"], "MHPR": ["sa-009-mhpr-merfish-serial-align"]}
    for dataset, tasks in direct_map.items():
        paper_sub = paper_scores[paper_scores["dataset"] == dataset]
        if paper_sub.empty:
            continue
        for task_id in tasks:
            task_sub = task_means[task_means["task_id"] == task_id]
            for metric in ["gene_pcc", "gene_cosine", "gene_ssim", "gene_mi", "matching_accuracy", "overlap_ratio_region"]:
                if metric not in task_sub or metric not in paper_sub:
                    continue
                valid = paper_sub.loc[pd.to_numeric(paper_sub[metric], errors="coerce").notna()]
                if valid.empty:
                    continue
                best = valid.loc[pd.to_numeric(valid[metric], errors="coerce").idxmax()]
                for _, row in task_sub.iterrows():
                    direct_metrics.append(
                        {
                            "dataset": dataset,
                            "task_id": task_id,
                            "prompt_variant": row["prompt_variant"],
                            "prompt_label": row["prompt_label"],
                            "metric": metric,
                            "agent_task_mean": row.get(metric, np.nan),
                            "paper_best_method": best["method"],
                            "paper_best_value": float(best[metric]),
                            "agent_minus_paper_best": row.get(metric, np.nan) - float(best[metric]),
                        }
                    )
    return paired, pd.DataFrame(direct_metrics)


def build_gold_standard_table(task_means: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([gold_standard_for_task(str(task_id)) for task_id in sorted(task_means["task_id"].unique())])


def load_old_selected() -> pd.DataFrame:
    df = pd.read_csv(OLD_SELECTED)
    df = df.rename(columns={"task": "task_id", "prompt": "prompt_variant"})
    df["prompt_label"] = df["prompt_variant"].map(PROMPT_LABELS).fillna(df["prompt_variant"])
    df["queue_source"] = "old_balanced_240"
    df["runner_status"] = df["status"]
    df["score_status"] = df["status"]
    df["assigned_zero"] = False
    df["score_success"] = df["status"].eq("scored")
    df["used_method_env_success"] = False
    for metric in METRICS:
        if metric not in df.columns:
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    keep = [
        "task_id",
        "prompt_variant",
        "prompt_label",
        "analysis_rep",
        "queue_source",
        "runner_status",
        "score_status",
        "assigned_zero",
        "score_success",
        "run_name",
        "run_dir",
        "workspace",
        "leak_findings",
        "high_leak_findings",
        "used_method_env_success",
        *METRICS,
    ]
    return df[keep]


def make_task_means(combined: pd.DataFrame) -> pd.DataFrame:
    value_cols = [metric for metric in METRICS if metric in combined.columns]
    grouped = (
        combined.groupby(["task_id", "prompt_variant", "prompt_label"], observed=True)
        .agg(
            n_runs=("run_name", "count"),
            score_success_runs=("score_success", "sum"),
            assigned_zero_runs=("assigned_zero", "sum"),
            used_method_env_success_runs=("used_method_env_success", "sum"),
            **{metric: (metric, "mean") for metric in value_cols},
        )
        .reset_index()
    )
    grouped["prompt_order"] = grouped["prompt_variant"].map({p: i for i, p in enumerate(PROMPT_ORDER)})
    return grouped.sort_values(["task_id", "prompt_order"]).drop(columns="prompt_order")


def summary_table(runs: pd.DataFrame, task_means: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt in PROMPT_ORDER:
        run_sub = runs[runs["prompt_variant"] == prompt]
        task_sub = task_means[task_means["prompt_variant"] == prompt]
        rows.append(
            {
                "prompt_variant": prompt,
                "prompt_label": PROMPT_LABELS[prompt],
                "run_rows": len(run_sub),
                "tasks": run_sub["task_id"].nunique(),
                "score_success_runs": int(run_sub["score_success"].sum()),
                "assigned_zero_runs": int(run_sub["assigned_zero"].sum()),
                "run_overall_mean": run_sub["overall_mean"].mean(),
                "run_overall_median": run_sub["overall_mean"].median(),
                "task_mean_overall_mean": task_sub["overall_mean"].mean(),
                "task_mean_overall_median": task_sub["overall_mean"].median(),
                "task_mean_overall_min": task_sub["overall_mean"].min(),
                "task_mean_overall_max": task_sub["overall_mean"].max(),
                "method_env_success_runs": int(run_sub["used_method_env_success"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_overall_task_distribution(task_means: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    rng = np.random.default_rng(12)
    positions = np.arange(len(PROMPT_ORDER))
    for idx, prompt in enumerate(PROMPT_ORDER):
        vals = task_means.loc[task_means["prompt_variant"] == prompt, "overall_mean"].dropna().to_numpy()
        color = PROMPT_COLORS[prompt]
        bp = ax.boxplot(
            vals,
            positions=[idx],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color, "linewidth": 1.8},
            medianprops={"color": color, "linewidth": 2.2},
            whiskerprops={"color": color, "linewidth": 1.6},
            capprops={"color": color, "linewidth": 1.6},
        )
        del bp
        x = rng.normal(idx, 0.045, size=len(vals))
        ax.scatter(x, vals, s=34, color=color, alpha=0.78, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(positions, [PROMPT_LABELS[p] for p in PROMPT_ORDER])
    ax.set_ylabel("Task-mean overall score")
    ax.set_title("Common 40 Tasks: Composite Metric")
    save_fig(fig, "01_task_mean_overall_distribution")


def plot_metric_small_multiples(task_means: pd.DataFrame) -> None:
    metrics = [m for m in METRICS[1:] if task_means[m].notna().any()]
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.2 * nrows), squeeze=False)
    rng = np.random.default_rng(13)
    for ax, metric in zip(axes.flat, metrics):
        for idx, prompt in enumerate(PROMPT_ORDER):
            vals = task_means.loc[task_means["prompt_variant"] == prompt, metric].dropna().to_numpy()
            if metric == "mae":
                vals = np.log1p(vals)
            color = PROMPT_COLORS[prompt]
            if len(vals):
                ax.boxplot(
                    vals,
                    positions=[idx],
                    widths=0.52,
                    patch_artist=True,
                    showfliers=False,
                    boxprops={"facecolor": color, "alpha": 0.20, "edgecolor": color, "linewidth": 1.5},
                    medianprops={"color": color, "linewidth": 2.0},
                    whiskerprops={"color": color, "linewidth": 1.3},
                    capprops={"color": color, "linewidth": 1.3},
                )
                ax.scatter(
                    rng.normal(idx, 0.045, size=len(vals)),
                    vals,
                    s=24,
                    color=color,
                    alpha=0.70,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
        label = METRIC_LABELS.get(metric, metric)
        if metric == "mae":
            label = "log1p(MAE)"
        ax.set_title(label)
        ax.set_xticks(range(len(PROMPT_ORDER)), [PROMPT_LABELS[p] for p in PROMPT_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Task mean")
    for ax in axes.flat[len(metrics) :]:
        ax.axis("off")
    fig.suptitle("Common 40 Tasks: Individual Metrics", fontsize=24, fontweight="bold")
    save_fig(fig, "02_task_mean_metric_distributions")


def plot_author_style_metric_boxplots(task_means: pd.DataFrame) -> None:
    """SABench-inspired horizontal boxplots with task dots for prompt families.

    The layout follows the original SABench plotting notebooks' repeated
    horizontal boxplot panels, but the plotted units here are agent prompt
    families rather than spatial-alignment methods.
    """
    metrics = ["gene_pcc", "gene_cosine", "gene_ssim", "gene_mi", "matching_accuracy", "overlap_ratio_region"]
    metrics = [m for m in metrics if m in task_means and task_means[m].notna().any()]
    if not metrics:
        return

    ncols = min(3, len(metrics))
    nrows = int(math.ceil(len(metrics) / ncols))
    fig, axes_grid = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.5 * nrows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes_grid).ravel()
    rng = np.random.default_rng(42)
    y_positions = np.arange(len(PROMPT_ORDER))
    prompt_tick_labels = ["Basic", "Package-aware", "Full + prior"]

    for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        data = [
            task_means.loc[task_means["prompt_variant"] == prompt, metric].dropna().to_numpy()
            for prompt in PROMPT_ORDER
        ]
        box = ax.boxplot(
            data,
            vert=False,
            positions=y_positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            boxprops={"linewidth": 1.8},
            medianprops={"color": "#222222", "linewidth": 2.1},
            whiskerprops={"color": "#444444", "linewidth": 1.4},
            capprops={"color": "#444444", "linewidth": 1.4},
        )
        for patch, prompt in zip(box["boxes"], PROMPT_ORDER):
            patch.set_facecolor(PROMPT_COLORS[prompt])
            patch.set_alpha(0.28)
            patch.set_edgecolor(PROMPT_COLORS[prompt])

        for idx, prompt in enumerate(PROMPT_ORDER):
            vals = data[idx]
            if len(vals) == 0:
                continue
            y = rng.normal(y_positions[idx], 0.055, size=len(vals))
            ax.scatter(
                vals,
                y,
                s=34,
                color=PROMPT_COLORS[prompt],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=25, fontweight="bold", pad=10)
        ax.set_xlabel("Task mean", fontsize=22, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0, 0.5, 1.0])
        ax.axvline(0.5, color="#777777", linestyle=(0, (18, 5)), linewidth=1.0, alpha=0.55)
        ax.set_ylim(-0.55, len(PROMPT_ORDER) - 0.45)
        ax.invert_yaxis()
        if ax_idx % ncols == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(prompt_tick_labels, fontsize=19, fontweight="bold")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    for ax in axes[len(metrics) :]:
        ax.axis("off")

    fig.suptitle("SABench-Style Prompt Distributions", fontsize=30, fontweight="bold")
    save_fig(fig, "19_author_style_metric_boxplots")


def plot_task_heatmap(task_means: pd.DataFrame) -> None:
    wide = task_means.pivot(index="task_id", columns="prompt_variant", values="overall_mean")
    wide = wide[PROMPT_ORDER]
    wide = wide.sort_values("full-package-paper-prior", ascending=False)
    data = wide.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 12))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PROMPT_ORDER)), [PROMPT_LABELS[p] for p in PROMPT_ORDER], rotation=25, ha="right")
    short_tasks = [task.replace("sa-", "").replace("-serial-align", "").replace("-3d-align", "") for task in wide.index]
    ax.set_yticks(range(len(short_tasks)), short_tasks)
    ax.tick_params(axis="y", labelsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Task-mean overall score")
    ax.set_title("Common Task Means")
    save_fig(fig, "03_task_mean_overall_heatmap")


def plot_package_use(package_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        package_df.groupby(["prompt_variant", "prompt_label", "package"], observed=True)
        .agg(
            n_rows=("run_name", "count"),
            signal_runs=("package_signal", "sum"),
            success_signal_runs=("package_success_signal", "sum"),
            method_env_success_runs=("method_env_command_success", "sum"),
        )
        .reset_index()
    )
    summary["signal_rate"] = summary["signal_runs"] / summary["n_rows"]
    summary["success_signal_rate"] = summary["success_signal_runs"] / summary["n_rows"]
    summary["method_env_success_rate"] = summary["method_env_success_runs"] / summary["n_rows"]

    totals = summary.groupby("package", observed=True)["signal_runs"].sum().sort_values(ascending=False)
    packages = totals[totals > 0].head(8).index.tolist()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(packages))
    height = 0.22
    offsets = [-height, 0, height]
    for idx, prompt in enumerate(PROMPT_ORDER):
        sub = summary[(summary["prompt_variant"] == prompt) & (summary["package"].isin(packages))]
        vals = sub.set_index("package").reindex(packages)["signal_rate"].fillna(0).to_numpy()
        bars = ax.barh(
            y + offsets[idx],
            vals,
            height=height,
            color=PROMPT_COLORS[prompt],
            alpha=0.84,
            edgecolor=PROMPT_COLORS[prompt],
            linewidth=0.9,
            label=PROMPT_LABELS[prompt],
        )
        for bar, value in zip(bars, vals):
            if value >= 0.08:
                ax.text(
                    value + 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0%}",
                    va="center",
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                    color=PROMPT_COLORS[prompt],
                )
    ax.set_yticks(y, [PACKAGE_DISPLAY_LABELS.get(package, package) for package in packages])
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of runs with observable package signal")
    ax.set_title("Package-use signals on common tasks")
    ax.set_xlim(0, max(0.05, min(1.0, summary["signal_rate"].max() * 1.28)))
    ax.legend(loc="lower right")
    save_fig(fig, "04_package_signal_by_prompt")

    latest = summary[summary["prompt_variant"] == "full-package-paper-prior"].copy()
    latest = latest.sort_values("method_env_success_runs", ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(
        [PACKAGE_DISPLAY_LABELS.get(package, package) for package in latest["package"]],
        latest["method_env_success_runs"],
        color=PROMPT_COLORS["full-package-paper-prior"],
        alpha=0.72,
    )
    ax.set_xlabel("Runs with successful method-env command")
    ax.set_title("Direct method-env success")
    save_fig(fig, "05_latest_method_env_success")
    return summary


def plot_endpoint_package_summary(task_means: pd.DataFrame, package_summary: pd.DataFrame) -> None:
    """Combined manuscript panel linking endpoint scores to package-use signals."""
    fig, (ax_score, ax_pkg) = plt.subplots(
        1,
        2,
        figsize=(11.6, 4.35),
        gridspec_kw={"width_ratios": [1.0, 1.36], "wspace": 0.30},
        constrained_layout=False,
    )
    rng = np.random.default_rng(12)
    positions = np.arange(len(PROMPT_ORDER))
    for idx, prompt in enumerate(PROMPT_ORDER):
        vals = task_means.loc[task_means["prompt_variant"] == prompt, "overall_mean"].dropna().to_numpy()
        color = PROMPT_COLORS[prompt]
        ax_score.boxplot(
            vals,
            positions=[idx],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.20, "edgecolor": color, "linewidth": 1.5},
            medianprops={"color": "#222222", "linewidth": 1.8},
            whiskerprops={"color": "#555555", "linewidth": 1.2},
            capprops={"color": "#555555", "linewidth": 1.2},
        )
        ax_score.scatter(
            rng.normal(idx, 0.045, size=len(vals)),
            vals,
            s=24,
            color=color,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax_score.set_xticks(positions, ["Basic", "Package-\naware", "Full +\nprior"])
    ax_score.set_ylabel("Task-mean composite", fontsize=10.5, labelpad=4)
    ax_score.set_title("A. Endpoint scores", loc="left", pad=6, fontsize=12.5)
    ax_score.tick_params(axis="both", labelsize=9.5)
    ax_score.grid(axis="y", alpha=0.14)

    totals = package_summary.groupby("package", observed=True)["signal_runs"].sum().sort_values(ascending=False)
    packages = totals[totals > 0].head(6).index.tolist()
    y = np.arange(len(packages))
    height = 0.20
    offsets = [-height, 0, height]
    for idx, prompt in enumerate(PROMPT_ORDER):
        sub = package_summary[
            (package_summary["prompt_variant"] == prompt) & (package_summary["package"].isin(packages))
        ]
        vals = sub.set_index("package").reindex(packages)["signal_rate"].fillna(0).to_numpy()
        ax_pkg.barh(
            y + offsets[idx],
            vals,
            height=height,
            color=PROMPT_COLORS[prompt],
            alpha=0.84,
            edgecolor=PROMPT_COLORS[prompt],
            linewidth=0.8,
            label=PROMPT_LABELS[prompt],
        )
        for ypos, value in zip(y + offsets[idx], vals):
            if value >= 0.10:
                ax_pkg.text(
                    min(value + 0.012, 0.98),
                    ypos,
                    f"{value:.0%}",
                    va="center",
                    ha="left",
                    fontsize=8.8,
                    fontweight="bold",
                    color=PROMPT_COLORS[prompt],
                )
    ax_pkg.set_yticks(y, [PACKAGE_DISPLAY_LABELS.get(package, package) for package in packages])
    ax_pkg.invert_yaxis()
    ax_pkg.set_xlabel("Runs with observable package signal", fontsize=10.5, labelpad=4)
    ax_pkg.set_title("B. Observable package use", loc="left", pad=6, fontsize=12.5)
    ax_pkg.set_xlim(0, max(0.05, min(1.0, package_summary["signal_rate"].max() * 1.25)))
    ax_pkg.tick_params(axis="both", labelsize=9.5)
    ax_pkg.grid(axis="x", alpha=0.14)
    ax_pkg.grid(axis="y", alpha=0.08)

    ax_pkg.legend(
        loc="lower right",
        prop={"family": AVENIR_BOLD_FAMILY, "size": 8.8},
        handlelength=1.1,
        handletextpad=0.45,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16)
    save_fig(fig, "20_endpoint_package_summary")


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    headers = [str(col) for col in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        vals = [str(row[col]) if not pd.isna(row[col]) else "" for col in out.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    runs: pd.DataFrame,
    task_means: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    package_summary: pd.DataFrame,
    queue_progress: pd.DataFrame,
    paper_pairing: pd.DataFrame,
    gold_standard: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Spatial common-prompt comparison",
        "",
        "Inputs:",
        f"- Old balanced matrix: `{OLD_SELECTED.relative_to(ROOT)}`",
        "- Latest full-package queue states:",
        *[f"  - `{entry['path']}` ({entry['source']})" for entry in FULL_PACKAGE_QUEUE_STATES],
        "",
        "## Progress",
        "",
        markdown_table(queue_progress),
        "",
        "## Common Subset",
        "",
        f"- Common tasks: {runs['task_id'].nunique()}",
        f"- Combined run rows on the common subset: {len(runs)}",
        "- Old matrix contributes 3 reps per prompt/task; latest full-package round contributes 2 reps per task.",
        "- Full-package reruns replace infrastructure/scorer failure rows for the same task/rep.",
        "- Task-level plots average reps within each task before comparing prompts.",
        "",
        "## Prompt Summary",
        "",
        markdown_table(prompt_summary),
        "",
        "## Package Use",
        "",
        "Package rows use observable code/report/command signals. The direct method-env success column is mostly relevant to the latest package-backed round. PASTE and PASTE2 share `.method-venv-ot`, so OT-env command success is a PASTE-family signal rather than a clean PASTE-vs-PASTE2 distinction.",
        "",
        markdown_table(package_summary.sort_values(["prompt_variant", "signal_runs"], ascending=[True, False]).head(30), ".3f"),
        "",
        "## Paper Benchmark Pairing",
        "",
        "Paper matches are exact only where the local source workbooks have direct per-method metric tables. Aggregate-rank rows are useful priors, not task-specific score targets.",
        "",
        markdown_table(
            paper_pairing[
                [
                    "task_id",
                    "paper_context",
                    "paper_match_type",
                    "paper_metric",
                    "paper_best_method",
                    "paper_best_value",
                    "full_package_paper_prior_overall",
                    "full_minus_basic",
                    "full_minus_rich_package",
                ]
            ].head(35),
            ".4f",
        ),
        "",
        "## Gold Standard",
        "",
        "This table describes the hidden scorer-side target, not data shown to agents. For blind solves, these fields are withheld from workspace `data/*.h5ad` files.",
        "",
        markdown_table(
            gold_standard[
                [
                    "task_id",
                    "metric_profile",
                    "gold_standard_type",
                    "gold_standard_source",
                    "paper_comparability",
                ]
            ].head(35),
            ".4f",
        ),
        "",
        "## Figures",
        "",
        "- `01_task_mean_overall_distribution`: task-level composite score distribution.",
        "- `02_task_mean_metric_distributions`: task-level individual metric distributions.",
        "- `03_task_mean_overall_heatmap`: per-task mean composite scores.",
        "- `04_package_signal_by_prompt`: observable package-use signal by prompt.",
        "- `05_latest_method_env_success`: latest-round direct package-env successes.",
        "- `19_author_style_metric_boxplots`: SABench-author-style horizontal task-dot boxplots.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_theme()
    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    queue_rows: list[dict[str, Any]] = []
    for entry in FULL_PACKAGE_QUEUE_STATES:
        if entry["path"].exists():
            state = read_json(entry["path"])
            jobs = state.get("jobs", [])
            queue_rows.append(
                {
                    "queue": entry["source"],
                    "started_at": state.get("started_at"),
                    "finished_at": state.get("finished_at"),
                    "jobs": len(jobs),
                    **Counter(job.get("status") for job in jobs),
                }
            )
        else:
            queue_rows.append({"queue": entry["source"], "started_at": "", "finished_at": "", "jobs": 0})
    queue_progress = pd.DataFrame(queue_rows).fillna(0)

    latest = load_latest_round()
    old = load_old_selected()
    latest_tasks = set(latest["task_id"])
    old_common = old[old["task_id"].isin(latest_tasks)].copy()
    common_tasks = set(old_common.groupby("task_id")["prompt_variant"].nunique().loc[lambda s: s >= 2].index) & latest_tasks
    runs = pd.concat(
        [
            old_common[old_common["task_id"].isin(common_tasks)],
            latest[latest["task_id"].isin(common_tasks)],
        ],
        ignore_index=True,
        sort=False,
    )
    runs["prompt_label"] = runs["prompt_variant"].map(PROMPT_LABELS).fillna(runs["prompt_label"])
    runs["prompt_variant"] = pd.Categorical(runs["prompt_variant"], categories=PROMPT_ORDER, ordered=True)
    runs = runs.sort_values(["task_id", "prompt_variant", "analysis_rep"])
    task_means = make_task_means(runs)
    prompt_summary = summary_table(runs, task_means)
    package_df = package_rows_for_runs(runs)
    package_summary = plot_package_use(package_df)
    plot_endpoint_package_summary(task_means, package_summary)
    paper_pairing, paper_direct_metrics = build_paper_pairing(task_means)
    gold_standard = build_gold_standard_table(task_means)

    latest.to_csv(table_dir / "latest_round_runs.csv", index=False)
    runs.to_csv(table_dir / "common_subset_runs.csv", index=False)
    task_means.to_csv(table_dir / "task_prompt_means.csv", index=False)
    prompt_summary.to_csv(table_dir / "prompt_summary.csv", index=False)
    package_df.to_csv(table_dir / "package_signal_by_run.csv", index=False)
    package_summary.to_csv(table_dir / "package_signal_summary.csv", index=False)
    paper_pairing.to_csv(table_dir / "paper_benchmark_pairing_by_task.csv", index=False)
    paper_direct_metrics.to_csv(table_dir / "paper_direct_metric_pairing_long.csv", index=False)
    gold_standard.to_csv(table_dir / "gold_standard_by_task.csv", index=False)
    queue_progress.to_csv(table_dir / "queue_progress.csv", index=False)
    error_rows = latest[latest["assigned_zero"] | latest["failure_message"].notna()][
        [
            "task_id",
            "analysis_rep",
            "queue_source",
            "runner_status",
            "score_status",
            "assigned_zero",
            "overall_mean",
            "failure_stage",
            "failure_message",
            "run_dir",
        ]
    ]
    error_rows.to_csv(table_dir / "latest_round_error_traces.csv", index=False)

    plot_overall_task_distribution(task_means)
    plot_metric_small_multiples(task_means)
    plot_author_style_metric_boxplots(task_means)
    plot_task_heatmap(task_means)
    write_report(runs, task_means, prompt_summary, package_summary, queue_progress, paper_pairing, gold_standard)

    print(f"Wrote {OUT_DIR}")
    print(prompt_summary.to_string(index=False))


if __name__ == "__main__":
    main()
