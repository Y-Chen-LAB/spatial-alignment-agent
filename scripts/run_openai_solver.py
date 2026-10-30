from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from genomics_agent_bench.containment import Containment, resolve_binary, stage_state
from genomics_agent_bench.metrics import ZERO_METRICS


DEVELOPER_INSTRUCTIONS = """You are solving a spatial-genomics benchmark task by writing code.
Return one complete Python script and no prose.
The script must run from the prepared task workspace.
It must read only files in the workspace, write coordinate CSVs under submission/aligned_coordinates, and preserve obs_id row order.
Prefer simple, robust methods over fragile assumptions.
Available packages include h5py, numpy, pandas, scipy, scikit-learn, and the local helper package genomics_agent_bench.
Useful helpers include genomics_agent_bench.h5ad.H5ADSlice, genomics_agent_bench.alignment.centroid_similarity_alignment, and genomics_agent_bench.io.write_coordinate_outputs.
Construct slices with pathlib paths, e.g. `H5ADSlice(Path("data/151673.h5ad"), slice_id="151673")`.
H5ADSlice exposes `.id`, `.obs_names`, `.spatial`, `.obs_column(key)`, `.has_obs_column(key)`, and `.expression_for_genes(genes)`.
It does not expose AnnData-style `.obs` or `.obsm` attributes.
write_coordinate_outputs expects `(coords_by_slice: dict[str, ndarray], slices: list[H5ADSlice], output_dir: Path)`.
"""

BLIND_DEVELOPER_INSTRUCTIONS = """You are solving a spatial-genomics benchmark task by writing code.
Return one complete Python script and no prose.
The script runs in an isolated workspace: read only the workspace-relative files listed in
the task prompt, and write coordinate CSVs under submission/aligned_coordinates while
preserving obs_id row order. The workspace has Python plus h5py, numpy, pandas, scipy, and
scikit-learn. Implement any data-loading helpers your script needs; do not rely on a package
outside the workspace.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use an OpenAI model to generate, run, and score a task solver.")
    parser.add_argument("task", type=Path, help="Path to task.toml")
    parser.add_argument("--provider", choices=["openai", "azure"], default="openai",
                        help="Responses API provider. Azure uses an /openai/v1 endpoint and AZURE_OPENAI_API_KEY.")
    parser.add_argument("--model", default="gpt-5-mini",
                        help="OpenAI model, or Azure deployment name when --provider azure.")
    parser.add_argument("--reasoning-effort", default="xhigh", help="Reasoning effort to send to the Responses API. Use 'none' to omit for models without reasoning controls.")
    parser.add_argument("--max-output-tokens", type=int, default=64000,
                        help="Cap on output tokens, reasoning included; 0 means the model's own maximum. "
                             "At xhigh effort 8000 was used up by reasoning before any code was written.")
    parser.add_argument("--workspace-root", type=Path, default=Path("agent_workspaces"))
    parser.add_argument("--runs-root", type=Path, default=Path("agent_runs/openai_codegen"))
    parser.add_argument("--timeout-minutes", type=float, default=20,
                        help="Wall-clock limit for executing the generated solver.")
    parser.add_argument("--api-timeout-seconds", type=float, default=3600,
                        help="How long to wait for the model's response, polling in background mode.")
    parser.add_argument("--force", action="store_true", help="Recreate generated workspace files before running.")
    parser.add_argument("--dry-run", action="store_true", help="Write the prompt and command metadata without calling the API.")
    parser.add_argument("--no-execute", action="store_true", help="Generate the solver but do not execute or score it.")
    parser.add_argument(
        "--blind-solve",
        action="store_true",
        help="Send only a sanitized workspace prompt to the model and execute its solver in a container.",
    )
    parser.add_argument(
        "--blind-mode",
        choices=["container", "none"],
        default="container",
        help="Isolation for --blind-solve; container is required for comparable cluster runs.",
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get("P2B_SINGULARITY_IMAGE", "docker://python:3.11-slim"),
        help="Pinned Singularity/Apptainer image used to execute a blind solver.",
    )
    parser.add_argument(
        "--container-bin",
        default=os.environ.get("P2B_SINGULARITY_BIN", "singularity"),
        help="Container runtime; Apptainer is accepted and takes the same flags.",
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_local_env() -> None:
    env_files = [
        _repo_root() / ".env",
        _repo_root() / ".env.local",
        Path.home() / ".config" / "paper2bench" / "env",
        Path.home() / ".paper2bench.env",
    ]
    explicit = os.environ.get("PAPER2BENCH_ENV_FILE")
    if explicit:
        env_files.insert(0, Path(explicit))

    for env_file in env_files:
        if not env_file.exists() or not env_file.is_file():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _run_prepare(task: Path, workspace_root: Path, force: bool, clean_submission: bool, *, blind_solve: bool) -> None:
    command = [
        sys.executable,
        "scripts/prepare_agent_task.py",
        str(task),
        "--workspace-root",
        str(workspace_root),
    ]
    if force:
        command.append("--force")
    if clean_submission:
        command.append("--clean-submission")
    if blind_solve:
        command.append("--blind-solve")
    subprocess.run(command, cwd=_repo_root(), check=True)


def _workspace_for_task(task: Path, workspace_root: Path) -> Path:
    task_text = task.read_text(encoding="utf-8")
    match = re.search(r'^id\s*=\s*"([^"]+)"', task_text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find task id in {task}")
    return workspace_root / match.group(1)


def _build_prompt(workspace: Path) -> str:
    task_md = workspace / "TASK.md"
    if not task_md.exists():
        raise FileNotFoundError(task_md)
    data_listing = "\n".join(f"- {path.name}" for path in sorted((workspace / "data").glob("*")))
    return f"""Write a Python solver for this benchmark workspace.

Workspace-relative data files:
{data_listing}

Task prompt:
{task_md.read_text(encoding="utf-8")}

Return only Python code. Do not include markdown fences.
"""


def _extract_text(response: dict) -> str:
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason", "unknown")
        usage = response.get("usage") or {}
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        raise RuntimeError(
            f"Response incomplete ({reason}): {usage.get('output_tokens')} output tokens, "
            f"{reasoning} of them reasoning, no code returned. Raise --max-output-tokens "
            f"or lower --reasoning-effort."
        )
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text") if isinstance(content, dict) else None
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise RuntimeError(f"Unrecognized response shape: {json.dumps(response)[:400]}")
    return "\n".join(parts)


def _extract_python(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text.strip() + "\n"


def _build_openai_payload(model: str, prompt: str, reasoning_effort: str, max_output_tokens: int | None,
                          developer_instructions: str | None = DEVELOPER_INSTRUCTIONS) -> dict:
    input_items: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    if developer_instructions is not None:
        input_items.insert(0, {"role": "developer", "content": developer_instructions})
    payload: dict = {
        "model": model,
        "input": input_items,
    }
    if max_output_tokens:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def _responses_endpoint(provider: str) -> tuple[str, dict[str, str]]:
    if provider == "azure":
        # AZURE_* is the portable convention. MOOD_* keeps the local
        # Databricks/Foundry credential naming usable without copying a key.
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("MOOD_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("MOOD_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise RuntimeError(
                "Azure configuration is incomplete; set AZURE_OPENAI_ENDPOINT and "
                "AZURE_OPENAI_API_KEY (or their MOOD_OPENAI_* equivalents). The --model "
                "value must be the Azure deployment name."
            )
        base_url = endpoint.rstrip("/")
        # Azure examples use a resource root, while Azure/OpenAI SDK examples
        # conventionally pass a base URL that already ends in /openai/v1.
        # Accept either form without producing /openai/v1/openai/v1/responses.
        if not base_url.endswith("/openai/v1"):
            base_url += "/openai/v1"
        url = base_url + "/responses"
        return url, {"api-key": api_key}

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing local OpenAI credential; set it in your shell or a supported local env file.")
    return "https://api.openai.com/v1/responses", {"Authorization": f"Bearer {api_key}"}


def _http_json(url: str, auth_header: dict[str, str], method: str, body: dict | None,
               timeout_seconds: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json", **auth_header},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Responses API error HTTP {exc.code}: {text[:1200]}") from exc


def _call_responses(provider: str, model: str, prompt: str, reasoning_effort: str,
                    max_output_tokens: int | None, timeout_seconds: float,
                    developer_instructions: str | None = DEVELOPER_INSTRUCTIONS,
                    poll_seconds: float = 15.0) -> dict:
    """Submit in background mode and poll until the response settles.

    A synchronous request holds one HTTP connection open for as long as the model reasons.
    At xhigh effort with no output cap that ran past whatever sits between us and the model,
    which dropped the connection without closing it; the client then waited for bytes until
    its own 90-minute timeout, six times in a row. Background mode returns an id at once and
    each poll is a short call, so the wait is bounded by timeout_seconds and nothing else.
    """
    url, auth_header = _responses_endpoint(provider)
    payload = _build_openai_payload(
        model, prompt, reasoning_effort, max_output_tokens, developer_instructions,
    )
    payload["background"] = True

    deadline = time.monotonic() + timeout_seconds
    response = _http_json(url, auth_header, "POST", payload, timeout_seconds=120)
    response_id = response.get("id")
    if not response_id:
        raise RuntimeError(f"Responses API returned no id: {json.dumps(response)[:400]}")

    polls = 0
    while response.get("status") in ("queued", "in_progress"):
        if time.monotonic() > deadline:
            try:
                _http_json(f"{url}/{response_id}/cancel", auth_header, "POST", None, timeout_seconds=60)
            except RuntimeError:
                pass
            raise RuntimeError(
                f"Responses API call still {response['status']} after {timeout_seconds:.0f}s "
                f"({polls} polls); cancelled. Raise --api-timeout-seconds or cap --max-output-tokens."
            )
        time.sleep(poll_seconds)
        polls += 1
        try:
            response = _http_json(f"{url}/{response_id}", auth_header, "GET", None, timeout_seconds=60)
        except (RuntimeError, OSError) as exc:
            # One failed poll is a network blip, not a failed run; the next poll retries.
            print(f"poll {polls} failed ({exc}); retrying", file=sys.stderr)
    return response


def _run_solver(
    workspace: Path,
    solver_path: Path,
    timeout_seconds: float,
    *,
    blind_solve: bool,
    blind_mode: str,
    container_image: str,
    container_bin: str,
) -> subprocess.CompletedProcess[str]:
    if blind_solve:
        if blind_mode != "container":
            raise RuntimeError("Blind Azure runs require --blind-mode container; refusing an uncontained run.")
        binary = resolve_binary(container_bin)
        if binary is None:
            raise RuntimeError(
                f"Blind Azure runs require {container_bin!r} (or Apptainer) on PATH; load the site module first."
            )
        containment = Containment(
            image=container_image,
            workspace=workspace.resolve(),
            state_dir=stage_state("azure-openai"),
            agent="azure-openai",
            binary=binary,
        )
        try:
            return subprocess.run(
                containment.wrap(["python", solver_path.name]),
                cwd=workspace,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        finally:
            shutil.rmtree(containment.state_dir, ignore_errors=True)

    env = os.environ.copy()
    repo_src = str(_repo_root() / "src")
    env["PYTHONPATH"] = repo_src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else repo_src
    return subprocess.run(
        [sys.executable, solver_path.name],
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )


def _run_score(task: Path, workspace: Path, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["P2B_ALLOW_EVAL_SCORING"] = "1"
    return subprocess.run(
        [
            # The interpreter already running this harness, not "uv run python": the cluster
            # environment is a plain venv with no uv, and scoring is the last step of a run
            # that has already cost an hour of agent time.
            sys.executable,
            "scripts/score_agent_submission.py",
            str(task),
            str(workspace / "submission"),
            "--output",
            str(workspace / "scores.json"),
        ],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )


def _maybe_run_hidden_grid(task: Path, solver_path: Path, run_dir: Path) -> dict | None:
    """If the task declares a hidden_perturbation_grid_file, re-execute the solver across the grid.

    Returns a small summary dict (or None if the task has no hidden grid).
    """
    data_manifest_path = task.parent / "data" / "manifest.json"
    if not data_manifest_path.exists():
        return None
    try:
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    grid_field = data_manifest.get("hidden_perturbation_grid_file")
    if not grid_field:
        return None

    public_slices = data_manifest.get("public_slices") or []
    if len(public_slices) < 2:
        return None
    reference_spec = f"{public_slices[0]['id']}={public_slices[0]['path']}"

    public_report = data_manifest.get("public_instance", {}).get("report_json")
    if not public_report:
        return None
    report = json.loads((_repo_root() / public_report).read_text(encoding="utf-8"))
    moving_source_path = report.get("input")
    if not moving_source_path:
        return None
    moving_source_spec = f"{public_slices[1]['id']}={moving_source_path}"

    grid_dir = run_dir / "hidden_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_hidden_perturbation_grid.py",
        "--task", str(task),
        "--solver", str(solver_path.resolve()),
        "--reference-slice", reference_spec,
        "--moving-source-slice", moving_source_spec,
        "--output-dir", str(grid_dir.resolve()),
    ]
    try:
        result = subprocess.run(cmd, cwd=_repo_root(), text=True, capture_output=True, timeout=60 * 60)
    except subprocess.TimeoutExpired as exc:
        (grid_dir / "stderr.txt").write_text(str(exc), encoding="utf-8")
        return {"status": "timeout", "message": str(exc)}
    (grid_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (grid_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        return {"status": "failed", "returncode": result.returncode}
    summary_path = grid_dir / "hidden_grid_scores.json"
    if not summary_path.exists():
        return {"status": "missing_summary"}
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    return {
        "status": "scored",
        "n_cells": summary.get("n_cells"),
        "n_scored": summary.get("n_scored"),
        "failure_rate": summary.get("failure_rate"),
        "summary_path": str(summary_path),
    }


def _write_failure_score(
    run_dir: Path,
    workspace: Path,
    *,
    stage: str,
    message: str,
) -> None:
    payload = {
        "task_id": workspace.name,
        "submission": str(workspace / "submission"),
        "status": "failed",
        "failure_stage": stage,
        "failure_message": message[:4000],
        "assigned_zero": True,
        "scores": {
            "aggregate": dict(ZERO_METRICS),
            "pairs": [],
            "n_slices": None,
            "n_pairs": 0,
            "n_marker_genes_requested": None,
        },
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    (run_dir / "scores.json").write_text(text, encoding="utf-8")
    workspace_score = workspace / "scores.json"
    workspace_score.parent.mkdir(parents=True, exist_ok=True)
    workspace_score.write_text(text, encoding="utf-8")


def main() -> None:
    _load_local_env()
    args = parse_args()
    repo = _repo_root()
    task = args.task if args.task.is_absolute() else repo / args.task
    task_arg = task.relative_to(repo) if task.is_relative_to(repo) else task
    _run_prepare(
        task_arg,
        args.workspace_root,
        args.force,
        clean_submission=not args.dry_run,
        blind_solve=args.blind_solve,
    )
    workspace = _workspace_for_task(task, args.workspace_root)
    prompt = _build_prompt(workspace)

    run_name = f"{workspace.name}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    request_payload = _build_openai_payload(args.model, prompt, args.reasoning_effort, args.max_output_tokens)
    (run_dir / "request_payload.json").write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    manifest = {
        "runner": f"{args.provider}_codegen_local_exec",
        "provider": args.provider,
        "model": args.model,
        "task": str(task),
        "workspace": str(workspace),
        "dry_run": args.dry_run,
        "no_execute": args.no_execute,
        "blind_solve": args.blind_solve,
        "blind_mode": args.blind_mode if args.blind_solve else "none",
        "container_image": args.container_image if args.blind_solve else None,
        "started_at": datetime.now().isoformat(),
    }

    if args.dry_run:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Dry run wrote {run_dir / 'prompt.md'}")
        return

    start = time.perf_counter()
    try:
        response = _call_responses(
            args.provider,
            args.model,
            prompt,
            args.reasoning_effort,
            args.max_output_tokens,
            timeout_seconds=args.api_timeout_seconds,
            developer_instructions=BLIND_DEVELOPER_INSTRUCTIONS if args.blind_solve else DEVELOPER_INSTRUCTIONS,
        )
    except Exception as exc:
        manifest["api_seconds"] = time.perf_counter() - start
        manifest["status"] = "failed"
        manifest["failure_stage"] = "api"
        manifest["failure_message"] = str(exc)
        manifest["finished_at"] = datetime.now().isoformat()
        _write_failure_score(run_dir, workspace, stage="api", message=str(exc))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote failed run artifacts to {run_dir}")
        return
    manifest["api_seconds"] = time.perf_counter() - start
    (run_dir / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")

    solver_code = _extract_python(_extract_text(response))
    solver_path = workspace / "solver_from_openai.py"
    solver_path.write_text(solver_code, encoding="utf-8")
    shutil.copy2(solver_path, run_dir / "solver_from_openai.py")

    if args.no_execute:
        manifest["status"] = "generated"

    if not args.no_execute:
        try:
            solver_result = _run_solver(
                workspace,
                solver_path,
                timeout_seconds=args.timeout_minutes * 60,
                blind_solve=args.blind_solve,
                blind_mode=args.blind_mode,
                container_image=args.container_image,
                container_bin=args.container_bin,
            )
            (run_dir / "solver_stdout.txt").write_text(solver_result.stdout, encoding="utf-8")
            (run_dir / "solver_stderr.txt").write_text(solver_result.stderr, encoding="utf-8")
            manifest["solver_returncode"] = solver_result.returncode
        except subprocess.TimeoutExpired as exc:
            manifest["solver_returncode"] = None
            manifest["status"] = "failed"
            manifest["failure_stage"] = "solver_timeout"
            manifest["failure_message"] = str(exc)
            (run_dir / "solver_stdout.txt").write_text(str(exc.stdout or ""), encoding="utf-8")
            (run_dir / "solver_stderr.txt").write_text(str(exc.stderr or exc), encoding="utf-8")
            _write_failure_score(run_dir, workspace, stage="solver_timeout", message=str(exc))
            manifest["finished_at"] = datetime.now().isoformat()
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Wrote failed run artifacts to {run_dir}")
            return
        except RuntimeError as exc:
            manifest["solver_returncode"] = None
            manifest["status"] = "failed"
            manifest["failure_stage"] = "solver_container"
            manifest["failure_message"] = str(exc)
            _write_failure_score(run_dir, workspace, stage="solver_container", message=str(exc))
            manifest["finished_at"] = datetime.now().isoformat()
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Wrote failed run artifacts to {run_dir}")
            return

        try:
            score_result = _run_score(task_arg, workspace, timeout_seconds=5 * 60)
            (run_dir / "score_stdout.txt").write_text(score_result.stdout, encoding="utf-8")
            (run_dir / "score_stderr.txt").write_text(score_result.stderr, encoding="utf-8")
            manifest["score_returncode"] = score_result.returncode
        except subprocess.TimeoutExpired as exc:
            manifest["score_returncode"] = None
            manifest["status"] = "failed"
            manifest["failure_stage"] = "score_timeout"
            manifest["failure_message"] = str(exc)
            _write_failure_score(run_dir, workspace, stage="score_timeout", message=str(exc))
            manifest["finished_at"] = datetime.now().isoformat()
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Wrote failed run artifacts to {run_dir}")
            return

        workspace_scores = workspace / "scores.json"
        if solver_result.returncode != 0:
            message = (solver_result.stderr or solver_result.stdout or "Solver exited non-zero.").strip()
            manifest["status"] = "failed"
            manifest["failure_stage"] = "solver"
            manifest["failure_message"] = message[:4000]
            _write_failure_score(run_dir, workspace, stage="solver", message=message)
        elif score_result.returncode != 0:
            message = (score_result.stderr or score_result.stdout or "Scorer exited non-zero.").strip()
            manifest["status"] = "failed"
            manifest["failure_stage"] = "score"
            manifest["failure_message"] = message[:4000]
            _write_failure_score(run_dir, workspace, stage="score", message=message)
        elif workspace_scores.exists():
            manifest["status"] = "scored"
            shutil.copy2(workspace_scores, run_dir / "scores.json")
            if args.blind_solve:
                # The grid harness re-invokes model-generated code against host-side sources.
                # Do not quietly bypass the solver container; a contained grid path belongs in
                # a dedicated follow-up implementation.
                grid_manifest = task.parent / "data" / "manifest.json"
                try:
                    has_hidden_grid = bool(json.loads(grid_manifest.read_text()).get("hidden_perturbation_grid_file"))
                except (OSError, json.JSONDecodeError):
                    has_hidden_grid = False
                if has_hidden_grid:
                    manifest["hidden_grid"] = {"status": "skipped", "reason": "requires contained re-execution"}
            else:
                grid_summary = _maybe_run_hidden_grid(task, solver_path, run_dir)
                if grid_summary:
                    manifest["hidden_grid"] = grid_summary
        else:
            message = f"Scorer succeeded but did not write {workspace_scores}."
            manifest["status"] = "failed"
            manifest["failure_stage"] = "missing_score"
            manifest["failure_message"] = message
            _write_failure_score(run_dir, workspace, stage="missing_score", message=message)

    manifest["finished_at"] = datetime.now().isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
