from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from genomics_agent_bench.metrics import failure_aggregate
from genomics_agent_bench.usage import finalise, for_run
from genomics_agent_bench.containment import (
    Containment, codex_azure_config, codex_provider, resolve_binary, secret_routes, stage_state,
)
from genomics_agent_bench.task_spec import load_task_spec
from genomics_agent_bench.prompt_variants import (
    DEFAULT_PROMPT_VARIANT,
    NO_METHOD_HELPER_PROMPT_VARIANTS,
    PROMPT_VARIANTS,
    method_prebootstrap_target,
    prompt_variant_tag,
    render_cli_prompt_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local coding-agent CLI against one prepared benchmark workspace.")
    parser.add_argument("task", type=Path, help="Path to task.toml")
    parser.add_argument("--agent", choices=["codex", "claude", "gemini", "mini-swe"], default="codex")
    parser.add_argument(
        "--model",
        required=True,
        help="Model name passed to the agent CLI. Required: there is no default, because a stale one silently launches a whole campaign on a model that no longer exists.",
    )
    parser.add_argument("--reasoning-effort", default="xhigh", help="Reasoning effort for Codex CLI.")
    parser.add_argument(
        "--claude-effort", default=None, choices=["low", "medium", "high", "xhigh", "max"],
        help="Effort level for the Claude CLI (--effort). Unset leaves the CLI default, as earlier runs had.",
    )
    parser.add_argument(
        "--warm-start-from", type=Path, default=None,
        help="Directory of coordinate CSVs from a previous run to place in the workspace as the current answer "
             "(the warm-start condition). Usually that model's own bare submission for this task.",
    )
    parser.add_argument(
        "--uv-cache", type=Path, default=None,
        help="Host directory bound into the container as the uv cache, shared by every run on the node so "
             "method environments hard-link packages instead of copying them.",
    )
    parser.add_argument(
        "--codex-provider", choices=["auto", "chatgpt", "azure"], default="auto",
        help="How the codex CLI reaches its model: the ChatGPT login in ~/.codex/auth.json, or an Azure "
             "OpenAI deployment by API key (files azure-openai-api-key and azure-openai-endpoint in "
             "the gen2bench config dir). auto picks azure when that key file exists.",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path("agent_workspaces"))
    parser.add_argument("--runs-root", type=Path, default=Path("agent_runs/local_coding_agent"))
    parser.add_argument("--timeout-minutes", type=float, default=60, help="Agent runtime timeout in minutes. Use 0 to disable this timeout.")
    parser.add_argument(
        "--score-timeout-seconds",
        type=float,
        default=300,
        help="Timeout for the post-agent scoring command.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=DEFAULT_PROMPT_VARIANT,
        help="Optional prompt enrichment variant.",
    )
    parser.add_argument(
        "--prebootstrap-method-venv",
        choices=["auto", "never", "always"],
        default="auto",
        help=(
            "Prebuild workspace .method-venv with common paper-method packages. "
            "auto enables this for package-backed prompt variants."
        ),
    )
    parser.add_argument(
        "--method-bootstrap-timeout-seconds",
        type=float,
        default=1200,
        help="Timeout for prebuilding the package-backed method venv. Use 0 to disable this timeout.",
    )
    parser.add_argument("--force", action="store_true", help="Recreate generated workspace files before running.")
    parser.add_argument("--dangerous-bypass", action="store_true", help="Use noninteractive agent flags that bypass approvals/sandboxes.")
    parser.add_argument(
        "--blind-solve",
        action="store_true",
        help="Run the agent in a copied-input, solve-only workspace without local scorer/source access.",
    )
    parser.add_argument(
        "--blind-mode",
        choices=["auto", "seatbelt", "container", "none"],
        default="auto",
        help=(
            "How --blind-solve is enforced. `seatbelt` is the macOS read-deny profile, "
            "`container` is Singularity/Apptainer containment for Linux, `none` relies on the "
            "workspace copy alone and is not blind. `auto` picks seatbelt on macOS and "
            "container elsewhere, and fails rather than silently falling back to none."
        ),
    )
    parser.add_argument(
        "--container-image",
        default=os.environ.get("P2B_SINGULARITY_IMAGE", "docker://python:3.11-slim"),
        help="Image for --blind-mode container. Pin an approved .sif for reproducible runs.",
    )
    parser.add_argument(
        "--container-bin",
        default=os.environ.get("P2B_SINGULARITY_BIN", "singularity"),
        help="Container runtime; Apptainer is accepted and takes the same flags.",
    )
    parser.add_argument("--gpu", action="store_true", help="Expose the assigned GPU inside the container (--nv).")
    parser.add_argument(
        "--agent-runtime",
        default=os.environ.get("P2B_AGENT_RUNTIME"),
        help="Install prefix holding the agent CLI, bound read-only into the container. "
             "Defaults to the prefix of whichever binary is on PATH. Required when the image "
             "does not already contain the CLI.",
    )
    parser.add_argument(
        "--guidance-dir",
        type=Path,
        default=None,
        help="A synthesized strategy (PLAYBOOK.md and policy/) from the generalization experiment, copied into "
             "the workspace as strategy_guide/ and described in the prompt. No prior alignment is placed on disk.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write prompt and command without invoking the agent.")
    return parser.parse_args()


def _auth_source(agent: str) -> Path | None:
    """The one credential file the CLI needs copied into private state, if it exists."""
    candidates = {
        "codex": Path.home() / ".codex" / "auth.json",
        "claude": Path.home() / ".claude" / ".credentials.json",
        "gemini": Path.home() / ".gemini",
    }
    candidate = candidates.get(agent)
    return candidate if candidate and candidate.exists() else None


def _resolve_blind_mode(args) -> str:
    """Decide how blind solving is enforced, refusing to downgrade silently.

    A run that says it is blind but has no boundary is worse than one that fails, because its
    scores enter the analysis looking comparable to genuinely isolated runs.
    """
    if not args.blind_solve:
        return "none"
    mode = args.blind_mode
    if mode == "auto":
        mode = "seatbelt" if sys.platform == "darwin" else "container"
    if mode == "seatbelt" and sys.platform != "darwin":
        raise SystemExit("--blind-mode seatbelt needs macOS; use --blind-mode container on Linux")
    if mode == "container" and resolve_binary(args.container_bin) is None:
        raise SystemExit(
            f"--blind-mode container needs {args.container_bin!r} (or apptainer) on PATH. "
            "Load the site module first, or pass --blind-mode none and label the runs as "
            "workspace-isolated rather than blind."
        )
    return mode


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src = str(SRC_ROOT)
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def _task_id(task: Path) -> str:
    match = re.search(r'^id\s*=\s*"([^"]+)"', task.read_text(encoding="utf-8"), flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find task id in {task}")
    return match.group(1)


def _run_prepare(
    task: Path,
    workspace_root: Path,
    force: bool,
    clean_submission: bool,
    blind_solve: bool,
    no_method_helper: bool,
    warm_start_from: Path | None = None,
) -> None:
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
    if no_method_helper:
        command.append("--no-method-helper")
    if warm_start_from is not None:
        command += ["--warm-start-from", str(warm_start_from)]
    subprocess.run(command, cwd=_repo_root(), check=True, env=_repo_python_env())


def _prompt(workspace: Path, prompt_variant: str, blind_solve: bool) -> str:
    task_md = (workspace / "TASK.md").read_text(encoding="utf-8")
    extra_context = render_cli_prompt_context(prompt_variant)
    extra_block = f"\n\nAdditional context:\n{extra_context}\n" if extra_context else ""
    if blind_solve:
        return f"""You are running inside a blind benchmark workspace.

Read TASK.md, inspect only the data files under data/, and write the required coordinate CSVs under submission/aligned_coordinates/.
No local scorer is available during solving. Do not search parent directories, repository paths, prior runs/results, benchmark source files, scorer code, Downloads caches, or any location outside this workspace.
Allowed inputs are the sanitized H5AD expression data, observation IDs, `obsm/spatial` coordinates, and slice identities described in TASK.md. Do not use or search for hidden aligned coordinates, ground-truth transforms, perturbation metadata, scorer labels, evaluation landmarks, split manifests, or private source paths.
You may create scripts or notes inside this workspace, but do not modify source data.
When finished, give a concise final answer describing the submitted files.

TASK.md:
{task_md}{extra_block}
"""
    return f"""You are running inside a benchmark workspace.

Read TASK.md, inspect the data files under data/, and write the required coordinate CSVs under submission/aligned_coordinates/.
You may run ./score.sh for debugging, but the benchmark runner will score deterministically after your agent process exits.
You may create scripts or notes inside this workspace, but do not modify source data.
When finished, give a concise final answer describing the submitted files.

TASK.md:
{task_md}{extra_block}
"""


def _add_guidance(workspace: Path, guidance: Path) -> str:
    """Copy a synthesized strategy into the workspace and return the prompt paragraph that points to it.

    The strategy was distilled from other agents' runs on other tasks (never this one). It is offered as a
    starting point, not as an answer to improve: nothing is written to submission/.
    """
    target = workspace / "strategy_guide"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    shutil.copy2(guidance / "PLAYBOOK.md", target / "PLAYBOOK.md")
    if (guidance / "policy").is_dir():
        shutil.copytree(guidance / "policy", target / "policy")
    return (
        "\n\nStrategy guide:\n"
        "`strategy_guide/` holds a strategy for slice alignment that was synthesized from earlier agents' work on "
        "other alignment tasks (not this one). `strategy_guide/PLAYBOOK.md` explains it: the approaches that "
        "worked, when to use each, and checks for a bad alignment. `strategy_guide/policy/` implements it "
        "(`bash strategy_guide/policy/run.sh <workspace>` writes a submission for a workspace laid out like this "
        "one). Use it as your starting point for solving this task.\n"
    )


def _elapsed(manifest: dict) -> float | None:
    """Run duration in seconds.

    Recorded rather than left to be reconstructed from the directory-name timestamp and
    finished_at, which is what analysis had to do before and which silently drops any run
    whose name does not parse.
    """
    try:
        start = datetime.fromisoformat(manifest["started_at"])
        end = datetime.fromisoformat(manifest["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return round((end - start).total_seconds(), 2)


def _cli_version(agent: str) -> str | None:
    """The agent CLI's own version string.

    A CLI upgrade part-way through a campaign changes behaviour and is otherwise invisible in
    the results, so it is recorded per run rather than assumed constant.
    """
    binary = {"codex": "codex", "claude": "claude", "gemini": "gemini", "mini-swe": "mini"}.get(agent)
    if binary is None or shutil.which(binary) is None:
        return None
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0][:120] or None


def _capture_usage(args, model: str, stdout: str, run_dir: Path, manifest: dict) -> None:
    """Record tokens and cost in the manifest, from the trace this run just produced.

    Written here rather than left to the backfill so a live run is self-describing; the
    backfill exists for the history that predates it.
    """
    usage = for_run(args.agent, model, stdout, trajectory=run_dir / "mini_trajectory.json")
    if usage is None:
        return
    manifest["usage"] = finalise(usage)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _capture_mini_trajectory(args, workspace: Path, run_dir: Path, manifest: dict) -> None:
    """Copy mini's trajectory out of the workspace and record its usage.

    The trajectory is written inside the workspace because that is the only writable path
    under containment, but the workspace is disposable, so it has to be lifted into the run
    directory before anything cleans up.
    """
    if args.agent != "mini-swe":
        return
    trajectory = workspace.resolve() / ".mini_trajectory.json"
    if trajectory.is_file():
        shutil.copy2(trajectory, run_dir / "mini_trajectory.json")


def _path_safe_model(model: str) -> str:
    """`anthropic/claude-opus-4` to `anthropic__claude-opus-4`.

    litellm model names carry a provider prefix separated by a slash, which would otherwise
    turn a run directory name into a nested path.
    """
    return model.replace("/", "__")


def _read_mini_trajectory(path: Path) -> dict:
    """Usage and exit status from mini's trajectory file.

    DefaultAgent prints almost nothing to stdout, so the trajectory is the run's only durable
    record. Token counts are not in `info.model_stats`; they sit on each assistant message
    under `extra.response.usage`, so they are summed here.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    info = data.get("info") or {}
    stats = info.get("model_stats") or {}
    usage = {
        "cost_usd": float(stats.get("instance_cost") or 0.0),
        "api_calls": int(stats.get("api_calls") or 0),
        "exit_status": info.get("exit_status") or "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
    }
    for message in data.get("messages") or []:
        api_usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        usage["input_tokens"] += int(api_usage.get("prompt_tokens") or 0)
        usage["output_tokens"] += int(api_usage.get("completion_tokens") or 0)
        usage["cached_input_tokens"] += int(api_usage.get("cache_read_input_tokens") or 0)
    return usage


def _mini_builtin_config() -> str:
    """Absolute path to mini's own bundled mini.yaml.

    Passing any `-c` replaces mini's default config list rather than adding to it, and
    `system_template` and `instance_template` have no built-in defaults, so the bundled file
    has to be named explicitly whenever an override is passed. Resolved from the interpreter
    in the `mini` shebang rather than from our venv, because mini is installed as its own tool.
    """
    mini = shutil.which("mini")
    if mini is None:
        raise SystemExit("mini-swe-agent is not on PATH; install it before using --agent mini-swe")
    shebang = Path(mini).read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    interpreter = Path(shebang.lstrip("#!").strip()) if shebang.startswith("#!") else Path(sys.executable)
    root = interpreter.parent.parent
    matches = sorted(root.glob("lib/python3.*/site-packages/minisweagent/config/mini.yaml"))
    for candidate in reversed(matches):
        if candidate.is_file():
            return str(candidate)
    raise SystemExit(f"mini-swe-agent bundled config not found under {root}")


def _mini_command(model: str, prompt: str, workspace: Path, trajectory: Path,
                  timeout_minutes: float) -> list[str]:
    """mini's argv for an unattended run.

    Every flag here is load-bearing. `--agent-class default` because upstream defaults to the
    interactive class, which blocks forever without a terminal. `--environment-class local`
    because isolation is the outer sandbox, not mini's own. `-y` and `--exit-immediately` to
    suppress the two confirmation prompts. `-o` because DefaultAgent prints almost nothing to
    stdout, so the trajectory file is the only durable record of the run. `-l 0` because the
    upstream cost limit is $3.00, which silently truncates a run that then looks complete.
    `environment.timeout` because upstream kills any single shell command after 30 seconds and
    loading one of our h5ad slices takes longer than that, so every real step would fail.
    """
    wall_limit = int(timeout_minutes * 60) - 300 if timeout_minutes else 28500
    return [
        "mini",
        "--agent-class", "default",
        "--environment-class", "local",
        "-y",
        "--exit-immediately",
        "-m", model,
        "-l", os.environ.get("MINI_COST_LIMIT_USD", "0"),
        "-o", str(trajectory),
        "-c", _mini_builtin_config(),
        "-c", f"agent.wall_time_limit_seconds={max(wall_limit, 600)}",
        "-c", f"environment.timeout={os.environ.get('MINI_COMMAND_TIMEOUT_SECONDS', '3600')}",
        "-c", f"environment.cwd={workspace}",
        "-t", prompt,
    ]


def _sandbox_profile_literal(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _write_blind_read_deny_profile(run_dir: Path, repo: Path) -> Path | None:
    """Deny known private benchmark roots to the Codex process and all child tools.

    Codex's built-in workspace-write sandbox still allows reads outside the workspace on macOS.
    This outer Seatbelt profile keeps the repo/scorer/source paths unreadable during solving,
    while the parent harness remains outside the sandbox and can score after Codex exits.
    """
    if sys.platform != "darwin" or shutil.which("sandbox-exec") is None:
        return None

    profile_root = Path("/tmp/p2b_spatial_blind_profiles") / run_dir.name
    profile_root.mkdir(parents=True, exist_ok=True)

    raw_deny_roots = [
        repo.resolve(),
        Path.home() / "Downloads" / "paper2bench-data",
        # The SABench checkout contains public benchmark code and result tables.
        # It is a reference artifact, not an allowed solve-time input.
        Path.home() / "Downloads" / "SABench",
        Path("/tmp/p2b_spatial_prior_artifacts"),
        Path("/private/tmp/p2b_spatial_prior_artifacts"),
        Path("/tmp/p2b_spatial_blind_profiles"),
        Path("/private/tmp/p2b_spatial_blind_profiles"),
    ]
    deny_roots: list[Path] = []
    seen_roots: set[str] = set()
    for root in raw_deny_roots:
        resolved = root.resolve()
        if str(resolved) not in seen_roots:
            seen_roots.add(str(resolved))
            deny_roots.append(resolved)
    read_filters = "\n".join(f'  (subpath "{_sandbox_profile_literal(path)}")' for path in deny_roots)
    profile = f"""(version 1)
(allow default)
(deny file-read-data
{read_filters})
(deny file-read-metadata
{read_filters})
(deny file-write*
{read_filters})
"""
    profile_path = (profile_root / "blind_read_deny.sb").resolve()
    profile_path.write_text(profile, encoding="utf-8")
    return profile_path


def _build_command(
    args: argparse.Namespace,
    workspace: Path,
    prompt: str,
    *,
    sandbox_profile: Path | None,
) -> list[str]:
    model = args.model
    if args.agent == "codex":
        # A bare name: under a container the binary is the image's, and a host path resolved
        # here would not exist inside it.
        codex_bin = "codex" if args.blind_mode == "container" else (shutil.which("codex") or "codex")
        command = [
            codex_bin,
            "exec",
            "--json",
            "--model",
            model,
            "-c",
            f"model_reasoning_effort={args.reasoning_effort}",
            "--skip-git-repo-check",
        ]
        if args.blind_solve:
            command.extend(["--cd", str(workspace.resolve()), "--dangerously-bypass-approvals-and-sandbox"])
        if args.dangerous_bypass:
            command.append("--dangerously-bypass-approvals-and-sandbox")
        command.append(prompt)
    elif args.agent == "mini-swe":
        # The trajectory goes inside the workspace because that is the only writable path
        # under containment, and it is copied into the run directory afterwards.
        command = _mini_command(model, prompt, workspace.resolve(),
                                workspace.resolve() / ".mini_trajectory.json",
                                args.timeout_minutes)
    elif args.agent == "gemini":
        # Gemini CLI documents --prompt as headless mode and stream-json as a
        # machine-readable output format.  Use PATH so a container image can
        # provide the CLI rather than relying on a host-specific absolute path.
        command = [
            "gemini",
            "--prompt", prompt,
            "--output-format", "stream-json",
            "--model", model,
            "--skip-trust",
        ]
        # The outer Seatbelt/container boundary is the protection in blind
        # mode, analogous to Codex's unattended inner invocation.
        if args.blind_solve or args.dangerous_bypass:
            command.append("--yolo")
    else:
        command = [
            "claude",
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--model",
            model,
            "--add-dir",
            str(workspace),
        ]
        if args.claude_effort:
            command.extend(["--effort", args.claude_effort])
        # Same reasoning as Codex and Gemini above: in blind mode the boundary is the outer
        # Seatbelt profile or container, not the CLI's own approval prompt. Without this the
        # agent is asked to confirm every Write and Bash, gets no answer because nothing is
        # attached to stdin, and finishes having written no submission at all.
        if args.blind_solve or args.dangerous_bypass:
            command.append("--dangerously-skip-permissions")
        command.extend(["-p", prompt])

    # On macOS apply the same outer filesystem boundary to every local agent,
    # rather than treating Codex as the sole agent that needs isolation.
    if args.blind_solve and sandbox_profile is not None:
        return ["/usr/bin/sandbox-exec", "-f", str(sandbox_profile), *command]
    return command


def _run_score(task: Path, workspace: Path, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    env = _repo_python_env()
    env["P2B_ALLOW_EVAL_SCORING"] = "1"
    # Current SABench-faithful tasks use `all`; legacy hidden-split tasks retain
    # their explicit eval setting.  Do not hard-code eval here, or an all-spot
    # result is misleadingly labelled as an eval score in its provenance.
    score_split = load_task_spec(task).score_split
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
            "--score-split",
            score_split,
        ],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=env,
    )


def _prebootstrap_target(args: argparse.Namespace) -> str | None:
    if args.prompt_variant in NO_METHOD_HELPER_PROMPT_VARIANTS:
        return None
    target = method_prebootstrap_target(args.prompt_variant)
    if args.prebootstrap_method_venv == "always":
        return target or "paper-common"
    if args.prebootstrap_method_venv == "never":
        return None
    return target


def _wheelhouse() -> Path | None:
    """Prebuilt wheels for method packages the container cannot compile (see Containment.wheelhouse):
    P2B_WHEELHOUSE if set, else the JHPCE copy when it exists."""
    if os.environ.get("P2B_METHOD_ENV_PROFILE") == "legacy":
        return None
    path = Path(os.environ.get("P2B_WHEELHOUSE", "/dcs07/chen/data/alignment-agent/p2b-wheelhouse"))
    return path.resolve() if path.is_dir() and any(path.glob("*.whl")) else None


def _prebootstrap_method_venv(workspace: Path, target: str, timeout_seconds: float,
                              containment=None) -> subprocess.CompletedProcess[str]:
    """Build the method venv where the agent will run.

    Under a container that means inside it: the image has uv and the python the venv must
    point at, and the workspace is bound at the same path, so the venv the agent finds is
    the one built here. Built on the host instead, it needs a uv the host may not have and
    links to an interpreter the container cannot see.
    """
    script = workspace.resolve() / "bootstrap_method_venv.sh"
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    command = ["bash", str(script), target]
    if containment is not None:
        command = containment.wrap(command)
    return subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


MINI_PROVIDER_KEYS = {
    "anthropic": ("anthropic-api-key", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini-api-key", "GEMINI_API_KEY"),
    "openai": ("openai-api-key", "OPENAI_API_KEY"),
}


def _mini_provider_key(env: dict[str, str], model: str) -> dict[str, str]:
    """Put back exactly the one API key mini needs, and nothing else.

    Blind mode strips every credential from the environment, which is right for claude, codex
    and gemini because they authenticate by OAuth and have no business seeing stray keys. mini
    authenticates only by key, so the strip left it with no way to reach any provider and every
    run died with APIConnectionError.

    The key is read from the config directory rather than inherited from the ambient
    environment, so a run depends on a file someone deliberately placed, and only the provider
    named by the model prefix is restored. This mirrors the container, which mounts one
    credential per agent under /run/secrets.
    """
    provider = model.split("/", 1)[0] if "/" in model else ""
    entry = MINI_PROVIDER_KEYS.get(provider)
    if entry is None:
        raise SystemExit(
            f"mini-swe model {model!r} has no provider prefix I know. Use one of "
            f"{sorted(MINI_PROVIDER_KEYS)}, for example 'gemini/gemini-3.5-flash'.")
    filename, var = entry
    config_dir = Path(os.environ.get("GEN2BENCH_CONFIG_DIR", Path.home() / ".config" / "gen2bench"))
    key_file = config_dir / filename
    if not key_file.is_file():
        raise SystemExit(
            f"mini-swe needs {key_file} for {provider}. Subscription tokens do not work through "
            "litellm; see docs/CREDENTIALS.md.")
    env[var] = key_file.read_text(encoding="utf-8").strip()
    return env


def _mini_environment(env: dict[str, str], state_root: Path) -> dict[str, str]:
    """Settings mini needs to run unattended, wherever it runs.

    Without MSWEA_CONFIGURED mini calls `configure_if_first_time`, which opens a
    prompt_toolkit prompt and dies with `OSError: [Errno 22]` the moment stdin is not a
    terminal, so every batch run fails before the first model call. Its config directory has
    to be writable and private to the run, and the bundled litellm cost map avoids a network
    fetch at import time.
    """
    env["MSWEA_CONFIGURED"] = "1"
    env["MSWEA_SILENT_STARTUP"] = "1"
    env["MSWEA_GLOBAL_CONFIG_DIR"] = str(state_root)
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    # litellm refuses to run when its price map has no entry for the model, which is the case
    # for anything newer than its pinned map. Cost then comes from our own table instead.
    env["MSWEA_COST_TRACKING"] = "ignore_errors"
    return env


AGENT_BINARIES = {"codex": "codex", "claude": "claude", "gemini": "gemini", "mini-swe": "mini"}


def _image_has_cli(agent: str, image: str, container_bin: str) -> bool:
    """Whether the image already ships the agent CLI, in which case nothing is bound in."""
    binary = resolve_binary(container_bin)
    if binary is None or agent not in AGENT_BINARIES:
        return False
    probe = subprocess.run(
        [binary, "exec", "--containall", "--cleanenv", image, "sh", "-c",
         f"command -v {AGENT_BINARIES[agent]}"],
        capture_output=True, text=True, timeout=120,
    )
    return probe.returncode == 0


def _agent_runtime_prefix(agent: str, override: str | None, *,
                          image: str | None = None, container_bin: str = "singularity") -> Path | None:
    """The install prefix to bind read-only into the container so the CLI exists inside it.

    A container that starts from a plain python image sees nothing of the host under
    --containall, so the agent binary is simply absent and every job dies with "not found".
    Binding a host install in is the fallback. The image built from containers/gen2bench.def
    already carries every CLI, and that is preferred: nothing is bound, and every job runs
    the version the image pins rather than whatever the submit host happens to have.

    The prefix is the parent of the directory holding the binary, so bin/ and the lib/ or
    node_modules/ beside it come in together: the claude and gemini CLIs are Node programs and
    do not work from a bare bin/ directory alone.
    """
    if override:
        return Path(override).resolve()
    if image and _image_has_cli(agent, image, container_bin):
        return None
    found = shutil.which(AGENT_BINARIES.get(agent, ""))
    if not found:
        return None
    return Path(found).resolve().parent.parent


def _agent_binary_dir(agent: str) -> str | None:
    """Directory holding the agent CLI, resolved before the PATH is narrowed.

    Blind runs hand the agent a fixed minimal PATH, which silently excludes anything installed
    under ~/.local/bin. `claude` survived that only because Homebrew also puts a copy in
    /opt/homebrew/bin; `mini` has no such copy and every run died instantly with
    "sandbox-exec: execvp() of 'mini' failed", which reads like a missing install rather than a
    PATH problem. Resolving the directory here and adding it back keeps the narrow PATH without
    depending on which installer a host happened to use.
    """
    binary = {"codex": "codex", "claude": "claude", "gemini": "gemini", "mini-swe": "mini"}.get(agent)
    found = shutil.which(binary) if binary else None
    return str(Path(found).resolve().parent) if found else None


def _agent_environment(repo: Path, workspace: Path, *, blind_solve: bool, agent: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if blind_solve:
        # Blind solve agents should not inherit external model/API credentials.
        # Codex CLI authentication is handled by the user's Codex login, not by
        # exposing provider keys inside the benchmark workspace shell.
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "OPENAI_ORGANIZATION",
            "OPENAI_PROJECT",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "MOOD_OPENAI_API_KEY",
            "MOOD_OPENAI_ENDPOINT",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY",
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
            "DEEPSEEK_API_KEY",
            "XAI_API_KEY",
        ]:
            env.pop(key, None)
        private_tokens = [
            str(repo.resolve()),
            str((Path.home() / "Downloads" / "paper2bench-data").resolve()),
            str((Path.home() / "Downloads" / "SABench").resolve()),
            str(Path("/tmp/p2b_spatial_prior_artifacts").resolve()),
            str(Path("/tmp/p2b_spatial_blind_profiles").resolve()),
        ]
        for key, value in list(env.items()):
            if any(token and token in value for token in private_tokens):
                env.pop(key, None)
        path_entries = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
        binary_dir = _agent_binary_dir(agent) if agent else None
        if binary_dir and binary_dir not in path_entries:
            path_entries.insert(0, binary_dir)
        env["PATH"] = ":".join(path_entries)
        env["PWD"] = str(workspace.resolve())
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"
        env.setdefault("TMPDIR", "/tmp")
        return env

    repo_src = str(repo / "src")
    env["PYTHONPATH"] = repo_src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else repo_src
    return env


def _snapshot_workspace(run_dir: Path, workspace: Path) -> Path:
    snapshot_dir = run_dir / "workspace_snapshot"
    shutil.rmtree(snapshot_dir, ignore_errors=True)

    excluded_root_names = {"data", ".method-cache", ".uv-cache", "__pycache__"}

    def _ignore(path: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__"} & set(names)
        if Path(path).resolve() == workspace.resolve():
            ignored |= excluded_root_names & set(names)
            ignored |= {name for name in names if name == ".method-venv" or name.startswith(".method-venv-")}
        return ignored

    shutil.copytree(workspace, snapshot_dir, ignore=_ignore)
    return snapshot_dir


def _write_leakage_audit(task: Path, workspace: Path, run_dir: Path, manifest: dict) -> None:
    """Record a non-fatal leakage audit beside the run artifacts."""
    try:
        from audit_spatial_agent_leakage import audit_run

        audit = audit_run(task, workspace, run_dir)
        (run_dir / "leakage_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
        manifest["leakage_audit"] = str(run_dir / "leakage_audit.json")
        manifest["leakage_status"] = audit["status"]
        manifest["leakage_findings"] = audit["summary"]["total_findings"]
        manifest["leakage_high_findings"] = audit["summary"]["high_findings"]
    except Exception as exc:
        manifest["leakage_audit_error"] = str(exc)[:1000]


_FAILURE_SLICES: list | None = None   # set in main(), so a failed run gets identity MAE


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
            "aggregate": failure_aggregate(_FAILURE_SLICES),
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
    args = parse_args()
    if args.blind_solve and args.dangerous_bypass:
        raise SystemExit("--blind-solve cannot be combined with --dangerous-bypass")
    repo = _repo_root()
    task = args.task if args.task.is_absolute() else repo / args.task
    task_arg = task.relative_to(repo) if task.is_relative_to(repo) else task
    task_id = _task_id(task)
    workspace = args.workspace_root / task_id
    global _FAILURE_SLICES
    try:
        _FAILURE_SLICES = load_task_spec(task).runnable_slices
    except Exception:
        _FAILURE_SLICES = None
    if args.blind_solve and workspace.resolve().is_relative_to(repo.resolve()):
        raise SystemExit("--blind-solve requires --workspace-root outside the repository so private repo paths can be read-denied")
    no_method_helper = args.prompt_variant in NO_METHOD_HELPER_PROMPT_VARIANTS
    _run_prepare(
        task_arg,
        args.workspace_root,
        args.force,
        clean_submission=not args.dry_run,
        blind_solve=args.blind_solve,
        no_method_helper=no_method_helper,
        warm_start_from=args.warm_start_from,
    )
    prompt = _prompt(workspace, args.prompt_variant, args.blind_solve)
    if args.guidance_dir is not None:
        prompt += _add_guidance(workspace, args.guidance_dir)

    model = args.model
    run_name = f"{task_id}_{args.agent}_{_path_safe_model(model)}{prompt_variant_tag(args.prompt_variant)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    blind_mode = _resolve_blind_mode(args)
    sandbox_profile = (
        _write_blind_read_deny_profile(run_dir, repo)
        if blind_mode == "seatbelt"
        else None
    )
    if blind_mode == "seatbelt" and sandbox_profile is None:
        raise SystemExit("--blind-mode seatbelt requires macOS sandbox-exec on this host")
    command = _build_command(args, workspace, prompt, sandbox_profile=sandbox_profile)

    containment = None
    provider = codex_provider(args.codex_provider) if args.agent == "codex" else None
    if blind_mode == "container":
        containment = Containment(
            image=args.container_image,
            workspace=workspace.resolve(),
            state_dir=stage_state(
                args.agent, auth_source=_auth_source(args.agent),
                codex_config=codex_azure_config() if provider == "azure" else None),
            agent=args.agent,
            binary=resolve_binary(args.container_bin),
            use_gpu=args.gpu,
            secrets=secret_routes(args.agent, provider=provider),
            uv_cache=args.uv_cache.resolve() if args.uv_cache else None,
            wheelhouse=_wheelhouse(),
            glibc_execstack=os.environ.get("P2B_METHOD_ENV_PROFILE") != "legacy",
            runtime_dir=_agent_runtime_prefix(args.agent, args.agent_runtime,
                                              image=args.container_image, container_bin=args.container_bin),
        )
        if containment.runtime_dir is None and not _image_has_cli(args.agent, args.container_image, args.container_bin):
            raise SystemExit(
                f"container mode needs the {args.agent} CLI inside the container. Either pass "
                "--agent-runtime with the install prefix, or build an image that already has it "
                "(see containers/README.md)."
            )
        # the agent sees the node's shared uv cache read-only (see Containment.binds); the prebootstrap,
        # which builds the method environments, still writes it
        command = containment.wrap(command, uv_cache_readonly=True)
    prebootstrap_target = _prebootstrap_target(args)
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")

    manifest = {
        "runner": "local_coding_agent",
        "cli_version": _cli_version(args.agent),
        "blind_mode": blind_mode,
        "container_image": args.container_image if blind_mode == "container" else None,
        "agent": args.agent,
        "codex_provider": provider,
        "warm_start_from": str(args.warm_start_from) if args.warm_start_from else None,
        "guidance_dir": str(args.guidance_dir) if args.guidance_dir else None,
        "model": model,
        "reasoning_effort": args.reasoning_effort if args.agent == "codex" else args.claude_effort if args.agent == "claude" else None,
        "prompt_variant": args.prompt_variant,
        "task": str(task),
        "workspace": str(workspace),
        "dangerous_bypass": args.dangerous_bypass,
        "blind_solve": args.blind_solve,
        "blind_read_deny_profile": str(sandbox_profile) if sandbox_profile is not None else None,
        "blind_environment_sanitized": args.blind_solve,
        "blind_agent_provider_keys_stripped": args.blind_solve,
        "blind_codex_execution_mode": "outer-sandbox-with-unattended-bypass" if args.blind_solve and args.agent == "codex" else None,
        "blind_gemini_execution_mode": "outer-sandbox-with-yolo" if args.blind_solve and args.agent == "gemini" else None,
        "dry_run": args.dry_run,
        "method_env_profile": os.environ.get("P2B_METHOD_ENV_PROFILE", "current"),
        "uv_cache_agent_mount": "ro" if (containment is not None and containment.uv_cache) else None,
        "wheelhouse": sorted(p.name for p in containment.wheelhouse.glob("*.whl")) if (containment is not None and containment.wheelhouse) else None,
        "method_env_prebootstrap_mode": args.prebootstrap_method_venv,
        "method_env_prebootstrap_requested": prebootstrap_target is not None,
        "method_env_prebootstrap_target": prebootstrap_target,
        "method_env_prebootstrap_timeout_seconds": args.method_bootstrap_timeout_seconds,
        "method_helper_exposed": not no_method_helper,
        "started_at": datetime.now().isoformat(),
    }

    if args.dry_run:
        manifest["method_env_prebootstrap_skipped"] = "dry_run"
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["elapsed_seconds"] = _elapsed(manifest)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Dry run wrote {run_dir / 'command.json'}")
        return

    if prebootstrap_target is not None:
        try:
            bootstrap_result = _prebootstrap_method_venv(
                workspace, prebootstrap_target, args.method_bootstrap_timeout_seconds, containment=containment)
            (run_dir / "method_env_prebootstrap_stdout.txt").write_text(bootstrap_result.stdout, encoding="utf-8")
            (run_dir / "method_env_prebootstrap_stderr.txt").write_text(bootstrap_result.stderr, encoding="utf-8")
            manifest["method_env_prebootstrap"] = {
                "target": prebootstrap_target,
                "returncode": bootstrap_result.returncode,
                "python": str((workspace / ".method-venv" / "bin" / "python").resolve()),
            }
            if bootstrap_result.returncode != 0:
                manifest["status"] = "failed"
                manifest["failure_stage"] = "method_env_prebootstrap"
                manifest["failure_message"] = (bootstrap_result.stderr or bootstrap_result.stdout or "method venv prebootstrap failed").strip()[:4000]
                _write_failure_score(run_dir, workspace, stage="method_env_prebootstrap", message=manifest["failure_message"])
                manifest["finished_at"] = datetime.now().isoformat()
                manifest["elapsed_seconds"] = _elapsed(manifest)
                (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"Wrote failed run artifacts to {run_dir}")
                return
        except subprocess.TimeoutExpired as exc:
            manifest["status"] = "failed"
            manifest["failure_stage"] = "method_env_prebootstrap_timeout"
            manifest["failure_message"] = str(exc)
            (run_dir / "method_env_prebootstrap_stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
            (run_dir / "method_env_prebootstrap_stderr.txt").write_text(exc.stderr or str(exc), encoding="utf-8")
            _write_failure_score(run_dir, workspace, stage="method_env_prebootstrap_timeout", message=str(exc))
            manifest["finished_at"] = datetime.now().isoformat()
            manifest["elapsed_seconds"] = _elapsed(manifest)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Wrote failed run artifacts to {run_dir}")
            return

    env = _agent_environment(repo, workspace, blind_solve=args.blind_solve, agent=args.agent)
    if args.agent == "gemini":
        # A private, empty home per run. The CLI prefers cached OAuth in ~/.gemini over
        # GEMINI_API_KEY, so an interactive login that has exhausted its quota makes every
        # batch run fail at onboardUser even when a working key is set. An empty home falls
        # through to the key, and also keeps the interactive account's history out of the run.
        gemini_home = run_dir / "gemini-home"
        gemini_home.mkdir(parents=True, exist_ok=True)
        env["GEMINI_CLI_HOME"] = str(gemini_home)
    if args.agent == "mini-swe":
        env = _mini_environment(env, run_dir / "mini-config")
        env = _mini_provider_key(env, model)
        (run_dir / "mini-config").mkdir(parents=True, exist_ok=True)
    result: subprocess.CompletedProcess[str] | None = None
    agent_timeout_seconds = None if args.timeout_minutes <= 0 else args.timeout_minutes * 60
    if containment is not None:
        # Private state (HOME, caches, the staged credential) is per run; drop it with the run.
        import atexit
        atexit.register(shutil.rmtree, containment.state_dir, True)
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=agent_timeout_seconds,
            env=env,
        )
        manifest["agent_returncode"] = result.returncode
        (run_dir / "raw_stdout.jsonl").write_text(result.stdout, encoding="utf-8")
        _capture_mini_trajectory(args, workspace, run_dir, manifest)
        _capture_usage(args, model, result.stdout, run_dir, manifest)
        (run_dir / "raw_stderr.txt").write_text(result.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        manifest["agent_returncode"] = None
        manifest["status"] = "failed"
        manifest["failure_stage"] = "agent_timeout"
        manifest["failure_message"] = str(exc)
        (run_dir / "raw_stdout.jsonl").write_text(str(exc.stdout or ""), encoding="utf-8")
        (run_dir / "raw_stderr.txt").write_text(str(exc.stderr or exc), encoding="utf-8")
        _write_failure_score(run_dir, workspace, stage="agent_timeout", message=str(exc))
        _write_leakage_audit(task, workspace, run_dir, manifest)
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["elapsed_seconds"] = _elapsed(manifest)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote failed run artifacts to {run_dir}")
        return
    except FileNotFoundError as exc:
        manifest["agent_returncode"] = None
        manifest["agent_error"] = f"{args.agent!r} CLI not found: {exc}"
        manifest["status"] = "failed"
        manifest["failure_stage"] = "agent_cli_missing"
        manifest["failure_message"] = str(exc)
        (run_dir / "raw_stdout.jsonl").write_text("", encoding="utf-8")
        (run_dir / "raw_stderr.txt").write_text(str(exc), encoding="utf-8")

    try:
        score_result = _run_score(task_arg, workspace, timeout_seconds=args.score_timeout_seconds)
        manifest["score_returncode"] = score_result.returncode
        (run_dir / "score_stdout.txt").write_text(score_result.stdout, encoding="utf-8")
        (run_dir / "score_stderr.txt").write_text(score_result.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        manifest["score_returncode"] = None
        manifest["status"] = "failed"
        manifest["failure_stage"] = "score_timeout"
        manifest["failure_message"] = str(exc)
        _write_failure_score(run_dir, workspace, stage="score_timeout", message=str(exc))
        _write_leakage_audit(task, workspace, run_dir, manifest)
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["elapsed_seconds"] = _elapsed(manifest)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote failed run artifacts to {run_dir}")
        return

    workspace_scores = workspace / "scores.json"
    if result is None:
        message = str(manifest.get("failure_message", "Agent did not run."))
        _write_failure_score(run_dir, workspace, stage=str(manifest.get("failure_stage", "agent")), message=message)
    elif result.returncode != 0:
        message = (result.stderr or result.stdout or "Agent exited non-zero.").strip()
        manifest["status"] = "failed"
        manifest["failure_stage"] = "agent"
        manifest["failure_message"] = message[:4000]
        _write_failure_score(run_dir, workspace, stage="agent", message=message)
    elif score_result.returncode != 0:
        message = (score_result.stderr or score_result.stdout or "Scorer exited non-zero.").strip()
        manifest["status"] = "failed"
        manifest["failure_stage"] = "score"
        manifest["failure_message"] = message[:4000]
        _write_failure_score(run_dir, workspace, stage="score", message=message)
    elif workspace_scores.exists():
        manifest["status"] = "scored"
        shutil.copy2(workspace_scores, run_dir / "scores.json")
    else:
        message = f"Scorer succeeded but did not write {workspace_scores}."
        manifest["status"] = "failed"
        manifest["failure_stage"] = "missing_score"
        manifest["failure_message"] = message
        _write_failure_score(run_dir, workspace, stage="missing_score", message=message)

    try:
        manifest["workspace_snapshot"] = str(_snapshot_workspace(run_dir, workspace))
    except Exception as exc:
        manifest["workspace_snapshot_error"] = str(exc)[:1000]

    _write_leakage_audit(task, workspace, run_dir, manifest)

    # Did this run's method environments work? (src/genomics_agent_bench/env_health.py; env_broken means the run hit
    # damaged packages and should be rerun, the other fields record limitations that are the same on every node)
    try:
        from genomics_agent_bench.env_health import scan_run
        manifest["env_health"] = scan_run(run_dir)
    except Exception as exc:
        manifest["env_health"] = {"error": str(exc)[:300]}

    manifest["finished_at"] = datetime.now().isoformat()
    manifest["elapsed_seconds"] = _elapsed(manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
