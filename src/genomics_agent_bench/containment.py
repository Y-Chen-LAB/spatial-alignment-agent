"""Linux containment for blind solving, ported in shape from the perturbation benchmark.

Blind solving on macOS uses a Seatbelt read-deny profile, which does not exist on Linux, so
cluster runs had no enforcement at all: the workspace copy and the stripped environment are
conventions, not boundaries. This module supplies the boundary with Singularity or Apptainer.

The containment is `--containall --cleanenv --no-home`, so nothing from the host is visible
except what is bound explicitly:

    workspace        read-write, and the working directory
    state directory  read-write at /agent-state, holding HOME, TMPDIR and the CLI's config
    secrets          read-only under /run/secrets, one file per credential
    agent runtime    read-only, when a prepared environment is supplied

The repository, the unsanitised slices, the split manifests and the scorer are simply absent,
which is the property the Seatbelt profile was approximating. `prepare_agent_task.py
--blind-solve` already writes a self-contained workspace, with real copies of the sanitised
slices rather than symlinks, so binding the workspace alone is sufficient.

Ported in shape from perturbation-coding-agent/run_benchmark_slurm.py, with our flag names
and our provider set. The credential table drives both the binds and the environment exports
from one place, so a route cannot be half-added.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

SECRET_ROOT = "/run/secrets"
STATE_MOUNT = "/agent-state"
UV_CACHE_MOUNT = "/uv-cache"
WHEELHOUSE_MOUNT = "/p2b-wheels"


@dataclass(frozen=True)
class SecretRoute:
    """One credential: where it lives on the host, and which variable the CLI reads."""
    source: Path
    name: str
    env_var: str

    @property
    def container_path(self) -> str:
        return f"{SECRET_ROOT}/{self.name}"


def config_root(config_dir: Path | None = None) -> Path:
    return config_dir or Path(os.environ.get("GEN2BENCH_CONFIG_DIR",
                                            Path.home() / ".config" / "gen2bench"))


def codex_provider(requested: str = "auto", *, config_dir: Path | None = None) -> str:
    """Which backend the codex CLI authenticates against.

    "chatgpt" is the subscription login in ~/.codex/auth.json. "azure" is the Azure OpenAI
    deployment reached with an API key, which needs no login and bills the Azure resource;
    the CLI runs its full agent loop either way. "auto" picks azure when the key file exists,
    since a login is the thing a compute node can never perform.
    """
    if requested != "auto":
        return requested
    return "azure" if (config_root(config_dir) / "azure-openai-api-key").is_file() else "chatgpt"


def codex_azure_config(config_dir: Path | None = None) -> str:
    """The codex config.toml that points the CLI at the Azure resource."""
    endpoint_file = config_root(config_dir) / "azure-openai-endpoint"
    if not endpoint_file.is_file():
        raise RuntimeError(f"codex on Azure needs the endpoint in {endpoint_file}; see docs/CREDENTIALS.md")
    base = endpoint_file.read_text(encoding="utf-8").strip().rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    return (
        'model_provider = "azure"\n\n'
        "[model_providers.azure]\n"
        'name = "Azure OpenAI"\n'
        f'base_url = "{base}"\n'
        'env_key = "AZURE_OPENAI_API_KEY"\n'
        'wire_api = "responses"\n'
    )


def secret_routes(agent: str, *, config_dir: Path | None = None, provider: str | None = None) -> list[SecretRoute]:
    """Credentials the named agent needs, restricted to those that exist on this host."""
    config = config_root(config_dir)
    candidates: list[tuple[str, Path, str, str]] = [
        ("codex", config / "openai-api-key", "openai-api-key", "OPENAI_API_KEY"),
        ("codex-azure", config / "azure-openai-api-key", "azure-openai-api-key", "AZURE_OPENAI_API_KEY"),
        # No ANTHROPIC_API_KEY for claude: the CLI prefers it over the subscription login that
        # the arm is defined by, and answered 401 the moment it was exported. The setup token
        # is the same login in portable form and stays.
        ("claude", Path.home() / ".claude" / "setup-token", "claude-oauth-token", "CLAUDE_CODE_OAUTH_TOKEN"),
        ("gemini", config / "gemini-api-key", "gemini-api-key", "GEMINI_API_KEY"),
        # mini-swe reaches Anthropic through litellm, which needs a pay-per-token key. The
        # Claude Code subscription token is gated to that client and returns masked 429s
        # through litellm, so it must not be routed here.
        ("mini-swe", config / "anthropic-api-key", "anthropic-api-key", "ANTHROPIC_API_KEY"),
        ("mini-swe", config / "gemini-api-key", "gemini-api-key", "GEMINI_API_KEY"),
    ]
    owner_key = "codex-azure" if agent == "codex" and provider == "azure" else agent
    if agent == "codex" and provider == "chatgpt":
        # The ChatGPT login (auth.json) is the credential; passing OPENAI_API_KEY as well could make the CLI bill
        # the API account instead, so it is not injected for this provider.
        return []
    return [SecretRoute(path.resolve(), name, env)
            for owner, path, name, env in candidates if owner == owner_key and path.is_file()]


def temporary_root() -> Path | None:
    """Node-local scratch when SLURM offers it, failing closed if it is misconfigured.

    Falling back to a shared filesystem silently would put credentials and agent state on
    storage other jobs can reach, so a broken SLURM_TMPDIR is an error rather than a default.
    """
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if not slurm_tmpdir:
        return None
    root = Path(slurm_tmpdir).resolve()
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise RuntimeError(f"SLURM_TMPDIR is not a writable directory: {slurm_tmpdir}")
    return root


def stage_state(agent: str, *, auth_source: Path | None = None, codex_config: str | None = None) -> Path:
    """Private per-run state holding HOME, TMPDIR and only the CLI's own credentials."""
    state = Path(tempfile.mkdtemp(prefix=f"p2b-{agent}-state-", dir=temporary_root()))
    state.chmod(0o700)
    for sub in ("home", "tmp", "pip-cache"):
        (state / sub).mkdir(mode=0o700)

    if agent == "codex" and codex_config is not None:
        # Azure: a provider config instead of a login file. CODEX_HOME points here.
        target = state / "codex"
        target.mkdir(mode=0o700)
        (target / "config.toml").write_text(codex_config, encoding="utf-8")
        (target / "config.toml").chmod(0o600)
        return state

    if auth_source is None or not auth_source.exists():
        return state
    # Copy only the credential the CLI needs, never a whole config directory, which would
    # carry histories and settings from the interactive session into the run.
    if agent == "codex":
        target = state / "codex"
        target.mkdir(mode=0o700)
        shutil.copy2(auth_source, target / "auth.json")
        (target / "auth.json").chmod(0o600)
    elif agent == "claude":
        target = state / "claude"
        target.mkdir(mode=0o700)
        shutil.copy2(auth_source, target / ".credentials.json")
        (target / ".credentials.json").chmod(0o600)
    elif agent == "gemini":
        # Deliberately no OAuth material. The CLI prefers cached OAuth over GEMINI_API_KEY, so
        # copying it in makes the container inherit whatever quota state the interactive login
        # is in; an empty home falls through to the key mounted under /run/secrets.
        (state / ".gemini").mkdir(mode=0o700)
    return state


@dataclass
class Containment:
    """Builds the `singularity exec` argv that wraps an agent command."""

    image: str
    workspace: Path
    state_dir: Path
    agent: str
    binary: str = "singularity"
    use_gpu: bool = False
    runtime_dir: Path | None = None
    secrets: list[SecretRoute] = field(default_factory=list)
    uv_cache: Path | None = None      # a host directory shared by every run on the node
    # prebuilt wheels for packages that only ship sources needing a compiler the image lacks (annoy, hnswlib,
    # torch-scatter, torch-sparse for STAligner), offered to uv and pip as find-links
    wheelhouse: Path | None = None
    glibc_execstack: bool = True      # False reproduces the legacy environment (torch 2.3 fails to load under glibc 2.41)

    def binds(self, uv_cache_readonly: bool = False) -> list[str]:
        mounts = [
            f"{self.workspace}:{self.workspace}:rw",
            f"{self.state_dir}:{STATE_MOUNT}:rw",
        ]
        mounts += [f"{r.source}:{r.container_path}:ro" for r in self.secrets]
        if self.runtime_dir:
            mounts.append(f"{self.runtime_dir}:{self.runtime_dir}:ro")
        if self.uv_cache:
            # Read-write while the method environments are built, read-only while the agent runs: the
            # environments are symlinks into this cache, which every run on the node shares, so anything an
            # agent deleted or edited inside one of them used to change it for every later run there
            # (2026-10-23: six, python-dateutil, scipy, numpy lost most of their files on compute-128/170/174).
            mounts.append(f"{self.uv_cache}:{UV_CACHE_MOUNT}:{'ro' if uv_cache_readonly else 'rw'}")
        if self.wheelhouse:
            mounts.append(f"{self.wheelhouse}:{WHEELHOUSE_MOUNT}:ro")
        return mounts

    def environment(self, uv_cache_readonly: bool = False) -> list[str]:
        path_parts = []
        if self.runtime_dir:
            path_parts.append(str(self.runtime_dir / "bin"))
        path_parts += ["/usr/local/bin", "/usr/bin", "/bin"]
        env = [
            f"HOME={STATE_MOUNT}/home",
            f"TMPDIR={STATE_MOUNT}/tmp",
            f"PIP_CACHE_DIR={STATE_MOUNT}/pip-cache",
            # uv's cache: shared per node when the host provides one, so the thirteen method
            # environments of every run on the node hard-link one copy of each package
            # (it holds public wheels only); otherwise private to the run. Never the
            # container's own /tmp, which --containall makes small.
            # an agent that installs more packages gets a private cache when the shared one is read-only
            f"UV_CACHE_DIR={UV_CACHE_MOUNT if self.uv_cache and not uv_cache_readonly else STATE_MOUNT + '/pip-cache/uv'}",
            # The cache and the workspace are separate bind mounts, and the kernel refuses
            # hard-links between them (EXDEV), so uv would silently copy: 34 GB per equipped
            # run. Symlinks work across mounts; the environments only run in here.
            *(["UV_LINK_MODE=symlink"] if self.uv_cache else []),
            *([f"UV_FIND_LINKS={WHEELHOUSE_MOUNT}", f"PIP_FIND_LINKS={WHEELHOUSE_MOUNT}"] if self.wheelhouse else []),
            f"PATH={':'.join(path_parts)}",
            "PYTHONNOUSERSITE=1",
            # The image's glibc (2.41) refuses to dlopen a library that asks for an executable stack, which torch
            # 2.3's libtorch_cpu.so does ("cannot enable executable stack"), so every torch-based method failed to
            # import (STAligner, SLAT, CAST, GPSA). This tunable restores the old behaviour for those libraries.
            *(["GLIBC_TUNABLES=glibc.rtld.execstack=2"] if self.glibc_execstack else []),
        ]
        if self.agent == "codex":
            env.append(f"CODEX_HOME={STATE_MOUNT}/codex")
        elif self.agent == "claude":
            env.append(f"CLAUDE_CONFIG_DIR={STATE_MOUNT}/claude")
        elif self.agent == "gemini":
            env.append(f"GEMINI_CLI_HOME={STATE_MOUNT}")
        elif self.agent == "mini-swe":
            # Without MSWEA_CONFIGURED mini runs an interactive first-run setup that aborts
            # when stdin is not a terminal. Its config dir must sit in writable state because
            # the runtime is bound read-only, and the bundled cost map avoids a network fetch
            # at import time.
            env += [
                "MSWEA_CONFIGURED=1",
                "MSWEA_SILENT_STARTUP=1",
                f"MSWEA_GLOBAL_CONFIG_DIR={STATE_MOUNT}/mini",
                "LITELLM_LOCAL_MODEL_COST_MAP=True",
                # litellm aborts when its price map lacks the model, which it does for
                # anything newer than its pinned map; our own table prices the run instead.
                "MSWEA_COST_TRACKING=ignore_errors",
            ]
        return env

    def wrap(self, command: list[str], uv_cache_readonly: bool = False) -> list[str]:
        argv = [self.binary, "exec", "--containall", "--cleanenv", "--no-home"]
        if self.use_gpu:
            # SLURM grants the device; --nv exposes the assigned driver interface without
            # adding host filesystem mounts.
            argv.append("--nv")
        argv += ["--pwd", str(self.workspace)]
        for bind in self.binds(uv_cache_readonly):
            argv += ["--bind", bind]
        argv += [self.image, "/usr/bin/env", *self.environment(uv_cache_readonly)]
        if self.secrets:
            # The key values are read inside the container, from the mounted files, so they
            # never appear on the host command line, in `ps`, or in the recorded command.
            argv += ["sh", "-c", self._secret_prelude() + 'exec "$@"', "sh"]
        return argv + command

    def _secret_prelude(self) -> str:
        return "".join(f'export {r.env_var}="$(cat {r.container_path})"; ' for r in self.secrets)

    def secret_exports(self) -> dict[str, str]:
        """Which variable each mounted secret file is exported as, inside the container."""
        return {r.env_var: r.container_path for r in self.secrets}


def resolve_binary(preferred: str = "singularity") -> str | None:
    """Singularity if present, else Apptainer, which takes the same flags."""
    for candidate in (preferred, "singularity", "apptainer"):
        found = shutil.which(candidate)
        if found:
            return found
    return None
