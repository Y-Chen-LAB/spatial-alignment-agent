"""Report credential status for every agent, and optionally prove each one with a live call.

Run before submitting a campaign. Credentials here expire on very different clocks, and the
failure mode is a queue of jobs that all die hours later with an auth error rather than a
result, so it is worth thirty seconds up front.

Two kinds of credential are in play, and the distinction matters for long runs:

  OAuth, used by the claude and gemini CLIs. Short-lived access tokens with a longer
  refresh token behind them. The CLI refreshes itself while the refresh token is valid; once
  that expires, someone has to log in interactively, which nothing on a compute node can do.

  API keys, used by mini-swe through litellm. No expiry. For a campaign spanning days this is
  the more robust path, and it is the reason the gemini runner prefers the key over cached
  OAuth.

The codex CLI's ChatGPT login is deliberately not checked: the GPT arm runs through the Azure
OpenAI key (azure-openai below) until that budget is used up, so a missing ~/.codex/auth.json
is not a blocker.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("GEN2BENCH_CONFIG_DIR", Path.home() / ".config" / "gen2bench"))
NOW = datetime.datetime.now(datetime.timezone.utc)


def _hours(ts: datetime.datetime) -> float:
    return (ts - NOW).total_seconds() / 3600


def _fmt(ts: datetime.datetime | None) -> str:
    if ts is None:
        return "unknown"
    h = _hours(ts)
    return f"{ts:%Y-%m-%d %H:%M} UTC ({h:+.1f} h)" + ("  EXPIRED" if h < 0 else "")


def _jwt_expiry(token: str) -> datetime.datetime | None:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return datetime.datetime.fromtimestamp(exp, datetime.timezone.utc) if exp else None
    except Exception:
        return None


def claude_status() -> dict:
    out = {"agent": "claude", "mechanism": "subscription OAuth"}
    if sys.platform == "darwin":
        try:
            raw = subprocess.run(
                ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
                capture_output=True, text=True, timeout=15).stdout
            data = json.loads(raw).get("claudeAiOauth", {})
        except Exception:
            return {**out, "ok": False, "detail": "no Keychain entry; run `claude` once to log in"}
        out["location"] = "macOS Keychain"
    else:
        path = Path.home() / ".claude" / ".credentials.json"
        if not path.is_file():
            return {**out, "ok": False,
                    "detail": f"{path} absent. On Linux the CLI writes this file; log in on the "
                              "submit host before queueing, since a compute node cannot."}
        data = json.loads(path.read_text()).get("claudeAiOauth", {})
        out["location"] = str(path)
    access = data.get("expiresAt")
    refresh = data.get("refreshTokenExpiresAt")
    to_dt = lambda v: datetime.datetime.fromtimestamp(float(v) / 1000, datetime.timezone.utc) if v else None
    out["access_expires"] = to_dt(access)
    out["refresh_expires"] = to_dt(refresh)
    out["ok"] = bool(refresh) and _hours(to_dt(refresh)) > 0
    out["plan"] = data.get("subscriptionType")
    return out


def gemini_status() -> dict:
    out = {"agent": "gemini", "mechanism": "API key (preferred over cached OAuth)"}
    key_file = CONFIG_DIR / "gemini-api-key"
    has_env = bool(os.environ.get("GEMINI_API_KEY"))
    out["location"] = str(key_file) if key_file.is_file() else ("GEMINI_API_KEY in environment" if has_env else "none")
    out["ok"] = key_file.is_file() or has_env
    cached = Path.home() / ".gemini" / "oauth_creds.json"
    if cached.is_file():
        out["note"] = ("cached OAuth present; the CLI prefers it over the key, so runs use a "
                       "private empty GEMINI_CLI_HOME to bypass it")
    if not out["ok"]:
        out["detail"] = "no GEMINI_API_KEY; see docs/CREDENTIALS.md"
    return out


def azure_status() -> dict:
    """Azure OpenAI, which the Responses-API solver uses.

    Both halves come from either the AZURE_ or the MOOD_ prefix. The key alone is not enough
    and a missing endpoint fails before any call, which is easy to misread as a bad key.
    """
    out = {"agent": "azure-openai", "mechanism": "Azure OpenAI, Responses API"}
    key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("MOOD_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("MOOD_OPENAI_ENDPOINT")
    out["location"] = endpoint or "no endpoint set"
    out["ok"] = bool(key and endpoint)
    if not out["ok"]:
        missing = [n for n, v in (("key", key), ("endpoint", endpoint)) if not v]
        out["detail"] = ("missing " + " and ".join(missing) +
                         "; set MOOD_OPENAI_* or AZURE_OPENAI_* in ~/.config/paper2bench/env")
    else:
        out["note"] = "deployment names are resource-specific; probe with scripts/check_azure_openai_access.py"
    return out


def _load_env_file() -> None:
    """Read the same env file the solver reads, so this check sees what a run would."""
    for candidate in (Path.home() / ".config" / "paper2bench" / "env", Path(".env")):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip().removeprefix("export ").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def mini_status() -> dict:
    out = {"agent": "mini-swe", "mechanism": "litellm, pay-per-token API keys", "location": str(CONFIG_DIR)}
    present = [p.name for p in sorted(CONFIG_DIR.glob("*-api-key"))] if CONFIG_DIR.is_dir() else []
    out["keys"] = present
    out["ok"] = bool(present)
    missing = {"anthropic-api-key", "gemini-api-key", "openai-api-key"} - set(present)
    if missing:
        out["detail"] = ("missing " + ", ".join(sorted(missing)) +
                         "; subscription tokens do not work through litellm")
    return out


PROBES = {
    "claude": ["claude", "--print", "--model", "{model}", "Reply with only the word OK."],
    "gemini": ["gemini", "--skip-trust", "--model", "{model}", "-p", "Reply with only the word OK."],
}
DEFAULT_PROBE_MODELS = {"claude": "claude-opus-5", "gemini": "gemini-3.5-flash"}


def probe(agent: str, model: str) -> tuple[bool, str]:
    if agent not in PROBES or shutil.which(PROBES[agent][0]) is None:
        return False, "CLI not on PATH"
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)
    if agent == "gemini":
        import tempfile
        env["GEMINI_CLI_HOME"] = tempfile.mkdtemp(prefix="p2b-gemini-probe-")
    cmd = [part.format(model=model) for part in PROBES[agent]]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    text = (out.stdout + out.stderr).strip()
    return ("OK" in out.stdout), text.splitlines()[-1][:110] if text else "no output"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", action="store_true", help="Make one real call per agent. Costs a few cents.")
    ap.add_argument("--warn-hours", type=float, default=24.0,
                    help="Warn when a refresh token expires within this many hours.")
    args = ap.parse_args()

    # Loaded once, before any check: the file is read by the solver at run time, so a check
    # that ignores it reports a credential as missing when a real run would find it.
    _load_env_file()
    rows = [claude_status(), gemini_status(), azure_status(), mini_status()]
    failures = []
    for row in rows:
        mark = "ok  " if row.get("ok") else "FAIL"
        print(f"[{mark}] {row['agent']:9s} {row['mechanism']}")
        print(f"         at {row.get('location')}")
        if row.get("access_expires") is not None:
            print(f"         access token  {_fmt(row['access_expires'])}")
        if row.get("refresh_expires") is not None:
            print(f"         refresh token {_fmt(row['refresh_expires'])}")
            if _hours(row["refresh_expires"]) < args.warn_hours:
                print("         WARNING: refresh token expires soon; log in again before queueing")
        for key in ("plan", "keys", "note", "detail"):
            if row.get(key):
                print(f"         {key}: {row[key]}")
        if not row.get("ok"):
            failures.append(row["agent"])

    if args.probe:
        print("\nlive probes:")
        for agent, model in DEFAULT_PROBE_MODELS.items():
            ok, detail = probe(agent, model)
            print(f"  [{'ok  ' if ok else 'FAIL'}] {agent:9s} {model:20s} {detail}")
            if not ok and agent not in failures:
                failures.append(agent)

    if failures:
        print(f"\nnot ready: {', '.join(sorted(set(failures)))}")
        sys.exit(1)
    print("\nall agents have usable credentials")


if __name__ == "__main__":
    main()
