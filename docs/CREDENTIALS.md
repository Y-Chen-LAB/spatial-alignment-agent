# Agent authentication

Authenticate the chosen agent CLI on the host before starting a run. The runner uses
host-side credentials to reach the model provider and gives each run private CLI state.
Credential files and API keys must remain outside the repository and published artifacts.

| Runner | Supported credential source |
|---|---|
| `claude` | Claude Code CLI login |
| `codex` | CLI login, or Azure credentials selected with `--codex-provider` |
| `gemini` | API key or CLI login, subject to the runner's private-state configuration |
| `mini-swe` | Provider API keys through its configured backend |

For file-based provider configuration, `GEN2BENCH_CONFIG_DIR` selects the directory
(default `~/.config/gen2bench`). The runner recognizes provider-specific files such as
`anthropic-api-key`, `gemini-api-key`, `openai-api-key`, `azure-openai-api-key` and
`azure-openai-endpoint`. Restrict permissions on credential files to the owning user.
Azure deployment names must match deployments available in the selected resource.

Inspect the runner's `--help` and `scripts/check_access.py --help` for configuration and
access checks. A probe makes a real provider request and may incur a charge. Provider
availability, authentication behavior and quotas depend on the installed CLI and account;
historical campaign settings do not establish present access.
