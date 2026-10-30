# Credentials and access

Each agent authenticates differently, and the differences matter more than they look: two of
them do not use an API key at all, and one prefers a stale cached login over a working key.

## Where each agent gets its credentials

| Agent | Mechanism | Location |
| --- | --- | --- |
| `claude` | Claude Code subscription OAuth | macOS Keychain, service `Claude Code-credentials`. On Linux, `~/.claude/.credentials.json` |
| `codex` | ChatGPT subscription OAuth, or an Azure OpenAI key (`--codex-provider`, default auto) | `~/.codex/auth.json`, or `azure-openai-api-key` + `azure-openai-endpoint` in the gen2bench config dir |
| `gemini` | `GEMINI_API_KEY`, or cached OAuth in `~/.gemini/oauth_creds.json` | see the warning below |
| `mini-swe` | litellm, pay-per-token API keys only | `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` |
| `azure-openai` | Azure OpenAI Responses API key | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |

`claude` and `codex` need no API key on a machine where you have logged in interactively.
Setting `ANTHROPIC_API_KEY` is not required for `claude` and does not replace the login.

## codex on Azure

The codex CLI needs no ChatGPT login: pointed at an Azure OpenAI deployment it runs its full
agent loop, billed to the Azure key. The runner stages a provider config into the run's
private `CODEX_HOME` and mounts the key as a secret. It needs two files, one value each,
no trailing newline:

```bash
CFG=${GEN2BENCH_CONFIG_DIR:-$HOME/.config/gen2bench}
printf '%s' "$AZURE_OPENAI_API_KEY"  > "$CFG/azure-openai-api-key"
printf '%s' "$AZURE_OPENAI_ENDPOINT" > "$CFG/azure-openai-endpoint"    # resource root or .../openai/v1
chmod 600 "$CFG"/azure-openai-*
```

With the key file present `--codex-provider auto` picks Azure; `chatgpt` forces the login.
Verified 2026-09-16 on JHPCE: gpt-5.3-codex, gpt-5.5 and gpt-6-astra each wrote a file, ran
it and read the output back through codex 0.154.0 inside the container.

## Azure OpenAI code-generation runs (one-shot baseline, not an agent)

Keep the endpoint and key in a mode-600 file outside the checkout, at
`~/.config/paper2bench/env`: the solver, the Azure probe and `check_access.py` all read that
path by default. (`PAPER2BENCH_ENV_FILE` can point the solver at another file, but
`check_access.py` does not honour it, so a setup that runs can still report FAIL.) The key is
used only by the host-side Responses API request; the generated solver runs in the blind
container and does not receive either Azure variable.

```bash
chmod 600 ~/.config/paper2bench/env
./.venv/bin/python scripts/check_azure_openai_access.py --deployment gpt-6-astra
scripts/slurm/submit_alignment.sh --agent azure-openai --task sa-001-dlpfc-serial-align --dry-run
```

The deployment value is an Azure deployment name, not necessarily the underlying model slug.

## Gemini prefers cached OAuth over your key

If `~/.gemini/oauth_creds.json` exists, the CLI uses it and ignores `GEMINI_API_KEY`. When
that account is out of quota every run fails at `onboardUser` with "Resource has been
exhausted", even though a working key is set. The runner therefore gives each gemini run a
private empty `GEMINI_CLI_HOME` so it falls through to the key, and the container does not
copy OAuth material in.

## mini-swe needs real API keys

mini reaches every provider through litellm, so subscription tokens do not work: the Claude
Code setup token is gated to that client and returns masked 429s through litellm. Put
pay-per-token keys in the config directory, which the container mounts read-only one file at a
time under `/run/secrets/`:

```bash
CFG=${GEN2BENCH_CONFIG_DIR:-$HOME/.config/gen2bench}
mkdir -p "$CFG" && chmod 700 "$CFG"

printf '%s' "$ANTHROPIC_API_KEY" > "$CFG/anthropic-api-key"
printf '%s' "$GEMINI_API_KEY"    > "$CFG/gemini-api-key"
printf '%s' "$OPENAI_API_KEY"    > "$CFG/openai-api-key"
chmod 600 "$CFG"/*-api-key
```

Files, never values in a script. `printf` rather than `echo` so no trailing newline ends up in
the key.

mini also needs `MSWEA_COST_TRACKING=ignore_errors`, which the runner sets. Without it litellm
aborts on any model missing from its pinned price map, which includes anything recent. Cost
then comes from `config/model_prices.json` instead.

## Checking access before a campaign

```bash
claude --print --model claude-opus-5 "Reply with only the word OK."
codex exec --model gpt-5.5 --skip-git-repo-check "Reply with only the word OK."
GEMINI_CLI_HOME=$(mktemp -d) gemini --skip-trust --model gemini-3.5-flash -p "Reply OK."
```

Verified working on 2026-09-15: claude via Keychain OAuth, codex on gpt-5.5 and gpt-6-astra,
gemini on gemini-3.5-flash via API key, and mini-swe on gemini/gemini-3.5-flash reaching
`exit_status: Submitted` over three API calls.

## For the cluster

`claude` is the one to check first. On macOS its credentials live in the Keychain, so there is
no `~/.claude/.credentials.json` for the container to copy. On Linux the CLI does write that
file, but it has to exist on the submit host before any job runs. Log in there interactively
once and confirm the file is present, or cell A will fail on every node with no credentials
and no obvious reason.

## Refresh cadence

Measured on 2026-09-15:

| Credential | Access token | Refresh token | Refreshes itself |
| --- | --- | --- | --- |
| claude | 5.6 h | 29 days, to 2026-10-15 | yes, while the refresh token is valid |
| codex | 10 days | present | yes; `last_refresh` showed 2 h earlier |
| gemini OAuth | 48 min | present | yes, but bypassed in favour of the key |
| gemini / openai API keys | no expiry | n/a | n/a |

Nothing needs refreshing by hand on a normal cadence. The CLIs renew their own access tokens,
so the only hard deadline is the refresh token: for claude that is 2026-10-15, and after it
someone has to log in interactively, which no compute node can do.

Two consequences for a campaign that spans days.

A job copies the credential into private state when it starts, so a token that was valid at
submission can be stale by the time a queued job runs. The CLI renews it inside the job using
the copied refresh token, and the renewed token is discarded with the state afterwards, which
is correct but means the renewal never reaches the host. The host's own token only refreshes
when someone uses the CLI there.

API keys do not expire, which is why they are the sturdier choice for long unattended runs.
That is an argument for the mini-swe arm on the cluster over the native CLIs, not only a
harness detail.

Run `scripts/check_access.py` before submitting; add `--probe` to make one real call per agent
rather than only reading expiry times. It exits non-zero when an agent is unusable and warns
when a refresh token is within a day of expiring.

## Verified 2026-09-15

| Path | Result |
| --- | --- |
| `claude` CLI, `claude-opus-5` | works, Keychain OAuth |
| `codex` CLI, `gpt-5.5` and `gpt-6-astra` | works, ChatGPT subscription OAuth |
| `gemini` CLI, `gemini-3.5-flash` | works via `GEMINI_API_KEY`. Note the CLI echoes whatever name it is given in its init event and maps an unknown one to a model it has: `gemini-3.5-flash`, which does not exist, ran as `gemini-3.5-flash`. The served model is in the `stats.models` block of the result event; the usage parser records it |
| `mini-swe`, `gemini/gemini-3.5-flash`, through the harness in blind mode | works, 3 API calls, served model confirmed |
| `mini-swe`, `openai/...` | blocked: the API key has no credits |

The OpenAI key is valid and can list 143 models including `gpt-5.5` and `gpt-6-astra`, but a
chat call returns HTTP 429, "You have no credits remaining". This does not affect the codex
CLI, which authenticates by ChatGPT subscription OAuth and not by the key, so cell B is
unaffected. It does block any API-keyed OpenAI path, which is the mini-swe OpenAI arm.

One detail worth knowing if the replication cell matters: `gpt-5.3-codex` is absent from the
codex CLI's model cache but still listed by the API, so that arm is reachable by key and not
by the CLI, once the key has credits.

## Azure OpenAI

The Responses-API solver reads either prefix, `AZURE_OPENAI_*` or `MOOD_OPENAI_*`, and needs
both halves. A key without an endpoint fails before any call with "configuration is
incomplete", which is easy to misread as a bad key.

```bash
cat >> ~/.config/paper2bench/env <<'ENV'
export MOOD_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"
ENV
chmod 600 ~/.config/paper2bench/env
```

The resource root works too; the code appends `/openai/v1` if it is not already there.

Deployment names are per resource, not the vendor's catalogue, so probe rather than assume,
and re-probe rather than trust a previous answer: a deployment can be added at any time. On the
MOOD resource, checked 2026-09-15:

| Deployment | |
| --- | --- |
| `gpt-5.5` | deployed |
| `gpt-6-astra` | deployed |
| `gpt-5.6-sol` | deployed |
| `gpt-5.3-codex` | deployed |
| `gpt-5.6-terra`, `gpt-5.6-luna` | DeploymentNotFound |

```bash
./.venv/bin/python scripts/check_azure_openai_access.py --deployment gpt-6-astra
```

Both models the generation contrast rests on, `gpt-5.5` and `gpt-6-astra`, are now reachable by
either route, so that cell can run on Azure or through the codex CLI. `gpt-5.3-codex` is the
more interesting one: every one of the 531 historical runs used it, it has since dropped out of
the codex CLI's model cache, and it is still deployed here. The replication arm is reachable
only by Azure.
