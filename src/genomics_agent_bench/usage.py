"""Token and cost accounting for agent runs.

Nothing was recorded before this: none of the 530 historical runs carries a usage block, so
there is no way to say what a task cost or how many tokens an arm spent. The information was
never lost though, only unread. Codex writes a `turn.completed` event into the stream we
already capture, and 455 of 527 historical runs contain one, so the whole history can be
priced retroactively without re-running anything.

Cost is deliberately split three ways. `reported_cost_usd` is what the vendor itself said, and
is used when available because it accounts for cache tiers and discounts we cannot see.
`estimated_cost_usd` is tokens times a local price table. `cost_usd` prefers the reported
figure and falls back to the estimate. An arm whose cost is estimated rather than reported
should not be compared to the cent with one that is.

Prices are not baked in. They change, a stale table silently produces confident wrong numbers,
and a token count is useful on its own. PRICES starts empty and is filled from
`config/model_prices.json` when that file exists; without it, tokens are still recorded and
`estimated_cost_usd` stays None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRICE_FILE = Path("config/model_prices.json")


def load_prices(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Per-million-token prices as {model: {input, cached_input, output}}."""
    target = path or PRICE_FILE
    if not target.is_file():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def empty_usage(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        # Anthropic bills a cache write at about 1.25x input and a cache read at 10% of it, a
        # twelvefold gap, so the two cannot be lumped together and costed. OpenAI does not bill
        # writes at all, so this stays 0 for codex, which is correct rather than missing.
        "cache_creation_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "api_calls": 0,
        "reported_cost_usd": None,
        "estimated_cost_usd": None,
        "cost_usd": None,
        "source": None,
    }


def _estimate(usage: dict[str, Any], prices: dict[str, dict[str, float]]) -> float | None:
    price = prices.get(str(usage.get("model")))
    if not price:
        return None
    # Cached input is billed separately and far cheaper, so it is subtracted from the input
    # count rather than double-charged.
    fresh_input = max(
        int(usage["input_tokens"]) - int(usage["cached_input_tokens"]) - int(usage["cache_creation_tokens"]), 0)
    return round(
        fresh_input / 1e6 * price.get("input", 0.0)
        + int(usage["cached_input_tokens"]) / 1e6 * price.get("cached_input", price.get("input", 0.0))
        + int(usage["cache_creation_tokens"]) / 1e6 * price.get("cache_write", price.get("input", 0.0))
        + int(usage["output_tokens"]) / 1e6 * price.get("output", 0.0),
        6,
    )


def finalise(usage: dict[str, Any], prices: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    usage["estimated_cost_usd"] = _estimate(usage, prices or load_prices())
    reported = usage.get("reported_cost_usd")
    usage["cost_usd"] = reported if reported not in (None, 0.0) else usage["estimated_cost_usd"]
    return usage


def from_codex_stream(text: str, model: str) -> dict[str, Any] | None:
    """Codex emits one `turn.completed` per turn; the counts are cumulative per turn."""
    usage = empty_usage(model)
    found = False
    for line in text.splitlines():
        line = line.strip()
        if '"turn.completed"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("usage") or {}
        if not payload:
            continue
        found = True
        usage["api_calls"] += 1
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
            usage[key] += int(payload.get(key) or 0)
    if not found:
        return None
    usage["source"] = "codex_turn_completed"
    return usage


def from_claude_stream(text: str, model: str) -> dict[str, Any] | None:
    """Claude's stream-json carries usage on assistant messages and a cost on the result event."""
    usage = empty_usage(model)
    found = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            cost = event.get("total_cost_usd", event.get("cost_usd"))
            if cost is not None:
                usage["reported_cost_usd"] = float(cost)
                found = True
        payload = (event.get("message") or {}).get("usage") or event.get("usage") or {}
        if payload:
            found = True
            usage["api_calls"] += 1
            usage["input_tokens"] += int(payload.get("input_tokens") or 0)
            usage["output_tokens"] += int(payload.get("output_tokens") or 0)
            usage["cached_input_tokens"] += int(
                payload.get("cache_read_input_tokens") or payload.get("cached_input_tokens") or 0)
            usage["cache_creation_tokens"] += int(payload.get("cache_creation_input_tokens") or 0)
    if not found:
        return None
    usage["source"] = "claude_stream_json"
    return usage


def from_mini_trajectory(path: Path, model: str) -> dict[str, Any] | None:
    """mini prints almost nothing, so its trajectory file is the only record."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    usage = empty_usage(model)
    stats = (data.get("info") or {}).get("model_stats") or {}
    usage["api_calls"] = int(stats.get("api_calls") or 0)
    if stats.get("instance_cost"):
        usage["reported_cost_usd"] = float(stats["instance_cost"])
    for message in data.get("messages") or []:
        payload = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        usage["input_tokens"] += int(payload.get("prompt_tokens") or 0)
        usage["output_tokens"] += int(payload.get("completion_tokens") or 0)
        usage["cached_input_tokens"] += int(payload.get("cache_read_input_tokens") or 0)
    usage["source"] = "mini_trajectory"
    return usage


def from_gemini_stream(text: str, model: str, prices: dict[str, dict[str, float]] | None = None) -> dict[str, Any] | None:
    """Gemini's stream-json reports usage once, in `stats` on the final result event.

    The counts are broken down by the model that actually served them, which is not the
    name that was requested: the CLI echoes the requested name in its init event and maps
    an unknown one to something it has (gemini-3.8-flash ran as gemini-3.5-flash), and it
    uses a smaller model for its own auxiliary calls. `served_models` records the split and
    the cost is summed per served model, so the figure is for what ran, not what was asked.
    """
    stats = None
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('{"type":"result"') and '"type": "result"' not in line[:30]:
            continue
        try:
            stats = json.loads(line).get("stats")
        except json.JSONDecodeError:
            continue
    if not stats:
        return None
    usage = empty_usage(model)
    usage["input_tokens"] = int(stats.get("input_tokens") or 0)
    usage["cached_input_tokens"] = int(stats.get("cached") or 0)
    usage["output_tokens"] = int(stats.get("output_tokens") or 0)
    usage["api_calls"] = int(stats.get("tool_calls") or 0)
    served = stats.get("models") or {}
    usage["served_models"] = {name: {"input_tokens": int(m.get("input_tokens") or 0),
                                     "cached_input_tokens": int(m.get("cached") or 0),
                                     "output_tokens": int(m.get("output_tokens") or 0)}
                              for name, m in served.items()}
    table = prices or load_prices()
    per_model = []
    for name, m in usage["served_models"].items():
        part = _estimate({"model": name, "input_tokens": m["input_tokens"], "cached_input_tokens": m["cached_input_tokens"],
                          "cache_creation_tokens": 0, "output_tokens": m["output_tokens"]}, table)
        per_model.append(part)
    if per_model and all(x is not None for x in per_model):
        usage["reported_cost_usd"] = round(sum(per_model), 6)
    usage["source"] = "gemini_result_stats"
    return usage


def for_run(agent: str, model: str, stdout: str, trajectory: Path | None = None) -> dict[str, Any] | None:
    if agent == "codex":
        return from_codex_stream(stdout, model)
    if agent == "claude":
        return from_claude_stream(stdout, model)
    if agent == "gemini":
        return from_gemini_stream(stdout, model)
    if agent == "mini-swe" and trajectory is not None:
        return from_mini_trajectory(trajectory, model)
    return None
