#!/usr/bin/env python3
"""Verify an Azure OpenAI deployment before submitting a benchmark array.

This is a lightweight credential/deployment preflight. It prints no secret material and is
tracked separately from users' local API experimentation helpers.
"""

from __future__ import annotations

import argparse
import json

from run_openai_solver import _call_responses, _extract_text, _load_local_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment",
        default="gpt-6-astra",
        help="Azure deployment name, not necessarily the underlying model slug.",
    )
    parser.add_argument(
        "--input",
        default="Reply with only the word OK.",
        help="Short prompt used only to verify the deployment.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    args = parser.parse_args()

    _load_local_env()
    try:
        response = _call_responses(
            "azure",
            args.deployment,
            args.input,
            "none",
            args.max_output_tokens,
            args.timeout_seconds,
            developer_instructions=None,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    result = {
        "ok": response.get("status") == "completed",
        "deployment": args.deployment,
        "response_id": response.get("id"),
        "model": response.get("model"),
        "status": response.get("status"),
        "output_text": _extract_text(response) if response.get("status") == "completed" else None,
        "incomplete_details": response.get("incomplete_details"),
    }
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        parser.error(f"Azure response did not complete: {json.dumps(response.get('incomplete_details'))}")


if __name__ == "__main__":
    main()
