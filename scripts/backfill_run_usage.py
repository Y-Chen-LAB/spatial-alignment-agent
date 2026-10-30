"""Write a usage block into historical run manifests, from traces already on disk.

No agent time is needed: Codex recorded its token counts in the stream we captured, and mini
records everything in its trajectory. Runs whose trace predates that event, or that died
before the first turn completed, simply get no usage block rather than a fabricated one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from genomics_agent_bench.usage import finalise, for_run, load_prices


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=Path("agent_runs/local_coding_agent"))
    ap.add_argument("--prices", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prices = load_prices(args.prices)
    written = skipped = already = 0
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    for manifest_path in sorted(args.runs_root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        if "usage" in manifest and not args.dry_run:
            already += 1
            continue
        run_dir = manifest_path.parent
        stdout_path = run_dir / "raw_stdout.jsonl"
        stdout = stdout_path.read_text(errors="ignore") if stdout_path.is_file() else ""
        usage = for_run(
            manifest.get("agent", ""), manifest.get("model", ""), stdout,
            trajectory=run_dir / "mini_trajectory.json",
        )
        if usage is None:
            skipped += 1
            continue
        usage = finalise(usage, prices)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            totals[key] += usage[key]
        if usage["cost_usd"]:
            totals["cost_usd"] += usage["cost_usd"]
        written += 1
        if args.dry_run:
            continue
        manifest["usage"] = usage
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb} usage for {written} runs; {skipped} had no recoverable usage; {already} already had one")
    print(f"  input tokens  : {totals['input_tokens']:,} ({totals['cached_input_tokens']:,} cached)")
    print(f"  output tokens : {totals['output_tokens']:,}")
    print(f"  cost          : {'$%.2f' % totals['cost_usd'] if totals['cost_usd'] else 'not priced (no config/model_prices.json)'}")


if __name__ == "__main__":
    main()
