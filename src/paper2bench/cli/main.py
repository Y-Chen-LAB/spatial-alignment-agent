from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2bench.core.hashing import file_fingerprint, sha256_file
from paper2bench.core.ir import TaskIR
from paper2bench.core.task_card import compile_task_card
from paper2bench.core.validation import summarize_reports, validate_task_file


def _write_or_print(payload: dict, output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


def cmd_inspect(args: argparse.Namespace) -> int:
    task = TaskIR.from_toml(args.task)
    _write_or_print(task.definition_payload, args.output)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    reports = [
        validate_task_file(
            task_path,
            strict_data=args.strict_data,
            include_content_hashes=args.content_hashes,
        )
        for task_path in args.tasks
    ]
    payload = summarize_reports(reports)
    _write_or_print(payload, args.output)
    return 0 if payload["n_failed"] == 0 else 1


def cmd_validate_suite(args: argparse.Namespace) -> int:
    task_paths = sorted(Path().glob(args.task_glob))
    if not task_paths:
        raise SystemExit(f"No tasks matched {args.task_glob!r}")
    reports = [
        validate_task_file(
            task_path,
            strict_data=args.strict_data,
            include_content_hashes=args.content_hashes,
        )
        for task_path in task_paths
    ]
    payload = summarize_reports(reports)
    _write_or_print(payload, args.output)
    return 0 if payload["n_failed"] == 0 else 1


def cmd_hash(args: argparse.Namespace) -> int:
    path = args.path
    payload = file_fingerprint(path, include_content_hash=args.content)
    if args.content and path.exists() and path.is_file():
        payload["sha256"] = sha256_file(path)
    _write_or_print(payload, args.output)
    return 0


def cmd_compile_card(args: argparse.Namespace) -> int:
    compiled = compile_task_card(args.card, args.out_dir, force=args.force)
    payload = {"compiled": compiled.to_json()}
    if args.validate:
        report = validate_task_file(
            compiled.task_toml,
            strict_data=args.strict_data,
            include_content_hashes=args.content_hashes,
        )
        payload["validation"] = report.to_json()
    _write_or_print(payload, args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p2b", description="paper2bench compiler utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a task TOML as normalized model-free task IR")
    inspect_parser.add_argument("task", type=Path)
    inspect_parser.add_argument("--output", type=Path)
    inspect_parser.set_defaults(func=cmd_inspect)

    validate_parser = subparsers.add_parser("validate", help="Validate one or more task TOML files")
    validate_parser.add_argument("tasks", nargs="+", type=Path)
    validate_parser.add_argument("--strict-data", action="store_true", help="Treat missing local data assets as errors.")
    validate_parser.add_argument("--content-hashes", action="store_true", help="Hash asset file contents; can be slow.")
    validate_parser.add_argument("--output", type=Path)
    validate_parser.set_defaults(func=cmd_validate)

    suite_parser = subparsers.add_parser("validate-suite", help="Validate a suite of task TOML files")
    suite_parser.add_argument("--task-glob", default="benchmarks/spatial_alignment/tasks/*/task.toml")
    suite_parser.add_argument("--strict-data", action="store_true", help="Treat missing local data assets as errors.")
    suite_parser.add_argument("--content-hashes", action="store_true", help="Hash asset file contents; can be slow.")
    suite_parser.add_argument("--output", type=Path)
    suite_parser.set_defaults(func=cmd_validate_suite)

    hash_parser = subparsers.add_parser("hash", help="Fingerprint a path")
    hash_parser.add_argument("path", type=Path)
    hash_parser.add_argument("--content", action="store_true", help="Include file content SHA256.")
    hash_parser.add_argument("--output", type=Path)
    hash_parser.set_defaults(func=cmd_hash)

    card_parser = subparsers.add_parser(
        "compile-card",
        help="Compile a reviewed TOML/JSON task card into a model-free task package",
    )
    card_parser.add_argument("card", type=Path)
    card_parser.add_argument("--out-dir", required=True, type=Path, help="Directory to receive task.toml and instruction.md.")
    card_parser.add_argument("--force", action="store_true", help="Overwrite compiled task files if present.")
    card_parser.add_argument("--validate", action="store_true", help="Validate the compiled task package.")
    card_parser.add_argument("--strict-data", action="store_true", help="Treat missing local data assets as errors during validation.")
    card_parser.add_argument("--content-hashes", action="store_true", help="Hash asset file contents during validation; can be slow.")
    card_parser.add_argument("--output", type=Path)
    card_parser.set_defaults(func=cmd_compile_card)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
