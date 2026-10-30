from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a tree of Jupyter notebooks into plain .py exports.")
    parser.add_argument("src", type=Path, help="Root directory containing .ipynb files.")
    parser.add_argument("dst", type=Path, help="Output directory for converted .py files.")
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include .ipynb_checkpoints notebooks. Default is to skip them.",
    )
    return parser.parse_args()


def iter_notebooks(root: Path, *, include_checkpoints: bool) -> list[Path]:
    notebooks = []
    for path in sorted(root.rglob("*.ipynb")):
        if not include_checkpoints and ".ipynb_checkpoints" in path.parts:
            continue
        notebooks.append(path)
    return notebooks


def cell_to_text(cell: dict, index: int) -> str:
    cell_type = cell.get("cell_type", "unknown")
    source = "".join(cell.get("source", []))
    header = f"# %% [cell {index}: {cell_type}]\n"
    if cell_type == "markdown":
        body = "\n".join(f"# {line}" if line else "#" for line in source.splitlines())
        return header + body + "\n\n"
    return header + source.rstrip() + "\n\n"


def convert_notebook(src: Path, dst: Path) -> dict[str, object]:
    payload = json.loads(src.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    rendered = []
    rendered.append(f"# Converted from {src}\n")
    rendered.append(f"# Notebook cell count: {len(cells)}\n\n")
    for idx, cell in enumerate(cells, start=1):
        rendered.append(cell_to_text(cell, idx))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(rendered), encoding="utf-8")
    return {
        "source": str(src),
        "output": str(dst),
        "n_cells": len(cells),
    }


def main() -> None:
    args = parse_args()
    src_root = args.src.expanduser().resolve()
    dst_root = args.dst.expanduser().resolve()
    notebooks = iter_notebooks(src_root, include_checkpoints=args.include_checkpoints)
    converted = []
    for notebook in notebooks:
        relative = notebook.relative_to(src_root)
        out_path = dst_root / relative.with_suffix(".py")
        converted.append(convert_notebook(notebook, out_path))

    manifest = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "n_converted": len(converted),
        "include_checkpoints": bool(args.include_checkpoints),
        "files": converted,
    }
    manifest_path = dst_root / "conversion_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"n_converted": len(converted), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
