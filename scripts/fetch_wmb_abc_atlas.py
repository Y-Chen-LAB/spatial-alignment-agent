from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    "Zhuang-ABCA-1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Allen ABC atlas whole-mouse-brain MERFISH source used for "
            "SABench Data99-227 and summarize available sections. By default this follows "
            "the paper-faithful Allen source route for the 129-slice benchmark family: "
            "Zhuang-ABCA-1 with the SABench annotation filter."
        )
    )
    parser.add_argument(
        "--download-base",
        type=Path,
        default=Path.home() / "Downloads/paper2bench-data/allen_abc_atlas",
        help="Local ABC atlas cache root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="ABC atlas WMB datasets to fetch. Default is the paper-faithful Zhuang-ABCA-1 route.",
    )
    parser.add_argument(
        "--manifest",
        default="latest",
        help="Manifest to load, for example releases/20250531/manifest.json or 'latest'.",
    )
    parser.add_argument(
        "--matrix-kind",
        choices=["none", "raw", "log2", "both"],
        default="raw",
        help="Which expression matrices to fetch for each dataset.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Pass skip_hash_check=True to the ABC cache methods.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even if they already exist in the cache.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("results/wmb_abc_atlas_download_summary.json"),
        help="Where to write a local summary of downloaded datasets and section counts.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print what would be downloaded and exit without touching the cache.",
    )
    return parser.parse_args()


def _matrix_kinds(kind: str) -> list[str]:
    if kind == "both":
        return ["raw", "log2"]
    if kind == "none":
        return []
    return [kind]


def _print_plan(args: argparse.Namespace) -> None:
    print("Allen ABC WMB fetch plan")
    print(f"download_base={args.download_base}")
    print(f"manifest={args.manifest}")
    print(f"datasets={','.join(args.datasets)}")
    print(f"matrix_kind={args.matrix_kind}")
    print("metadata:")
    print("  per-dataset: cell_metadata, cell_metadata_with_cluster_annotation, gene")
    print("  per-dataset CCF: ccf_coordinates")
    print(
        "  shared: WMB-taxonomy cluster_to_cluster_annotation_membership_pivoted/color; "
        "Allen-CCF-2020 parcellation_to_parcellation_term_membership_acronym/color"
    )


def main() -> int:
    args = parse_args()
    _print_plan(args)
    if args.plan_only:
        return 0

    try:
        import pandas as pd  # noqa: F401
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    except Exception as exc:  # pragma: no cover - user-machine dependency
        print(
            "Missing dependency: install the Allen ABC atlas client first:\n"
            '  uv pip install "abc_atlas_access[notebooks] @ '
            'git+https://github.com/alleninstitute/abc_atlas_access.git"',
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    args.download_base.mkdir(parents=True, exist_ok=True)
    abc_cache = AbcProjectCache.from_cache_dir(args.download_base)

    if args.manifest == "latest":
        abc_cache.load_latest_manifest()
    else:
        abc_cache.load_manifest(args.manifest)

    summary: dict[str, object] = {
        "download_base": str(args.download_base),
        "manifest": abc_cache.current_manifest,
        "datasets": {},
        "shared_metadata": {},
    }

    shared_specs = [
        ("WMB-taxonomy", "cluster_to_cluster_annotation_membership_pivoted"),
        ("WMB-taxonomy", "cluster_to_cluster_annotation_membership_color"),
        ("Allen-CCF-2020", "parcellation_to_parcellation_term_membership_acronym"),
        ("Allen-CCF-2020", "parcellation_to_parcellation_term_membership_color"),
    ]
    for directory, file_name in shared_specs:
        df = abc_cache.get_metadata_dataframe(
            directory=directory,
            file_name=file_name,
            force_download=args.force_download,
            skip_hash_check=args.skip_hash_check,
        )
        summary["shared_metadata"][f"{directory}/{file_name}"] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
        }

    for dataset in args.datasets:
        cell = abc_cache.get_metadata_dataframe(
            directory=dataset,
            file_name="cell_metadata",
            dtype={"cell_label": str},
            force_download=args.force_download,
            skip_hash_check=args.skip_hash_check,
        )
        cell_annot = abc_cache.get_metadata_dataframe(
            directory=dataset,
            file_name="cell_metadata_with_cluster_annotation",
            dtype={"cell_label": str},
            force_download=args.force_download,
            skip_hash_check=args.skip_hash_check,
        )
        gene = abc_cache.get_metadata_dataframe(
            directory=dataset,
            file_name="gene",
            force_download=args.force_download,
            skip_hash_check=args.skip_hash_check,
        )
        ccf = abc_cache.get_metadata_dataframe(
            directory=f"{dataset}-CCF",
            file_name="ccf_coordinates",
            dtype={"cell_label": str},
            force_download=args.force_download,
            skip_hash_check=args.skip_hash_check,
        )

        matrix_paths: dict[str, str] = {}
        for matrix_kind in _matrix_kinds(args.matrix_kind):
            file_path = abc_cache.get_file_path(
                directory=dataset,
                file_name=f"{dataset}/{matrix_kind}",
                force_download=args.force_download,
                skip_hash_check=args.skip_hash_check,
            )
            matrix_paths[matrix_kind] = str(file_path)

        section_counts = (
            cell.groupby("brain_section_label", observed=True).size().sort_index().astype(int).to_dict()
        )
        summary["datasets"][dataset] = {
            "cells": int(len(cell)),
            "annotated_cells": int(len(cell_annot)),
            "genes": int(len(gene)),
            "ccf_rows": int(len(ccf)),
            "section_count": int(cell["brain_section_label"].nunique()),
            "sections": section_counts,
            "matrix_paths": matrix_paths,
        }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
