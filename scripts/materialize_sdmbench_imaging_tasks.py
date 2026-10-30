from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.splits import write_default_split_manifest


BARISTASEQ_SPECS = [
    ("Data36_slice1", "Slice_1_removed.h5ad", 36),
    ("Data37_slice2", "Slice_2_removed.h5ad", 37),
    ("Data38_slice3", "Slice_3_removed.h5ad", 38),
]

STARMAP_SPECS = [
    ("Data39_slice1", "20180417_BZ5_control.h5ad", 39),
    ("Data40_slice2", "20180419_BZ9_control.h5ad", 40),
    ("Data41_slice3", "20180424_BZ14_control.h5ad", 41),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize SABench Data36-41 BaristaSeq/STARmap tasks from SDMBench zip bundles."
    )
    parser.add_argument("--baristaseq-zip", type=Path, default=Path("/Users/yiqun/Downloads/22550074.zip"))
    parser.add_argument("--starmap-zip", type=Path, default=Path("/Users/yiqun/Downloads/22565200.zip"))
    parser.add_argument("--compiled-root", type=Path, default=Path("benchmarks/spatial_alignment/compiled_data/sdmbench"))
    parser.add_argument("--tasks-root", type=Path, default=Path("benchmarks/spatial_alignment/tasks"))
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp/sdmbench_data36_41"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing compiled assets and task directories.")
    return parser.parse_args()


def _to_toml(payload: dict[str, object]) -> str:
    lines = []
    for key, value in payload.items():
        lines.append(f"{key} = {_format_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _extract_zip(zip_path: Path, out_dir: Path, *, force: bool) -> None:
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as handle:
        handle.extractall(out_dir)


def _compile_perturbed_slice(
    *,
    source: Path,
    output: Path,
    report_json: Path,
    preserve_obs_cols: list[str],
    seed: int,
    force: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/compile_perturbed_h5ad.py",
        "--input",
        str(source),
        "--output",
        str(output),
        "--random-rigid-seed",
        str(seed),
        "--report-json",
        str(report_json),
    ]
    for col in preserve_obs_cols:
        cmd.extend(["--preserve-obs-col", col])
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)


def _write_marker_gene_file(path: Path, slice_path: Path) -> None:
    slice_ = H5ADSlice(slice_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene"])
        for gene in slice_.var_names.tolist():
            writer.writerow([str(gene)])


def _write_manifest(
    path: Path,
    *,
    task_id: str,
    raw_inputs: list[Path],
    assets: list[tuple[str, Path, int, int]],
    source: str,
    compiler: str,
) -> None:
    payload = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": task_id,
        "source": source,
        "compiler": compiler,
        "raw_inputs": [str(item) for item in raw_inputs],
        "assets": [
            {
                "asset_id": asset_id,
                "format": "h5ad",
                "path": str(path_),
                "role": "slice",
                "visibility": "public",
                "metadata": {"n_obs": n_obs, "n_var": n_var},
            }
            for asset_id, path_, n_obs, n_var in assets
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _slice_stats(path: Path) -> tuple[int, int]:
    slice_ = H5ADSlice(path)
    return len(slice_.obs_names), len(slice_.var_names)


def _materialize_task(
    *,
    task_dir: Path,
    task_id: str,
    name: str,
    difficulty: str,
    slices: list[tuple[str, Path]],
    paper_data_ids: list[str],
    paper_source: str,
    raw_inputs: list[Path],
    instruction_body: str,
    force: bool,
) -> None:
    if task_dir.exists() and force:
        shutil.rmtree(task_dir)
    elif task_dir.exists():
        raise FileExistsError(f"{task_dir} already exists; pass --force to overwrite.")

    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    marker_gene_file = data_dir / "marker_genes.csv"
    _write_marker_gene_file(marker_gene_file, slices[0][1])
    split_manifest = data_dir / "split_manifest.json"
    write_default_split_manifest(split_manifest, task_id, train_fraction=0.5, seed=0)

    assets = []
    for slice_id, slice_path in slices:
        n_obs, n_var = _slice_stats(slice_path)
        assets.append((slice_id, slice_path, n_obs, n_var))
    _write_manifest(
        data_dir / "manifest.json",
        task_id=task_id,
        raw_inputs=raw_inputs,
        assets=assets,
        source=paper_source,
        compiler="scripts/materialize_sdmbench_imaging_tasks.py",
    )

    task_payload = {
        "id": task_id,
        "name": name,
        "domain": "spatial_genomics",
        "modality": "spatial_transcriptomics",
        "task_type": "serial_alignment",
        "difficulty": difficulty,
        "status": "executable_local",
        "input_contract": "h5ad_spatial_v1",
        "output_contract": "aligned_coordinate_csv_v1",
        "metric_profile": "sabench_light_alignment_v1",
        "label_key": "Region",
        "marker_gene_file": str(marker_gene_file),
        "max_marker_genes": assets[0][3],
        "split_manifest": "data/split_manifest.json",
        "score_split": "eval",
        "public_score_split": "train",
        "split_visibility": "scorer_only",
        "slices": [f"{slice_id}={slice_path}" for slice_id, slice_path in slices],
        "paper_data_ids": paper_data_ids,
        "paper_source": paper_source,
    }
    (task_dir / "task.toml").write_text(_to_toml(task_payload), encoding="utf-8")
    (task_dir / "instruction.md").write_text(instruction_body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.compiled_root.mkdir(parents=True, exist_ok=True)
    reports_dir = args.compiled_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    barista_extract = args.tmp_root / "baristaseq"
    starmap_extract = args.tmp_root / "starmap"
    _extract_zip(args.baristaseq_zip, barista_extract, force=args.force)
    _extract_zip(args.starmap_zip, starmap_extract, force=args.force)

    compiled_barista = []
    for slice_id, filename, seed in BARISTASEQ_SPECS:
        output = args.compiled_root / f"{slice_id.lower()}_seed{seed}_rigid.h5ad"
        report = reports_dir / f"{slice_id.lower()}_rigid.json"
        _compile_perturbed_slice(
            source=barista_extract / filename,
            output=output,
            report_json=report,
            preserve_obs_cols=["Region", "layer", "Slice", "x", "y", "x_um", "y_um", "Angle", "Dist to bottom", "Dist to pia", "depth_um"],
            seed=seed,
            force=args.force,
        )
        compiled_barista.append((slice_id, output))

    compiled_starmap = []
    for slice_id, filename, seed in STARMAP_SPECS:
        output = args.compiled_root / f"{slice_id.lower()}_seed{seed}_rigid.h5ad"
        report = reports_dir / f"{slice_id.lower()}_rigid.json"
        _compile_perturbed_slice(
            source=starmap_extract / filename,
            output=output,
            report_json=report,
            preserve_obs_cols=["Region", "region", "ct"],
            seed=seed,
            force=args.force,
        )
        compiled_starmap.append((slice_id, output))

    barista_instruction = """# BaristaSeq Mouse Primary Cortex 3-Slice Serial Alignment (Data36-38)

Three BaristaSeq slices from the mouse primary cortex from the SABench paper (Yan et al. 2026, Data36-38), sourced from the SDMBench processed release.

In the paper's evaluation pipeline, random rigid perturbations are applied to disrupt the original coordinate frame before alignment benchmarking. This local task uses deterministic seeded rigid perturbations compiled into the repo-local `.h5ad` files referenced by `task.toml`.

## Task

Align the three slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- This family retains published `obs/Region` and `obs/layer` annotations, so the paper-aligned SABench landmark and region-overlap metrics are enabled in addition to the gene-based metrics.
- Source bundle: `~/Downloads/22550074.zip` from the SDMBench dataset release.
- The exact perturbation parameters are recorded in `benchmarks/spatial_alignment/compiled_data/sdmbench/reports/`.
"""
    starmap_instruction = """# STARmap Mouse Medial Prefrontal Cortex 3-Slice Serial Alignment (Data39-41)

Three STARmap slices from the mouse medial prefrontal cortex from the SABench paper (Yan et al. 2026, Data39-41), sourced from the SDMBench processed release.

As described in the SABench paper, random rigid perturbations are applied before benchmarking. This local task uses deterministic seeded rigid perturbations compiled into the repo-local `.h5ad` files referenced by `task.toml`.

## Task

Align the three slices into a shared 2D coordinate system. Submit one coordinate CSV per slice with columns `obs_id,x,y`. Row order and `obs_id` values must match the source `.h5ad` observations.

## Notes

- `obs/Region` is preserved from the SDMBench STARmap bundle, so the task supports the paper-aligned SABench landmark and region-overlap metrics in addition to gene-based metrics.
- Source bundle: `~/Downloads/22565200.zip` from the SDMBench dataset release.
- The exact perturbation parameters are recorded in `benchmarks/spatial_alignment/compiled_data/sdmbench/reports/`.
"""

    _materialize_task(
        task_dir=args.tasks_root / "sa-017-baristaseq-mpc-serial-align",
        task_id="sa-017-baristaseq-mpc-serial-align",
        name="BaristaSeq mouse primary cortex 3-slice serial alignment (Data36-38)",
        difficulty="medium",
        slices=compiled_barista,
        paper_data_ids=["Data36", "Data37", "Data38"],
        paper_source="https://doi.org/10.1038/s41592-024-02215-8 (SDMBench processed BaristaSeq slices used by SABench Data36-38)",
        raw_inputs=[args.baristaseq_zip],
        instruction_body=barista_instruction,
        force=args.force,
    )

    _materialize_task(
        task_dir=args.tasks_root / "sa-018-starmap-mmpc-serial-align",
        task_id="sa-018-starmap-mmpc-serial-align",
        name="STARmap mouse medial prefrontal cortex 3-slice serial alignment (Data39-41)",
        difficulty="medium",
        slices=compiled_starmap,
        paper_data_ids=["Data39", "Data40", "Data41"],
        paper_source="https://doi.org/10.1038/s41592-024-02215-8 (SDMBench processed STARmap slices used by SABench Data39-41)",
        raw_inputs=[args.starmap_zip],
        instruction_body=starmap_instruction,
        force=args.force,
    )

    summary = {
        "created_tasks": [
            "sa-017-baristaseq-mpc-serial-align",
            "sa-018-starmap-mmpc-serial-align",
        ],
        "compiled_assets": [str(path) for _, path in compiled_barista + compiled_starmap],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
