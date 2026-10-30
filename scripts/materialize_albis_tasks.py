"""Materialize spatial-alignment tasks from Albis-simulated datasets.

Albis (github.com/JanMatthias1/Albis) simulates a 3D tissue, sections it, and gives
every section its own random in-plane rigid offset, keeping the true frame. Its
`scripts/alignment_benchmark.py` exports each dataset as per-slice h5ads in this
suite's convention (obsm/spatial = what the agent sees; obs/x_GT, y_GT; obs/Region)
plus a catalog.json. This script turns each catalog into a task package.

Ground truth lives in slice s00's frame, so s00's input already is its answer and the
MAE (no rigid fit) measures how well every other slice was brought onto s00.

    ./.venv/bin/python scripts/materialize_albis_tasks.py \
        --data-root ~/p2b-data/paper2bench-data/albis_sim --force
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# dataset -> (task number, slug, difficulty, platform description for the prompt)
TASKS = {
    "strong_mix_cell": (301, "albis-strongmix-cell", "starter", "single-cell resolution (Xenium-like)"),
    "xenium_cell": (302, "albis-xenium-cell", "standard", "single-cell resolution (Xenium-like)"),
    "visium_spot": (303, "albis-visium-spot", "standard", "55 um capture spots on a 100 um lattice (Visium-like)"),
    "visiumhd_bin16": (304, "albis-visiumhd-bin16", "standard", "16 um square bins (Visium HD-like)"),
    "visiumhd_bin8": (305, "albis-visiumhd-bin8", "hard", "8 um square bins (Visium HD-like)"),
    # One factor at a time: grid fixed to each slide (no lattice leak; same transforms as
    # sa-303/304), non-adjacent sections on cells, sparse tissue after QC, and sparse + non-adjacent.
    "visium_spot_slide": (306, "albis-visium-spot-slide", "standard", "55 um capture spots on a 100 um lattice (Visium-like)"),
    "visiumhd_bin16_slide": (307, "albis-visiumhd-bin16-slide", "standard", "16 um square bins (Visium HD-like)"),
    "strong_mix_cell_gap": (308, "albis-strongmix-cell-gap", "standard", "single-cell resolution (Xenium-like)"),
    "visium_spot_sparse": (309, "albis-visium-spot-sparse", "hard", "55 um capture spots on a 100 um lattice (Visium-like)"),
    "visium_spot_sparse_gap": (310, "albis-visium-spot-sparse-gap", "hard", "55 um capture spots on a 100 um lattice (Visium-like)"),
    # Harder landmarks (suggested by the Albis authors): weak domain composition differences, strong
    # per-section batch effects and coarse 32 um bins; and mirror-image decoy domains with sections
    # randomly mounted face-down.
    "visiumhd_bin32_lowsignal_slide": (311, "albis-visiumhd-bin32", "hard", "32 um square bins (Visium HD-like)"),
    "strong_mix_cell_decoy_flip": (312, "albis-strongmix-cell-7domain", "hard", "single-cell resolution (Xenium-like)"),
}


# Prompt titles that would otherwise name the difficulty (the default title spells out the dataset name).
NEUTRAL_TITLES = {
    "visiumhd_bin32_lowsignal_slide": "Albis Visium HD 32um bin serial alignment",
    "strong_mix_cell_decoy_flip": "Albis single-cell serial alignment",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory holding <dataset>/catalog.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "benchmarks/spatial_alignment/tasks")
    parser.add_argument("--dataset", action="append", choices=sorted(TASKS), help="Repeatable; default: every catalog found.")
    parser.add_argument("--id-suffix", default="", help="Appended to task IDs, e.g. '-smoke' for a smoke-scale build.")
    parser.add_argument("--force", action="store_true", help="Overwrite task directories if they already exist.")
    return parser.parse_args()


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value) and len(value) > 3:
            return "[\n" + "".join(f"  {json.dumps(item)},\n" for item in value) + "]"
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _instruction(title: str, platform: str, slices: list[dict], gap: bool = False, qc: dict | None = None,
                 flip: bool = False) -> str:
    # Lines naming labels, the reference frame's origin or the offsets are dropped from the
    # agent prompt by prepare_agent_task.py, so this text avoids those words entirely.
    n = len(slices)
    rows = "\n".join(f"- `{s['slice']}`: {s['n_obs']:,} observations" for s in slices)
    kind = "sections" if gap else "serial sections"
    extra = ""
    if gap:
        extra += ("\nThe sections are listed in block order, but their spacing is irregular and not given: some\n"
                  "consecutive sections in this list are adjacent in the block, others have sections missing between them.\n")
    # flip is deliberately not mentioned: undoing a face-down section is part of the task
    if qc:
        extra += (f"\nThe data passed standard QC: observations with fewer than {qc['min_genes']} detected genes were removed.\n")
    return f"""# {title}

{n} {kind} of one simulated tissue block, measured at {platform}. Each section was
placed on the slide independently, so the sections do not share a coordinate system.
{extra}
{rows}

Bring every section into the coordinate system of `s00`. Keep `s00` where it is: its submitted
coordinates should equal its input coordinates. Coordinates are in microns, and the physical
scale of each section is correct as given.

Submit one coordinate CSV per section with columns `obs_id,x,y`. The row order and `obs_id`
values must match the source `.h5ad` file.

The score compares each section's submitted coordinates with where its observations truly lie in
the `s00` frame, and compares marker-gene expression between consecutive sections on a shared grid.
"""


def materialize(catalog_path: Path, output_root: Path, id_suffix: str, *, force: bool) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    dataset = catalog["dataset"]
    number, slug, difficulty, platform = TASKS[dataset]
    task_id = f"sa-{number:03d}-{slug}-serial-align{id_suffix}"
    task_dir = output_root / task_id
    if task_dir.exists():
        if not force:
            raise FileExistsError(f"{task_dir} already exists; pass --force to overwrite.")
        shutil.rmtree(task_dir)
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True)

    marker_file = data_dir / "marker_genes.csv"
    shutil.copyfile(catalog_path.parent / "marker_genes.csv", marker_file)
    n_markers = sum(1 for _ in marker_file.open(encoding="utf-8")) - 1

    slices = catalog["slices"]
    manifest = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": task_id,
        "source": "Albis simulator (scripts/alignment_benchmark.py)",
        "albis_dataset": dataset,
        "albis_scale": catalog["scale"],
        "albis_base_dataset": catalog.get("base_dataset", dataset),
        "slide_frame_grid": catalog.get("slide_frame", False),
        "albis_slice_ids": [s["albis_slice_id"] for s in catalog["slices"]],
        "qc": catalog.get("qc"),
        "albis_catalog": str(catalog_path.resolve()),
        "assets": [
            {
                "asset_id": s["slice"],
                "role": "slice",
                "path": s["path"],
                "format": "h5ad",
                "visibility": "public",
                "metadata": {"n_obs": s["n_obs"], "is_reference": s["is_reference"]},
            }
            for s in slices
        ],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    config = catalog["config"]
    title = NEUTRAL_TITLES.get(dataset, f"Albis {dataset.replace('_', ' ')} serial alignment") + (" (smoke)" if catalog["scale"] == "smoke" else "")
    payload = {
        "id": task_id,
        "name": title,
        "domain": "spatial_genomics",
        "modality": "spatial_transcriptomics",
        "task_type": "serial_alignment_simulated_ground_truth",
        "difficulty": difficulty,
        "status": "executable_local",
        "input_contract": "h5ad_spatial_v1",
        "output_contract": "aligned_coordinate_csv_v1",
        "metric_profile": "albis_alignment_v1",
        "label_key": "Region",
        "marker_gene_file": str(marker_file),
        "max_marker_genes": n_markers,
        "slices": [f"{s['slice']}={s['path']}" for s in slices],
        "paper_source": (
            f"Albis simulation, dataset {dataset} ({catalog['description']}); scale={catalog['scale']}; "
            f"n_cells={config['n_cells']}, sphere_R_um={config['sphere_R_um']:.1f}, n_slices={config['n_slices']}, "
            f"batch_sigma={config['batch_sigma']}, max_deg={config['max_deg']}, max_shift={config['max_shift']:.1f}. "
            + (f"Albis slices {[s['albis_slice_id'] for s in slices]}. " if catalog.get("slice_subset") else "")
            + ("Bins/spots gridded in each slide's frame. " if catalog.get("slide_frame") else "")
            + (f"QC {catalog['qc']}. " if catalog.get("qc") else "")
            + ("Half of the non-reference sections mirrored. " if catalog.get("flip") else "")
            + "Ground truth in slice s00's frame."
        ),
    }
    toml = "\n".join(f"{key} = {_format_toml_value(value)}" for key, value in payload.items()) + "\n"
    (task_dir / "task.toml").write_text(toml, encoding="utf-8")
    (task_dir / "instruction.md").write_text(_instruction(title, platform, slices, gap=bool(catalog.get("slice_subset")), qc=catalog.get("qc"),
                                                         flip=bool(catalog.get("flip"))), encoding="utf-8")
    return {"task_id": task_id, "dataset": dataset, "n_slices": len(slices), "task_dir": str(task_dir)}


def main() -> None:
    args = parse_args()
    datasets = args.dataset or [d for d in TASKS if (args.data_root / d / "catalog.json").exists()]
    created = [
        materialize(args.data_root / d / "catalog.json", args.output_root, args.id_suffix, force=args.force)
        for d in datasets
    ]
    print(json.dumps({"created_tasks": created, "n_tasks": len(created)}, indent=2))


if __name__ == "__main__":
    main()
