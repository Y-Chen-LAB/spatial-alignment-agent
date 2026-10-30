"""Build spatial-alignment task packages from the declarative registry.

Replaces the four `materialize_*_tasks.py` scripts, which duplicated a TOML writer, a
marker-gene writer, a manifest writer, the task payload, the overwrite guard, slice-spec
formatting, slice stat lookup, the instruction writer and the summary print between them.
Everything here is driven by `benchmarks/spatial_alignment/task_registry.toml` plus the
slice-isolation table, so rebuilding a slice never requires editing a task.

Three entry shapes are supported:

    [[task]]       an explicit slice list
    [[generated]]  kind = "all_pairs", expanded to every unordered pair of its members
    [[sweep]]      a reference slice plus one perturbed copy per level, scored star-wise

Metric profile and label key are derived from the `labels` and `gt_coords` columns of
`paper_slice_isolation_status.csv` rather than probed ad hoc, and the marker gene list
prefers SABench's own definition over anything computed here (see
`genomics_agent_bench.markers.resolve_marker_genes`).

Usage:
    uv run python scripts/build_tasks.py --check-coverage
    uv run python scripts/build_tasks.py --only mob_cross_platform --force
    uv run python scripts/build_tasks.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.markers import resolve_marker_genes, write_marker_gene_file
from paper2bench.core.task_card import _to_toml

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks/spatial_alignment/task_registry.toml"
SLICE_TABLE = REPO / "benchmarks/spatial_alignment/campaigns/paper_slice_isolation_status.csv"
TASKS_ROOT = REPO / "benchmarks/spatial_alignment/tasks"
SWEEP_DATA_ROOT = REPO / "benchmarks/spatial_alignment/compiled_data/sweeps"

# Key order in the emitted task.toml, matching the 40 hand-authored files.
TASK_KEY_ORDER = (
    "id", "name", "domain", "modality", "task_type", "difficulty", "status",
    "input_contract", "output_contract", "metric_profile", "label_key",
    "marker_gene_file", "max_marker_genes", "split_manifest", "score_split",
    "public_score_split", "split_visibility", "pair_mode", "slices",
    "paper_data_ids", "paper_source",
)


@dataclass
class SliceRef:
    """One registry slice alias resolved against the slice-isolation table."""

    alias: str
    data_id: str
    path: Path
    labels: str
    gt_coords: str
    family: str
    platform_class: str = ""

    @property
    def has_labels(self) -> bool:
        return bool(self.labels)

    @property
    def has_ground_truth(self) -> bool:
        return "x_GT" in self.gt_coords and "y_GT" in self.gt_coords

    @property
    def label_key(self) -> str:
        for candidate in ("Region", "Cluster", "Region_proxy"):
            if candidate in self.labels:
                return candidate
        return "Region"


@dataclass
class TaskPlan:
    task_id: str
    name: str
    family: str
    slices: list[SliceRef]
    task_type: str
    difficulty: str
    paper_source: str = ""
    paper_reference: str = ""
    marker_source: str | None = None
    pair_mode: str = "consecutive"
    sweep: dict[str, Any] | None = None
    extra_manifest: dict[str, Any] = field(default_factory=dict)


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_slice_refs(registry: dict[str, Any]) -> dict[str, SliceRef]:
    table = pd.read_csv(SLICE_TABLE).fillna("")
    by_id = {str(row.data_id): row for row in table.itertuples()}
    platform_class = registry.get("platform_class", {})
    refs: dict[str, SliceRef] = {}
    for alias, data_id in registry.get("slices", {}).items():
        row = by_id.get(str(data_id))
        if row is None:
            raise SystemExit(f"registry slice {alias!r} names unknown data_id {data_id!r}")
        if not row.local_file:
            raise SystemExit(f"registry slice {alias!r} ({data_id}) has no local file; rebuild it first")
        refs[alias] = SliceRef(
            alias=alias,
            data_id=str(data_id),
            path=Path(str(row.local_file).replace("~", str(Path.home()))),
            labels=str(row.labels),
            gt_coords=str(row.gt_coords),
            family=str(row.family),
            platform_class=str(platform_class.get(alias, "")),
        )
    return refs


def resolve_by_data_id(data_id: str) -> SliceRef:
    """Look up a slice the registry names by data_id rather than alias (sweeps do this)."""
    table = pd.read_csv(SLICE_TABLE).fillna("")
    match = table[table.data_id == data_id]
    if match.empty:
        raise SystemExit(f"unknown data_id {data_id!r}")
    row = match.iloc[0]
    return SliceRef(
        alias=data_id.replace("/", "_").lower(),
        data_id=data_id,
        path=Path(str(row.local_file).replace("~", str(Path.home()))),
        labels=str(row.labels),
        gt_coords=str(row.gt_coords),
        family=str(row.family),
    )


def metric_profile_for(slices: Sequence[SliceRef], *, sweep: bool) -> tuple[str, str]:
    """(metric_profile, label_key) from what the slices actually carry."""
    if all(s.has_ground_truth for s in slices):
        return "sabench_mae_gene_efficiency_v1", slices[0].label_key
    if all(s.has_labels for s in slices):
        profile = "sabench_light_robustness_v1" if sweep else "sabench_light_alignment_v1"
        return profile, slices[0].label_key
    return "sabench_gene_only_alignment_v1", slices[0].label_key


def expand_plans(registry: dict[str, Any], refs: dict[str, SliceRef]) -> list[TaskPlan]:
    plans: list[TaskPlan] = []

    for entry in registry.get("generated", []):
        if entry.get("kind") != "all_pairs":
            raise SystemExit(f"unsupported generated kind {entry.get('kind')!r}")
        members = list(entry["members"])
        for index, (a, b) in enumerate(combinations(members, 2), start=1):
            short_a, short_b = a.split("_", 1)[-1], b.split("_", 1)[-1]
            classes = {refs[a].platform_class, refs[b].platform_class}
            plans.append(
                TaskPlan(
                    task_id=entry["id_template"].format(index=index, a=short_a, b=short_b),
                    name=entry["name_template"].format(a=short_a, b=short_b),
                    family=entry["family"],
                    slices=[refs[a], refs[b]],
                    task_type=entry["task_type"],
                    difficulty=entry["difficulty"],
                    paper_source=entry.get("paper_source", ""),
                    paper_reference=entry.get("paper_reference", ""),
                    extra_manifest={
                        "platform_pair": [short_a, short_b],
                        "platform_pair_class": "mixed" if len(classes) > 1 else classes.pop(),
                    },
                )
            )

    for entry in registry.get("task", []):
        plans.append(
            TaskPlan(
                task_id=entry["id"],
                name=entry["name"],
                family=entry["family"],
                slices=[refs[a] for a in entry["slices"]],
                task_type=entry["task_type"],
                difficulty=entry["difficulty"],
                paper_source=entry.get("paper_source", ""),
                paper_reference=entry.get("paper_reference", ""),
                marker_source=entry.get("marker_source"),
                pair_mode=entry.get("pair_mode", "consecutive"),
            )
        )

    for entry in registry.get("sweep", []):
        reference = resolve_by_data_id(entry["reference_slice"])
        moving = resolve_by_data_id(entry["moving_slice"])
        plans.append(
            TaskPlan(
                task_id=entry["id"],
                name=entry["name"],
                family=entry["family"],
                slices=[reference],  # perturbed levels are appended when they are generated
                task_type="robustness_sweep",
                difficulty=entry["difficulty"],
                paper_source=entry.get("paper_source", ""),
                paper_reference=entry.get("paper_reference", ""),
                marker_source=entry.get("marker_source"),
                pair_mode="star",
                sweep={"moving": moving, "perturbation": entry["perturbation"], "levels": list(entry["levels"])},
            )
        )
    return plans


def generate_sweep_slices(plan: TaskPlan, *, force: bool) -> list[SliceRef]:
    """One perturbed copy of the moving slice per level, via compile_perturbed_h5ad.py.

    The perturbation recipes are SABench's own (`Robustness/*_rotation.py` rotates the
    target clockwise about its centroid; `*_crop.py` keeps the left x-fraction), so the
    levels here reproduce the grid the paper reported in MOESM8.
    """
    assert plan.sweep is not None
    moving = plan.sweep["moving"]
    kind = plan.sweep["perturbation"]
    out_dir = SWEEP_DATA_ROOT / plan.task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[SliceRef] = []
    for level in plan.sweep["levels"]:
        tag = f"{kind}{level:03d}"
        out_path = out_dir / f"{moving.alias}_{tag}.h5ad"
        if not out_path.exists() or force:
            cmd = [
                sys.executable, str(REPO / "scripts/compile_perturbed_h5ad.py"),
                "--input", str(moving.path), "--output", str(out_path), "--force",
                "--report-json", str(out_dir / f"{moving.alias}_{tag}.json"),
            ]
            for column in ("Region", "Cluster", "x_GT", "y_GT", "z_GT"):
                cmd += ["--preserve-obs-col", column]
            if kind == "rotation":
                cmd += ["--rotate-degrees", str(level)]
            elif kind == "crop":
                cmd += ["--crop-overlap-percent", str(level)]
            else:
                raise SystemExit(f"unsupported sweep perturbation {kind!r}")
            subprocess.run(cmd, check=True, cwd=REPO, capture_output=True)
        produced.append(
            SliceRef(
                alias=f"{moving.alias}_{tag}",
                data_id=moving.data_id,
                path=out_path,
                labels=moving.labels,
                gt_coords=moving.gt_coords,
                family=moving.family,
            )
        )
    return produced


def instruction_for(plan: TaskPlan, profile: str, label_key: str, marker_info: dict[str, Any]) -> str:
    slice_lines = "\n".join(f"- `{s.alias}` at `{s.path}`" for s in plan.slices)
    if plan.pair_mode == "star":
        scoring = (
            f"Each perturbed slice is compared against `{plan.slices[0].alias}`, the unperturbed "
            "reference, so one score is produced per perturbation level."
        )
    else:
        scoring = "Consecutive slices are compared in the order listed above."
    label_note = (
        f"Region labels are held by the scorer under `obs/{label_key}`; they are not in your workspace."
        if profile != "sabench_gene_only_alignment_v1"
        else "This task is scored on gene-expression agreement only; no region labels are used."
    )
    return f"""# {plan.name}

Align the slices below into a shared coordinate system.

## Slices

{slice_lines}

## Scoring

{scoring} {label_note}

Metric profile `{profile}`. Gene-based metrics are computed over a fixed marker-gene list
held by the scorer ({marker_info['n_present_on_all_slices']} genes, source
`{marker_info['marker_source']}`); you do not choose it and it is not in your workspace.

## Output

Write one CSV per slice to `submission/aligned_coordinates/<slice_id>.csv` with columns
`obs_id,x,y`, one row per observation in the slice's original order.
"""


def build_one(plan: TaskPlan, registry: dict[str, Any], *, force: bool) -> dict[str, Any]:
    defaults = registry["defaults"]
    task_dir = TASKS_ROOT / plan.task_id
    if task_dir.exists():
        if not force:
            raise SystemExit(f"{task_dir} exists; pass --force to rebuild")
        shutil.rmtree(task_dir)
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True)

    slices = list(plan.slices)
    if plan.sweep is not None:
        slices += generate_sweep_slices(plan, force=force)
        plan.slices = slices  # so the instruction lists every level, not just the reference

    profile, label_key = metric_profile_for(slices, sweep=plan.pair_mode == "star")
    loaded = [H5ADSlice(s.path, slice_id=s.alias) for s in slices]

    marker_cfg = (registry.get("marker_sources", {}) or {}).get(plan.marker_source or "", {})
    panel = marker_cfg.get("panel")
    source_data = marker_cfg.get("source_data")
    genes, marker_info = resolve_marker_genes(
        loaded,
        panel_csv=Path(str(panel).replace("~", str(Path.home()))) if panel else None,
        source_data_csv=REPO / source_data if source_data else None,
        top_n=int(defaults.get("marker_top_n", 100)),
    )
    marker_file = data_dir / "marker_genes.csv"
    write_marker_gene_file(marker_file, genes)
    # No split manifest: scoring uses the full slice, as SABench does. A random half-split
    # depressed every gene metric by about 0.1 through simple attenuation, because halving the
    # cells halves each subgrid's occupancy and dilutes the per-grid mean expression the gene
    # metrics correlate. It protected only the solve-time self-scorer, which blind runs
    # withhold and which 7 of 527 historical runs ever invoked.

    payload: dict[str, Any] = {
        "id": plan.task_id,
        "name": plan.name,
        "domain": defaults["domain"],
        "modality": defaults["modality"],
        "task_type": plan.task_type,
        "difficulty": plan.difficulty,
        "status": defaults["status"],
        "input_contract": defaults["input_contract"],
        "output_contract": defaults["output_contract"],
        "metric_profile": profile,
        "label_key": label_key,
        "marker_gene_file": str(marker_file.relative_to(REPO)),
        "max_marker_genes": max(1, len(genes)),
        "slices": [f"{s.alias}={_relative(s.path)}" for s in slices],
        "paper_data_ids": sorted({s.data_id for s in slices}),
        "paper_source": plan.paper_source,
    }
    if plan.pair_mode != "consecutive":
        payload["pair_mode"] = plan.pair_mode
    ordered = {key: payload[key] for key in TASK_KEY_ORDER if key in payload}
    (task_dir / "task.toml").write_text(_to_toml(ordered), encoding="utf-8")
    (task_dir / "instruction.md").write_text(
        instruction_for(plan, profile, label_key, marker_info), encoding="utf-8"
    )

    manifest = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": plan.task_id,
        "family": plan.family,
        "compiler": "scripts/build_tasks.py",
        "registry": str(REGISTRY.relative_to(REPO)),
        "paper_reference": plan.paper_reference,
        "marker_genes": marker_info,
        "pair_mode": plan.pair_mode,
        "supersedes": (registry.get("supersedes", {}) or {}).get(plan.task_id),
        "assets": [
            {
                "asset_id": s.alias,
                "format": "h5ad",
                "path": _relative(s.path),
                "role": "slice",
                "visibility": "public",
                "metadata": {
                    "paper_data_id": s.data_id,
                    "n_obs": int(len(loaded[i].obs_names)),
                    "n_var": int(len(loaded[i].var_names)),
                },
            }
            for i, s in enumerate(slices)
        ],
        **plan.extra_manifest,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"task_id": plan.task_id, "family": plan.family, "n_slices": len(slices),
            "metric_profile": profile, "n_marker_genes": len(genes),
            "marker_source": marker_info["marker_source"]}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _derivation_map() -> dict[str, set[str]]:
    """Derived slice file name -> every source it was ever generated from.

    Several tasks use a perturbed copy of a paper slice rather than the slice itself
    (`compile_perturbed_h5ad.py` writes a report recording input and output), so a paper
    slice can be covered by a task that never names its file directly. The value is a set
    because the same output basename can be regenerated from a different source: the
    sagittal sections were rebuilt from Space Ranger 1.0.0 matrices under the names the
    2.0.0 build had already used.
    """
    derived: dict[str, set[str]] = {}
    for report in (REPO / "benchmarks/spatial_alignment/compiled_data").glob("**/reports/*.json"):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        source, output = payload.get("input"), payload.get("output")
        if source and output:
            derived.setdefault(Path(str(output)).name, set()).add(Path(str(source)).name)
    return derived


def covered_slice_paths(task_globs: Iterable[str]) -> set[str]:
    """Every h5ad file name a task uses, plus the source of any perturbed copy it uses."""
    import re

    derived = _derivation_map()
    covered: set[str] = set()
    for pattern in task_globs:
        for task in REPO.glob(pattern.strip()):
            text = task.read_text(encoding="utf-8")
            for _, path in re.findall(r'"([^"=]+)=([^"]+\.h5ad)"', text):
                pending = [Path(path).name]
                while pending:
                    name = pending.pop()
                    if name in covered:
                        continue
                    covered.add(name)
                    pending.extend(derived.get(name, ()))
    return covered


def check_coverage(registry: dict[str, Any]) -> int:
    table = pd.read_csv(SLICE_TABLE).fillna("")
    globs = registry.get("legacy", {}).get("task_glob", "").split(",")
    globs = [g for g in globs if g.strip()] + ["benchmarks/spatial_alignment/tasks/*/task.toml"]
    covered = covered_slice_paths(globs)
    # A sweep covers its moving slice through the perturbed copies, which carry a
    # different file name; treat the source slice as covered when a sweep names it.
    for entry in registry.get("sweep", []):
        for key in ("reference_slice", "moving_slice"):
            ref = resolve_by_data_id(entry[key])
            covered.add(ref.path.name)
    table["covered"] = table.local_file.apply(lambda p: Path(str(p)).name in covered if p else False)
    missing = table[~table.covered]
    print(f"coverage: {int(table.covered.sum())} of {len(table)} paper slices appear in at least one task")
    if not missing.empty:
        print("\nnot covered:")
        print(missing.groupby("family").size().to_string())
        print(missing[["data_id", "family", "paper_file", "status"]].to_string(index=False))
    return 0 if missing.empty else 1



# --------------------------------------------------------------------------- paper pairs
# The paper's unit of work is a slice pair, not a dataset: "a total of 295 tasks were
# executed, including 260 pairs of real slices and 35 pairs of simulated slices"
# (Methods, lines 157-159). The 260 decompose as adjacent pairs within each serial family,
# two pairs for BCA (the Visium slice against each Xenium replicate), and all 45 unordered
# pairs of the ten AMBC platforms. The 35 simulated pairs are the 25 rotation levels and 10
# overlap levels of the DLPFC robustness grid.
#
# Our task packages group those pairs: one serial task scores every adjacent pair inside it
# in a single agent run, and the scorer already reports each pair separately in
# `scores.json["pairs"]`. So the pairs below are registered for coverage and reporting, not
# built as 295 separate directories, which would multiply agent cost by about five for the
# same numbers.

PAPER_SERIAL_FAMILIES = (
    ("DLPFC_shuffled", "adjacent"),
    ("MHPR_processed", "adjacent"),
    ("SCC_P2_processed", "adjacent"),
    ("SCC_P5_processed", "adjacent"),
    ("E14_16", "adjacent"),
    ("MBSA_processed", "adjacent"),
    ("MBSP_processed", "adjacent"),
    ("MPC_processed", "adjacent"),
    ("MMPC_processed", "adjacent"),
    ("MBRM_processed", "adjacent"),
    ("MOp_processed", "adjacent"),
    ("MLN_processed", "adjacent"),
    ("WMB_processed", "adjacent"),
    ("MOB_processed", "adjacent"),
)


def paper_pairs() -> list[dict[str, str]]:
    """The paper's 295 alignment tasks, as (left, right) slice-file pairs."""
    table = pd.read_csv(SLICE_TABLE).fillna("")
    rows: list[dict[str, str]] = []

    for family, _ in PAPER_SERIAL_FAMILIES:
        members = table[table.family == family]
        files = [Path(str(f)).name for f in members.local_file]
        ids = list(members.data_id)
        for i in range(len(files) - 1):
            rows.append({"kind": "serial", "family": family, "left_id": ids[i], "right_id": ids[i + 1],
                         "left": files[i], "right": files[i + 1]})

    bca = table[table.family == "BCA_processed"]
    bca_files = {str(r.data_id): Path(str(r.local_file)).name for r in bca.itertuples()}
    for replicate in ("Data229", "Data230"):
        rows.append({"kind": "cross_platform", "family": "BCA_processed", "left_id": "Data228",
                     "right_id": replicate, "left": bca_files["Data228"], "right": bca_files[replicate]})

    ambc = table[table.family == "AMBC_processed"]
    ambc_files = [(str(r.data_id), Path(str(r.local_file)).name) for r in ambc.itertuples()]
    for (aid, a), (bid, b) in combinations(ambc_files, 2):
        rows.append({"kind": "cross_platform", "family": "AMBC_processed", "left_id": aid,
                     "right_id": bid, "left": a, "right": b})

    for level in range(0, 361, 15):
        rows.append({"kind": "simulated", "family": "DLPFC_rotation", "left_id": "Data1",
                     "right_id": f"Data2+rotation{level:03d}", "left": "151673.h5ad",
                     "right": f"data2_rotation{level:03d}.h5ad"})
    for level in range(10, 101, 10):
        rows.append({"kind": "simulated", "family": "DLPFC_crop", "left_id": "Data1",
                     "right_id": f"Data2+crop{level:03d}", "left": "151673.h5ad",
                     "right": f"data2_crop{level:03d}.h5ad"})
    return rows


def suite_pairs() -> dict[tuple[str, str], list[str]]:
    """Every slice pair the built tasks score, mapped to the tasks that score it."""
    import tomllib as _tomllib

    scored: dict[tuple[str, str], list[str]] = {}
    for task in sorted(TASKS_ROOT.glob("*/task.toml")):
        payload = _tomllib.loads(task.read_text(encoding="utf-8"))
        specs = payload.get("slices") or payload.get("raw_slices") or []
        files = [Path(str(spec).split("=", 1)[1]).name for spec in specs]
        if not files:
            continue
        mode = payload.get("pair_mode", "consecutive")
        pairs = list(zip(files[:-1], files[1:])) if mode == "consecutive" else [(files[0], other) for other in files[1:]]
        for left, right in pairs:
            scored.setdefault(tuple(sorted((left, right))), []).append(str(payload["id"]))
    return scored


def check_pair_coverage(out_path: Path) -> int:
    wanted = paper_pairs()
    scored = suite_pairs()
    derived = _derivation_map()

    def resolve(name: str) -> set[str]:
        """A paper slice is also matched by any perturbed copy built from it."""
        names = {name}
        for output, sources in derived.items():
            if name in sources:
                names.add(output)
        return names

    records = []
    for row in wanted:
        candidates = {tuple(sorted((a, b))) for a in resolve(row["left"]) for b in resolve(row["right"])}
        tasks = sorted({t for key in candidates for t in scored.get(key, [])})
        records.append({**row, "scored_by": ";".join(tasks), "covered": bool(tasks)})
    frame = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print(f"paper pairs registered: {len(frame)} (paper states 295)")
    print(frame.groupby("kind").agg(pairs=("covered", "size"), covered=("covered", "sum")).to_string())
    missing = frame[~frame.covered]
    if not missing.empty:
        print("\nnot scored by any task:")
        print(missing.groupby("family").size().to_string())
        print(missing[["family", "left_id", "right_id"]].head(20).to_string(index=False))
    print(f"\nwrote {out_path}")
    return 0 if missing.empty else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], help="Build only these families or task ids.")
    ap.add_argument("--force", action="store_true", help="Rebuild task directories that already exist.")
    ap.add_argument("--check-coverage", action="store_true", help="Report slice coverage and exit.")
    ap.add_argument("--check-pair-coverage", action="store_true", help="Report coverage of the paper's 295 slice pairs and exit.")
    ap.add_argument("--list", action="store_true", help="List the planned tasks without building.")
    args = ap.parse_args()

    registry = load_registry()
    if args.check_coverage:
        return check_coverage(registry)
    if args.check_pair_coverage:
        return check_pair_coverage(REPO / "benchmarks/spatial_alignment/campaigns/paper_pair_coverage.csv")

    refs = load_slice_refs(registry)
    plans = expand_plans(registry, refs)
    if args.only:
        wanted = set(args.only)
        plans = [p for p in plans if p.family in wanted or p.task_id in wanted]
        if not plans:
            raise SystemExit(f"no registry entry matches {sorted(wanted)}")

    if args.list:
        for plan in plans:
            print(f"{plan.task_id:52s} {plan.family:22s} {len(plan.slices)} declared slices")
        print(f"\n{len(plans)} tasks planned")
        return 0

    built = [build_one(plan, registry, force=args.force) for plan in plans]
    for row in built:
        print(
            f"{row['task_id']:52s} {row['family']:22s} slices={row['n_slices']:3d} "
            f"{row['metric_profile']:34s} markers={row['n_marker_genes']:4d} ({row['marker_source']})"
        )
    print(f"\nbuilt {len(built)} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
