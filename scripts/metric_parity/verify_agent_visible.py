"""Sanitise one slice from every task and assert the agent-visible file carries no answers.

The sanitiser is an allowlist: it builds a fresh H5AD and copies only X, var, layers,
obs/_index and obsm/spatial. This checks the allowlist actually behaves that way on real
files rather than trusting the reading, and it runs on every task so a task whose slices
have an unusual structure cannot slip through.
"""
from __future__ import annotations

import re, sys, tempfile, tomllib
from pathlib import Path

sys.path.insert(0, "scripts")
import h5py
from prepare_agent_task import _write_sanitized_h5ad

GT = re.compile(r"GT|ccf|CCF|3d_aligned|new_[xyz]|raw_[xyz]", re.I)
LABELS = {"Region", "Cluster", "annotation", "Cell_class", "Neuron_cluster_ID", "layer", "Layer"}


def inspect(path: Path) -> dict:
    with h5py.File(path, "r") as h:
        return {
            "top": sorted(h.keys()),
            "obs": sorted(k for k in h.get("obs", {}).keys() if not k.startswith("_")),
            "obsm": sorted(h.get("obsm", {}).keys()),
            "uns": sorted(h.get("uns", {}).keys()) if "uns" in h else [],
            "obsp": sorted(h.get("obsp", {}).keys()) if "obsp" in h else [],
            "varm": sorted(h.get("varm", {}).keys()) if "varm" in h else [],
        }


def main() -> None:
    tasks = sorted(Path("benchmarks/spatial_alignment/tasks").glob("*/task.toml"))
    failures, checked, protected = [], 0, 0
    with tempfile.TemporaryDirectory(prefix="p2b_visible_") as tmp:
        for tp in tasks:
            spec = tomllib.load(open(tp, "rb"))
            # sa-005 names its inputs raw_slices and downsampled_slices rather than slices,
            # so collect from every key that ends in "slices" and check one of each list.
            lists = [v for k, v in spec.items() if k.endswith("slices") and isinstance(v, list) and v]
            if not lists:
                continue
            sources = [Path(group[0].split("=", 1)[1]) for group in lists]
            source = sources[0]
            before = inspect(source)
            if any(GT.search(c) for c in before["obs"]) or (set(before["obs"]) & LABELS):
                protected += 1

            problems_all = []
            for index, source in enumerate(sources):
                dest = Path(tmp) / f"{spec['id']}_{index}.h5ad"
                try:
                    _write_sanitized_h5ad(source, dest)
                except Exception as exc:
                    problems_all.append(f"sanitise failed on {source.name}: {exc}")
                    continue
                after = inspect(dest)
                dest.unlink(missing_ok=True)
                if after["obs"]:
                    problems_all.append(f"{source.name}: obs columns survived {after['obs']}")
                if [k for k in after["obsm"] if k != "spatial"]:
                    problems_all.append(f"{source.name}: extra obsm {after['obsm']}")
                for key in ("uns", "obsp", "varm"):
                    if after[key]:
                        problems_all.append(f"{source.name}: {key} survived {after[key]}")
                if "spatial" not in after["obsm"]:
                    problems_all.append(f"{source.name}: spatial missing, task unsolvable")
            checked += 1
            if problems_all:
                failures.append((spec["id"], "; ".join(problems_all)))
            continue

    print(f"tasks checked: {checked} of {len(tasks)}")
    print(f"  whose source slice carried GT or labels: {protected}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for task, why in failures:
            print(f"  {task}: {why}")
        sys.exit(1)
    print("\nall clear: every agent-visible file exposes only obs/_index, obsm/spatial, X and var")


if __name__ == "__main__":
    main()
