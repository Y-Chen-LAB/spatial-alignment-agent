"""Stage exactly the slices the tasks need into one directory, ready to rsync.

Task files name their slices from two different roots: an absolute path under
paper2bench-data, and a repository-relative path under
benchmarks/spatial_alignment/compiled_data, which is itself a symlink to data kept outside
the repo and excluded from git. A cluster needs both, and a `git clone` brings neither.

The bundle mirrors that split so the layout on the far side is predictable:

    <bundle>/paper2bench-data/...   the absolute references, rebased after transfer
    <bundle>/compiled_data/...      symlinked back into the repo after transfer

Files are hardlinked rather than copied when the bundle sits on the same filesystem as the
source, so staging 10 GB costs no extra disk and no time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tomllib
from pathlib import Path

TASKS = Path("benchmarks/spatial_alignment/tasks")
PAPER_DATA = Path.home() / "Downloads" / "paper2bench-data"
COMPILED = Path("benchmarks/spatial_alignment/compiled_data")


def sidecars() -> set[Path]:
    """Files the tooling reads that no task names, so walking the task files misses them.

    compile_perturbed_h5ad.py writes a report recording which paper slice each perturbed copy
    came from, and the coverage check reads those to trace a perturbed slice back to its
    source. They are a few kilobytes each and no task.toml mentions them, so a bundle built
    only from task references leaves them behind and coverage silently drops on the far side.
    """
    return set(COMPILED.glob("**/reports/*.json"))


def referenced() -> set[Path]:
    out: set[Path] = set()
    for task in sorted(TASKS.glob("*/task.toml")):
        spec = tomllib.load(open(task, "rb"))
        for key, value in spec.items():
            if key.endswith("slices") and isinstance(value, list):
                for entry in value:
                    out.add(Path(entry.split("=", 1)[1]))
    return out


def destination(source: Path, bundle: Path) -> Path | None:
    """Where a slice belongs in the bundle, keyed by the path the task names.

    Placement uses the unresolved path on purpose. Ten slices under paper2bench-data are
    symlinks into the unpacked SABench tutorial archive, so resolving first would put them
    outside both known roots and drop them from the bundle. The content still comes from the
    resolved target; only the layout follows the name the task uses.
    """
    for root, part in ((PAPER_DATA, "paper2bench-data"), (COMPILED, "compiled_data")):
        for candidate in (source, Path(os.path.abspath(source))):
            try:
                return bundle / part / candidate.relative_to(root if root.is_absolute() else Path(os.path.abspath(root)))
            except ValueError:
                continue
    # compiled_data is itself a symlink, so a task path may already be the resolved form.
    try:
        return bundle / "compiled_data" / source.resolve().relative_to(COMPILED.resolve())
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path.home() / "Downloads" / "p2b-cluster-bundle")
    ap.add_argument("--copy", action="store_true", help="Copy instead of hardlink.")
    args = ap.parse_args()

    files = sorted(referenced() | sidecars())
    args.bundle.mkdir(parents=True, exist_ok=True)
    linked = copied = skipped = 0
    total = 0
    unplaced: list[Path] = []

    for source in files:
        if not source.exists():
            skipped += 1
            continue
        target = destination(source, args.bundle)
        if target is None:
            unplaced.append(source)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            total += target.stat().st_size
            continue
        try:
            if args.copy:
                raise OSError
            os.link(source.resolve(), target)
            linked += 1
        except OSError:
            shutil.copy2(source.resolve(), target)
            copied += 1
        total += target.stat().st_size

    print(f"bundle: {args.bundle}")
    print(f"  {linked} hardlinked, {copied} copied, {skipped} missing")
    print(f"  {total / 1e9:.2f} GB")
    for part in ("paper2bench-data", "compiled_data"):
        sub = args.bundle / part
        if sub.exists():
            size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file())
            count = sum(1 for f in sub.rglob("*") if f.is_file())
            print(f"  {part:20s} {count:4d} files  {size / 1e9:5.2f} GB")
    if unplaced:
        print(f"\n  {len(unplaced)} file(s) under neither known root:")
        for f in unplaced[:5]:
            print(f"    {f}")


if __name__ == "__main__":
    main()
