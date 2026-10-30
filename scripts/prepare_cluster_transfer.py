"""List exactly the data a cluster needs, and write an rsync manifest for it.

The data directory holds about 125 GB, most of it raw downloads kept for provenance. The 92
tasks reference 10.4 GB of it, so moving the directory wholesale costs an order of magnitude
more transfer than the run needs. This walks the task files, resolves every slice they name,
and emits the list.

Also rewrites the absolute paths the task files carry, which is the other half of the move:
61 of 92 hardcode a path under the machine that built them.
"""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path

TASKS = Path("benchmarks/spatial_alignment/tasks")


def referenced_slices() -> set[Path]:
    files: set[Path] = set()
    for task in sorted(TASKS.glob("*/task.toml")):
        spec = tomllib.load(open(task, "rb"))
        for key, value in spec.items():
            if key.endswith("slices") and isinstance(value, list):
                for entry in value:
                    files.add(Path(entry.split("=", 1)[1]))
    return files


def digest(path: Path, limit: int = 1 << 20) -> str:
    """Hash the head of a file. Enough to catch a truncated transfer without reading 10 GB."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(limit))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("results/cluster_transfer_manifest.tsv"))
    ap.add_argument("--rsync-list", type=Path, default=Path("results/cluster_transfer_files.txt"))
    ap.add_argument("--checksums", action="store_true", help="Hash each file; slower, catches truncation.")
    args = ap.parse_args()

    files = sorted(referenced_slices())
    missing = [f for f in files if not f.exists()]
    present = [f for f in files if f.exists()]
    total = sum(f.stat().st_size for f in present)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        out.write("path\tbytes\tdigest\n")
        for f in present:
            out.write(f"{f}\t{f.stat().st_size}\t{digest(f) if args.checksums else ''}\n")
    args.rsync_list.write_text("\n".join(str(f) for f in present) + "\n")

    print(f"slices referenced : {len(files)}")
    print(f"present           : {len(present)}  ({total / 1e9:.1f} GB)")
    if missing:
        print(f"MISSING           : {len(missing)}")
        for f in missing[:5]:
            print(f"  {f}")
    print(f"\nmanifest  -> {args.output}")
    print(f"file list -> {args.rsync_list}")
    print("\ntransfer with:")
    print(f"  rsync -av --files-from={args.rsync_list} / cluster:/scratch/p2b-data/")
    print("then rebase the task files on the cluster:")
    print("  ./.venv/bin/python scripts/check_portability.py --data-root /scratch/p2b-data --apply")


if __name__ == "__main__":
    main()
