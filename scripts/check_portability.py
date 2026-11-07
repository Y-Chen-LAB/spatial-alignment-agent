"""Report paths that would not resolve on another machine.

Task files name their slices by absolute path, which is fine on the machine that built them
and fatal anywhere else. This lists what would have to be rebased before a cluster transfer,
and with --data-root rewrites them.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

ABSOLUTE = re.compile(r'(/users/[^"=\s]+|/Users/[^"=\s]+|/home/[^"=\s]+)')
TASKS = Path("benchmarks/spatial_alignment/tasks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", help="Rewrite the shared data prefix to this path.")
    ap.add_argument("--old-root", default="/users/ychen5/p2b-data/paper2bench-data")
    ap.add_argument("--apply", action="store_true", help="Write the rewrite; otherwise report only.")
    args = ap.parse_args()

    roots, files = Counter(), []
    for task in sorted(TASKS.glob("*/task.toml")):
        text = task.read_text()
        hits = ABSOLUTE.findall(text)
        if not hits:
            continue
        files.append(task)
        for hit in hits:
            roots["/".join(hit.split("/")[:5])] += 1

    print(f"{len(files)} of {len(list(TASKS.glob('*/task.toml')))} task files carry an absolute path")
    for root, count in roots.most_common():
        print(f"  {count:5d}  {root}")

    if not args.data_root:
        print("\nPass --data-root <path> --apply to rebase them for another machine.")
        return

    changed = 0
    for task in files:
        text = task.read_text()
        new = text.replace(args.old_root, args.data_root.rstrip("/"))
        if new != text:
            changed += 1
            if args.apply:
                task.write_text(new)
    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {changed} task files: {args.old_root} -> {args.data_root}")


if __name__ == "__main__":
    main()
