#!/usr/bin/env python3
"""Run one of the four reference aligners on a task workspace.

    python3 run_reference.py WORKSPACE --method {identity,centroid_similarity,pca_similarity,icp_similarity}
                             [--out DIR]

Reads the sections listed under "Required Output" in WORKSPACE/TASK.md (data/<section>.h5ad), aligns them
in that order, each to the previously aligned one (the first section stays where it is), and writes
<section>.csv files (obs_id,x,y) to DIR (default WORKSPACE/submission/aligned_coordinates).

- identity: leaves every section where it is.
- centroid_similarity: matches per-label centroids when the sections carry a shared region label; the solve
  inputs carry none, so in practice it centres and rescales each section onto the previous one.
- pca_similarity: aligns principal axes (rotation, reflection and scale), keeping the candidate whose bounding
  box overlaps the previous section most.
- icp_similarity: iterative closest point with a similarity transform, started from the PCA candidates.

Needs numpy, scipy, pandas and h5py (run it with the workspace's agent_python.sh).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refalign.alignment import (centroid_similarity_alignment, icp_similarity_alignment,  # noqa: E402
                                identity_alignment, pca_similarity_alignment)
from refalign.h5ad import H5ADSlice  # noqa: E402
from refalign.io import write_coordinate_outputs  # noqa: E402

METHODS = {"identity": identity_alignment, "centroid_similarity": centroid_similarity_alignment,
           "pca_similarity": pca_similarity_alignment, "icp_similarity": icp_similarity_alignment}


def sections(workspace: Path) -> list[str]:
    text = (workspace / "TASK.md").read_text()
    block = text.split("## Required Output", 1)[-1]
    found = re.findall(r"submission/aligned_coordinates/([^`/\s]+)\.csv", block)
    return list(dict.fromkeys(found))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    ids = sections(a.workspace)
    if not ids:
        raise SystemExit("no required output CSVs found in TASK.md")
    slices = [H5ADSlice(a.workspace / "data" / f"{s}.h5ad", s) for s in ids]
    coords = METHODS[a.method](slices)
    out = a.out or a.workspace / "submission" / "aligned_coordinates"
    out.mkdir(parents=True, exist_ok=True)
    write_coordinate_outputs(coords, slices, out)
    print(f"{a.method}: wrote {len(ids)} sections to {out}")


if __name__ == "__main__":
    main()
