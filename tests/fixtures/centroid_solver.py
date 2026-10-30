"""Minimal solver used to smoke-test the hidden-perturbation harness.

Reads data/reference.h5ad + data/moving.h5ad relative to cwd, runs the in-repo
centroid_similarity_alignment baseline, writes submission/aligned_coordinates/*.csv.
"""
from __future__ import annotations
from pathlib import Path

from genomics_agent_bench.alignment import centroid_similarity_alignment
from genomics_agent_bench.h5ad import H5ADSlice
from genomics_agent_bench.io import write_coordinate_outputs

slices = [
    H5ADSlice(Path("data/reference.h5ad"), slice_id="reference"),
    H5ADSlice(Path("data/moving.h5ad"),    slice_id="moving"),
]
coords = centroid_similarity_alignment(slices, label_key="Region")
write_coordinate_outputs(coords, slices, Path("submission/aligned_coordinates"))
