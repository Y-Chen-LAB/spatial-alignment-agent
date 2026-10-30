#!/usr/bin/env python3
"""Inspect the Open-ST metastatic lymph node H5AD for SABench task planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _decode_label(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def build_summary(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        var = handle["var"]
        obsm = handle["obsm"]

        sections = obs["n_section"][:]
        spatial = obsm["spatial"][:]
        spatial_3d = obsm["spatial_3d_aligned"][:]

        annotation = obs["annotation"]
        categories = [_decode_label(value) for value in annotation["categories"][:]]
        codes = annotation["codes"][:]

        uniq_sections, section_counts = np.unique(sections, return_counts=True)

        annotation_counts = {}
        for code, count in zip(*np.unique(codes[codes >= 0], return_counts=True)):
            annotation_counts[categories[int(code)]] = int(count)

        per_section = []
        for section_id, n_cells in zip(uniq_sections, section_counts):
            mask = sections == section_id
            xy = spatial[mask]
            xyz = spatial_3d[mask]
            section_codes = codes[mask]
            valid_codes = section_codes[section_codes >= 0]
            label_counts = {}
            for code, count in zip(*np.unique(valid_codes, return_counts=True)):
                label_counts[categories[int(code)]] = int(count)

            per_section.append(
                {
                    "section_id": int(section_id),
                    "n_cells": int(n_cells),
                    "n_labels": len(label_counts),
                    "z_values": sorted(np.unique(xyz[:, 2]).astype(float).tolist()),
                    "xy_min": [float(xy[:, 0].min()), float(xy[:, 1].min())],
                    "xy_max": [float(xy[:, 0].max()), float(xy[:, 1].max())],
                    "xyz_min": [
                        float(xyz[:, 0].min()),
                        float(xyz[:, 1].min()),
                        float(xyz[:, 2].min()),
                    ],
                    "xyz_max": [
                        float(xyz[:, 0].max()),
                        float(xyz[:, 1].max()),
                        float(xyz[:, 2].max()),
                    ],
                    "top_labels": [
                        {"label": label, "count": count}
                        for label, count in sorted(
                            label_counts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:5]
                    ],
                }
            )

        ordered_sections = [int(section_id) for section_id in uniq_sections.tolist()]
        adjacent_pairs = [
            [ordered_sections[index], ordered_sections[index + 1]]
            for index in range(len(ordered_sections) - 1)
        ]
        five_slice_windows = [
            ordered_sections[index : index + 5]
            for index in range(len(ordered_sections) - 4)
        ]

        return {
            "source_path": str(path),
            "n_cells": int(len(obs["_index"])),
            "n_genes": int(len(var["_index"])),
            "obs_columns": sorted(obs.keys()),
            "var_columns": sorted(var.keys()),
            "obsm_keys": sorted(obsm.keys()),
            "section_ids": ordered_sections,
            "n_sections": len(ordered_sections),
            "annotation_categories": categories,
            "annotation_counts": annotation_counts,
            "per_section": per_section,
            "candidate_task_families": {
                "full_serial_19": ordered_sections,
                "adjacent_pairs": adjacent_pairs,
                "five_slice_windows": five_slice_windows,
                "three_block_starters": [
                    ordered_sections[:5],
                    ordered_sections[7:12],
                    ordered_sections[-5:],
                ],
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    summary = build_summary(args.input_h5ad)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2))
    print(args.json_out)


if __name__ == "__main__":
    main()
