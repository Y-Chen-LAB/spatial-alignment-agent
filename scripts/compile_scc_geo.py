"""Compile SCC (Ji et al. 2020) GEO RAW tar (Data10-15 in SABench paper) into 6 paper2bench h5ad slices.

GSE144240_RAW.tar ships per-replicate pairs:
    GSM*_<patient>_ST_<rep>_stdata.tsv.gz                 spot-by-gene expression
    GSM*_spot_data-selection-<patient>_ST_<rep>.tsv.gz    in-tissue spot coords

We materialize P2_rep1/2/3 (Data10-12) and P5_rep1/2/3 (Data13-15) as separate h5ads using
pixel_x/pixel_y as obsm/spatial. The raw bundle ships no curated cluster annotations, so
obs/Region is omitted; tasks score under sabench_gene_only_alignment_v1.
"""

from __future__ import annotations

import argparse
import gzip
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


REPLICATE_RE = re.compile(r"_(P\d+)_ST_(rep\d+)_stdata\.tsv\.gz$")


def _members_for_replicate(tar: tarfile.TarFile, patient: str, rep: str) -> tuple[str, str] | None:
    stdata = None
    spot = None
    for m in tar.getmembers():
        if not m.isfile():
            continue
        if m.name.endswith(f"_{patient}_ST_{rep}_stdata.tsv.gz"):
            stdata = m.name
        if m.name.endswith(f"-{patient}_ST_{rep}.tsv.gz") and "spot_data-selection" in m.name:
            spot = m.name
    if not (stdata and spot):
        return None
    return stdata, spot


def _read_gz_tsv(tar: tarfile.TarFile, member_name: str, **kwargs) -> pd.DataFrame:
    raw = tar.extractfile(member_name)
    assert raw is not None
    decompressed = gzip.GzipFile(fileobj=raw)
    return pd.read_csv(decompressed, sep="\t", **kwargs)


def _write_h5ad(path: Path, spot_ids: np.ndarray, gene_names: np.ndarray, expression: np.ndarray, spatial: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        obs = f.create_group("obs")
        obs.attrs["encoding-type"] = "dataframe"
        obs.attrs["_index"] = "_index"
        obs.create_dataset("_index", data=spot_ids.astype("S"))
        obs.attrs["column-order"] = np.array([], dtype="S1")

        var = f.create_group("var")
        var.attrs["encoding-type"] = "dataframe"
        var.attrs["_index"] = "_index"
        var.create_dataset("_index", data=gene_names.astype("S"))
        var.attrs["column-order"] = np.array([], dtype="S1")

        obsm = f.create_group("obsm")
        obsm.create_dataset("spatial", data=spatial.astype(np.float64))

        f.create_dataset("X", data=expression.astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile SCC GEO tar into paper2bench h5ad slices.")
    parser.add_argument("--tar", type=Path, default=Path("/Users/yiqun/Downloads/paper2bench-data/sabench_full/Data10-15_SCC/GSE144240_RAW.tar"))
    parser.add_argument("--patients", nargs="+", default=["P2", "P5"], help="Patient prefixes to extract; default the SABench Data10-15 set.")
    parser.add_argument("--reps", nargs="+", default=["rep1", "rep2", "rep3"], help="Replicate suffixes per patient; default rep1-3.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    with tarfile.open(args.tar, "r") as tar:
        for patient in args.patients:
            for rep in args.reps:
                pair = _members_for_replicate(tar, patient, rep)
                if pair is None:
                    print(f"  skip {patient} {rep}: no matching members in tar")
                    continue
                stdata_name, spot_name = pair
                spot_df = _read_gz_tsv(tar, spot_name)
                stdata = _read_gz_tsv(tar, stdata_name, index_col=0)
                # spot_df: x, y, new_x, new_y, pixel_x, pixel_y
                spot_df["spot_id"] = spot_df.apply(lambda r: f"{int(r['x'])}x{int(r['y'])}", axis=1)
                spot_df = spot_df.set_index("spot_id")
                shared = sorted(set(spot_df.index).intersection(stdata.index))
                if not shared:
                    print(f"  skip {patient} {rep}: no shared spots between stdata and spot_data-selection")
                    continue
                stdata = stdata.loc[shared]
                spot_df = spot_df.loc[shared]
                slice_id = f"SCC_{patient}_{rep}"
                out_path = args.output_dir / f"{slice_id}.h5ad"
                _write_h5ad(
                    out_path,
                    spot_ids=np.array(shared),
                    gene_names=stdata.columns.to_numpy().astype(str),
                    expression=stdata.to_numpy(dtype=np.float32),
                    spatial=spot_df[["pixel_x", "pixel_y"]].to_numpy(dtype=float),
                )
                print(f"  wrote {out_path.name}: spots={len(shared)} genes={stdata.shape[1]}")
                written.append(slice_id)

    print(f"compiled {len(written)} slices: {written}")


if __name__ == "__main__":
    main()
