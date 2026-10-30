from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy import sparse


def _decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"S", "O"}:
        return np.array([value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values])
    return values


def _read_string_dataset(handle: h5py.File, key: str) -> np.ndarray:
    return _decode_array(handle[key][()])


def _read_obs_column(handle: h5py.File, key: str) -> np.ndarray:
    node = handle[f"obs/{key}"]
    if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") == "categorical":
        categories = _decode_array(node["categories"][()])
        codes = node["codes"][()]
        out = np.full(codes.shape, "", dtype=object)
        mask = codes >= 0
        out[mask] = categories[codes[mask]]
        return out.astype(str)
    return _decode_array(node[()]).astype(str)


@dataclass(frozen=True)
class H5ADSlice:
    """Small reader for the AnnData fields needed by alignment scorers."""

    path: Path
    slice_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    @cached_property
    def id(self) -> str:
        return self.slice_id or self.path.stem

    @cached_property
    def obs_names(self) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            return _read_string_dataset(handle, "obs/_index")

    @cached_property
    def var_names(self) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            return _read_string_dataset(handle, "var/_index")

    @cached_property
    def spatial(self) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            return np.asarray(handle["obsm/spatial"][()], dtype=float)

    def obs_column(self, key: str) -> np.ndarray:
        with h5py.File(self.path, "r") as handle:
            if f"obs/{key}" not in handle:
                raise KeyError(f"{key!r} is not present in obs for {self.path}")
            return _read_obs_column(handle, key)

    def has_obs_column(self, key: str) -> bool:
        with h5py.File(self.path, "r") as handle:
            return f"obs/{key}" in handle

    def expression_for_genes(self, genes: Iterable[str]) -> tuple[list[str], np.ndarray]:
        requested = list(dict.fromkeys(genes))
        gene_to_index = {gene: idx for idx, gene in enumerate(self.var_names)}
        present = [gene for gene in requested if gene in gene_to_index]
        if not present:
            return [], np.empty((self.spatial.shape[0], 0), dtype=float)

        columns = np.array([gene_to_index[gene] for gene in present], dtype=int)
        sort_order = np.argsort(columns, kind="stable")
        sorted_columns = columns[sort_order]
        inverse_order = np.empty_like(sort_order)
        inverse_order[sort_order] = np.arange(sort_order.size)
        with h5py.File(self.path, "r") as handle:
            node = handle["X"]
            if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") in {"csr_matrix", "csc_matrix"}:
                shape = tuple(int(x) for x in node.attrs["shape"])
                matrix = sparse.csr_matrix((node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape)
                values = matrix[:, sorted_columns].toarray()
            else:
                values = np.asarray(node[:, sorted_columns], dtype=float)
        values = values[:, inverse_order]
        # Deliberately no upcast. SABench reads `adata.X[:, mask].toarray()`, so the grid means
        # it feeds to the metrics carry the stored dtype (float32 for every slice we hold).
        # PCC, Cos_Sim and SSIM are insensitive to this, but sklearn's k-NN mutual-information
        # estimator is not: on byte-identical float32 and float64 inputs it returns values up to
        # 0.03 apart. Upcasting here would silently shift gene_mi away from the published numbers.
        return present, values


class MaskedSlice:
    """H5ADSlice view restricted to a boolean observation mask. Optional `spatial` overrides the base coords."""

    def __init__(self, base: H5ADSlice, mask: np.ndarray, spatial: np.ndarray | None = None):
        self.base = base
        self.mask = mask
        self.id = base.id
        self.path = base.path
        self.obs_names = base.obs_names[mask]
        self.spatial = spatial if spatial is not None else base.spatial[mask]

    def has_obs_column(self, key: str) -> bool:
        return self.base.has_obs_column(key)

    def obs_column(self, key: str) -> np.ndarray:
        return self.base.obs_column(key)[self.mask]

    def expression_for_genes(self, genes: Iterable[str]) -> tuple[list[str], np.ndarray]:
        present, values = self.base.expression_for_genes(genes)
        return present, values[self.mask]
