"""Plain-h5py slice reader shared by the scripted method baselines (scripts/run_<method>_baseline.py).

Each method lives in its own venv with its own anndata version, and some task files were written by old anndata
releases (dataframes without an encoding version), which newer `read_elem` refuses. This reader needs only h5py,
numpy and scipy, reads X (dense, CSR or CSC), the var index and obsm/spatial, and never touches obs.

Cross-platform and cross-resolution tasks pair slices with different gene panels. `set_task_genes(slices)` records,
per slice file, the columns of the genes shared by every slice of the task (in the first slice's order); after it is
called, `read_slice` returns X restricted to those genes, so every runner sees identical var names on every slice.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

_GENE_COLUMNS: dict[str, np.ndarray] = {}
_GENE_INFO: dict = {}


def _decode(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])
    return values.astype(str)


def read_var_names(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        g = f["var"]
        key = g.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        node = g[key]
        if isinstance(node, h5py.Group):  # categorical index (rare)
            cats = _decode(node["categories"][()])
            return cats[node["codes"][()]]
        return _decode(node[()])


def _read_X(node) -> sparse.csr_matrix:
    if isinstance(node, h5py.Dataset):
        return sparse.csr_matrix(np.asarray(node[()], dtype=np.float32))
    enc = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", "csr_matrix"))
    if isinstance(enc, bytes):
        enc = enc.decode()
    shape = tuple(int(x) for x in node.attrs.get("shape", node.attrs.get("h5sparse_shape")))
    parts = (node["data"][()].astype(np.float32), node["indices"][()], node["indptr"][()])
    if str(enc).startswith("csc"):
        return sparse.csc_matrix(parts, shape=shape).tocsr()
    return sparse.csr_matrix(parts, shape=shape)


def read_slice_h5(path: Path):
    """X (CSR float32, restricted to the task's shared genes if set_task_genes was called), var names, spatial."""
    with h5py.File(path, "r") as f:
        X = _read_X(f["X"])
        sp = f["obsm"]["spatial"]
        spatial = np.asarray(sp[()], dtype=np.float64)[:, :2]
    var = read_var_names(path)
    cols = _GENE_COLUMNS.get(str(Path(path)))
    if cols is not None:
        X = X[:, cols].tocsr()
        var = var[cols]
    return X, var, spatial


def set_task_genes(slices) -> dict:
    """Restrict every later read_slice of these slices to the genes shared by all of them.

    Returns a small summary for the run manifest. Raises SystemExit when the slices share no feature (e.g. RNA
    genes vs chromatin peaks), so the runner records the task as not applicable rather than producing nonsense.
    """
    _GENE_COLUMNS.clear()
    _GENE_INFO.clear()
    names = [read_var_names(s.path) for s in slices]
    has_dups = any(len(set(v)) < len(v) for v in names)   # duplicate gene names break name-based matching
    if not has_dups and all(len(v) == len(names[0]) and (v == names[0]).all() for v in names[1:]):
        _GENE_INFO.update(mode="identical", n_genes=int(len(names[0])))
        return dict(_GENE_INFO)
    shared = set(names[0]).intersection(*map(set, names[1:]))
    if not shared:
        raise SystemExit("NOT_APPLICABLE: the slices share no feature names (e.g. genes vs genomic peaks); "
                         "this method needs a shared expression feature space")
    order = [g for g in dict.fromkeys(names[0]) if g in shared]
    for s, v in zip(slices, names):
        first = {}
        for i, g in enumerate(v):
            first.setdefault(g, i)
        _GENE_COLUMNS[str(Path(s.path))] = np.array([first[g] for g in order], dtype=np.int64)
    _GENE_INFO.update(mode="intersection" + ("+dedup" if has_dups else ""), n_genes=len(order),
                      n_genes_per_slice=[int(len(v)) for v in names])
    print(f"[genes] panels {[len(v) for v in names]}{' (duplicate names dropped, first kept)' if has_dups else ''}; "
          f"using the {len(order)} shared unique genes",
          flush=True)
    return dict(_GENE_INFO)


def gene_info() -> dict:
    return dict(_GENE_INFO)
