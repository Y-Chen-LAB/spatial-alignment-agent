"""Marker-gene selection shared by the task builder and the standalone CLI.

The SABench gene-based metrics (PCC, Cos_Sim, SSIM, MI) are computed per marker gene over
an overlapping grid, so a task needs a gene list that exists on every one of its slices.
Where the source publication supplies a panel we use it; otherwise we rank genes by their
log1p variance and keep the top N.

`select_marker_genes` is memory-bounded, which matters because some slices are large
(Visium HD carries 453,820 bins x 32,245 genes, about 355M nonzeros). Two things keep it
bounded. Sparse matrices are never densified: log1p and per-cell scaling both act on the
nonzero values alone, and per-gene variance follows from column sums of x and x^2. Rows
are subsampled as a set of evenly spaced contiguous blocks, so only a few percent of the
CSR payload is read, in a handful of contiguous reads rather than thousands of scattered
ones. Evenly spaced blocks rather than one block because a contiguous run of cells is a
tissue subregion, and ranking variance inside one subregion would bias the panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .h5ad import H5ADSlice

DEFAULT_TOP_N = 100
DEFAULT_MAX_CELLS = 20_000
DEFAULT_SAMPLE_BLOCKS = 20


def intersect_var_names(slices: Sequence[H5ADSlice]) -> list[str]:
    """Gene names present on every slice, in the first slice's order."""
    if not slices:
        return []
    common: set[str] | None = None
    for slice_ in slices:
        names = set(slice_.var_names.tolist())
        common = names if common is None else (common & names)
    assert common is not None
    return [gene for gene in slices[0].var_names.tolist() if gene in common]


def _row_blocks(n_rows: int, max_cells: int, n_blocks: int) -> list[tuple[int, int]]:
    """Evenly spaced contiguous row ranges covering at most `max_cells` rows in total."""
    if n_rows <= max_cells:
        return [(0, n_rows)]
    n_blocks = max(1, min(n_blocks, max_cells))
    block = max(1, max_cells // n_blocks)
    starts = np.linspace(0, n_rows - block, num=n_blocks, dtype=int)
    return [(int(s), int(s) + block) for s in starts]


def _column_variance(matrix: sparse.csr_matrix, columns: np.ndarray) -> np.ndarray:
    """Per-gene variance of log1p(CP10K) values, computed without densifying."""
    sub = matrix[:, columns].tocsc()
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    totals[totals == 0] = 1.0
    scaled = sub.tocoo()
    values = np.log1p(scaled.data.astype(float) / totals[scaled.row] * 1e4)
    scaled = sparse.coo_matrix((values, (scaled.row, scaled.col)), shape=sub.shape).tocsc()
    n = sub.shape[0]
    mean = np.asarray(scaled.sum(axis=0)).ravel() / n
    mean_sq = np.asarray(scaled.multiply(scaled).sum(axis=0)).ravel() / n
    return np.maximum(mean_sq - mean**2, 0.0)


def _slice_variance(slice_: H5ADSlice, genes: Sequence[str], *, max_cells: int, n_blocks: int) -> pd.Series:
    gene_to_index = {gene: idx for idx, gene in enumerate(slice_.var_names.tolist())}
    present = [gene for gene in genes if gene in gene_to_index]
    if not present:
        return pd.Series(dtype=float, name=slice_.id)
    columns = np.array([gene_to_index[gene] for gene in present], dtype=int)

    with h5py.File(slice_.path, "r") as handle:
        node = handle["X"]
        if isinstance(node, h5py.Group) and node.attrs.get("encoding-type") in {"csr_matrix", "csc_matrix"}:
            shape = tuple(int(x) for x in node.attrs["shape"])
            if node.attrs.get("encoding-type") == "csc_matrix":
                matrix = sparse.csc_matrix(
                    (node["data"][()], node["indices"][()], node["indptr"][()]), shape=shape
                ).tocsr()
            else:
                indptr = node["indptr"][()]
                blocks = _row_blocks(shape[0], max_cells, n_blocks)
                parts = []
                for start, stop in blocks:
                    lo, hi = int(indptr[start]), int(indptr[stop])
                    parts.append(
                        sparse.csr_matrix(
                            (node["data"][lo:hi], node["indices"][lo:hi], indptr[start : stop + 1] - lo),
                            shape=(stop - start, shape[1]),
                        )
                    )
                matrix = sparse.vstack(parts, format="csr") if len(parts) > 1 else parts[0]
            variances = _column_variance(matrix, columns)
        else:
            n_rows = node.shape[0]
            blocks = _row_blocks(n_rows, max_cells, n_blocks)
            rows = np.concatenate([np.arange(a, b) for a, b in blocks])
            dense = np.asarray(node[rows.min() : rows.max() + 1, :], dtype=float)[rows - rows.min()][:, columns]
            totals = dense.sum(axis=1, keepdims=True)
            totals[totals == 0] = 1.0
            variances = np.log1p(dense / totals * 1e4).var(axis=0)
    return pd.Series(variances, index=present, name=slice_.id)


def select_marker_genes(
    slices: Sequence[H5ADSlice],
    *,
    top_n: int = DEFAULT_TOP_N,
    max_cells: int = DEFAULT_MAX_CELLS,
    n_blocks: int = DEFAULT_SAMPLE_BLOCKS,
    candidates: Iterable[str] | None = None,
) -> tuple[list[str], int]:
    """Return (marker genes, size of the cross-slice gene intersection).

    Genes are ranked by the mean of their per-slice variance ranks, so a gene has to be
    variable on every slice of the task to be selected, not only on the largest one.
    """
    common = list(candidates) if candidates is not None else intersect_var_names(slices)
    if not common:
        return [], 0
    frames = [_slice_variance(s, common, max_cells=max_cells, n_blocks=n_blocks) for s in slices]
    variance = pd.concat(frames, axis=1).dropna(how="any")
    if variance.empty:
        return [], len(common)
    ordered = variance.rank(ascending=False).mean(axis=1).sort_values()
    return ordered.head(top_n).index.tolist(), len(common)


def write_marker_gene_file(path: Path, genes: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene": list(genes)}).to_csv(path, index=False)


def load_slices(specs: Iterable[str]) -> list[H5ADSlice]:
    """Parse repeated ``ID=PATH`` specs into slices."""
    out: list[H5ADSlice] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"slice spec must be ID=PATH, got {spec!r}")
        slice_id, path = spec.split("=", 1)
        out.append(H5ADSlice(Path(path), slice_id=slice_id))
    return out


def _strip_pandas_dedup_suffix(name: str) -> str:
    """`CARTPT.1` back to `CARTPT`.

    The paper's source-data workbooks repeat a few marker genes as extra columns, and
    pandas disambiguates them on read. The gene itself is the part before the suffix.
    """
    head, sep, tail = name.rpartition(".")
    return head if sep and tail.isdigit() else name


def paper_gene_list_from_source_data(path: Path) -> list[str]:
    """The gene list SABench scored, taken from a source-data table's column header.

    MOESM7 and MOESM10 report one column per marker gene (DLPFC 123 columns covering 117
    genes, MHPR 155, MOB 7), so the header is the paper's own definition of which genes the
    gene-based metrics were computed on.
    """
    header = pd.read_csv(path, index_col=0, nrows=0).columns.tolist()
    seen: dict[str, None] = {}
    for column in header:
        seen.setdefault(_strip_pandas_dedup_suffix(str(column)), None)
    return list(seen)


def paper_gene_list_from_panel(path: Path) -> list[str]:
    """The curated panel shipped with the SABench code archive (single-column CSV)."""
    frame = pd.read_csv(path)
    return [str(gene) for gene in frame.iloc[:, 0].dropna().tolist()]


def resolve_marker_genes(
    slices: Sequence[H5ADSlice],
    *,
    explicit: Sequence[str] | None = None,
    panel_csv: Path | None = None,
    source_data_csv: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    max_cells: int = DEFAULT_MAX_CELLS,
    n_blocks: int = DEFAULT_SAMPLE_BLOCKS,
) -> tuple[list[str], dict[str, object]]:
    """Pick the gene list a task is scored on, preferring SABench's own definition.

    Order of preference: an explicit list in the registry, the column header of a paper
    source-data table, the curated panel from the code archive, and only then a variance
    ranking computed here. The returned provenance dict is written into the task's data
    manifest so a reader can tell which of the four was used.

    The source-data header outranks the panel because it is the list SABench actually
    scored, after their own filtering. FeatureSimilarity/GeneBased.ipynb reads the raw
    panel, drops any gene absent from a slice, then drops three more by hard-coded index
    (`gene_list = [x for i, x in enumerate(gene_list) if i not in [72, 77, 120]]`). The
    DLPFC panel holds 124 genes and the published DLPFC results hold 117, so preferring
    the panel would score 7 genes the paper excluded.
    """
    common = set(intersect_var_names(slices))
    n_common = len(common)

    for genes, origin, detail in (
        (list(explicit or []), "registry_explicit", None),
        (paper_gene_list_from_source_data(source_data_csv) if source_data_csv else [], "paper_source_data", str(source_data_csv) if source_data_csv else None),
        (paper_gene_list_from_panel(panel_csv) if panel_csv else [], "sabench_panel", str(panel_csv) if panel_csv else None),
    ):
        if not genes:
            continue
        kept = [gene for gene in genes if gene in common]
        provenance = {
            "marker_source": origin,
            "marker_source_path": detail,
            "n_requested": len(genes),
            "n_present_on_all_slices": len(kept),
            "n_common_genes": n_common,
        }
        if kept:
            return kept, provenance
        provenance["fallback_reason"] = "no requested gene is present on every slice"

    genes, n_common = select_marker_genes(slices, top_n=top_n, max_cells=max_cells, n_blocks=n_blocks)
    return genes, {
        "marker_source": "log1p_variance_ranking",
        "marker_source_path": None,
        "n_requested": top_n,
        "n_present_on_all_slices": len(genes),
        "n_common_genes": n_common,
    }
