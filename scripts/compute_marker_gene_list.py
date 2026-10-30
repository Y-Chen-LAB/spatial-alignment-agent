"""Generate a paper2bench-compatible marker_genes.csv by picking top-N highly variable genes
intersected across the supplied slices. Used for tasks that lack a curated panel from the
source publication (e.g. mouse brain Visium slices), so the SABench gene-based metrics
(PCC, Cos_Sim, SSIM, MI) can still fire on real data.

Variance is computed on log1p-transformed per-cell-normalized counts. Output is a single-column
CSV with header `gene` so genomics_agent_bench.metrics.load_marker_genes reads it correctly.

The selection itself lives in `genomics_agent_bench.markers` so that
`scripts/build_tasks.py` uses exactly the same ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from genomics_agent_bench.markers import (
    DEFAULT_MAX_CELLS,
    DEFAULT_SAMPLE_BLOCKS,
    load_slices,
    select_marker_genes,
    write_marker_gene_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a top-N HVG marker gene list intersected across slices.")
    parser.add_argument("--slice", action="append", required=True, help="ID=PATH spec; repeat for multiple slices.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS, help="Cells subsampled per slice when ranking variance.")
    parser.add_argument("--sample-blocks", type=int, default=DEFAULT_SAMPLE_BLOCKS, help="Evenly spaced row blocks the cell subsample is drawn from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slices = load_slices(args.slice)
    genes, n_common = select_marker_genes(
        slices,
        top_n=args.top_n,
        max_cells=args.max_cells,
        n_blocks=args.sample_blocks,
    )
    write_marker_gene_file(args.output, genes)
    print(f"wrote {len(genes)} genes to {args.output} (cross-slice intersection: {n_common})")


if __name__ == "__main__":
    main()
