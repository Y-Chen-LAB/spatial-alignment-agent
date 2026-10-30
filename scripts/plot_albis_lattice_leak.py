#!/usr/bin/env python3
"""Show that Albis's bin/spot tasks leak each slice's rotation through the point lattice.

Albis builds the bin or spot grid in the true frame and then applies the slice's rigid
offset to the centres. The grid therefore rotates with the slice, so the directions to a
point's nearest neighbours reveal that rotation modulo the lattice symmetry (90 degrees for
square bins, 60 for hexagonal spots). Any point-set registration locks onto it, and because
every slice carries the same underlying grid, the lattices also snap at whole-bin offsets.
Alignment ability cannot be measured on such a task.

Produces a four-panel figure and a CSV of the per-slice numbers behind it.

    scripts/plot_albis_lattice_leak.py --data-root ~/p2b-data/paper2bench-data/albis_sim

Each dataset directory needs slices/sNN.h5ad holding obsm/spatial (what a solver sees) and
obs/x_GT, obs/y_GT (the true frame), which is what the comparison needs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1-5 of the reference palette, in fixed order, validated for CVD.
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#1a1a19", "#52514e", "#e4e3e0", "#fcfcfb"

DATASETS = [
    ("strong_mix_cell", "sa-301 cells (strong mix)", 90.0),
    ("xenium_cell",     "sa-302 cells (Xenium)",     90.0),
    ("visium_spot",     "sa-303 spots (Visium)",     60.0),
    ("visiumhd_bin16",  "sa-304 bins (16 um)",       90.0),
    ("visiumhd_bin8",   "sa-305 bins (8 um)",        90.0),
]


def read_slice(path: Path, n: int = 40000, seed: int = 0):
    with h5py.File(path) as f:
        xy = np.asarray(f["obsm/spatial"][()], dtype=float)[:, :2]
        gt = np.c_[f["obs/x_GT"][()], f["obs/y_GT"][()]].astype(float)
    if len(xy) > n:
        idx = np.random.default_rng(seed).choice(len(xy), n, replace=False)
        xy, gt = xy[idx], gt[idx]
    return xy, gt


def neighbour_angles(xy: np.ndarray, period: float, k: int = 8) -> np.ndarray:
    """Directions to nearest neighbours at the modal spacing, folded into one lattice period."""
    tree = cKDTree(xy)
    d, j = tree.query(xy, k=k + 1)
    dd = d[:, 1:].ravel()
    vv = (xy[j[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
    modal = np.median(dd)
    keep = np.abs(dd - modal) < 0.15 * modal
    return np.degrees(np.arctan2(vv[keep, 1], vv[keep, 0])) % period


def lattice_peak(angles: np.ndarray, period: float, bins: int = 1800) -> tuple[float, float]:
    h, edges = np.histogram(angles, bins=bins, range=(0, period))
    return float((edges[h.argmax()] + edges[h.argmax() + 1]) / 2), float(h.max() / h.mean())


def lattice_residual(values: np.ndarray, period: float) -> tuple[float, float, float]:
    """How tightly the lattice fixes the rotation, allowing for the grid's own symmetry.

    A square grid's modal neighbour can be an axis or a diagonal, so slices may split into
    families a sub-period apart; a hexagonal grid does the same at 30 degrees. Folding by the
    best sub-period removes that discrete ambiguity, which a solver would resolve by trying
    both. The median absolute deviation is reported so one odd slice cannot hide a leak, with
    the fraction of slices inside 0.5 degrees.
    """
    best = (np.inf, period, 0.0)
    for div in (1, 2, 3, 4, 6):
        p_ = period / div
        v = values % p_
        centred = (v - np.median(v) + p_ / 2) % p_ - p_ / 2
        mad = float(np.median(np.abs(centred)))
        if mad < best[0]:
            best = (mad, p_, float(np.mean(np.abs(centred) < 0.5)))
    return best


def true_rotation(xy: np.ndarray, gt: np.ndarray) -> float:
    A = np.c_[xy, np.ones(len(xy))]
    coef, *_ = np.linalg.lstsq(A, gt, rcond=None)
    M = coef[:2].T
    return float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))


def collect(data_root: Path) -> pd.DataFrame:
    rows = []
    for dataset, label, period in DATASETS:
        slices = sorted((data_root / dataset / "slices").glob("s*.h5ad"))
        if not slices:
            print(f"  (skipping {dataset}: no slices found)")
            continue
        for path in slices:
            xy, gt = read_slice(path)
            peak, sharpness = lattice_peak(neighbour_angles(xy, period), period)
            rot = true_rotation(xy, gt)
            rows.append(dict(dataset=dataset, label=label, period=period, slice=path.stem,
                             lattice_angle=peak, sharpness=sharpness, true_rotation=rot,
                             sum_mod_period=(peak + rot) % period))
        print(f"  {label}: {len(slices)} slices")
    return pd.DataFrame(rows)


def panel_points(ax, data_root: Path, dataset: str, title: str, window: float, hue: str):
    """A patch of two slices with their neighbour edges drawn, so the grid and its angle show."""
    from matplotlib.lines import Line2D
    handles = []
    for i, (sl, colour) in enumerate([("s00", hue), ("s03", INK)]):
        path = data_root / dataset / "slices" / f"{sl}.h5ad"
        if not path.exists():
            continue
        with h5py.File(path) as f:
            xy = np.asarray(f["obsm/spatial"][()], dtype=float)[:, :2]
        c = xy.mean(axis=0)
        m = (np.abs(xy[:, 0] - c[0]) < window) & (np.abs(xy[:, 1] - c[1]) < window)
        pts = xy[m] - c
        tree = cKDTree(pts)
        d, j = tree.query(pts, k=5)
        modal = np.median(d[:, 1:])
        for a in range(len(pts)):                      # edges at the lattice pitch only
            for b in j[a, 1:][np.abs(d[a, 1:] - modal) < 0.15 * modal]:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], lw=0.7, color=colour, alpha=0.5, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color=colour, linewidths=0, zorder=2)
        handles.append(Line2D([], [], color=colour, lw=2, label=f"slice {sl}"))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("x offset (um)", fontsize=8, color=MUTED)
    ax.set_aspect("equal")
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=MUTED,
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "p2b-data/paper2bench-data/albis_sim")
    ap.add_argument("--out", type=Path, default=Path("results/albis_lattice_leak.png"))
    args = ap.parse_args()

    print("reading slices ...")
    df = collect(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    # A: the point pattern itself
    panel_points(axes[0, 0], args.data_root, "visiumhd_bin16",
                 "A. Bin centres sit on a grid that turns with the slice (sa-304, 16 um)", 60, HUES[3])
    axes[0, 0].set_ylabel("y offset (um)", fontsize=8, color=MUTED)

    # B: neighbour-direction histograms, one line per slice
    ax = axes[0, 1]
    for i, sl in enumerate(["s00", "s02", "s04", "s06"]):
        path = args.data_root / "visiumhd_bin16" / "slices" / f"{sl}.h5ad"
        if not path.exists():
            continue
        xy, _ = read_slice(path, n=20000, seed=i)
        ang = neighbour_angles(xy, 90.0)
        h, edges = np.histogram(ang, bins=360, range=(0, 90), density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.plot(centres, h, lw=2, color=HUES[i % len(HUES)], label=f"slice {sl}")
        ax.annotate(sl, (centres[h.argmax()], h.max()), textcoords="offset points",
                    xytext=(4, 2), fontsize=8, color=MUTED)
    ax.set_title("B. Every slice's neighbour directions form one sharp peak,\n    at a different angle (sa-304)",
                 fontsize=10, color=INK, loc="left")
    ax.set_xlabel("direction to nearest neighbours, folded mod 90 deg", fontsize=8, color=MUTED)
    ax.set_ylabel("density", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)

    # C: the leak, stated as an invariant
    ax = axes[1, 0]
    for i, (dataset, label, period) in enumerate(DATASETS):
        d = df[df.dataset == dataset]
        if d.empty:
            continue
        _, sub, _ = lattice_residual(d.sum_mod_period.to_numpy(), period)
        y = d.sum_mod_period.to_numpy() % sub          # folded by the grid's own symmetry
        ax.plot(range(len(y)), y, "o-", ms=6, lw=1.8, color=HUES[i],
                label=f"{label}  (folded mod {sub:.0f} deg)")
    ax.set_title("C. Lattice angle + true rotation is the same on every slice for bins and\n"
                 "    spots: the grid fixes the rotation. Cells scatter.", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("slice index", fontsize=8, color=MUTED)
    ax.set_ylabel("(lattice angle + true rotation), folded, deg", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=MUTED, loc="upper left", ncol=1)
    ax.set_ylim(-2, 40)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    # D: how much rotation is given away, in degrees
    ax = axes[1, 1]
    labels, spreads, colours, fracs = [], [], [], []
    for i, (dataset, label, period) in enumerate(DATASETS):
        d = df[df.dataset == dataset]
        if d.empty:
            continue
        mad, _, frac = lattice_residual(d.sum_mod_period.to_numpy(), period)
        labels.append(label); spreads.append(max(mad, 1e-3)); colours.append(HUES[i]); fracs.append(frac)
    ypos = np.arange(len(labels))
    ax.barh(ypos, spreads, color=colours, height=0.6)
    for y, s, fr in zip(ypos, spreads, fracs):
        ax.annotate(f"{s:.3f} deg   {fr:.0%} of slices exact", (s, y), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=8, color=INK)
    ax.set_yticks(ypos, labels, fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(right=200)
    ax.set_xlabel("residual after using the lattice to predict rotation (deg, log scale)", fontsize=8, color=MUTED)
    ax.set_title("D. On bins and spots the prediction is exact to ~0.01 deg;\n    on cells it is no better than guessing",
                 fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    fig.suptitle("Albis simulated slices: the bin and spot lattice reveals each slice's rotation",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.005, "Grid built in the true frame, then rotated with the slice. Fix: bin the molecules "
             "in each slice's own frame and invert the transform to recover true centres.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)
    print(f"\nwrote {args.out} and {args.out.with_suffix('.csv')}")
    for _, label, period in DATASETS:
        d = df[df.label == label]
        if d.empty:
            continue
        mad, sub, frac = lattice_residual(d.sum_mod_period.to_numpy(), period)
        print(f"  {label:28s} slices {len(d):3d}  sharpness {d.sharpness.mean():7.1f}  "
              f"residual {mad:7.3f} deg (folded by {sub:.1f} deg, {frac:.0%} of slices exact)")


if __name__ == "__main__":
    main()
