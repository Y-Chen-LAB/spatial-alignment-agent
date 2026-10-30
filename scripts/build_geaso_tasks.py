"""Build the sa-401 to sa-410 alignment tasks from the GEASO data release.

Source: Zenodo 18811760, the data for Wang, Liu & Ma, "Network model for alignment, stitching and
slice-to-volume 3D reconstruction of large-scale spatially resolved slices", Nat Commun 2026
(github.com/xkmaxidian/GEASO). Download and unzip it, then:

    .method-venv-spacel/bin/python scripts/build_geaso_tasks.py \
        --source /fastscratch/myscratch/ychen5/geaso/zenodo \
        --out ~/p2b-data/paper2bench-data/geaso_tasks

Tasks (all coordinates in µm):

  sa-401  STARmap PLUS atlas section sagittal3 (Shi et al. 2023; 207,591 cells, 1,022 genes),
          split by depth into 4 slices as GEASO does (split_slice, axis=2). s00 stays; s01-s03 are
          each rotated and shifted. Truth: every cell's original position. MAE, region, gene.
  sa-402  GEASO's stitching tutorial exactly: depth slices 1 and 2, cropped to 30% overlap
          (crop_slices, axis=1), and the second rotated 90 degrees about its centre. MAE, region, gene.
  sa-403  Overlap sweep: depth slice 0 whole as the reference; depth slice 1 cropped to 10-100%
          of its extent and rotated, one moving slice per level, scored against the reference
          (pair_mode star, like sa-041). MAE, region, gene.
  sa-404  STARmap vs RIBOmap on adjacent sections (Fig5_STAR_RIBO; 12 cell types). The RIBOmap
          section is moved by a known rigid transform; truth is its released position, i.e. the
          authors' registration. MAE (recovery), cell-type label metrics, gene.
  sa-405  STARmap PLUS, Alzheimer's-model study, two 8-month control mice (cr1, cr2; 4 regions).
          cr2 is moved by a known rigid transform; truth is its released (pre-registered)
          position. MAE (recovery), region, gene. X is log-normalised as released; no raw counts.
  sa-406  GEASO Fig. 6C-F: RNA of the P22 H3K27ac section vs H3K4me3 peaks of the neighbouring section
          (Zhang et al. 2023 spatial CUT&Tag-RNA-seq), as released. No coordinate truth and no shared
          features, so it is scored on label consistency only (GEASO's CLC). The labels are shared regions
          from joint clustering of both sections' RNA, because the released per-file clusters do not correspond.
  sa-407  GEASO Fig. 2A-B: STARmap PLUS cr1 fixed, cr2 rotated 30, 60, 90, 120, 135, 150, 180 degrees (star).
  sa-408  GEASO Fig. 2C-D: sagittal3 depth slices 1 and 2 stitched at 20% / 50% / 80% overlap
  sa-410  (one task each), each with the source rotated 30, 60 and 90 degrees (star).
          Grids are the paper's own, from its source data (41467_2026_71042_MOESM4_ESM.xlsx).

Unit assumptions (recorded in each manifest): STARmap PLUS pixels are 0.0946 µm (consistent with
sagittal3's depth range of about 27 z-steps x 0.35 µm, i.e. a ~10 µm section). The STARmap/RIBOmap
coordinates are taken to be µm already. The P22 barcode grid pitch is 20 µm.
Rows are shuffled and observation names anonymised, so row order carries no spatial information.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from materialize_albis_tasks import _format_toml_value  # noqa: E402

STARMAP_UM_PER_PX = 0.0946
P22_UM_PER_PIXEL = 20.0
SEED = 20260924


def rotate(xy: np.ndarray, theta_deg: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Rotate about the centroid, then shift."""
    c = xy.mean(0)
    t = np.deg2rad(theta_deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (xy - c) @ r.T + c + np.asarray(shift)


def counts_matrix(a: ad.AnnData, layer: str | None = None):
    x = a.layers[layer] if layer else a.X
    return sparse.csr_matrix(x, dtype=np.float32)


def select_marker_genes(parts: list, genes, labels: list[np.ndarray] | None, n: int = 100) -> list[str]:
    """Genes whose mean log-normalised expression differs most across labels; top variance if unlabelled.
    `parts` are expression matrices (cells x genes) sharing the gene order `genes`."""
    X = sparse.vstack([sparse.csr_matrix(p) for p in parts]).tocsr()
    tot = np.asarray(X.sum(1)).ravel()
    X = sparse.diags(np.divide(1e4, tot, out=np.zeros_like(tot), where=tot > 0)) @ X
    X.data = np.log1p(X.data)
    genes = np.asarray(genes)
    if labels is not None:
        lab = np.concatenate(labels)
        means = np.vstack([np.asarray(X[lab == g].mean(0)).ravel() for g in np.unique(lab)])
        spread = means.var(0)
    else:
        mean = np.asarray(X.mean(0)).ravel()
        spread = np.asarray(X.multiply(X).mean(0)).ravel() - mean ** 2
    return [str(genes[i]) for i in np.argsort(-spread)[:n]]


def write_slice(path: Path, name: str, X, genes, xy_input, xy_truth, region, rng, extra_obs=None) -> int:
    perm = rng.permutation(X.shape[0])
    obs = {k: pd.Categorical(np.asarray(v, dtype=str)[perm]) for k, v in (extra_obs or {}).items()}
    if region is not None:
        obs["Region"] = pd.Categorical(np.asarray(region, dtype=str)[perm])
    if xy_truth is not None:
        obs["x_GT"] = xy_truth[perm, 0]
        obs["y_GT"] = xy_truth[perm, 1]
    a = ad.AnnData(X=X[perm], obs=pd.DataFrame(obs, index=[f"{name}_{i:07d}" for i in range(X.shape[0])]),
                   var=pd.DataFrame(index=pd.Index(genes, dtype=str)))
    a.obsm["spatial"] = np.asarray(xy_input[perm], dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)
    return int(X.shape[0])


def emit_task(task_id, title, platform, difficulty, profile, label_key, slices, markers, meta, instruction,
              data_root: Path, pair_mode="consecutive", paper_label_key=None, scale_as_given=True):
    task_dir = REPO / "benchmarks/spatial_alignment/tasks" / task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    (task_dir / "data").mkdir(parents=True)
    marker_file = task_dir / "data/marker_genes.csv"
    pd.DataFrame({"gene": markers}).to_csv(marker_file, index=False)
    manifest = {
        "schema_version": "paper2bench.data_manifest.v0.1",
        "task_id": task_id,
        "source": "GEASO data release, Zenodo 18811760 (Wang, Liu & Ma, Nat Commun 2026)",
        "build_script": "scripts/build_geaso_tasks.py",
        **meta,
        "assets": [{"asset_id": sid, "role": "slice", "path": str(p), "format": "h5ad", "visibility": "public",
                    "metadata": {"n_obs": n, "is_reference": i == 0}} for i, (sid, p, n) in enumerate(slices)],
    }
    (task_dir / "data/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payload = {
        "id": task_id, "name": title, "domain": "spatial_genomics", "modality": "spatial_transcriptomics",
        "task_type": meta.get("task_type", "serial_alignment"), "difficulty": difficulty,
        "status": "executable_local", "input_contract": "h5ad_spatial_v1",
        "output_contract": "aligned_coordinate_csv_v1", "metric_profile": profile, "label_key": label_key,
        "marker_gene_file": str(marker_file), "max_marker_genes": len(markers),
        "slices": [f"{sid}={p}" for sid, p, _ in slices],
        "paper_source": meta["paper_source"],
        "paper_metrics": "geaso_v1",   # the scorer also reports GEASO's PCC, CLC, AAS and MAE
    }
    if pair_mode != "consecutive":
        payload["pair_mode"] = pair_mode
    if paper_label_key:
        payload["paper_label_key"] = paper_label_key   # labels the GEASO metrics (CLC) use, if not label_key
    (task_dir / "task.toml").write_text("\n".join(f"{k} = {_format_toml_value(v)}" for k, v in payload.items()) + "\n")
    rows = "\n".join(f"- `{sid}`: {n:,} observations" for sid, _, n in slices)
    instruction_text = (
        f"# {title}\n\n{instruction.strip()}\n\nMeasured at {platform}. Coordinates are in microns"
        + (", and each section's physical scale is correct as given." if scale_as_given else
           "; sections may differ slightly in scale from one another.")
        + f"\n\n{rows}\n\nKeep `{slices[0][0]}` where it is: its submitted coordinates "
        f"should equal its input coordinates. Submit one coordinate CSV per section with columns `obs_id,x,y`, in the "
        f"row order of the source `.h5ad` file.\n")
    # the agent-prompt sanitizer drops any line that matches its private-word patterns (e.g. "rotation",
    # "translation"); a task description must never lose a line that way
    import re as _re
    from prepare_agent_task import PRIVATE_INSTRUCTION_PATTERNS
    dropped = [l for l in instruction_text.splitlines()
               if l.strip() and any(_re.search(q, l, _re.IGNORECASE) for q in PRIVATE_INSTRUCTION_PATTERNS)]
    if dropped:
        raise ValueError(f"{task_id}: the agent-prompt sanitizer would drop these instruction lines: {dropped}")
    (task_dir / "instruction.md").write_text(instruction_text)
    print(task_id, [(s, n) for s, _, n in slices], flush=True)


def build_sagittal3(src: Path, out: Path, rng):
    a = ad.read_h5ad(src / "Fig2_stitching.h5ad")
    X = counts_matrix(a, "counts")
    genes = np.asarray(a.var_names)
    raw = np.asarray(a.obsm["spatial_raw"], float)
    xy = raw[:, :2] * STARMAP_UM_PER_PX
    region = a.obs["Main_molecular_tissue_region"].astype(str).to_numpy()
    order = np.argsort(raw[:, 2], kind="stable")
    per = len(order) // 4
    depth = [order[i * per:(i + 1) * per] for i in range(4)]   # GEASO split_slice(split_num=4, axis=2)
    markers = select_marker_genes([X[d] for d in depth], genes, [region[d] for d in depth])
    genes_l = list(genes)
    meta_common = {"dataset": "STARmap PLUS mouse CNS atlas, section sagittal3 (Shi et al. 2023)",
                   "um_per_pixel": STARMAP_UM_PER_PX,
                   "paper_source": "Shi et al. 2023 STARmap PLUS mouse CNS atlas, section sagittal3, via the GEASO data release "
                                   "(Zenodo 18811760); split by depth as in GEASO's stitching tutorial."}

    # sa-401: 4-slice depth stack
    root = out / "sa-401"; slices = []
    angles = rng.uniform(-180, 180, 4); shifts = rng.uniform(-1500, 1500, (4, 2))
    for i, d in enumerate(depth):
        sid = f"s{i:02d}"
        inp = xy[d] if i == 0 else rotate(xy[d], angles[i], shifts[i])
        n = write_slice(root / f"{sid}.h5ad", sid, X[d], genes_l, inp, xy[d], region[d], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-401-starmapplus-sagittal3-depth-stack", "STARmap PLUS sagittal section, four-slice depth stack",
              "single-cell resolution (STARmap PLUS, 1,022 genes)", "standard", "sabench_mae_region_gene_v1", "Region",
              slices, markers,
              {**meta_common, "task_type": "serial_alignment", "rotation_degrees": angles[1:].round(2).tolist(),
               "shift_um": shifts[1:].round(1).tolist()},
              "Four thin sections from one block of mouse brain tissue, in order, each measured at single-cell "
              "resolution. Each section was placed on the slide independently, so they do not share a coordinate "
              "system. Bring every section into the coordinate system of `s00`.", out)

    # sa-402: GEASO's stitching tutorial exactly: depth slices 1 and 2, cropped to 30% overlap along y
    # (crop_slices, axis=1), and the second rotated by 90 degrees about its centre, no shift
    s1, s2 = depth[1], depth[2]
    ov = 0.30
    y1 = xy[s1, 1]; y_min, y_max = y1.min(), y1.max()
    low = (1 - ov) * (y_max - y_min) / (2 - ov) + y_min
    high = y_max - (1 - ov) * (y_max - y_min) / (2 - ov)
    a_idx, b_idx = s1[xy[s1, 1] > low], s2[xy[s2, 1] < high]
    theta = 90.0; shift = np.zeros(2)
    root = out / "sa-402"; slices = []
    for sid, idx, inp in [("s00", a_idx, xy[a_idx]), ("s01", b_idx, rotate(xy[b_idx], theta, shift))]:
        n = write_slice(root / f"{sid}.h5ad", sid, X[idx], genes_l, inp, xy[idx], region[idx], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-402-starmapplus-sagittal3-overlap-pair", "STARmap PLUS sagittal sections with 30% overlap",
              "single-cell resolution (STARmap PLUS, 1,022 genes)", "standard", "sabench_mae_region_gene_v1", "Region",
              slices, markers, {**meta_common, "task_type": "partial_overlap_alignment", "overlap_fraction": ov,
                                "rotation_degrees": round(theta, 2), "shift_um": shift.round(1).tolist()},
              "Two neighbouring thin sections from one block of mouse brain tissue that only partly overlap: each "
              "covers a different part of the tissue, and they share a band of roughly 30%. The second section was "
              "placed on the slide independently. Bring `s01` into the coordinate system of `s00`.", out)

    # sa-403: overlap sweep, reference = depth slice 0 whole; moving = depth slice 1 cropped from the top
    s1, s2 = depth[0], depth[1]
    root = out / "sa-403"; slices = []
    n = write_slice(root / "reference.h5ad", "reference", X[s1], genes_l, xy[s1], xy[s1], region[s1], rng)
    slices.append(("reference", root / "reference.h5ad", n))
    y2 = xy[s2, 1]; lo, hi = y2.min(), y2.max()
    levels = list(range(10, 101, 10)); angles = rng.uniform(-180, 180, len(levels))
    for k, pct in enumerate(levels):
        keep = s2[y2 <= lo + (hi - lo) * pct / 100]
        sid = f"moving_overlap{pct:03d}"
        n = write_slice(root / f"{sid}.h5ad", sid, X[keep], genes_l, rotate(xy[keep], angles[k]), xy[keep], region[keep], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-403-starmapplus-sagittal3-overlap-sweep", "STARmap PLUS overlap sweep (10 to 100 percent)",
              "single-cell resolution (STARmap PLUS, 1,022 genes)", "medium", "sabench_mae_region_gene_v1", "Region",
              slices, markers, {**meta_common, "task_type": "robustness_sweep", "overlap_percent": levels,
                                "rotation_degrees": angles.round(2).tolist()},
              "A reference section and ten versions of the neighbouring section from the same block of mouse brain "
              "tissue. Each version keeps a different fraction of the neighbouring section (10% to 100% of its "
              "height) and was placed on the slide independently. Align every `moving_*` section to `reference`, "
              "independently of the others.", out, pair_mode="star")


def build_star_ribo(src: Path, out: Path, rng):
    a = ad.read_h5ad(src / "Fig5_STAR_RIBO.h5ad")
    X = counts_matrix(a); genes = list(a.var_names)
    ct = a.obs["cell_type"].astype(str).to_numpy()
    xy = a.obs[["x", "y"]].to_numpy(float)
    st, rb = np.where(a.obs["sample"] == "STAR_rep1")[0], np.where(a.obs["sample"] == "RIBO_rep1")[0]
    theta = float(rng.uniform(-180, 180)); shift = rng.uniform(-1500, 1500, 2)
    markers = select_marker_genes([X[st], X[rb]], genes, [ct[st], ct[rb]])
    root = out / "sa-404"; slices = []
    for sid, idx, inp in [("starmap", st, xy[st]), ("ribomap", rb, rotate(xy[rb], theta, shift))]:
        n = write_slice(root / f"{sid}.h5ad", sid, X[idx], genes, inp, xy[idx], ct[idx], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-404-starmap-ribomap-cross-modality-pair", "STARmap (RNA) to RIBOmap (ribosome-bound RNA) cross-modality alignment",
              "single-cell resolution (STARmap and RIBOmap, 5,413 genes)", "medium", "sabench_mae_region_gene_v1", "Region",
              slices, markers,
              {"dataset": "STARmap and RIBOmap on adjacent mouse brain sections (GEASO Fig. 5)", "task_type": "cross_modality_alignment",
               "truth": "released coordinates of the RIBOmap section (the authors' registration), recovered after a known rigid move",
               "rotation_degrees": round(theta, 2), "shift_um": shift.round(1).tolist(), "units": "released x, y taken as µm",
               "paper_source": "STARmap/RIBOmap adjacent mouse brain sections, via the GEASO data release (Zenodo 18811760, Fig. 5)."},
              "Two adjacent sections of mouse brain. `starmap` measures RNA; `ribomap` measures ribosome-bound RNA "
              "(protein synthesis) for the same genes. The two sections were placed on their slides independently. Bring "
              "`ribomap` into the coordinate system of `starmap`.", out)


def build_ad_pair(src: Path, out: Path, rng):
    parts = [ad.read_h5ad(src / f"Fig2_alignment/8month_cr{k}_example_align.h5ad") for k in (1, 2)]
    genes = list(parts[0].var_names)
    theta = float(rng.uniform(-180, 180)); shift = rng.uniform(-500, 500, 2)
    root = out / "sa-405"; slices = []
    regions = [p.obs["region"].astype(str).to_numpy() for p in parts]
    markers = select_marker_genes([counts_matrix(p) for p in parts], genes, regions)
    for k, (sid, p) in enumerate([("mouse_cr1", parts[0]), ("mouse_cr2", parts[1])]):
        xy = np.asarray(p.obsm["spatial"], float) * STARMAP_UM_PER_PX
        inp = xy if k == 0 else rotate(xy, theta, shift)
        n = write_slice(root / f"{sid}.h5ad", sid, counts_matrix(p), genes, inp, xy, regions[k], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-405-starmapplus-ad-cross-animal-pair", "STARmap PLUS cross-animal alignment (two 8-month control mice)",
              "single-cell resolution (STARmap PLUS, 2,766 genes; expression is log-normalised as released)", "medium",
              "sabench_mae_region_gene_v1", "Region", slices, markers,
              {"dataset": "STARmap PLUS, Alzheimer's-model study, 8-month control replicates cr1 and cr2 (as used by SANTO and GEASO)",
               "task_type": "cross_animal_alignment", "um_per_pixel": STARMAP_UM_PER_PX,
               "truth": "released (pre-registered) coordinates of cr2, recovered after a known rigid move",
               "rotation_degrees": round(theta, 2), "shift_um": shift.round(1).tolist(),
               "paper_source": "STARmap PLUS Alzheimer's-model mouse brain, 8-month controls, via the GEASO data release (Zenodo 18811760, Fig. 2)."},
              "Matching sections of the cortex and hippocampus from two different mice. The two sections were placed "
              "on their slides independently. Bring `mouse_cr2` into the coordinate system of `mouse_cr1`.", out)


def joint_region_labels(parts: list, xys: list, n_regions: int = 10, n_pcs: int = 30, smooth_k: int = 6, seed: int = 0):
    """Shared region labels for sections measured on the same genes: log1p CPM, per-section
    per-gene z-score (removes section offsets), joint PCA, within-section spatial kNN smoothing,
    joint KMeans. Scorer-side only; used where released cluster labels do not correspond."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    zs = []
    for X in parts:
        X = sparse.csr_matrix(X, dtype=np.float64)
        tot = np.asarray(X.sum(1)).ravel()
        X = sparse.diags(np.divide(1e4, tot, out=np.zeros_like(tot), where=tot > 0)) @ X
        X.data = np.log1p(X.data)
        D = X.toarray()
        zs.append((D - D.mean(0)) / (D.std(0) + 1e-6))
    keep = np.all([z.std(0) > 0 for z in zs], axis=0)
    pca = PCA(n_components=n_pcs, random_state=seed).fit(np.vstack([z[:, keep] for z in zs]))
    embs = []
    for z, xy in zip(zs, xys):
        e = pca.transform(z[:, keep])
        _, nn = NearestNeighbors(n_neighbors=smooth_k).fit(xy).kneighbors(xy)
        embs.append(e[nn].mean(1))
    km = KMeans(n_clusters=n_regions, n_init=10, random_state=seed).fit(np.vstack(embs))
    cuts = np.cumsum([0] + [len(e) for e in embs])
    return [np.array([f"P{k}" for k in km.labels_[cuts[i]:cuts[i + 1]]]) for i in range(len(embs))]


def build_p22(src: Path, out: Path, rng):
    """GEASO Fig. 6C-F: the RNA slice (the H3K27ac section's RNA, 9,370 spots x 23,415 genes) and the
    neighbouring section's H3K4me3 histone-mark slice (9,548 spots), aligned as released."""
    d_rna, d_epi = "Dataset9_Mouse_Brain_H3K27ac", "Dataset8_Mouse_Brain_H3K4me3"
    rna = ad.read_h5ad(src / f"P22Mouse/{d_rna}/adata_RNA.h5ad")
    epi = ad.read_h5ad(src / f"P22Mouse/{d_epi}/adata_peaks_normalized.h5ad")
    epi_rna = ad.read_h5ad(src / f"P22Mouse/{d_epi}/adata_RNA.h5ad")   # same spots as `epi`, for labels only
    assert list(epi.obs_names) == list(epi_rna.obs_names)
    xy_rna = np.asarray(rna.obsm["spatial"], float) * P22_UM_PER_PIXEL
    xy_epi = np.asarray(epi.obsm["spatial"], float) * P22_UM_PER_PIXEL
    common = sorted(set(map(str, rna.var_names)) & set(map(str, epi_rna.var_names)))
    hv = select_marker_genes([counts_matrix(rna[:, common]), counts_matrix(epi_rna[:, common])], common, None, n=2000)
    lab_rna, lab_epi = joint_region_labels([counts_matrix(rna[:, hv]), counts_matrix(epi_rna[:, hv])], [xy_rna, xy_epi])
    root = out / "sa-406"; slices = []
    n = write_slice(root / "rna.h5ad", "rna", counts_matrix(rna), list(map(str, rna.var_names)), xy_rna, None, lab_rna, rng)
    slices.append(("rna", root / "rna.h5ad", n))
    n = write_slice(root / "h3k4me3.h5ad", "h3k4me3", sparse.csr_matrix(epi.X, dtype=np.float32),
                    list(map(str, epi.var_names)), xy_epi, None, lab_epi, rng)
    slices.append(("h3k4me3", root / "h3k4me3.h5ad", n))
    old = REPO / "benchmarks/spatial_alignment/tasks/sa-406-p22-mouse-spatial-multiomics-sections"
    if old.exists():
        shutil.rmtree(old)
    emit_task("sa-406-p22-mouse-rna-h3k4me3-cross-modality-pair",
              "P22 mouse brain RNA to H3K4me3 cross-modality alignment",
              "20 µm barcoded pixels (spatial CUT&Tag-RNA-seq)", "hard", "sabench_label_only_v1", "Region", slices, [],
              {"dataset": "P22 mouse brain spatial epigenome-transcriptome co-profiling (Zhang et al. 2023): RNA of the "
                          "H3K27ac section vs H3K4me3 peaks of the neighbouring section, as in GEASO Fig. 6C-F",
               "task_type": "cross_modality_alignment", "um_per_pixel": P22_UM_PER_PIXEL, "truth": "none (coordinates)",
               "labels": "scorer-only shared regions P0-P9 from joint clustering of both sections' RNA (the H3K4me3 "
                         "section was co-profiled for RNA); the released per-file clusters do not correspond",
               "paper_source": "Zhang et al. 2023 P22 mouse brain spatial CUT&Tag-RNA-seq (GSE165217), via the GEASO "
                               "data release (Zenodo 18811760); GEASO Fig. 6C-F and Supplementary Fig. 18."},
              "Two neighbouring sections of postnatal day 22 mouse brain measured on barcoded grids with different "
              "assays. `rna` holds RNA counts per spot (genes). `h3k4me3` holds normalised histone H3K4me3 signal per "
              "spot over genomic peaks (features are genomic intervals, not genes). The sections were placed "
              "independently. Bring `h3k4me3` into the coordinate system of `rna`.", out)


def build_rotation_sweep(src: Path, out: Path, rng):
    """GEASO Fig. 2a-b, exactly as its R2S10 notebook: the target is cr2, unmoved; the source is cr1,
    rotated by theta about its own centre and then placed on cr2's centre. Angles from the paper's
    source data. Truth for the source is cr1's released (pre-registered) position. The paper's PCC and
    CLC are per source cell against its nearest target cell, CLC on `top_level_cell_type`."""
    cr1, cr2 = [ad.read_h5ad(src / f"Fig2_alignment/8month_cr{k}_example_align.h5ad") for k in (1, 2)]
    genes = list(cr2.var_names)
    xy1 = np.asarray(cr1.obsm["spatial"], float) * STARMAP_UM_PER_PX
    xy2 = np.asarray(cr2.obsm["spatial"], float) * STARMAP_UM_PER_PX
    reg1, reg2 = (p.obs["region"].astype(str).to_numpy() for p in (cr1, cr2))
    ct1, ct2 = (p.obs["top_level_cell_type"].astype(str).to_numpy() for p in (cr1, cr2))
    markers = select_marker_genes([counts_matrix(cr2), counts_matrix(cr1)], genes, [reg2, reg1])
    root = out / "sa-407"
    if root.exists():
        shutil.rmtree(root)
    slices = []
    n = write_slice(root / "target.h5ad", "target", counts_matrix(cr2), genes, xy2, xy2, reg2, rng, {"cell_type": ct2})
    slices.append(("target", root / "target.h5ad", n))
    angles = [30, 60, 90, 120, 135, 150, 180]   # the paper's source data (Fig. 2a-b)
    for ang in angles:
        sid = f"source_rot{ang:03d}"
        inp = rotate(xy1, ang) - xy1.mean(0) + xy2.mean(0)   # R2S10: rotate about own centre, place on target centre
        n = write_slice(root / f"{sid}.h5ad", sid, counts_matrix(cr1), genes, inp, xy1, reg1, rng, {"cell_type": ct1})
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-407-starmapplus-ad-rotation-sweep", "STARmap PLUS cross-animal angle sweep (30 to 180 degrees)",
              "single-cell resolution (STARmap PLUS, 2,766 genes; expression is log-normalised as released)", "medium",
              "sabench_mae_region_gene_v1", "Region", slices, markers,
              {"dataset": "STARmap PLUS, Alzheimer's-model study (Zeng et al. 2023), 8-month controls: target cr2, source cr1",
               "task_type": "robustness_sweep", "rotation_degrees": angles, "um_per_pixel": STARMAP_UM_PER_PX,
               "protocol": "GEASO R2S10 notebook: source cr1 rotated about its centre and placed on the target's centre",
               "truth": "released (pre-registered) coordinates of cr1",
               "paper_source": "Zeng et al. 2023 STARmap PLUS (SCP1375), via the GEASO data release (Zenodo 18811760); GEASO Fig. 2a-b."},
              "A target section and seven rotated copies of a matching section from a second mouse (cortex and "
              "hippocampus). Each copy was rotated by a different, unknown angle. Align every `source_*` section to "
              "`target`, independently of the others.", out, pair_mode="star", paper_label_key="cell_type")


def build_stitching_grid(src: Path, out: Path, rng):
    """GEASO Fig. 2C-D: stitching two depth slices of sagittal3 at several overlap ratios and angles."""
    a = ad.read_h5ad(src / "Fig2_stitching.h5ad")
    X = counts_matrix(a, "counts"); genes = list(a.var_names)
    raw = np.asarray(a.obsm["spatial_raw"], float); xy = raw[:, :2] * STARMAP_UM_PER_PX
    region = a.obs["Main_molecular_tissue_region"].astype(str).to_numpy()
    order = np.argsort(raw[:, 2], kind="stable"); per = len(order) // 4
    s1, s2 = order[per:2 * per], order[2 * per:3 * per]   # GEASO's tutorial pair: depth slices 1 and 2
    markers = select_marker_genes([X[s1], X[s2]], genes, [region[s1], region[s2]])
    angles = [30, 60, 90]   # the paper's source data (Fig. 2c-d): overlaps {20,50,80}% x rotations {30,60,90}
    for number, ov in [(408, 0.20), (409, 0.50), (410, 0.80)]:
        y1 = xy[s1, 1]; y_min, y_max = y1.min(), y1.max()
        low = (1 - ov) * (y_max - y_min) / (2 - ov) + y_min
        high = y_max - (1 - ov) * (y_max - y_min) / (2 - ov)
        a_idx, b_idx = s1[xy[s1, 1] > low], s2[xy[s2, 1] < high]
        root = out / f"sa-{number}"; slices = []
        n = write_slice(root / "target.h5ad", "target", X[a_idx], genes, xy[a_idx], xy[a_idx], region[a_idx], rng)
        slices.append(("target", root / "target.h5ad", n))
        for ang in angles:
            sid = f"source_rot{ang:03d}"
            n = write_slice(root / f"{sid}.h5ad", sid, X[b_idx], genes, rotate(xy[b_idx], ang), xy[b_idx], region[b_idx], rng)
            slices.append((sid, root / f"{sid}.h5ad", n))
        pct = int(round(ov * 100))
        emit_task(f"sa-{number}-starmapplus-sagittal3-stitch-overlap{pct:02d}",
                  f"STARmap PLUS stitching at {pct}% overlap (30, 60 and 90 degrees)",
                  "single-cell resolution (STARmap PLUS, 1,022 genes)", "medium" if ov >= 0.5 else "hard",
                  "sabench_mae_region_gene_v1", "Region", slices, markers,
                  {"dataset": "STARmap PLUS mouse CNS atlas section sagittal3 (Shi et al. 2023), depth slices 1 and 2",
                   "task_type": "partial_overlap_alignment", "overlap_fraction": ov, "rotation_degrees": angles,
                   "um_per_pixel": STARMAP_UM_PER_PX,
                   "paper_mae_units": "GEASO reports MAE in the section's raw pixels; divide our µm by 0.0946 to compare",
                   "paper_source": "Shi et al. 2023 STARmap PLUS atlas (SCP1830), via the GEASO data release (Zenodo 18811760); GEASO Fig. 2C-D."},
                  f"A target section and three rotated copies of a neighbouring section from the same block of mouse "
                  f"brain tissue. The two sections only partly overlap (about {pct}% of the target's height), and each "
                  f"copy was rotated by a different, unknown angle. Align every `source_*` section to `target`, "
                  f"independently of the others.", out, pair_mode="star")


def build_merfish3d(src: Path, out: Path, rng):
    """GEASO Fig. 2e: slice-to-volume 3D reconstruction of the 129 MERFISH mouse hemibrain sections
    (Zhang et al. 2023, Zhuang ABCA-1) released with the paper. Follows the paper's own 3D pipeline
    (its OpenST notebook; no MERFISH notebook is released): sections in file-number order, the first is
    the reference, every other section rotated by a random angle about its own centre, and each section
    aligned to the previous one. Coordinates are the in-plane CCF coordinates `spatial_2D` (µm), which
    are also the truth. The paper's MAE is per consecutive pair (128 pairs)."""
    import re
    files = sorted((src / "Fig2_3d_MERFISH").glob("slice_*.h5ad"), key=lambda f: int(re.search(r"\d+", f.name).group()))
    parts = [ad.read_h5ad(f) for f in files]
    genes = list(parts[0].var_names)
    root = out / "sa-411"; slices = []
    angles = rng.uniform(0, 360, len(parts)); angles[0] = 0.0
    labels = [p.obs["cell_type"].astype(str).to_numpy() for p in parts]
    markers = select_marker_genes([counts_matrix(p) for p in parts[::8]], genes, labels[::8])
    for k, (f, p) in enumerate(zip(files, parts)):
        sid = f"s{k:03d}"
        truth = np.asarray(p.obsm["spatial_2D"], float)
        inp = truth if k == 0 else rotate(truth, angles[k])
        n = write_slice(root / f"{sid}.h5ad", sid, counts_matrix(p), genes, inp, truth, labels[k], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-411-merfish-hemibrain-129-section-3d-reconstruction",
              "MERFISH mouse hemibrain 3D reconstruction (129 sections)",
              "single-cell resolution (MERFISH, 1,122 genes)", "hard", "sabench_mae_region_gene_v1", "Region",
              slices, markers,
              {"dataset": "MERFISH whole mouse brain, Zhuang ABCA-1 animal 1 (Zhang et al. 2023), 129 sections as released "
                          "with GEASO (Fig2_3d_MERFISH, slice_9 to slice_137)",
               "task_type": "serial_3d_reconstruction", "n_sections": len(files),
               "section_files": [f.name for f in files], "rotation_degrees": angles.round(2).tolist(),
               "labels": "cell_type (as in the release)",
               "protocol": "first section is the reference; others rotated about their own centre (GEASO OpenST notebook); "
                           "each aligned to the previous one; paper MAE per consecutive pair",
               "discrepancy": "the paper text states 9.3 million cells; the released files hold about 1.6 million",
               "paper_source": "Zhang et al. 2023 MERFISH whole mouse brain via the GEASO data release (Zenodo 18811760); GEASO Fig. 2e."},
              "129 consecutive coronal sections through one mouse hemibrain, in order, each measured at single-cell "
              "resolution. Each section was placed on its slide independently, so they do not share a coordinate "
              "system. Reconstruct the block: bring every section into the coordinate system of `s000`.", out)


def _read_legacy_h5ad(path: Path, obs_cols: list[str]):
    """X (CSR), var names, obsm/spatial[:, :2] and obs columns from an older-format h5ad via h5py."""
    import h5py
    with h5py.File(path, "r") as f:
        g = f["X"]
        X = sparse.csr_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g.attrs["shape"]), dtype=np.float32)
        var = f["var"]; vkey = var.attrs.get("_index", "_index")
        genes = [x.decode() if isinstance(x, bytes) else str(x) for x in var[vkey][()]]
        xy = np.asarray(f["obsm"]["spatial"][()], float)[:, :2]
        obs = {}
        for c in obs_cols:
            d = f["obs"][c]
            if isinstance(d, h5py.Group):   # categorical
                cats = [x.decode() if isinstance(x, bytes) else str(x) for x in d["categories"][()]]
                obs[c] = np.asarray(cats, dtype=object)[d["codes"][()]]
            else:
                v = d[()]
                obs[c] = np.array([x.decode() if isinstance(x, bytes) else x for x in v]) if v.dtype.kind in "OS" else v
    return X, genes, xy, obs


def build_openst_paper(src: Path, out: Path, rng):
    """GEASO Fig. 2f, as its OpenST_3D_Rec notebook: the 19 Open-ST lymph node sections (same sections as
    sa-027), `obsm['spatial']` as the input, every section but the first rotated by a random angle about
    its own centre, each aligned to the previous one; CLC on `annotation`. The 3D-aligned coordinates
    (sa-027's x_GT/y_GT) are kept as truth for our MAE."""
    import tomllib
    spec = tomllib.loads(next((REPO / "benchmarks/spatial_alignment/tasks").glob("sa-027-*/task.toml")).read_text())
    entries = [e.split("=", 1) for e in spec["slices"]]
    root = out / "sa-412"; slices = []
    angles = rng.uniform(0, 360, len(entries)); angles[0] = 0.0
    parts = [_read_legacy_h5ad(REPO / p if not p.startswith("/") else Path(p), ["annotation", "x_GT", "y_GT"]) for _, p in entries]
    genes = parts[0][1]
    assert all(pt[1] == genes for pt in parts), "sections differ in genes"
    markers = select_marker_genes([pt[0] for pt in parts[::4]], genes, [pt[3]["annotation"] for pt in parts[::4]])
    for k, ((sid, _), (X, _, xy, obs)) in enumerate(zip(entries, parts)):
        truth = np.c_[obs["x_GT"], obs["y_GT"]].astype(float)
        inp = xy if k == 0 else rotate(xy, angles[k])
        # truth in the frame of section 1's input (the suite's MAE has no fitting step): section 1's
        # own rigid map from its 3D-aligned truth to its input, applied to every section's truth
        if k == 0:
            from numpy.linalg import svd
            ca, cb = truth.mean(0), xy.mean(0); u, _, vt = svd((truth - ca).T @ (xy - cb)); r0 = u @ vt
            if np.linalg.det(r0) < 0:
                u[:, -1] *= -1; r0 = u @ vt
            t0 = cb - ca @ r0
        n = write_slice(root / f"{sid}.h5ad", sid, X, genes, inp, truth @ r0 + t0, obs["annotation"], rng,
                        {"annotation": obs["annotation"]})
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-412-openst-mln-19-section-3d-reconstruction-paper",
              "Open-ST lymph node 3D reconstruction, GEASO protocol (19 sections)",
              "near single-cell resolution (Open-ST)", "hard", "sabench_mae_region_gene_v1", "Region", slices, markers,
              {"dataset": "Open-ST human metastatic lymph node, 19 sections (Schott et al. 2024; same sections as sa-027)",
               "task_type": "serial_3d_reconstruction", "rotation_degrees": angles.round(2).tolist(),
               "protocol": "GEASO OpenST_3D_Rec notebook: obsm['spatial'] input, sections after the first rotated about "
                           "their own centre, each aligned to the previous; paper metric CLC on `annotation` per pair",
               "truth": "3D-aligned coordinates (sa-027's x_GT/y_GT) mapped into section 1's input frame",
               "paper_source": "Schott et al. 2024 Open-ST (GSE251926); GEASO Fig. 2f."},
              "19 consecutive sections of a human metastatic lymph node, in order. Each section was placed on its "
              "slide independently, so they do not share a coordinate system. Reconstruct the block: bring every "
              "section into the coordinate system of the first section.", out, paper_label_key="annotation",
              scale_as_given=False)   # raw Open-ST sections differ from the aligned truth by up to 4% in scale


def build_star_ribo_released(src: Path, out: Path, rng):
    """GEASO Fig. 6a: STARmap (target) and RIBOmap (source) adjacent sections aligned as released, with no
    added transform. No coordinate truth; scored on label consistency, and the paper's cell-type
    concordance (GEASO 81.8%) is clc_k1."""
    a = ad.read_h5ad(src / "Fig5_STAR_RIBO.h5ad")
    X = counts_matrix(a); genes = list(a.var_names)
    ct = a.obs["cell_type"].astype(str).to_numpy()
    xy = a.obs[["x", "y"]].to_numpy(float)
    st, rb = np.where(a.obs["sample"] == "STAR_rep1")[0], np.where(a.obs["sample"] == "RIBO_rep1")[0]
    markers = select_marker_genes([X[st], X[rb]], genes, [ct[st], ct[rb]])
    root = out / "sa-413"; slices = []
    for sid, idx in [("starmap", st), ("ribomap", rb)]:
        n = write_slice(root / f"{sid}.h5ad", sid, X[idx], genes, xy[idx], None, ct[idx], rng)
        slices.append((sid, root / f"{sid}.h5ad", n))
    emit_task("sa-413-starmap-ribomap-as-released-pair", "STARmap to RIBOmap alignment, GEASO protocol (as released)",
              "single-cell resolution (STARmap and RIBOmap, 5,413 genes)", "medium", "sabench_label_only_v1", "Region",
              slices, markers,
              {"dataset": "STARmap and RIBOmap on adjacent mouse brain sections (Zeng et al. 2023), as released",
               "task_type": "cross_modality_alignment", "truth": "none (coordinates); labels are the 12 released cell types",
               "paper_result": "GEASO cell-type concordance 81.8% (paper text; Supplementary Fig. 17)",
               "paper_source": "Zeng et al. 2023 STARmap/RIBOmap (SCP1835), via the GEASO data release (Zenodo 18811760); GEASO Fig. 6a."},
              "Two adjacent sections of mouse brain. `starmap` measures RNA; `ribomap` measures ribosome-bound RNA "
              "(protein synthesis) for the same genes. Bring `ribomap` into the coordinate system of `starmap`.", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="Build just these (e.g. sa-411); the default builds sa-401 to sa-410")
    args = parser.parse_args()
    if args.only:
        builders = {"sa-407": build_rotation_sweep, "sa-411": build_merfish3d,
                    "sa-412": build_openst_paper, "sa-413": build_star_ribo_released}
        for t in args.only:
            builders[t](args.source, args.out, np.random.default_rng([SEED, int(t[3:])]))
        return
    rng = np.random.default_rng(SEED)
    build_sagittal3(args.source, args.out, rng)
    build_star_ribo(args.source, args.out, rng)
    build_ad_pair(args.source, args.out, rng)
    build_p22(args.source, args.out, rng)
    build_rotation_sweep(args.source, args.out, rng)
    build_stitching_grid(args.source, args.out, rng)


if __name__ == "__main__":
    main()
