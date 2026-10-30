# Data sources

Where every dataset behind tasks sa-001 to sa-417 came from, which script turns it into task
inputs, and where those inputs live. The Hicks-lab tasks (sa-5xx, sa-6xx) are not in this
release.

**Two kinds of data.**
- *Raw downloads* are the files as published (GEO tars, Dryad CSVs, 10x bundles). They were
  converted on the original workstation in September 2026, and most were never copied to the
  cluster. This release points to them by accession instead of redistributing them.
- *Task inputs* are the compiled per-slice `.h5ad` files the tasks read: about 34 GB, in the
  Zenodo record under `task_inputs/`, with `compiled_data/` and `paper2bench-data/` as on disk.
  Task files reference them as `/users/ychen5/p2b-data/...` or through the
  `benchmarks/spatial_alignment/compiled_data` symlink. To use another location, run
  `scripts/check_portability.py --data-root <dir> --apply` and re-point the symlink.

`SOURCES.tsv` in the Zenodo record has the same rows in machine-readable form, with md5 sums.

In the tables, `P/` is `task_inputs/paper2bench-data/` and `C/` is `task_inputs/compiled_data/`.
**TODO** marks a fact we have not recorded yet.

## Real data

| Dataset (SABench IDs) | Tasks | Raw source | Script | Task inputs |
|---|---|---|---|---|
| DLPFC 151673-76 (Data1-4) | 001-003, 040, 041 | SABench Zenodo 10.5281/zenodo.18605715, `SABench.zip`. Original: spatialLIBD (Maynard et al. 2021) | extracted by hand; `compile_perturbed_h5ad.py`, `build_tasks.py` (sweeps) | P/DLPFC, P/coarse_align, C/perturbed, C/sweeps |
| MHPR MERFISH (Data5-9) | 009, 042, 043 | Dryad 10.5061/dryad.8t8s248 (Moffitt et al. 2018) | `compile_mhpr_csv.py`, `attach_mhpr_bass_regions.py` | P/MHPR_paper, C/sweeps |
| SCC (Data10-15) | 011, 012 | GEO GSE144240 (Ji et al. 2020) | `compile_scc_geo.py` | P/SCC |
| Flysta3D E14-16h (Data16-31) | 019 | db.cngb.org/stomics/flysta3d | `compile_flysta3d_slices.py` | P/Flysta3D |
| MBSA / MBSP Visium (Data32-35) | 006, 007 | 10x Genomics datasets (URLs in each task's `paper_source`), Space Ranger 1.0.0 | `preprocess_10x_visium_to_h5ad.py`, `compile_perturbed_h5ad.py` | C/perturbed_sr100 |
| BaristaSeq / STARmap (Data36-41) | 017, 018 | SDMBench (Yuan et al. 2024, 10.1038/s41592-024-02215-8), files `22550074.zip`, `22565200.zip`. **TODO:** confirm these are figshare article IDs | `materialize_sdmbench_imaging_tasks.py` | C/sdmbench |
| Vizgen MERFISH receptor map (Data42-44) | 031 | Vizgen MERFISH Mouse Brain Receptor Map. **TODO:** exact file URL | `compile_paper_acquisitions_20260914.py` | P/MBRM |
| MOp MERFISH (Data45-77) | 020, 021, 101-112 | Zhang et al. 2021 (10.1038/s41586-021-03705-x), `processed_data/counts.h5ad`. **TODO:** URL (likely Brain Image Library) | `compile_mop_bil_processed.py`, `materialize_mop_tasks.py` | C/mop_paper |
| MOB Stereo-seq (Data78) | 030 | `Data78_MOB_SEDR/Stero-seq.tar.gz`. **TODO:** URL (likely the SEDR release) | `compile_mob_stereoseq_data78.py` | C/mob_stereoseq |
| MOB Slide-seqV2 (Data79) | 030 | Broad Single Cell Portal SCP815 | `compile_paper_acquisitions_20260914.py` | P/MOB_slideseqv2 |
| Open-ST lymph node (Data80-98) | 022, 025-027, 412 | GEO GSE251926 (Schott et al. 2024) | `compile_openst_mln_sections.py`, `materialize_openst_mln_tasks.py`; 412 by `build_geaso_tasks.py` | C/openst_mln, P/geaso_tasks/sa-412 |
| Whole mouse brain, Zhuang ABCA-1 (Data99-227) | 023, 028, 029, 414-417 | CELLxGENE collection 0cca8620-8dee-45d0-aef5-23f032a5cf09 (animal 1); alternative route Allen ABC atlas (`fetch_wmb_abc_atlas.py`) | `compile_wmb_cellxgene_paper.py`, `materialize_wmb_partial_tasks.py`, `build_wmb_realistic_tasks.py` | P/wmb_cellxgene_paper, C/wmb_partial, P/wmb_realistic |
| Breast cancer Xenium Rep1/2 (Data229-230) | 004, 005, 014, 032 | 10x Xenium FFPE Human Breast (Janesick et al. 2023, 10.1038/s41467-023-43458-x), via the SABench tutorial | extracted by hand | P/BCA_raw, P/BCA_downsampled, P/BCA_STAligner_DR |
| Breast cancer Visium (Data228) | 032 | 10x Genomics. **TODO:** dataset URL | `preprocess_10x_visium_to_h5ad.py` | P/BCA_visium |
| Adult mouse brain, Visium (Data231) | 010, 013, 015, 016, 201, 210-217 | 10x `V1_Adult_Mouse_Brain` | `preprocess_10x_visium_to_h5ad.py` | P/AMBC_Visium |
| Adult mouse brain, Visium HD (Data232) | 015, 202, 210, 218-224 | 10x Visium HD Mouse Brain (fresh frozen) | `preprocess_10x_visiumhd_to_h5ad.py` | C/visiumhd |
| Adult mouse brain, Xenium / Xenium Prime 5k (Data234-235) | 008, 010, 013, 203, 204, 211, 212, 218, 219, 225-235 | 10x (fresh-frozen mouse brain replicates; URL in sa-008's `paper_source`) | `compile_xenium_10x_bundle.py` | C/xenium |
| Adult mouse brain, STARmap PLUS well 11 (Data237) | 016, 206, 214, 221, 227, 232, 236, 240-242 | Zenodo 10.5281/zenodo.8327576 | `compile_starmap_plus_well11.py` | C/starmap |
| Adult mouse brain, ST / Slide-seq / MERFISH | 201-209, 213, 215, 220, 222, 226, 228, 231, 233, 236-240, 243, 244 | molecularatlas.org (ST 21A), braincelldata.org (Slide-seq puck 46), Vizgen portal (MERFISH S2R3). **TODO:** file URLs | `compile_paper_acquisitions_20260914.py` | P/AMBC_ST, P/AMBC_Slideseq, P/AMBC_MERFISH |
| Adult mouse brain, Stereo-seq T312 (Data239) | 208, 216, 223, 229, 234, 238, 241, 243, 245 | digital-brain.cn, mouse 1 section T312 | `compile_stereoseq_t312.py` | P/AMBC_Stereoseq |
| Adult mouse brain, CosMx | 209, 217, 224, 230, 235, 239, 242, 244, 245 | Bruker/NanoString CosMx Mouse Brain FFPE; we used the stand-in Run5642_S3_Quarter because the paper's sample is no longer offered. **TODO:** URL and compile script | none found | P/AMBC_CosMx |
| GEASO release | 401-411, 413 | Zenodo 10.5281/zenodo.18811760 (GEASO, Wang, Liu & Ma 2026). It bundles STARmap PLUS SCP1830, AD SCP1375, STAR/RIBOmap SCP1835, P22 CUT&Tag GSE165217 and a MERFISH hemibrain | `build_geaso_tasks.py` | P/geaso_tasks |

## Simulated data

| Dataset | Tasks | Generator | Script | Task inputs |
|---|---|---|---|---|
| Albis simulations | 301-312, plus smoke copies | Albis (github.com/JanMatthias1/Albis, MIT), `scripts/alignment_benchmark.py` | `materialize_albis_tasks.py` | P/albis_sim, P/albis_sim_smoke |

## Known gaps

1. **No download URL recorded:** MOp `counts.h5ad`, MOB Stereo-seq Data78, breast cancer
   Visium Data228, the SDMBench zips and the CosMx stand-in.
2. **Portal name only, no file URL:** the Vizgen receptor map and the adult-mouse-brain ST,
   Slide-seq and MERFISH files.
3. **Not scripted:** the DLPFC and breast-cancer Xenium inputs were taken from `SABench.zip`
   by hand; the CosMx stand-in has no compile script.
4. **Raw file not kept:** Open-ST GSE251926 was read from `/tmp`. It is re-downloadable from GEO.
5. **Wrong provenance:** the sa-414 manifest names GEASO and `build_geaso_tasks.py` as its
   source. It is built from Zhuang ABCA-1 by `build_wmb_realistic_tasks.py`.
6. **Licenses:** no dataset license has been recorded. Fill in `SOURCES.tsv` before release.
