# Scripts

One line per script, grouped by stage. Subdirectories: `slurm/` (cluster job scripts), `generalization/` (strategy synthesis on held-out folds), `sabench_comparison/` (re-running SABench's own code), `agent_strategies/` (trace extraction and score exports), `metric_parity/` (checks against the papers' metric code).

Run from the repo root with `PYTHONPATH=src`.

## Data: download and compile raw data into slices

- `attach_mhpr_bass_regions.py`: Write the anatomical domain labels SABench scored into the MHPR slices.
- `bundle_cluster_data.py`: Stage exactly the slices the tasks need into one directory, ready to rsync.
- `center_slice_coordinates.py`: Centre each slice's input and ground-truth coordinates on its own centroid.
- `compile_flysta3d_slices.py`: Compile the Flysta3D E14-16h h5ad (Data16-31 in SABench paper) into 16 paper2bench slices.
- `compile_mhpr_csv.py`: Compile the MHPR (Moffitt et al.) MERFISH all-cells CSV (Data5-9 in SABench paper) into 5 paper2bench h5ad slices.
- `compile_mob_stereoseq_data78.py`
- `compile_mop_bil_processed.py`
- `compile_openst_mln_sections.py`
- `compile_paper_acquisitions_20260914.py`: Compile the paper slices acquired on 2026-09-14 into minimal paper2bench h5ad files.
- `compile_perturbed_h5ad.py`
- `compile_scc_geo.py`: Compile SCC (Ji et al. 2020) GEO RAW tar (Data10-15 in SABench paper) into 6 paper2bench h5ad slices.
- `compile_starmap_plus_well11.py`: Compile STARmap+ well11 (Data237 in SABench paper) raw CSVs into a paper2bench h5ad.
- `compile_stereoseq_t312.py`: Compile Data239 (AMBC Stereo-seq, braindatacenter mouse1 section T312) into a paper2bench h5ad.
- `compile_wmb_cellxgene_paper.py`: Compile the paper-faithful WMB (Data99-227) sections from the cellxgene Zhuang animal-1 file.
- `compile_wmb_zhuang_abca1.py`: Split the Allen ABC atlas Zhuang-ABCA-1 cache into 129 paper2bench h5ad slices.
- `compile_xenium_10x_bundle.py`
- `compute_marker_gene_list.py`: Generate a paper2bench-compatible marker_genes.csv by picking top-N highly variable genes
- `extract_xlsx_source_data.py`
- `fetch_wmb_abc_atlas.py`
- `inspect_openst_mln.py`: Inspect the Open-ST metastatic lymph node H5AD for SABench task planning.
- `materialize_compatible_h5ad.py`
- `prepare_cluster_transfer.py`: List exactly the data a cluster needs, and write an rsync manifest for it.
- `preprocess_10x_visium_to_h5ad.py`: Convert a 10x Visium filtered_feature_bc_matrix.h5 + spatial.tar.gz pair into a minimal AnnData h5ad.
- `preprocess_10x_visiumhd_to_h5ad.py`
- `shuffle_perturbed_rows.py`: Shuffle the rows of already-compiled perturbed slices, in place, with a backup.

## Tasks: build task packages

- `build_geaso_tasks.py`: Build the sa-401 to sa-410 alignment tasks from the GEASO data release.
- `build_run_matrix.py`: Define the design subset and emit one row per planned agent run.
- `build_task_cards.py`: Generate docs/TASK_CARDS.md from the committed runnable tasks.
- `build_tasks.py`: Build spatial-alignment task packages from the declarative registry.
- `build_wmb_realistic_tasks.py`: Realistic-difficulty 3D reconstruction tasks from the Zhuang ABCA-1 MERFISH sections (sa-414 to sa-416).
- `check_portability.py`: Report paths that would not resolve on another machine.
- `check_task_scale_consistency.py`: Check that each task's prompt tells the truth about scale.
- `materialize_albis_tasks.py`: Materialize spatial-alignment tasks from Albis-simulated datasets.
- `materialize_mop_tasks.py`
- `materialize_openst_mln_tasks.py`
- `materialize_sdmbench_imaging_tasks.py`
- `materialize_wmb_partial_tasks.py`

## Agents: run coding agents in the blind container

- `bootstrap_spatial_method_venv.sh`
- `check_access.py`: Report credential status for every agent, and optionally prove each one with a live call.
- `check_azure_openai_access.py`: Verify an Azure OpenAI deployment before submitting a benchmark array.
- `find_bare_submission.py`: Print the path of a model's own scored bare submission for a task, for warm starting.
- `lib_env.sh`
- `notify.py`: Send a paper2bench notification.
- `prepare_agent_task.py`
- `refresh_claude_login.sh`: Make sure the claude login on disk will outlive the runs about to start.  Each container gets a copy of ~/.claude/.credentials.json. Its access token has a fixed
- `run_codex_concurrent_queue.py`
- `run_codex_repeat_batch.sh`
- `run_codex_repeat_matrix.sh`
- `run_codex_repeat_pipeline.sh`
- `run_e2e_smoke.sh`: One task, start to finish, with the checks that matter afterwards.  Run this before committing a shard to a queue. It exercises the whole chain on the cheapest
- `run_full_prompt_paper_smoke.sh`
- `run_local_coding_agent.py`
- `run_openai_solver.py`
- `run_spatial_alignment_agent_suite.sh`
- `setup_lima_containment.sh`: A Linux VM on macOS for testing the container boundary.  Apptainer and Singularity build isolation from Linux namespaces, so neither runs natively on
- `start_agent_benchmark.sh`
- `verify_containment.py`: Prove the blind boundary actually holds, on whichever platform this is.

## Baselines: published methods as scripted, non-agent runs

- `baseline_campaign_report.py`: Coverage matrix for the all-tasks method-baseline campaign.
- `baseline_h5.py`: Plain-h5py slice reader shared by the scripted method baselines (scripts/run_<method>_baseline.py).
- `build_albis_baselines_table.py`: Albis (sa-301..312) comparison table: published-method and scripted baselines against bare agent runs.
- `build_baseline_campaign_rows.py`: Rows for the all-tasks method-baseline campaign (scripts/slurm/run_baseline_cell.sbatch).
- `build_baselines_vs_paper.py`: Our method-baseline runs (results/baselines/<task>/<variant>/) against the numbers the papers report for the same
- `document_baseline_failures.py`: Every failed or not-applicable cell of the baselines campaign, with its cause and resolution.
- `extract_geaso_paper_baselines.py`: The GEASO paper's reported per-method results, mapped onto our tasks.
- `run_alignment_baseline.py`
- `run_cast_baseline.py`: CAST (Tang et al., Nat Methods 2024; CAST_MARK + CAST_STACK) as a scripted, non-agent alignment baseline.
- `run_geaso_baseline.py`: GEASO (Wang, Liu & Ma, Nat Commun 2026) as a scripted, non-agent alignment baseline.
- `run_gpsa_baseline.py`: GPSA (Gaussian process spatial alignment, Jones et al. 2023; package gpsa-engelhardt) as a scripted, non-agent baseline.
- `run_mnn_addon.py`: Add spatial MNN metrics (scripts/score_mnn_addon.py) to every final run on every task. Existing scores.json
- `run_moscot_baseline.py`: moscot AlignmentProblem (fused Gromov-Wasserstein, ott-jax) as a scripted, non-agent alignment baseline.
- `run_paste2_baseline.py`: PASTE2 (partial fused Gromov-Wasserstein, rigid) as a scripted, non-agent alignment baseline.
- `run_paste_baseline.py`: PASTE (paste-bio; fused Gromov-Wasserstein + generalized Procrustes, rigid) as a scripted, non-agent baseline.
- `run_santo_baseline.py`: SANTO (Li et al., Nat Commun 2024; coarse-to-fine alignment) as a scripted, non-agent alignment baseline.
- `run_slat_baseline.py`: SLAT (scSLAT; Xia et al., Nat Commun 2023) as a scripted, non-agent alignment baseline.
- `run_spacel_baseline.py`: SPACEL (Scube) as a scripted, non-agent alignment baseline.
- `run_spateo_baseline.py`: Spateo (Morpho, rigid + non-rigid) as a scripted, non-agent alignment baseline.
- `run_stair_baseline.py`: STAIR (Yu & Xie, Genome Biol 2025; STAIR-tools) as a scripted, non-agent alignment baseline.
- `run_stalign_baseline.py`: STalign (Clifton et al., Nat Commun 2023; LDDMM on rasterized cell positions) as a scripted, non-agent baseline.
- `run_staligner_baseline.py`: STAligner (Zhou et al., Nat Comput Sci 2023) as a scripted, non-agent alignment baseline.
- `run_task_baseline.py`

## Scoring and audits

- `audit_agent_readiness.py`: End-to-end agent-readiness audit on every task in benchmarks/spatial_alignment/tasks/.
- `audit_paper_slice_isolation.py`: Replication table: SABench Supplementary Table 1 (240 slices) side by side with our local rebuilds.
- `audit_run_conditions.py`: Check that every run was really the condition its matrix row says, from what it left behind.
- `audit_scorer_inputs.py`: Prove that nothing the scorer reads reaches the agent, task by task.
- `audit_spatial_agent_leakage.py`
- `backfill_run_usage.py`: Write a usage block into historical run manifests, from traces already on disk.
- `campaign_status.py`: One-screen status of the running campaign: queue, progress, failures, pushback, cost.
- `diagnose_stack_mae.py`: Where does a 3D-stack task's MAE come from: per-section error, drift, frame offset, or the task's design?
- `rank_sabench_cohort.py`: Rank a method cohort with the documented SABench-style procedure.
- `run_bca_resolution_recovery.py`
- `run_dlpfc_rotation_robustness.py`
- `run_hidden_perturbation_grid.py`: Re-execute an agent's solver across a hidden perturbation grid (paper Fig 3 protocol).
- `score_agent_submission.py`
- `score_dlpfc_recovery.py`: Perturbation-recovery MAE for the DLPFC tasks built by moving real sections.
- `score_mnn_addon.py`: Spatial MNN metrics for one submission, as an add-on to its existing scores.json (which is not touched).
- `score_sabench_paper_metrics.py`: Score submissions with the SABench-exact path and write them in the paper's table layout.

## Analysis and figures

- `analyze_spatial_run_structure.py`: Post-run structural analyses for spatial-alignment agent benchmarks.
- `classify_spatial_failure_modes_llm.py`: LLM-as-judge classification for spatial-alignment failure modes.
- `export_aligned_points.py`: Export the cells behind the aligned-result figures, so each panel can be redrawn on its own.
- `export_all_coordinates.py`: Every cell's predicted aligned coordinates from every baseline method and every scored agent run, per task.
- `export_viewer_points.py`: Export subsampled cells and every panel's coordinates for the explorable alignment viewer.
- `plot_albis_alignments.py`: 3D views of the Albis alignment tasks (sa-301 to sa-310), one figure per task and model.
- `plot_albis_lattice_leak.py`: Show that Albis's bin/spot tasks leak each slice's rotation through the point lattice.
- `plot_aligned_results.py`: Aligned-result figures for every scored spatial-alignment task: input, ground truth, published and
- `plot_dlpfc_alignments.py`: Input vs model alignment vs ground truth for the DLPFC tasks, one figure per task and model.
- `plot_parallel_trends_publication.py`: Per-task parallel-trends plots: composite + per-metric, publication style.
- `plot_spatial_actual_task_examples.py`: Plot actual public spatial coordinates for representative SA tasks.
- `plot_spatial_agent_alignment_examples.py`: Plot raw spatial coordinates against agent-submitted aligned coordinates.
- `plot_spatial_agent_alignment_examples_more.py`: Generate an extended plate of spatial alignment examples.
- `plot_spatial_agent_alignment_gold_examples.py`: Plot hidden-coordinate gold examples as an alignment reference column.
- `plot_spatial_all_paper_methods.py`: Plot all reported paper methods against spatial-agent prompt means.
- `plot_spatial_common_prompt_comparison.py`: Compare common-task spatial prompt runs, including the latest 30x2 round.
- `plot_spatial_human_benchmark_barplots.py`: Paper-benchmark barplots for spatial-agent prompt comparisons.
- `plot_spatial_identity_diagnostics.py`: Identity-tier diagnostic figures for the spatial-agent manuscript.
- `plot_spatial_identity_failure_examples.py`: Plot identity-strong tasks where agent transformations degrade alignment.
- `plot_spatial_mae_apples_to_apples.py`: Focused MAE applicability and apples-to-apples plots for spatial-agent runs.
- `plot_spatial_manuscript_overview.py`: Manuscript overview figures for spatial-agent benchmark results.
- `plot_spatial_trace_case_studies.py`: Create observable trace-vignette table and timeline figure for spatial runs.
- `plot_spatial_trace_strategy_analysis.py`: Plot observable strategy families from spatial agent traces.
- `plot_spatial_xhigh_analysis.py`: Plot and summarize the spatial xhigh benchmark matrix.
- `summarize_agent_runs.py`
- `summarize_alignment_runs.py`
- `summarize_conditions.py`: Per model and task: the best bare run, the best equipped run, and the warm start.

## Other

- `convert_ipynb_tree_to_py.py`
