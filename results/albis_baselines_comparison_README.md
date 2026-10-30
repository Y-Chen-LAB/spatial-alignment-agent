# Albis baselines vs agents (sa-301..312)

`albis_baselines_comparison.csv` is built by `scripts/build_albis_baselines_table.py`. Rerun that script after new runs land.

The table also includes every gene-grid metric and the four spatial MNN diagnostics,
with `_n_runs` columns showing available metric coverage. Missing MNN scores in old
JSON files remain blank; these saved submissions have not been rescored with the
new profile. Median component values are metric-wise medians; best component values
come from the same lowest-MAE run. Component summary means require all five tasks.
See [ALBIS scoring rules](../docs/ALBIS_SCORING.md).

- Each row is one (task, entry, stat). `mae_um` is the MAE in µm in s00's frame. `mae_reduction` is 1 − mae / mae_identity. Both come from the agents' scorer, `scripts/score_agent_submission.py`; every baseline was rescored with it.
- Baselines (`kind=baseline`) come from `results/baselines/<task>/<method>/scores.json`. Each method directory has a `run_manifest.json` with settings, versions and runtime. Each runner, `scripts/run_<method>_baseline.py`, has a docstring with the environment and the deviations from the method's defaults.
- Agents (`kind=agent (bare)`) are runs under `agent_runs/cluster/*/<task>*/` with `prompt_variant=full-basic` and `status=scored`.
  - `median` is the median MAE over runs. Its `mae_reduction` is derived from that MAE.
  - `best` is the run with the lowest MAE.
  - `n_runs` is the number of runs, usually 1–3, so treat medians as rough.
- Task groups:
  - **sa-301, sa-302**: cell-level sections, no lattice.
  - **sa-303..305**: Visium/HD bins on a shared lattice. This is a known leak that agents can invert exactly, so agent scores here are not comparable to the methods.
  - **sa-306..310**: the leak-fixed versions. Use these for comparisons.
  - **sa-311**: 32 µm bins with weak domain landmarks. New task type.
  - **sa-312** (`sa-312-albis-strongmix-cell-7domain-serial-align`): 4 of 9 non-reference sections are reflected. New task type.
- The rows with `task_id = mean_sa-306..310` average an entry over the five leak-fixed tasks. They appear only for entries present on all five. For them, `n_runs` and `runtime_s` are sums.
- Variant suffixes:
  - `_rigidpart`, `_affine`, `_init`, `_coarse`, `_nonrigid`: an intermediate or alternative stage of the same method.
  - `_allowflip`: reflections enabled (a deviation from the method's default).
  - `cast_rescale`: CAST_STACK with rescale=True.
  - `slat_lsq`: least-squares rigid fit on SLAT's matches.
  - `stalign_expr` and `stalign_noinit`: STalign with an expression-image start and with the identity start.
- PASTE2, Spateo and GEASO use the same settings as on sa-411/414..417. Spateo used a 5,000-cell subsample (morpho_align_ref) on sa-302 and sa-305, and all cells elsewhere. GEASO used the openst3d profile (rigid output), with `geaso_nonrigid` as the alternative output.
