# Spatial Alignment Run Notes

## 2026-04-28 Paper-Prior 10x3 Smoke

Launcher state:

- `/tmp/p2b_paper_prior_10x3_20260428-230423.json`
- `/tmp/p2b_paper_prior_10x3_20260428-230423.log`
- `/tmp/p2b_spatial_paper-prior-10x3-20260428-230423`

Prompt/model:

- `full-package-paper-prior`
- `gpt-5.3-codex`
- `xhigh`
- 10 tasks x 3 reps, queued at 3 concurrent runs

Harness note:

- `sa-001-dlpfc-serial-align` rep 1 and
  `sa-004-bca-downsampled-pair-align` rep 1 were launched before the
  `.method-venv-ot` PASTE/POT compatibility fix below was added. Treat those
  as pre-fix artifacts if analyzing package usability.
- The fix adds a `paste_pot_compat` post-install hook for the merged OT env.
  It patches `paste-bio`'s `paste/PASTE.py` line-search callback signature so
  `paste.pairwise_align` works with `POT 0.9.6+`.
- Already-created OT envs in the running queue were patched in place. Future
  queued reps will receive the patched bootstrap script and manifest at
  workspace preparation time.

Disposition:

- The mixed pre/post-fix temp queue above was stopped and its `/tmp` root,
  state, and log files were removed before analysis.
- Clean restart launched at 2026-04-28 23:51 local time:
  `/tmp/p2b_paper_prior_10x3_clean_20260428-235127.json`.
  This run uses the patched PASTE/POT bootstrap path from the first job onward.

## 2026-04-29 Paper-Prior Next10x2 Queue

Launcher state:

- `/tmp/p2b_paper_prior_next10x2_clean_20260429-000323.json`
- `/tmp/p2b_paper_prior_next10x2_clean_20260429-000323.log`
- `/tmp/p2b_spatial_paper-prior-next10x2-clean-20260429-000323`

Queue policy:

- Waits for `/tmp/p2b_paper_prior_10x3_clean_20260428-235127.json` to
  drain before launching any jobs.
- Runs 10 additional distinct `sa-*` tasks x 2 reps, queued at 3 concurrent
  runs.
- Uses `full-package-paper-prior`, `gpt-5.3-codex`, `xhigh`, blind solve, and
  the patched PASTE/POT bootstrap path.

Tasks:

- `sa-002-dlpfc-robust-perturb`
- `sa-003-dlpfc-coarse-shuffled-pair`
- `sa-005-bca-resolution-recovery`
- `sa-006-mbsa-serial-align`
- `sa-007-mbsp-serial-align`
- `sa-008-ambc-xenium-multisection-serial-align`
- `sa-011-scc-p2-serial-align`
- `sa-012-scc-p5-serial-align`
- `sa-013-ambc-visium-xenium5k-cross-platform`
- `sa-014-bca-xenium-serial-align`

## 2026-04-29 Package-Use Diagnosis

Across the 50 completed `full-package-paper-prior` runs, every workspace had
records for the method envs (`staligner`, `spateo`, `gpsa`, `cast`, `stair`,
`santo`, `spacel`, `scslat`/`slat`, `stalign`, `ot`/`paste`/`paste2`) and the
workspace `method_install_failures.tsv` files were empty. The underuse of
several packages is therefore not explained by broad prebootstrap failure.

Actual command traces favored the more discoverable/direct APIs:

- `.method-venv-ot/bin/python`: 492 command invocations
- `.method-venv-spacel/bin/python`: 180 command invocations
- `.method-venv-spateo/bin/python`: 157 command invocations
- `.method-venv-cast/bin/python`: 2 command invocations
- `.method-venv-staligner`, `.method-venv-stalign`,
  `.method-venv-scslat`, `.method-venv-stair`, `.method-venv-santo`,
  `.method-venv-gpsa`: 0 direct command invocations

Representative import/API probes on a completed workspace:

- `STAligner` imports and exposes `Cal_Spatial_Net`, `ICP_align`,
  `STALIGNER`, `train_STAligner`, and related helpers, but agents did not call
  its env directly.
- `STalign` imports as package `STalign`; the top-level package is sparse and
  functions live in submodules such as `STalign.STalign`.
- `STAIR` imports as uppercase `STAIR`; top-level attrs are sparse and useful
  code appears in submodules such as `emb_alignment`, `loc_alignment`, and
  `loc_prediction`.
- `SANTO` imports as lowercase `santo`; uppercase `SANTO` fails.
- `GPSA` imports and exposes `GPSA` / `VariationalGPSA`, but agents did not
  attempt it, likely because it looks like a model-training API rather than a
  direct CSV-coordinate alignment API.
- `CAST` imports and exposes `CAST_MARK`, `CAST_PROJECT`, and `CAST_STACK`;
  one agent probed it, but it was not used beyond API introspection.
- `scSLAT` was a real environment issue in this run: `import scSLAT` failed
  with `ImportError: zarr-python major version > 2 is not supported`. This is
  fixed for future bootstraps by adding a `zarr<3` post-install pin to the
  `scslat` method env. A fresh smoke workspace reported `scslat scSLAT OK`
  with `scSLAT 0.3.0` and `zarr 2.18.7`.

Observed agent messages repeatedly cite slow imports, heavy JIT/startup, OT
runtime, AnnData encoding compatibility, and lack of a direct transformed
coordinate path as reasons for falling back to PASTE-family methods, SPACEL,
spateo, or custom rigid/ICP-style alignments.

## 2026-04-29 Package-Fix Use Smoke 5x2

Launcher state:

- `/tmp/p2b_pkgfix_use_smoke_5x2_20260429-185452.json`
- `/tmp/p2b_pkgfix_use_smoke_5x2_20260429-185452.log`
- `/tmp/p2b_spatial_pkgfix-use-smoke-5x2-20260429-185452`

Queue policy:

- Runs 5 distinct `sa-*` tasks x 2 reps, queued at 3 concurrent runs.
- Uses `full-package`, `gpt-5.3-codex`, `xhigh`, blind solve, and patched
  method env bootstrap.
- Purpose is to test whether the `scSLAT`/`zarr<3` fix and slow-import prompt
  language increase real attempts of previously underused package families.
- Stopped after discovering the blind-solve agent shell inherited host
  `OPENAI_API_KEY`. Codex itself was logged in through ChatGPT (`codex login
  status` reported ChatGPT login), and the run path was `codex exec`, not the
  Python OpenAI API runner, but the inherited provider key made this smoke
  unsuitable as a strict no-API-key control.
- Patched `scripts/run_local_coding_agent.py` on 2026-04-29 to strip common LLM
  provider credentials from blind-solve agent environments before launching
  `codex exec`.

Tasks:

- `sa-001-dlpfc-serial-align`
- `sa-009-mhpr-merfish-serial-align`
- `sa-013-ambc-visium-xenium5k-cross-platform`
- `sa-019-flysta3d-e14-16-3d-serial-align`
- `sa-022-openst-mln-serial-align`

## 2026-04-29 Package-Fix Use Smoke 5x2 No-API-Key Relaunch

Launcher state:

- `/tmp/p2b_pkgfix_use_smoke_noapikey_5x2_20260429-192925.json`
- `/tmp/p2b_pkgfix_use_smoke_noapikey_5x2_20260429-192925.log`
- `/tmp/p2b_spatial_pkgfix-use-smoke-noapikey-5x2-20260429-192925`

Queue policy:

- Same 5 tasks x 2 reps, queued at 3 concurrent runs.
- Uses `full-package`, `gpt-5.3-codex`, `xhigh`, blind solve.
- Launched from an environment with common provider API keys unset, and the
  patched blind-solve harness strips provider keys again before `codex exec`.

## 2026-04-29 Full-Package-Paper-Prior Local-STalign 30x2 Queue

Launcher state:

- tmux session: `fpp-localstalign-30x2-20260429-232112`
- `/tmp/p2b_fpp_localstalign_30x2_20260429-232112.json`
- `/tmp/p2b_spatial_fpp_localstalign_30x2_20260429-232112`
- An earlier launch
  (`fpp-localstalign-30x2-20260429-231920`) was stopped immediately because
  the first two same-task reps started in the same second and could collide in
  `agent_runs/` run-directory naming. The corrected launcher starts all rep-1
  jobs before later reps.

Queue policy:

- Runs the first 30 sorted `benchmarks/spatial_alignment/tasks/*/task.toml`
  entries x 2 reps, queued at 3 concurrent runs.
- Uses `full-package-paper-prior`, `gpt-5.3-codex`, `xhigh`, blind solve,
  `paper-all` prebootstrap, 60-minute agent timeout, 600-second score timeout,
  and 1800-second method-bootstrap timeout.
- `benchmarks/spatial_alignment/spatial_method_envs.toml` is temporarily
  patched so `STalign` installs from
  `/tmp/p2b_method_sources/STalign-b2068edc98974efa54537eca194736e177bbe11d`
  rather than running a live GitHub fetch. Keep that patch while this queue is
  draining; revert it after all pending workspaces have been prepared or after
  the queue finishes.

Pre-launch package import probe on the local-STalign workspace passed for:

- `STAligner`, `spateo`, `gpsa`, `CAST`, `STAIR`, `santo`, `SPACEL`,
  `scSLAT` (`zarr 2.18.7`), `STalign`, `paste`, and `paste2`.

## 2026-09-14 Paper-Faithful Data Rebuild

- `sa-009` now uses `~/Downloads/paper2bench-data/MHPR_paper/` (Bregma -0.04 to
  -0.24, Ambiguous cells removed, `Fos` dropped; 5/5 paper cell counts).
- `sa-020`, `sa-021`, `sa-101` to `sa-112` now use
  `compiled_data/mop_paper/` (slice membership from `cell_labels.csv`; 33/33
  paper cell counts for mouse 1).
- `compiled_data/visiumhd/data232_visiumhd_08um.h5ad` exists but `sa-015` was
  not switched to it.
- `sa-022/data/` was regenerated after the directory went missing.
- Runs dated before 2026-09-14 on the tasks above used the old slices. See
  `PAPER_DATA_ISOLATION_AUDIT.md` section 7.
- Later on 2026-09-14: `sa-023` now uses `~/Downloads/paper2bench-data/wmb_cellxgene_paper/` (paper cell set, 3,698,530 cells); `sa-006`/`sa-007` now use `compiled_data/perturbed_sr100/` (Space Ranger 1.0.0 matrices, same perturbation seeds). Runs before this date on those tasks used the old slices.

