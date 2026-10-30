# Spatial Alignment Input Sanitization Audit

Generated from a source H5AD schema scan after changing `scripts/prepare_agent_task.py` to create sanitized agent-visible H5ADs.

## Sanitizer Contract

| Agent-visible | Withheld from agent-visible H5ADs |
|---|---|
| `X`, `var`, optional `layers`, `obs/_index`, `obsm/spatial` | All other `obs/*`, all `obsm/*` except `spatial`, scorer labels (`Region`, `Cluster`, etc.), GT/CCF coordinates, aligned 3D coordinates, transform/perturbation attrs, split manifests |

The scorer still uses the original task H5ADs through `task.toml`; only the workspace `data/*.h5ad` files are sanitized. Prompts are also redacted to remove references to scorer-only labels, ground truth coordinates, transforms, and raw absolute source paths.

## Run-Time Leakage Flagging

Each local coding-agent run now writes a non-fatal leakage audit artifact:

```text
agent_runs/local_coding_agent/<run>/leakage_audit.json
```

The audit checks three things:

- Workspace `data/*.h5ad` files expose only `obs/_index` and `obsm/spatial`, and do not symlink back to raw task/source H5ADs.
- The saved prompt, Codex JSON stream, stderr, and generated workspace snapshot are scanned for references to private source paths such as `benchmarks/spatial_alignment/compiled_data`, `split_manifest.json`, hidden perturbation files, or `~/Downloads/paper2bench-data`.
- The same artifacts are scanned for suspicious references to fields withheld for that task, including scorer labels (`Region`, `Cluster`, `Cell_class`, `class_label`, etc.), GT/CCF coordinates (`x_GT`, `X_CCF`, etc.), and aligned 3D arrays (`spatial_3d_aligned`).

Non-blind workspaces expose `score.sh`, which calls `python -m genomics_agent_bench.public_score`; solve-time scoring omits scorer-only paths such as `split_manifest` from `scores.json` and suppresses Python tracebacks that would otherwise expose scorer implementation paths.

## Blind Solve Harness

Clean Codex benchmark runs should use `scripts/run_local_coding_agent.py --blind-solve` with a workspace root outside the repository, for example under `/tmp`.

Blind solves add the following controls:

- Agent-visible H5ADs are copied into the task workspace instead of symlinked to the sanitized cache.
- `score.sh`, `bootstrap_method_venv.sh`, split manifests, scorer paths, task TOML paths, and raw source paths are not written into the solve workspace or prompt.
- Codex, Claude, and Gemini are wrapped in a macOS `sandbox-exec` profile that denies reads and writes to repo/source/prior-artifact paths, including the local `~/Downloads/SABench` reference checkout. Codex and Gemini run unattended only inside that outer filesystem sandbox; blind solves still reject a user-supplied `--dangerous-bypass` flag.
- The agent subprocess environment is sanitized so inherited variables such as `PWD`, `PYTHONPATH`, or `OLDPWD` do not expose the repository or prior-artifact paths.
- Final eval scoring is run only by the harness after the agent process exits.
- The prompt explicitly forbids searching parent directories, repo paths, prior runs/results, scorer code, source files, and Downloads caches.

The intended smoke command shape is:

```bash
uv run python scripts/run_local_coding_agent.py benchmarks/spatial_alignment/tasks/sa-001-dlpfc-serial-align/task.toml \
  --agent codex \
  --model gpt-5.3-codex \
  --reasoning-effort xhigh \
  --workspace-root /tmp/p2b_spatial_blind_workspaces \
  --runs-root agent_runs/local_coding_agent_xhigh_blind \
  --prompt-variant default \
  --blind-solve \
  --force
```

The repeat launchers `scripts/run_codex_repeat_batch.sh`, `scripts/run_codex_repeat_matrix.sh`, and `scripts/run_codex_repeat_pipeline.sh` now default to blind solve mode and `/tmp` workspace roots. They reject `--dangerous-bypass` unless explicitly switched to `--non-blind`.

This is a practical filesystem isolation layer for non-adversarial benchmark agents plus trace auditing. It is still not a formal security boundary against a fully adversarial local process or CLI bug, so any run that references repo/scorer/prior-artifact paths should be flagged and excluded.

## Prior Artifact Isolation

Prior spatial submissions, workspaces, and result tables were archived out of the repository before clean blind reruns:

```text
/tmp/p2b_spatial_prior_artifacts/20260426_1504
```

The move manifest is stored at `/tmp/p2b_spatial_prior_artifacts/20260426_1504/MOVED_PATHS.txt`. New blind runs also deny reads under `/tmp/p2b_spatial_prior_artifacts`, so archived prior submissions cannot be used as in-context examples or copied into new submissions.

## Summary

| Category | Count |
|---|---:|
| Tasks scanned | 40 |
| Source tasks where scorer label key exists in source H5ADs | 25 |
| Source tasks with GT/CCF/aligned coordinate fields | 8 |
| Source tasks with transform/perturbation attrs | 2 |
| Sanitized output schemas with anything beyond allowed fields | 0 by construction; representative workspaces verified with HDF5 inspection |

## Per-Task Audit

| Task | Profile | Label key | Slices | Withheld scored label? | Label-like obs withheld | GT/CCF/aligned withheld | Transform attrs withheld | Other obs metadata withheld | Agent-visible fields |
|---|---|---|---:|---|---|---|---|---:|---|
| sa-001-dlpfc-serial-align | `sabench_light_alignment_v1` | `Region` | 4 | yes | `Region` | - | - | 3 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-002-dlpfc-robust-perturb | `sabench_light_robustness_v1` | `Region` | 2 | yes | `Region` | - | - | 3 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-003-dlpfc-coarse-shuffled-pair | `sabench_light_alignment_v1` | `Region` | 2 | yes | `Region` | - | - | 3 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-004-bca-downsampled-pair-align | `sabench_gene_only_alignment_v1` | `Cluster` | 2 | no | - | - | - | 2 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-005-bca-resolution-recovery | `resolution_recovery_alignment_v1` | `Cluster` | 6 | yes | `Cluster` | - | - | 12 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-006-mbsa-serial-align | `sabench_gene_only_alignment_v1` | `Cluster` | 2 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-007-mbsp-serial-align | `sabench_gene_only_alignment_v1` | `Cluster` | 2 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-008-ambc-xenium-multisection-serial-align | `sabench_gene_only_alignment_v1` | `Cluster` | 3 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-009-mhpr-merfish-serial-align | `sabench_light_alignment_v1` | `Region` | 5 | yes | `Cell_class`, `Neuron_cluster_ID`, `Region` | - | - | 1 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-010-ambc-visium-xenium-cross-platform | `sabench_gene_only_alignment_v1` | `Cluster` | 2 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-011-scc-p2-serial-align | `sabench_gene_only_alignment_v1` | `Region` | 3 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-012-scc-p5-serial-align | `sabench_gene_only_alignment_v1` | `Region` | 3 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-013-ambc-visium-xenium5k-cross-platform | `sabench_gene_only_alignment_v1` | `Cluster` | 2 | no | - | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-014-bca-xenium-serial-align | `sabench_geometry_gene_alignment_v1` | `Cluster` | 2 | yes | `Cluster` | - | - | 9 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-015-ambc-visium-visiumhd-cross-resolution | `sabench_gene_only_alignment_v1` | `Region` | 2 | no | - | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-016-ambc-visium-starmapplus-cross-platform | `sabench_gene_only_alignment_v1` | `Region` | 2 | no | `Cell_type`, `Region`, `Sub_region` | - | - | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-017-baristaseq-mpc-serial-align | `sabench_light_alignment_v1` | `Region` | 3 | yes | `Region`, `layer` | - | - | 9 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-018-starmap-mmpc-serial-align | `sabench_light_alignment_v1` | `Region` | 3 | yes | `Region`, `region` | - | - | 1 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-019-flysta3d-e14-16-3d-serial-align | `sabench_mae_gene_efficiency_v1` | `Region` | 16 | no | `Region` | `x_GT`, `y_GT`, `z_GT` | `obs.attrs/perturbation` | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-020-mop-mouse1-atlas-serial-align | `sabench_light_alignment_v1` | `Cluster` | 33 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-021-mop-mouse2-atlas-serial-align | `sabench_light_alignment_v1` | `Cluster` | 31 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-022-openst-mln-serial-align | `sabench_light_alignment_v1` | `Region` | 19 | yes | `Region`, `annotation`, `annotation_key` | `x_GT`, `y_GT`, `z_GT`, `obsm/spatial_3d_aligned` | - | 14 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-023-wmb-zhuang-abca1-129-section-3d-mae | `sabench_mae_gene_efficiency_v1` | `Region` | 129 | no | `Region` | `x_GT`, `y_GT`, `z_GT` | `obs.attrs/perturbation` | 0 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-025-openst-mln-adjacent-02-03-3d-align | `sabench_mae_gene_efficiency_v1` | `Region` | 2 | no | `Region`, `annotation`, `annotation_key` | `x_GT`, `y_GT`, `z_GT`, `obsm/spatial_3d_aligned` | - | 14 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-026-openst-mln-early-window-02-06-3d-align | `sabench_mae_gene_efficiency_v1` | `Region` | 5 | no | `Region`, `annotation`, `annotation_key` | `x_GT`, `y_GT`, `z_GT`, `obsm/spatial_3d_aligned` | - | 14 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-027-openst-mln-full-19-slice-3d-align | `sabench_mae_gene_efficiency_v1` | `Region` | 19 | no | `Region`, `annotation`, `annotation_key` | `x_GT`, `y_GT`, `z_GT`, `obsm/spatial_3d_aligned` | - | 14 | `X`, `var`, `layers`, `obs/_index`, `obsm/spatial` |
| sa-028-wmb-animal1-window5-serial-align | `sabench_light_alignment_v1` | `Region` | 5 | yes | `Region`, `brain_section_label`, `ccf_region_name`, `cell_type`, +3 more | `x_ccf`, `y_ccf`, `z_ccf`, `obsm/X_CCF` | - | 3 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-029-wmb-animal1-window10-serial-align | `sabench_light_alignment_v1` | `Region` | 10 | yes | `Region`, `brain_section_label`, `ccf_region_name`, `cell_type`, +3 more | `x_ccf`, `y_ccf`, `z_ccf`, `obsm/X_CCF` | - | 3 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-101-mop-mouse1sample1-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-102-mop-mouse1sample2-serial-align | `sabench_light_alignment_v1` | `Cluster` | 5 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-103-mop-mouse1sample3-serial-align | `sabench_light_alignment_v1` | `Cluster` | 4 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-104-mop-mouse1sample4-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-105-mop-mouse1sample5-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-106-mop-mouse1sample6-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-107-mop-mouse2sample1-serial-align | `sabench_light_alignment_v1` | `Cluster` | 4 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-108-mop-mouse2sample2-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-109-mop-mouse2sample3-serial-align | `sabench_light_alignment_v1` | `Cluster` | 6 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-110-mop-mouse2sample4-serial-align | `sabench_light_alignment_v1` | `Cluster` | 5 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-111-mop-mouse2sample5-serial-align | `sabench_light_alignment_v1` | `Cluster` | 5 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |
| sa-112-mop-mouse2sample6-serial-align | `sabench_light_alignment_v1` | `Cluster` | 5 | yes | `Cluster`, `Region_proxy`, `class_label`, `label`, +1 more | - | - | 2 | `X`, `var`, `obs/_index`, `obsm/spatial` |

## Representative Validation

I prepared temporary sanitized workspaces for `sa-009`, `sa-019`, and `sa-028` and inspected their HDF5 schemas. Each checked H5AD had only `obs/_index` and `obsm/spatial` among observation/spatial fields; no `Region`, `Cluster`, `x_GT`, `y_GT`, `z_GT`, `x_ccf`, `X_CCF`, `spatial_3d_aligned`, or `obs.attrs/perturbation` remained.

I also verified the active restarted `sa-001` workspace has only `obs/_index` and `obsm/spatial` in its agent-visible H5ADs.

## Trace Audit Notes

- `sa-014-bca-xenium-serial-align` rich run `20260426_212229` produced one medium `Cluster` trace finding. Manual inspection showed this was the literal word `cluster` from an allowed expression-clustering heuristic (`MiniBatchKMeans`) in stdout, not access to `obs/Cluster` or any scorer-only label. The run's sanitized H5AD inputs exposed only `obs/_index` and `obsm/spatial` among observation/spatial fields, so this finding is treated as a benign false positive rather than label leakage.
