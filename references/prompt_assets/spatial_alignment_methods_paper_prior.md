## Benchmark Context (Paper-Informed Package Priority)

The benchmark paper compared multiple spatial-alignment method families on
related spatial-transcriptomics tasks. In this prompt variant, you are given
high-level paper-derived method priorities so you can spend your package-testing
budget on methods that were strongest in the paper. These are method priors,
not task labels, not ground-truth transforms, and not local scorer information.

You still must use only the allowed task inputs: sanitized AnnData/CSV files,
spot IDs, gene expression, original `obsm/spatial` x/y coordinates, and slice
identities from `TASK.md`. Do not search for or use paper result tables, hidden
aligned coordinates, benchmark source, scorer code, prior submissions, or
withheld labels.

First create a complete valid submission for every required output CSV. Then
try package-backed methods in a paper-informed order. A useful package attempt
means using the method-specific Python executable from `.method-cache/METHOD_ENVS.md`,
running an alignment API on the task data or a representative pair/subset, and
recording whether it produced usable coordinates. If a high-priority method
fails to import, has incompatible API/runtime behavior, exceeds a deliberate
timeout, or looks worse under allowed input-only diagnostics, move on and keep
the best current submission. Do not treat a quiet import as an immediate
failure: several Torch/numba-heavy method envs can spend a minute or two in
first-import/JIT startup. Wait unless the process errors or your bounded timeout
is genuinely too expensive for the task.

### Paper-Informed Priority By Task Family

| Task family / data shape | Try first | Then try | Notes |
|---|---|---|---|
| DLPFC-like Visium serial alignment | `spacel`, `paste2` | `cast`, `stair`, `spateo` | Direct paper DLPFC gene+landmark tables favored `SPACEL` and `PASTE2`; `CAST` and `STAIR` were also strong. |
| MHPR/MERFISH or image-like serial alignment | `spateo`, `spacel` | `paste2`, `stair`, `slat` | Direct paper MHPR tables favored `spateo` rigid/non-rigid and `SPACEL`; `PASTE2`, `STAIR`, and `SLAT` were competitive. |
| Generic NGS/Visium/STereo-seq slices | `paste2`, `spacel`, `spateo` | `stair`, `paste` | Aggregate NGS rankings favored `PASTE2`; `SPACEL` and `spateo` were consistently strong. `PASTE` is a useful faster fallback. |
| 3D serial / MAE-style stacks | `spacel` | `paste`, `paste2`, `stair`, `spateo` | Paper Flysta3D MAE favored `SPACEL`, then PASTE-family / `STAIR` / `spateo` methods. |
| MLN-like multi-section MAE tasks | `spateo` rigid | `slat`, `spateo` non-rigid, `stalign`, `spacel` | Paper MLN MAE favored `spateo` rigid and `SLAT`. |
| Unknown or mixed platform | `spateo`, `spacel`, `paste2` | `stair`, `cast`, `slat`, `paste` | Start with the robust paper winners, then adapt based on data size and import/API success. |

Do not treat this table as a hard rule. It is a search prior. If the dataset has
very large slices, use subsets to estimate transforms first. If a method needs
domain labels that are not present, you may create unsupervised domains from
allowed expression/coordinate data, but do not invent or recover withheld
anatomical labels.

### Practical Package Guidance

- Prefer `spateo` and `SPACEL` when tissue geometry, non-rigid deformation, or
  image-like spatial continuity looks important.
- Prefer `PASTE2` when partial overlap, NGS/Visium-style slices, or robust
  optimal-transport matching looks important.
- Keep `PASTE` as a simpler/faster fallback, especially for large serial stacks
  where full package methods are too slow.
- Use `CAST`, `STAIR`, or `SLAT` when their APIs are importable and the task
  looks compatible with embedding or graph matching.
- Treat `GPSA`, `STAligner`, `STalign`, and `SANTO` as optional method-specific
  attempts rather than first defaults unless their API/docs clearly fit the
  task.
- For slower method families, run a small representative pair/subset first and
  allow enough time for first import/JIT startup before deciding the method is
  infeasible.

### Required Reporting

In `submission/report.json` or a local note, record:

- which paper-prior methods you attempted,
- which package env executable you used,
- whether the package produced usable transformed coordinates,
- why you moved on from any failed high-priority method,
- which candidate you finally submitted.
