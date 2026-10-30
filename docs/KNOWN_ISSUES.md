# Known issues

Problems in the run environment that change whether a run finishes, with how they were found and how they are handled.

## Under Gemini CLI, SPACEL `Scube.align(n_threads>1)` can hang, and the CLI never times the command out

Found 2026-10-21, during the Gemini 4 Argon equipped batch (arrays 36519178/36519179).

**Symptom (Gemini runs only).** The run hits the 360 min agent limit (`failure_stage = agent_timeout`) after 10–38 minutes of normal
work. The last tool call in the trace is a `.method-venv-spacel/bin/python` script calling `Scube.align(...,
n_threads=4)`, and that call has no tool result. Nothing more is logged for the remaining 5+ hours.

**Cause.**
- `n_threads` is passed to `scipy.optimize.differential_evolution(workers=n_threads)` in
  `SPACEL/Scube/alignment.py` (`optimize()`, lines 94–112). With `workers > 1`, scipy starts a `multiprocessing.Pool`,
  which forks on Linux.
- A likely trigger: forking after the parent has used an OpenMP or BLAS thread pool can deadlock the workers, for example after
  scikit-learn `KMeans`, which agents often run first to make the `spatial_domain` labels SPACEL needs. 12 of the 13
  stuck calls ran KMeans earlier in the same script.
- Gemini CLI's `run_shell_command` has no per-command time limit, so a stuck command blocks the agent until the run
  limit.
- **Only Gemini runs hang.** Opus 5.5 called `Scube.align` with `n_threads>1` 55 times, in 46 runs (9 in the
  background), and 28 of those scripts ran KMeans first. All 55 returned, with no timeouts. Sonnet 4.6 used
  `n_threads>1` in 34 runs; GPT-5.5 and GPT-6 astra almost never did (0 and 2 runs). So KMeans followed by a fork does
  not hang by itself. Something in how Gemini CLI runs commands makes it hang. One candidate: Gemini runs commands
  in a pseudo-terminal, and Claude Code runs them over pipes. Not yet verified.

**Evidence** (every Gemini trace under `agent_runs/cluster`, 2026-10-21):

| `Scube.align` calls | Calls | Never returned |
|---|---|---|
| `n_threads=4` | 65 | 13 |
| `n_threads` 1, 2, 8, 16 or default (1) | 849 | 0 |

| Agent timeouts | Total | Stuck in SPACEL |
|---|---|---|
| Gemini 4 Argon equipped | 10 of the first 28 finished rows | 10 (in sa-011 a Spateo call was the newest open call; an earlier SPACEL call was also still open) |
| Gemini 3.5 Flash, all runs | 34 | 5 (3 of them `Scube.align` with `n_threads>1`). The rest are other causes |

The 10 Argon rows: sa-003, sa-004, sa-006, sa-011, sa-013, sa-014, sa-018, sa-019, sa-020, sa-101.

**Not a hang in the logs.** On a timeout, `scripts/run_local_coding_agent.py` writes `str(exc.stdout)`, the partial
output as an escaped bytes literal (`b'...'`) on one line, to `raw_stdout.jsonl`. These files look empty to line-based
tools (`wc -l` gives 0) but hold 100–310 KB of events. Decode them with `ast.literal_eval(text).decode()`.

**Handling.**
- 2026-10-21 (first 10 hangs): rerun unchanged, as array 36532929, the same way infrastructure failures are rerun.
- 2026-10-21 22:15 (after 21 of the first 44 finished Argon equipped rows had hung, and sa-011 hung again on its
  rerun): with the user's approval, `[methods.spacel]` in `spatial_method_envs.toml` gained
  `post_install = "spacel_single_worker"`. The bootstrap then rewrites `workers=n_threads` to `workers=1` in
  `SPACEL/Scube/alignment.py`, and records the result in `.method-cache/spacel_single_worker.txt`. `updating` still
  follows `n_threads` (`deferred` when it is above 1), so the search and its result are those of the parallel run,
  without the pool. The prompt is unchanged.
- Verified in live workspaces (36519178_52 and 36532929_7): `spacel_single_worker.txt` says `patched_2`, and both
  `workers=` lines read `workers=1`.
- Side effect, fixed 22:55: uv installs package files as links into the node's cache (`/tmp/p2b-uv-cache-ychen5`,
  `/uv-cache` in the container). The first version of the hook therefore wrote the patch into the cached SPACEL file on
  each node where it ran: compute-099, 123, 126, 170 and 174 (tasks set up 22:20–22:40). Any later SPACEL install on
  those nodes gets the patched file, whatever the config says. The hook now replaces the link with a patched copy and
  leaves the cache alone. Before a future campaign that should run unpatched SPACEL, clear those caches.
- Scope: every workspace set up after the change, meaning the rest of Argon equipped and all its reruns. Earlier runs
  of every model had the unpatched package. Outside Gemini, no `Scube.align` call ever hung, so those runs would not
  have gone differently.
- The hang is treated as an infrastructure failure: hung rows are rerun regardless of `--max-attempts`.
- Detection: a timed-out Gemini run whose last open tool call is `Scube.align` with `n_threads>1`.

**Possible fixes for future campaigns.**
- Wrap each Gemini shell command in a time limit.
- Write `exc.stdout` decoded on a timeout.

## Damaged shared uv cache breaks equipped method environments (found 2026-10-23)

**Symptom.** An equipped run scores normally, but its agent finds the prebuilt method environments unimportable
(`No module named 'six'`, `No module named 'scipy._lib._array_api'`) and spends its time rebuilding them, or works
without them. Only some nodes, only some runs.

**Cause.** Method environments are symlinks (`UV_LINK_MODE=symlink`) into one uv cache per node
(`/tmp/p2b-uv-cache-$USER`, mounted read-write at `/uv-cache`), shared by every run there. Package copies in it had
lost most of their files, so every environment linked to them was broken: on compute-128, 14 archives (six 3/5 files
missing, python-dateutil 20/24, scipy 1.17.1 1379/1424, scikit-learn 1.9.1 1002/1007, numpy 2.4.6 951/1041, ...);
compute-174 likewise; compute-096/121/166/175 clean. With the cache writable, a deletion or edit made through an
environment's symlinks changed the shared copy for every later run on the node.

**Evidence.** sa-5xx campaign: 12 of 20 equipped runs on compute-128/170 broken this way, 0 of 25 bare runs.
`src/genomics_agent_bench/env_health.py runs <root>` classifies each run; `... cache <dir>` checks a cache against
each wheel's RECORD.

**Fix (2026-10-23 ~16:00).** (1) The agent's container mounts the shared cache read-only; the prebootstrap that builds
the environments still writes it; an agent that installs more gets a private cache. Verified in gen2bench.sif:
imports work, deletes and writes through the links fail with EROFS, new installs work. Recorded per run as
`uv_cache_agent_mount: ro` in the manifest. (2) `alignment_array.sbatch` runs `env_health.py cache --repair` at job
start under a per-node lock, removing archives with missing files so uv extracts them again (verified), and keeps the
report as `uv_cache_check.txt` in the run folder. Runs that started before the fix use the old (writable) mount.

**Handling.** The 15 affected sa-5xx equipped runs were moved to `agent_runs/hicks_sa5xx_20261023_quarantine/` and
rerun; compute-128/146/170 excluded for that campaign (146: torch's `libtorch_cpu.so` cannot load, "cannot enable
executable stack"). The main campaign's runs used the same setup and are being scanned.

**Also found, the same on every node (so comparable, but each removes a method from the equipped toolkit):**
STAligner (annoy and hnswlib compile from source and the image has no `g++`), Spateo (`libGL.so.1` missing from the
image, and statsmodels no longer has `multipletests`), STalign (its git source path under `/tmp/p2b_method_sources`
is not visible inside the container). 24 of 24 sa-5xx equipped runs.

### Update 2026-10-23 evening: method environments fixed, legacy profile, reruns

- **Fixed (profile `current`, the default):** STAligner, Spateo, STalign, SLAT, STAIR, SANTO and CAST all import in the
  container (bootstrap `probe`: all 12 envs OK). Changes: `spatial_method_envs.toml` (STalign from GitHub at the same
  commit; Spateo statsmodels 0.14.6 and headless OpenCV; SLAT torch-scatter/sparse and headless OpenCV; STAIR scanpy,
  tqdm, torch 2.6, torch-geometric 2.8; CAST dgl 2.4.0 for torch 2.2), a wheelhouse of prebuilt wheels the image cannot
  compile (`/dcs07/chen/data/alignment-agent/p2b-wheelhouse`: annoy, hnswlib with libgomp bundled, torch-scatter,
  torch-sparse, dgl), and `GLIBC_TUNABLES=glibc.rtld.execstack=2` (torch 2.3 in SANTO's env).
- **Legacy profile:** `P2B_METHOD_ENV_PROFILE=legacy` selects `spatial_method_envs_legacy.toml` (the committed spec plus
  the SPACEL single-worker fix), no wheelhouse and no tunable: the environments the 121-task results ran with.
- **121-task results, strict damage rule** (a piece of an installed package missing: six, dateutil.*, sklearn.*,
  bundled libgfortran/libgomp): 53 runs, 50 of them Gemini 4 Argon equipped on compute-170/128/123 (0 of 74 Argon
  equipped runs elsewhere). Scores were not lower: equipped minus bare −0.008 damaged vs −0.011 clean. Rerun with the
  legacy profile (cache_damage_rerun_matrix_20261023.csv), queued after the sa-5xx arrays; old done markers in
  `agent_runs/cluster/done_superseded_cache_damage_20261023/`, old run dirs kept until the reruns score.
- **Ongoing check:** every run's manifest now carries `env_health` (env_health.scan_run). No nodes are excluded.

## Gemini: requested model vs served model, and thinking defaults (found 2026-10-23)

- **Flash was silently upgraded.** Runs request `--model gemini-3.5-flash`. The Gemini CLI (0.60.0 in gen2bench.sif, built
  2026-09-16 and unchanged since) passes that name through. The model the API reports serving changed between Oct 4 and
  Oct 17:
  - up to Oct 4: every run reports `gemini-3.5-flash` (684 runs, 2026-09-20 to 10-04);
  - from Oct 17: every run reports `gemini-3.6-flash` (13 rep-2 runs in the 121-task results, all sa-5xx Flash runs).

  The CLI does not remap the explicit name, so the redirect is on the API side. Argon is served as `gemini-4-argon`
  throughout.
- **Thinking.** No thinking flag is passed. The CLI gives `thinkingLevel: HIGH` only when `supportsModernFeatures(model)`
  holds, i.e. the name matches `^gemini-3` or does not start with `gemini-`. So:
  - `gemini-3.5-flash` gets HIGH;
  - `gemini-4-argon` gets the legacy `thinkingBudget: 8192` (DEFAULT_THINKING_MODE).

  This is the same in every campaign, so it is consistent, but not "high" for Argon.
- **OpenCV swap, 2026-10-23 ~21:00:** two Gemini Flash equipped runs, the first jobs on fresh node caches
  (compute-140, compute-107), had `opencv-python-headless` recorded as installed but `cv2/` missing (both wheels own the
  same files), so Spateo and SLAT could not import. The `opencv_headless` hook now checks `import cv2` after the swap and
  relinks the headless wheel when it fails, recording ok/relinked/cv2_import_failed in `.method-cache/opencv_headless.txt`.
  Both runs were quarantined (env_broken/) and rerun.
- **Cache repair widened (2026-10-23 ~23:50).** compute-170 still had a scipy 1.11.4 copy with 611 of 1267 files
  overwritten by a newer scipy (no files missing, so the first repair kept it; it fails as "No module named
  'scipy._lib._array_api'"), and a pandas 3.0.6 copy with no RECORD and no files. The repair now also removes archives
  with a lost RECORD, or with >= 10 changed files, or with >= 3 changed files that are >= 5% of the package. One or two
  changed files are our own PASTE/SPACEL patches and are kept, since running jobs may link to them. 170 was repaired by
  hand; other nodes are repaired as their next job starts. One Argon run (36601106_0) was quarantined and rerun.
- **compute-094** exits jobs on purpose when /tmp has < 40G free (the job-script guard); those rows are simply refilled.
