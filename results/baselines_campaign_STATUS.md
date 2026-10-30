# All-tasks method-baseline campaign: status

Goal: every non-agent method on every non-smoke task (121 tasks; sa-3xx Albis already complete), scored with the
agents' scorer, then checked against the papers' reported numbers.

Started 2026-10-05 23:10 (America/New_York). Campaign dir: `/dcs07/chen/data/alignment-agent/baselines_campaign/`.

## Submitted arrays

| Class | Array | Rows | Throttle | Resources |
|---|---|---|---|---|
| cpu_small | 36173568 | 833 | 36 | shared,scavenge; icelake/sapphirerapids; 8 CPU, 64G, 2 d |
| cpu_big | 36173569 | 276 | 14 | shared,scavenge; 8 CPU, 180G, 3 d |
| cpu_stack | 36173570 | 152 | 10 | shared,scavenge; 8 CPU, 200G, 3 d (fit on 5,000 obs/slice) |
| gpu_small | 36173571 | 207 | 5 | gpu, 1 GPU (not 117), 8 CPU, 96G, 1 d (GEASO, CAST, cast_rescale) |
| gpu_big | 36173572 | 106 | 3 | gpu, 1 GPU A100/H100, 8 CPU, 240G, 3 d |

At most 60 CPU and 8 GPU jobs run at once.

GPU note: the two GPU arrays were submitted to partition `gpu` only. After submission the V100 node compute-123 was
excluded from the pending gpu_small tasks (GEASO ran out of its 32 GB GPU memory: indices 1 and 4 failed, to be
retried). Moving the pending GPU tasks to `gpu,bstgpu,bstgpu2,bstgpu3` with `scontrol update` was blocked by the
permission system, so the running arrays still use `gpu` only (A100/H100, throttles 5 + 3). New GPU submissions
(retries) use `gpu,bstgpu,bstgpu2,bstgpu3`, throttles 8 + 4 (cap 12), via `submit_baseline_campaign.sh`. All exclude `P2B_EXCLUDE_NODES`.
Row files (array index = line number - 1): `baselines_campaign/rows/submitted_<class>_<array>.tsv`
(`variant  runner  task.toml  extra args`). Logs: `baselines_campaign/logs/p2b-bl-<class>-<array>_<index>.out`.
Per-cell status: `baselines_campaign/status/<task>__<variant>.json` (exit code, elapsed, node).
Not applicable (not submitted): `baselines_campaign/not_applicable.tsv` (sa-406: RNA genes vs H3K4me3 peaks share
no feature; only STalign's density image applies).

## Code changes made for this campaign (uncommitted)

- `scripts/baseline_h5.py`: plain-h5py reader (old-format task files that newer anndata refuses) and
  `set_task_genes`, which restricts every slice to the genes shared by all slices of the task (cross-platform pairs).
- `scripts/run_spacel_baseline.py:read_slice` delegates to it; each `scripts/run_<method>_baseline.py` calls
  `set_task_genes(slices)` after loading the task; GEASO's reader uses the same h5py path (it intersects per pair itself).
- `scripts/build_baseline_campaign_rows.py`, `scripts/slurm/run_baseline_cell.sbatch`,
  `scripts/slurm/submit_baseline_campaign.sh`.
- CPU jobs are constrained to icelake/sapphirerapids nodes: STAligner's `annoy` was built with native flags and
  dies with SIGILL (exit 132) on haswell.

## How to check progress

    sacct -j 36173568,36173569,36173570,36173571,36173572 -X --format=JobID%18,State,Elapsed,MaxRSS | awk '{print $2}' | sort | uniq -c

## How to resume

1. `.venv/bin/python scripts/build_baseline_campaign_rows.py` (skips every cell that already has `scores.json`).
2. Remove rows still running in the old arrays (or cancel only those arrays: they are ours), then
   `bash scripts/slurm/submit_baseline_campaign.sh [class ...]`.
3. OOM / timeout retries: resubmit those rows with more memory/time (`THROTTLE=` overrides the throttle).

## Requeue applied (2026-10-05 23:18, run outside the agent)

The pending tasks of 36173568/69/70 were cancelled and the 1,250 pending rows resubmitted as per-method arrays
`p2b-bl-<method>` (only STAligner constrained to icelake/sapphirerapids), arrays 36173629-36173651
(see `baselines_campaign/submissions.log`; rows `rows/submitted_<method>_<size>_<id>.tsv`). Throttles total 33,
plus the 12 GPSA tasks still running from 36173568 = 45 CPU jobs max (headroom left under 60). The script that did this
is kept as `rows/requeue_unconstrained.DO_NOT_RUN_already_applied.sh`; do not run it again.

## State

Arrays just submitted; nothing from the main arrays has finished yet. Paper check not started.

## 2026-10-05 23:25: re-budgeted resubmission (the user keeps 150-200 of their 400 shared CPUs free)

All still-pending CPU tasks were cancelled and resubmitted with fewer CPUs and tighter throttles.
Rows: `baselines_campaign/rows/budget_*.tsv`; IDs in `submissions.log`.

| Array | Rows | CPUs | Mem | At once | Start |
|---|---|---|---|---|---|
| 36173679 light_small (scripted, SANTO, STAIR, SLAT, Moscot, PASTE) | 343 | 2 | 32G | 10 | now |
| 36173680 light_large | 184 | 2 | 128G | 4 | now |
| 36173681 staligner (icelake/sapphirerapids) | 107 | 4 | 128G | 2 | now |
| 36173682 heavyA_small (SPACEL, Spateo) | 80 | 4 | 64G | 4 | now |
| 36173683 heavyA_large (SPACEL) | 31 | 4 | 200G | 2 | now |
| 36173684 heavyB_small (Spateo, PASTE2, STalign, GPSA) | 317 | 4 | 64G | 18 | +5 h |
| 36173685 heavyB_large | 175 | 4 | 200G | 5 | +5 h |

Still running from before: 12 GPSA tasks of 36173568 (8 CPUs each) and a few light tasks.
GPU arrays 36173571 and 36173572 are unchanged.
The earlier per-method arrays 36173629..36173651 have no pending tasks left.
