# Existing experiments: what stays valid, what must be re-run, what has never run

Written 2026-09-15 from `results/agent_run_inventory_20260915.csv` (554 run
directories under `agent_runs/`, 531 with a score; all dated 2026-04-26 to
2026-05-02). The balanced design that the current figures use is three prompt
variants (`default`, `paper-method-package-first`, `full-package-paper-prior`)
with three reps each on all 40 tasks, `gpt-5.3-codex` at `high`, blind solve.
`full-package` and `full-basic` exist only on a subset and are not part of the
comparison figures.

## 1. Runs invalidated by the data rebuilds (re-run)

The rebuilds on 2026-09-14 changed the input slices, so agent solutions and
their scores no longer refer to the current task data.

| Tasks | What changed | Existing scored runs | Re-run |
| --- | --- | --- | --- |
| sa-009 | MHPR: different Bregma set, Ambiguous cells removed, `Fos` dropped (5/5 paper counts) | 32 | 3 prompts x 3 reps = 9 |
| sa-020, sa-021, sa-101 to sa-112 (14 tasks) | MOp: slice membership from `cell_labels.csv`, 1 to 3% more cells per slice, all 280,186 cells placed | 132 | 14 x 9 = 126 |
| sa-023 | WMB: cellxgene cell set, 3,698,530 cells (about 30% more than before), 1,120 genes | 18 | 9 |
| sa-006, sa-007 | MBSA/MBSP: Space Ranger 1.0.0 matrices (2 to 4 spots and 1,232 genes differ). Old submissions cannot be re-scored because spot IDs changed. | 22 | 2 x 9 = 18 |

Total: 17 tasks, 162 runs. At the previous cadence (3 concurrent jobs, 60
minute agent timeout, roughly 30 minutes average) that is about 27 hours of
wall time.

The 204 stale runs stay in `agent_runs/local_coding_agent/` for the record;
they should be tagged as pre-rebuild in any analysis that reads the directory
(their dates, all before 2026-09-14, are the tag).

## 2. Runs that stay valid (no re-run)

23 tasks whose slices did not change: sa-001 to sa-005, sa-008, sa-010 to
sa-019, sa-022, sa-025 to sa-029. 327 scored runs.

Two caveats:

- sa-022: its `data/` directory had vanished and was regenerated. Verified
  on 2026-09-15: with `marker_genes.csv` rebuilt as the top-100 variance list
  (`scripts/compute_marker_gene_list.py --top-n 100`) and the historic 50%/seed-0
  split, re-scoring an April submission reproduces its original scores to
  four decimals on every metric, so the old sa-022 runs stand.
- sa-001 and the other DLPFC/MHPR/MOB/MLN/Flysta3D tasks will be re-scored,
  not re-run, once the paper-faithful scoring mode (plan item B1) exists.
  Submissions are coordinate CSVs, so re-scoring needs no agent time.

## 3. Runs that will be needed after planned rebuilds (do not run yet)

| Task | Pending change | Then |
| --- | --- | --- |
| sa-019 | input frame rebuild to centered `raw_x/raw_y` (plan B6) | 9 runs |
| sa-015 | optional switch to the 8 um Visium HD file | 9 runs if switched |
| sa-010 | optional switch to a hemisphere Xenium crop | 9 runs if switched |

## 4. Experiments that have never run

| Experiment | Tasks to build | Agent runs | Method-baseline runs | Paper reference |
| --- | --- | --- | --- | --- |
| Rotation and crop sweeps (plan B3) | DLPFC 25 + 10, MHPR 25 + 10 = 70 pair tasks | 70 x 2 reps = 140 (single-pair tasks, short) | 11 methods x 70 = 770 | MOESM8, Fig. 3 |
| MOB cross-platform pair (B4) | 1 | 9 | 11 | MOESM10, Fig. 5b |
| AMBC cross-platform pairs (B5) | 41 new pairs (45 total, 4 exist) | 41 x 2 = 82 | 11 x 45 = 495 | Fig. 5d |
| Method baselines on the existing 40 tasks (B2, R1, R2) | none | none | 11 x 40 = 440 | Fig. 4b ranks (9 tasks), per-task references (all) |
| Downstream clustering (B8, optional) | 1 | 9 | 11 | MOESM7 2E |

No paper method has been run on any task outside agent sessions; the only
non-agent baselines are the three geometry baselines in
`agent_runs/analysis/spatial_geometry_baselines/`.

## 5. Suggested order and commands

1. Re-run the 17 rebuilt tasks with the three prompt variants. The queue
   runner takes explicit task paths; one queue per prompt variant, 3 reps,
   3 concurrent:

   ```bash
   TASKS=$(for t in sa-006 sa-007 sa-009 sa-020 sa-021 sa-101 sa-102 sa-103 sa-104 sa-105 sa-106 sa-107 sa-108 sa-109 sa-110 sa-111 sa-112; do printf -- "--task benchmarks/spatial_alignment/tasks/%s*/task.toml " $t; done)
   for pv in default paper-method-package-first full-package-paper-prior; do
     uv run python scripts/run_codex_concurrent_queue.py $TASKS --count 3 --concurrency 3 \
       --prompt-variant $pv --workspace-root /tmp/p2b_rerun_rebuilt_$pv \
       --state /tmp/p2b_rerun_rebuilt_$pv.json
   done
   ```

   Run the three queues one after another, not in parallel, to keep 3
   concurrent jobs total.
2. Build B1 (paper-faithful scoring) and re-score all valid submissions on
   sa-001, sa-009 (new runs), sa-019, sa-025 to sa-027; verify sa-022.
3. Build B2 (method runner) and run the 11 methods on the 9 Fig. 4b tasks
   first (sa-001, sa-006, sa-007, sa-009, sa-011, sa-017, sa-018, sa-019,
   sa-020), check ranks against MOESM9, then on the rest.
4. Build B3 to B5 tasks and run them (agents and methods).
5. Rebuild sa-019 (B6), then re-run it.
