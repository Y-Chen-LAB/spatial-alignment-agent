# Moving this to a cluster

## What actually has to move

| Piece | Size | How |
| --- | --- | --- |
| The repository | 660 files, a few MB | `git clone` the `spatial-alignment` branch |
| Slice data the tasks reference | **10.4 GB**, 359 files | rsync, manifest below |
| Everything else in `paper2bench-data` | ~115 GB | leave it; raw downloads kept for provenance, no task reads them |
| `agent_runs` history | 8.9 GB | leave it; re-scoring runs on the machine that holds it |

`scripts/prepare_cluster_transfer.py` writes the file list and a manifest with sizes, so the
transfer carries a tenth of the data directory rather than all of it.

```bash
./.venv/bin/python scripts/prepare_cluster_transfer.py --checksums
rsync -av --files-from=results/cluster_transfer_files.txt / cluster:/scratch/p2b-data/
```

GitHub is not an option for the data. Individual slices exceed the 100 MB file limit, the
largest being `data232_visiumhd_08um.h5ad` at 0.84 GB, and 10.4 GB is past what Git LFS gives
you for free. The code belongs in git and the data does not.

## Then, in order

1. **Rebase the paths.** 61 of 92 task files hardcode a path under the machine that built
   them, 242 references in all. Every job would fail with file-not-found.
   ```bash
   ./.venv/bin/python scripts/check_portability.py --data-root /scratch/p2b-data --apply
   ./.venv/bin/python scripts/build_tasks.py --check-coverage    # 240 of 240
   ```

2. **Log in to claude on the submit host.** On macOS its credentials live in the Keychain, so
   there is no `~/.claude/.credentials.json` for a container to copy. On Linux the CLI writes
   that file, but only after an interactive login, which no compute node can do. Without it
   cell A fails everywhere with no obvious cause.
   ```bash
   ./.venv/bin/python scripts/check_access.py --probe
   ```

3. **Build and pin a container image.** The default is still the mutable `docker://` tag, so
   two jobs a week apart could run different environments.
   ```bash
   singularity build python-3.11-slim.sif docker://python:3.11-slim
   sha256sum python-3.11-slim.sif        # record this
   export P2B_SINGULARITY_IMAGE=$PWD/python-3.11-slim.sif
   ```

4. **Prove the containment, once.** The code fails closed correctly but has never executed:
   there is no container runtime on the Mac. Run one task and check the two properties that
   matter, that the agent can read `data/` and cannot read the repository.
   ```bash
   scripts/slurm/submit_alignment.sh --shard 1 --task sa-001-dlpfc-serial-align
   ```

5. **Then a shard.** `--shard 1 --dry-run` first; it prints the row count and the sbatch line.

## Known gaps

`mini-swe` on OpenAI is blocked: the API key is valid but has no credits. Codex is unaffected,
since it authenticates by subscription OAuth rather than the key.

`mini-swe` on Anthropic needs a pay-per-token key in `$GEN2BENCH_CONFIG_DIR/anthropic-api-key`.
The Claude Code subscription token is gated to that client and returns masked 429s through
litellm.

Codex cannot run blind on Linux at all: its read-deny profile is macOS `sandbox-exec`. The
submit script filters those rows out and the array script refuses them, so cell B stays on the
Mac.
