# Transfer to JHPCE

Run these in order. Steps 1 and 2 are on your Mac, 3 onward on the cluster.

## 1. Check there is room

The bundle is 10.4 GB. JHPCE home directories are quota'd, and `/users/ychen5` may not have
that much free. Check before starting a transfer that will otherwise die part-way:

```bash
ssh ychen5@jhpce03.jhsph.edu 'quota -s; df -h /users/ychen5'
```

If home is tight, put the data on project or scratch space and pass that path to the setup
script in step 4 instead. Nothing assumes the home directory.

## 2. Send the data

The bundle is already staged at `~/Downloads/p2b-cluster-bundle`, built by
`scripts/bundle_cluster_data.py`. It holds exactly the 359 slices the 92 tasks reference, in
two directories that mirror the two roots the task files use.

```bash
rsync -avh --partial --progress \
  ~/Downloads/p2b-cluster-bundle/ \
  ychen5@jhpce03.jhsph.edu:/users/ychen5/p2b-data/
```

`--partial` matters: 10 GB over a home connection can drop, and without it a retry restarts
from zero. Re-running the same command resumes and skips what already arrived.

To verify afterwards:

```bash
ssh ychen5@jhpce03.jhsph.edu 'find /users/ychen5/p2b-data -name "*.h5ad" | wc -l'   # expect 359
du -sh ~/Downloads/p2b-cluster-bundle                                              # expect ~10.4 GB
```

## 3. Clone the code

```bash
ssh ychen5@jhpce03.jhsph.edu
cd /users/ychen5
git clone -b spatial-alignment https://github.com/yiqunchen/Gen2Bench.git genomics-agent
cd genomics-agent
```

## 4. Build the environment

```bash
module avail python          # find what the site offers
module load python/3.11      # or whichever 3.11+ it names
scripts/slurm/bootstrap_env.sh
```

The script picks the newest Python 3.11 or later on PATH, creates `.venv`, installs the
package, and then imports every dependency to prove the install rather than trusting pip's
exit code. Override the interpreter with `P2B_PYTHON_BIN` if the module system names it
something unusual.

Nothing here compiles: h5py, numpy, scipy and scikit-learn all ship manylinux wheels, so no
HDF5 module needs loading. If pip starts building from source, something is bypassing the
wheel index rather than a library being missing.

## 5. Wire up the data

```bash
scripts/slurm/setup_on_cluster.sh /users/ychen5/p2b-data
```

This does the two things a clone cannot: link `benchmarks/spatial_alignment/compiled_data`,
which is excluded from git, and rebase the 242 absolute slice paths across 61 task files. It
then checks that all 240 slices resolve and that the suite validates 92 of 92.

## 6. Log in to claude, on the submit host

This is the step that fails quietly if skipped. On macOS claude keeps its credentials in the
Keychain, so there is no `~/.claude/.credentials.json` for a container to copy. On Linux the
CLI writes that file, but only after an interactive login, which no compute node can perform.

```bash
claude            # log in, then exit
ls -l ~/.claude/.credentials.json
.venv/bin/python scripts/check_access.py
```

## 7. Azure OpenAI key

The GPT arm runs through Azure, not the codex CLI, until that budget is used up. Put both
halves in `~/.config/paper2bench/env`: the solver, the Azure probe and `check_access.py` all
read that path by default, so nothing else has to be exported. (`PAPER2BENCH_ENV_FILE` also
works for the solver, but `check_access.py` does not honour it and will report FAIL for a
setup that runs.)

```bash
mkdir -p ~/.config/paper2bench && chmod 700 ~/.config/paper2bench
cat > ~/.config/paper2bench/env <<'ENV'
export MOOD_OPENAI_API_KEY="<key>"
export MOOD_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"
ENV
chmod 600 ~/.config/paper2bench/env

.venv/bin/python scripts/check_access.py                                        # azure-openai: ok
.venv/bin/python scripts/check_azure_openai_access.py --deployment gpt-6-astra  # one real call
```

Deployment names are per resource, not the vendor's catalogue; `docs/CREDENTIALS.md` lists
what the MOOD resource had on 2026-09-15.

## 8. Build the container image

Build it there, not here: a `.sif` is a Linux image and macOS has no Singularity to build one
with. Build from `containers/gen2bench.def`, not from `docker://python:3.11-slim`: Azure runs
execute the generated solver inside the container, so the image needs numpy, scipy, h5py and
the rest baked in, and slim has none of them.

```bash
module avail singularity apptainer          # JHPCE: singularity/3.11.4 (default), 4.5.1, apptainer/1.5.3
module load singularity
singularity build gen2bench.sif containers/gen2bench.def
sha256sum gen2bench.sif                     # record this in containers/README.md
```

If the build refuses for lack of privileges, retry with `--fakeroot`, or with
`module load apptainer/1.5.3` and `apptainer build`. Pin the built file rather than leaving a
`docker://` tag in place: the tag is mutable, so two jobs a week apart could otherwise run
different environments.

## 9. Site variables, then the boundary check

Nothing needs exporting on JHPCE: `site_config.sh` finds the repo from its own location,
uses `gen2bench.sif` in the repo root when it exists, and the array script loads
`singularity` (then `apptainer`) itself when no container runtime is on PATH. The overrides
exist for other sites:

```bash
export P2B_SINGULARITY_IMAGE=...      # an image somewhere else
export P2B_CONTAINER_MODULE=...       # a module with a non-standard name
export P2B_MAX_CONCURRENT=1           # for a first job; the default is 8
```

Then prove the boundary on real cluster hardware, which has not been done yet:

```bash
.venv/bin/python scripts/verify_containment.py --mode container    # should say boundary holds
```

## 10. One Azure job, end to end

`scripts/run_e2e_smoke.sh` cannot drive the Azure arm: it goes through
`run_local_coding_agent.py`, whose `--agent` choices are codex, claude, gemini and mini-swe,
and it passes no container flags. For Azure the end-to-end test is one SLURM job.

```bash
scripts/slurm/submit_alignment.sh --agent azure-openai --task sa-001-dlpfc-serial-align --dry-run   # 3 rows, reps 1-3
scripts/slurm/submit_alignment.sh --agent azure-openai --task sa-001-dlpfc-serial-align
squeue -u ychen5
tail -f logs/p2b-align-*.out
```

When it finishes, open `agent_runs/cluster/<jobid>_<index>/*/manifest.json` and check that
`status` is `scored` and `blind_mode` is `container`, and that `scores.json` sits beside it.
A `failed` status carries `failure_stage` (solver or score) and the first 4000 characters of
stderr. The Azure solver writes no `leakage_status`; that audit exists only in the CLI-agent
runner.

## 11. Then the shard

```bash
export P2B_MAX_CONCURRENT=8
scripts/slurm/submit_alignment.sh --agent azure-openai --shard 1 --dry-run
scripts/slurm/submit_alignment.sh --agent azure-openai --shard 1
```

## A second cluster, for the other shard

Shards 1 and 2 are halves of the design subset with no task in common, so one cluster runs
`--shard 1` and the other `--shard 2` and the results merge by copying directories. Pull
everything from JHPCE rather than from the Mac: the data there is already the exact bundle,
and the image must be the same bytes on both sides.

On the new cluster:

```bash
# code, then environment
git clone -b spatial-alignment https://github.com/yiqunchen/Gen2Bench.git genomics-agent
cd genomics-agent
module avail python && module load python/3.11          # whatever names a 3.11+
scripts/slurm/bootstrap_env.sh

# data (6.8 GB, 359 slices) and the image (800 MB), straight from JHPCE
rsync -avh --partial --exclude='*.bak' ychen5@jhpce03.jhsph.edu:/users/ychen5/p2b-data/ ~/p2b-data/   # the compiled files here carry the 2026-09-20 leak fixes
rsync -avh --partial ychen5@jhpce03.jhsph.edu:/users/ychen5/genomics-agent/gen2bench.sif .
sha256sum gen2bench.sif     # must match containers/README.md
scripts/slurm/setup_on_cluster.sh ~/p2b-data              # the committed task paths are JHPCE's; this rewrites them
# sa-005 also needs paper2bench-data/BCA_STAligner_DR/STAligner_BCA_new_slices_Rep{1,2}.h5ad, which the
# first bundle omitted; its rows are runnable_today=False in the matrix until they are in place.

# keys: copy the files, log claude in fresh
mkdir -p ~/.config/paper2bench ~/.config/gen2bench && chmod 700 ~/.config/paper2bench ~/.config/gen2bench
rsync -avh ychen5@jhpce03.jhsph.edu:.config/paper2bench/env ~/.config/paper2bench/env
rsync -avh ychen5@jhpce03.jhsph.edu:.config/gen2bench/ ~/.config/gen2bench/
claude                       # interactive login; do not copy .credentials.json, a refreshed token on one host can log the other out
```

Site names differ, and the defaults in `site_config.sh` are JHPCE's. Look them up and put
the overrides in `~/.bashrc`:

```bash
module avail singularity apptainer      # -> P2B_CONTAINER_MODULE
sinfo -s                                # -> P2B_PARTITION ("shared" is JHPCE's name)

export P2B_PARTITION=<partition>
export P2B_RUNS_ROOT=$HOME/genomics-agent/agent_runs/cluster-<site>   # distinct from JHPCE's agent_runs/cluster
export P2B_CONTAINER_MODULE=<module>    # only if it is not simply "singularity" or "apptainer"
```

`P2B_WORKSPACE_ROOT` defaults to `$SCRATCH`, then `$TMPDIR`, then `/tmp`; set it only if none
of those is a sensible scratch there. Then the same ladder as above, with a shard-2 task for
the test (sa-018 is in shard 1; sa-017 is its small imaging sibling):

```bash
.venv/bin/python scripts/check_access.py
.venv/bin/python scripts/verify_containment.py --mode container
scripts/slurm/submit_alignment.sh --task sa-017-baristaseq-mpc-serial-align --dry-run
scripts/slurm/submit_alignment.sh --task sa-017-baristaseq-mpc-serial-align
scripts/slurm/submit_alignment.sh --shard 2 --dry-run
scripts/slurm/submit_alignment.sh --shard 2
```

Do not commit the `task.toml` files `setup_on_cluster.sh` rewrites; they carry that site's
absolute paths. To bring results together, rsync `agent_runs/cluster-<site>/` next to
`agent_runs/cluster/`; the `done/` markers travel with it, so a resubmit anywhere still skips
what scored.

## The codex arm

Cell B is parked, not blocked. Its read-deny profile used to be macOS `sandbox-exec` only;
container mode now supplies the boundary on Linux and the array script accepts codex rows.
It is left out of the submissions above because the GPT arm is running through the Azure key
first, and `check_access.py` no longer checks for a codex login for the same reason.
