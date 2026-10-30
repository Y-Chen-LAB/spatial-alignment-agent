# Container image for blind solving on Linux

Blind solving on macOS uses a Seatbelt read-deny profile. Linux has no equivalent, so cluster
runs need a container or they are not blind: without one the agent can read the repository,
the unsanitised slices and the scorer. `scripts/run_local_coding_agent.py --blind-mode
container` supplies the boundary, and the runner exits rather than downgrading silently.

The image needs Python 3.11. For blind Azure code-generation runs it also needs the standard
scientific stack, because the generated solver is executed inside the image. Build the supplied
image rather than using `docker://python:3.11-slim` directly:

```bash
singularity build gen2bench.sif containers/gen2bench.def
export P2B_SINGULARITY_IMAGE=$PWD/gen2bench.sif
```

On a cluster with no `/etc/subuid` entry for you (JHPCE), `singularity build --fakeroot`
fails with "no mapping entry found". Use apptainer instead, which falls back to a root-mapped
user namespace on its own; the def file already turns off apt's privilege drop, which is the
one thing that breaks in that mode:

```bash
module load apptainer/1.5.3
export APPTAINER_CACHEDIR=/fastscratch/myscratch/$USER/apptainer-cache   # keep 2 GB off home quota
apptainer build --fakeroot gen2bench.sif containers/gen2bench.def
```

Built on JHPCE 2026-09-16 with apptainer 1.5.3, 1.06 GB, containing Python 3.11.16,
claude 2.1.273, gemini 0.60.0, codex 0.154.0, mini-swe-agent 2.4.6, and the scientific
stack plus scanpy:

    7c085ca2de4f05432282bb7026b353832606665507c84b128f5eb45d0b7deeae  gen2bench.sif

(An earlier build the same day without codex, 07db2300…d7ae, ran the first Azure one-shot
and claude/gemini test jobs.)

Copy that file to any second cluster rather than rebuilding, so both halves of a shard split
run the same bytes.

## Rebuilding instead

The def pins every CLI and library to the versions in that image, so a rebuild gives the same
software even though the bytes differ (the `python:3.11-slim` base and Debian packages are
whatever the day serves). What a rebuild needs from the site:

| Need | Check | If missing |
| --- | --- | --- |
| outbound network on the build host | `curl -sI https://deb.nodesource.com \| head -1` | build on a login or transfer node that has it |
| an `/etc/subuid` entry, or unprivileged user namespaces | `grep $USER /etc/subuid; unshare -U -r true` | neither: copy the image, or build on any Linux box with Docker and `apptainer build gen2bench.sif docker-daemon://…` |
| 3 GB of scratch for the cache | | `export APPTAINER_CACHEDIR=<scratch>/apptainer-cache APPTAINER_TMPDIR=<scratch>/apptainer-tmp` |

```bash
scripts/slurm/build_image.sh     # picks the tool, caches on scratch, builds, checks the pins, prints the hash
```

Record the hash it prints here, next to the JHPCE one.

Pin a built `.sif` and record its checksum rather than relying on the `docker://` tag, which
is mutable and would let the environment drift between runs in the same campaign:

```bash
sha256sum gen2bench.sif
```

Apptainer is accepted and takes the same flags; pass `--container-bin apptainer` or set
`P2B_SINGULARITY_BIN`. On a cluster that provides it as a module, set `P2B_CONTAINER_MODULE`
(for example `singularity/3.7.1`) and the array script loads it before running.

The sibling perturbation benchmark used Singularity 3.7.1 with an image recorded as
`650234e13a70b768285aaf6260da1a081879eefaaedbc7dcc1e138ddb5b7a5c1`. Reuse that image if you
want the two benchmarks byte-identical; it is not distributed with the code.

## What the container may see

`singularity exec --containall --cleanenv --no-home` plus, and only, these binds:

| Mount | Mode | Why |
| --- | --- | --- |
| the workspace | rw | the sanitised slices and the submission directory |
| `/agent-state` | rw | HOME, TMPDIR, pip cache and the CLI's own config |
| `/run/secrets/*` | ro | one file per credential, listed in `containment.secret_routes` |
| the agent runtime | ro | only when a prepared environment is passed |

The repository, the unsanitised slices, the split manifests and the scorer are absent. That
is the property the Seatbelt profile was approximating, now enforced rather than audited.

Credentials are copied into private per-run state, never bound from the interactive config
directory, so histories and settings do not enter the run. Under SLURM that state goes on
node-local `SLURM_TMPDIR`, and a misconfigured `SLURM_TMPDIR` is an error rather than a
silent fall back to shared storage.

## The image needs the agent CLI inside it

`--containall` means the container sees nothing of the host, so an image that is only
`python:3.11-slim` has no `claude`, `codex`, `gemini` or `mini` and every job fails with "not
found". There are two ways to solve it, and the runner refuses to start if neither is in place.

**Bake them in**, which is the sturdier option because a job then depends on nothing outside
the image:

```bash
singularity build gen2bench.sif containers/gen2bench.def
sha256sum gen2bench.sif
export P2B_SINGULARITY_IMAGE=$PWD/gen2bench.sif
```

**Or bind a host install** read-only with `--agent-runtime <prefix>` or `P2B_AGENT_RUNTIME`.
The prefix is the parent of the directory holding the binary, not the `bin` directory itself:
`claude` and `gemini` are Node programs and need the `lib/` or `node_modules/` beside `bin/`.
Without an override the runner infers the prefix from whichever binary is on PATH.

`codex` is the exception. It is distributed as a native binary tied to a ChatGPT login rather
than through npm, so `gen2bench.def` does not install it; use `--agent-runtime` pointing at a
host install, or keep the codex arm on macOS.

## Testing containment on macOS

Apptainer and Singularity are Linux-only: they build isolation out of Linux namespaces, and
Homebrew's formula says `Required: Linux` outright. There is no native macOS build, and the
old Singularity Desktop for Mac was discontinued. So a local test needs a Linux VM.

Lima is the lightest way to get one:

```bash
brew install lima
limactl start --name=p2b template://ubuntu-lts     # a few minutes, downloads an image
limactl shell p2b

# inside the VM
sudo apt-get update && sudo apt-get install -y apptainer
apptainer --version
```

Lima mounts your home directory into the VM by default, so the repo and the data bundle are
visible without copying, and `P2B_SINGULARITY_BIN=apptainer` makes the runner use it.

Whether this is worth doing depends on what you want to learn. The VM proves the plumbing:
that the container starts, that the agent CLI is reachable inside it, that the workspace is
writable and the repository is not. It does not prove the cluster works, because the module
system, the filesystem layout and `SLURM_TMPDIR` are all different there. If the cluster is
where the runs will happen, one real single-task job there tests more than a local VM does and
takes less setup.

### Verifying the boundary

`scripts/verify_containment.py` runs a probe through the same isolation a real run uses and
asserts the two properties that make a run blind: the agent can read its workspace, and cannot
read the repository, which holds the unsanitised slices, the task files and the scorer. The
probe is `/bin/sh`, so no model is involved and the test costs nothing.

```bash
./.venv/bin/python scripts/verify_containment.py                  # this platform
./.venv/bin/python scripts/verify_containment.py --mode both      # seatbelt and container
```

`scripts/setup_lima_containment.sh` builds a Linux VM so the container path can be exercised
from macOS. Verified on 2026-09-15: the Seatbelt boundary holds on macOS, and the container
boundary holds under Apptainer 1.4.5 on Ubuntu, both reporting workspace readable and
repository blocked.

One VM-only wrinkle: Ubuntu 24.04 and later block unprivileged user namespaces through
AppArmor, so a container builds and then dies at exec with "Failed to create container
process: Operation not permitted". The setup script clears
`kernel.apparmor_restrict_unprivileged_userns`. Clusters usually ship a setuid apptainer and
are unaffected, so do not carry that change onto a shared machine.
