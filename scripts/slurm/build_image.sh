#!/usr/bin/env bash
# Build the agent container image on a cluster, rootless, and check what came out.
#   scripts/slurm/build_image.sh              # -> gen2bench.sif in the repo root
#   scripts/slurm/build_image.sh other.sif
#
# Picks apptainer if the site has it (it builds without an /etc/subuid entry; singularity
# does not), keeps the multi-GB build cache off the home quota, and after the build checks
# that the CLI and library versions inside match the pins in containers/gen2bench.def.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$REPO/gen2bench.sif}"
DEF="$REPO/containers/gen2bench.def"

# Container tool: prefer apptainer, fall back to singularity, loading a module if needed.
if ! command -v apptainer >/dev/null && ! command -v singularity >/dev/null; then
    if type module >/dev/null 2>&1; then
        module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    fi
fi
BIN=$(command -v apptainer || command -v singularity || true)
[ -n "$BIN" ] || { echo "no apptainer or singularity on PATH; check 'module avail apptainer singularity'" >&2; exit 1; }
echo "using $BIN ($("$BIN" --version))"

# Cache and temp: 3 GB or so, transient. Scratch if the site has one, else /tmp.
SCRATCH_DIR=""
for c in "${SCRATCH:-}" "/fastscratch/myscratch/$USER" "/scratch/$USER" "${TMPDIR:-}" "/tmp/$USER"; do
    [ -n "$c" ] && mkdir -p "$c" 2>/dev/null && [ -w "$c" ] && { SCRATCH_DIR="$c"; break; }
done
export APPTAINER_CACHEDIR="$SCRATCH_DIR/apptainer-cache" APPTAINER_TMPDIR="$SCRATCH_DIR/apptainer-tmp"
export SINGULARITY_CACHEDIR="$APPTAINER_CACHEDIR" SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
echo "cache in $SCRATCH_DIR"

# Never overwrite an image file: /users is a network filesystem, and replacing a file that
# running containers have open leaves their mapped pages stale, which killed every running
# agent with SIGBUS the one time it was tried. Each build gets its own file, and $OUT is a
# symlink that new jobs follow; a running job keeps the file it opened.
STAMP="$(date +%Y%m%d-%H%M%S)"
VERSIONED="${OUT%.sif}-$STAMP.sif"
echo "building $VERSIONED from $DEF ..."
"$BIN" build --fakeroot "$VERSIONED" "$DEF"
if [ -e "$OUT" ] && [ ! -L "$OUT" ]; then
    echo "$OUT is a regular file; leaving it in place as ${OUT%.sif}-previous.sif" >&2
    mv "$OUT" "${OUT%.sif}-previous.sif"
fi
ln -sfn "$(basename "$VERSIONED")" "$OUT"
echo "$OUT -> $(readlink "$OUT")"

echo
echo "checking versions inside the image against the pins in the def:"
"$BIN" exec "$OUT" bash -c '
    ok=1
    check() { if [ "$2" = "$3" ]; then echo "  [ok  ] $1 $2"; else echo "  [FAIL] $1 got $2, def pins $3"; ok=0; fi; }
    check python  "$(python -c "import sys; print(sys.version.split()[0])")" 3.11.16
    check claude  "$(claude --version 2>/dev/null | cut -d" " -f1)" 2.1.273
    check gemini  "$(gemini --version 2>/dev/null | head -1)" 0.60.0
    check codex   "$(codex --version 2>/dev/null | awk "{print \$2}")" 0.154.0
    check mini    "$(mini --version 2>/dev/null | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")" 2.4.6
    for p in numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 h5py==3.16.0 pandas==2.3.3 anndata==0.12.19 scanpy==1.11.5; do
        check "${p%%==*}" "$(pip show "${p%%==*}" 2>/dev/null | awk "/^Version:/{print \$2}")" "${p##*==}"
    done
    [ $ok = 1 ]
' 2>/dev/null | grep -v gocryptfs || { echo "version check failed; see above" >&2; exit 1; }

echo
sha256sum "$VERSIONED"
echo "record that hash in containers/README.md. $OUT now points at it; older images stay until"
echo "no job started before this build is still running, then they can be deleted."
