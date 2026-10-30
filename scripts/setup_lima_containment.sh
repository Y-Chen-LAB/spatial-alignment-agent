#!/usr/bin/env bash
# A Linux VM on macOS for testing the container boundary.
#
# Apptainer and Singularity build isolation from Linux namespaces, so neither runs natively on
# macOS and the container path cannot be exercised here without a VM. Lima mounts the host home
# read-only at the same path inside the guest, so the repository is visible without copying.
set -euo pipefail

VM="${1:-p2b}"

command -v limactl >/dev/null || { echo "brew install lima" >&2; exit 2; }

if ! limactl list --quiet 2>/dev/null | grep -qx "$VM"; then
    echo "creating VM $VM"
    limactl start --name="$VM" --tty=false template://ubuntu-lts
fi

limactl shell "$VM" -- bash -lc '
set -eux
sudo apt-get update -qq
sudo apt-get install -y -qq apptainer squashfuse fuse2fs uidmap

# Ubuntu 24.04 and later block unprivileged user namespaces through AppArmor, so a container
# builds and then dies at exec with "Failed to create container process: Operation not
# permitted". Clusters usually ship a setuid apptainer and are not affected; this is a
# local-VM concession, not something to copy to a shared machine.
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0

apptainer --version
'

echo
echo "verify the boundary inside the VM:"
echo "  limactl shell $VM -- bash -lc 'cd $PWD && \\"
echo "    apptainer build --force /tmp/test.sif docker://python:3.11-slim && \\"
echo "    P2B_SINGULARITY_IMAGE=/tmp/test.sif python3 scripts/verify_containment.py \\"
echo "      --mode container --container-bin apptainer'"
