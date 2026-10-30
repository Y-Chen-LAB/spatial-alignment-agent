#!/usr/bin/env bash
# Remove dead p2b workspaces and state from this node's /tmp, leaving live jobs alone.
# Run as a tiny job pinned to each node. A workspace is dead when its job is not in the
# queue; state dirs cannot be mapped to jobs, so only ones untouched for 8 h go.
echo "$(hostname): before $(df -h /tmp | tail -1 | awk '{print $4}') free"
for d in /tmp/p2b_workspaces/*/; do
    [ -d "$d" ] || continue
    id=$(basename "$d")
    if squeue -h -j "${id%_*}_${id#*_}" -o %i 2>/dev/null | grep -q .; then continue; fi
    rm -rf "$d" && echo "removed $id"
done
find /tmp -maxdepth 1 -name 'p2b-*-state-*' -mmin +480 -exec rm -rf {} + 2>/dev/null
echo "after $(df -h /tmp | tail -1 | awk '{print $4}') free"
