#!/usr/bin/env bash
# Make sure the claude login on disk will outlive the runs about to start.
#
# Each container gets a copy of ~/.claude/.credentials.json. Its access token has a fixed
# expiry, and a copy cannot reliably refresh itself: refresh tokens rotate, so once any
# process (an interactive session, another run) has refreshed, the other copies fail with
# "OAuth session expired and could not be refreshed". So refresh on the host before
# submitting, whenever less than P2B_CLAUDE_MIN_HOURS (3) remain, with one cheap call.
set -euo pipefail
: "${P2B_CLAUDE_MIN_HOURS:=0.33}"  # only block a launch that would die at startup; a run that outlives the token fails cheaply and is retried
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hours_left() {
    "$REPO/.venv/bin/python" - <<'PY' 2>/dev/null
import json, base64, datetime, sys
from pathlib import Path
p = Path.home() / ".claude" / ".credentials.json"
d = json.loads(p.read_text()).get("claudeAiOauth") or {}
exp = d.get("expiresAt")
if not exp: print(0); sys.exit()
now = datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000
print(round((exp - now) / 3.6e6, 2))
PY
}
left=$(hours_left)
echo "claude access token: ${left}h left"
if [ "$(echo "${left:-0} < $P2B_CLAUDE_MIN_HOURS" | bc -l)" = 1 ]; then
    echo "refreshing the login with one call..."
    CLAUDECODE= claude --print --model claude-sonnet-4-6 "Reply with only the word OK." >/dev/null 2>&1 || true
    left=$(hours_left); echo "after refresh: ${left}h left"
    [ "$(echo "${left:-0} < $P2B_CLAUDE_MIN_HOURS" | bc -l)" = 1 ] && { echo "still short; log in interactively with 'claude' on a node with a terminal" >&2; exit 1; }
fi
exit 0
