#!/usr/bin/env bash
#
# Stop the remote image-to-3d server.
#
# Usage:
#   ./disconnect.sh              # uses default host
#   ./disconnect.sh spark-1      # specify SSH host
#
set -euo pipefail

REMOTE_HOST="${1:-${REMOTE_HOST:-spark-1}}"

echo ""
echo -e "  \033[2mStopping server on ${REMOTE_HOST}...\033[0m"

ssh "${REMOTE_HOST}" 'pkill -f "python server.py" 2>/dev/null && echo "  ✓ Server stopped" || echo "  · No server was running"'
echo ""
