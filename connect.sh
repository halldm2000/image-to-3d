#!/usr/bin/env bash
#
# Connect to a remote DGX running image-to-3d and open the viewer locally.
#
# Usage:
#   ./connect.sh                  # uses default host from REMOTE_HOST or config
#   ./connect.sh spark-1          # specify SSH host
#   ./connect.sh user@10.0.0.5    # full SSH target
#
# What it does:
#   1. SSHs to the remote, starts server.py if not already running
#   2. Sets up port forwarding (remote 8090 → local 8090)
#   3. Opens http://localhost:8090 in your browser
#   4. Ctrl-C to disconnect
#
set -euo pipefail

REMOTE_HOST="${1:-${REMOTE_HOST:-spark-1}}"
REMOTE_PORT="${REMOTE_PORT:-8090}"
LOCAL_PORT="${LOCAL_PORT:-8090}"
REMOTE_DIR="${REMOTE_DIR:-~/PROJECTS/image-to-3d}"
CONDA_ENV="image-to-3d"

BOLD="\033[1m"
DIM="\033[2m"
CYAN="\033[36m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

echo ""
echo -e "  ${CYAN}${BOLD}image-to-3d${RESET}  remote viewer"
echo -e "  ${DIM}Connecting to ${REMOTE_HOST}...${RESET}"
echo ""

# ── Check if local port is already in use ────────────────────────────────
if lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN &>/dev/null; then
    echo -e "  ${YELLOW}!${RESET} Port ${LOCAL_PORT} already in use locally"
    echo -e "  ${DIM}  If that's a previous session, kill it or use: LOCAL_PORT=8091 ./connect.sh${RESET}"
    echo ""
fi

# ── Start server on remote if not running ────────────────────────────────
echo -e "  ${DIM}Starting server on ${REMOTE_HOST}...${RESET}"
ssh "${REMOTE_HOST}" bash -s "${REMOTE_DIR}" "${CONDA_ENV}" "${REMOTE_PORT}" <<'REMOTE_SCRIPT'
REMOTE_DIR="$1"
CONDA_ENV="$2"
PORT="$3"

# Check if server is already running
if ss -tlnp 2>/dev/null | grep -q ":${PORT} " || \
   netstat -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "SERVER_ALREADY_RUNNING"
    exit 0
fi

# Find conda
for CONDA_PATH in ~/miniconda3 ~/anaconda3 ~/opt/anaconda3 /opt/conda; do
    if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_PATH}/etc/profile.d/conda.sh"
        break
    fi
done

cd "${REMOTE_DIR}" || exit 1

# Start server in background, detached from this SSH session
nohup bash -c "
    source ${CONDA_PATH}/etc/profile.d/conda.sh 2>/dev/null
    conda activate ${CONDA_ENV} 2>/dev/null
    python server.py
" > /tmp/image-to-3d-server.log 2>&1 &

echo "SERVER_STARTED (pid $!)"
sleep 1
REMOTE_SCRIPT

echo -e "  ${GREEN}✓${RESET} Server running on ${REMOTE_HOST}:${REMOTE_PORT}"
echo ""

# ── Open browser ─────────────────────────────────────────────────────────
echo -e "  ${GREEN}✓${RESET} Opening ${CYAN}http://localhost:${LOCAL_PORT}${RESET}"
echo ""

# Open browser (macOS or Linux)
if command -v open &>/dev/null; then
    open "http://localhost:${LOCAL_PORT}" 2>/dev/null &
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:${LOCAL_PORT}" 2>/dev/null &
fi

# ── SSH tunnel (foreground — Ctrl-C to stop) ─────────────────────────────
echo -e "  ${DIM}Port forwarding: localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT}${RESET}"
echo -e "  ${DIM}Press Ctrl-C to disconnect${RESET}"
echo ""

ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${REMOTE_HOST}"
