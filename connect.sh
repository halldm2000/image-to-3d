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
#   1. Kills any stale local listeners on the port
#   2. SSHs to the remote, starts server.py if not already running
#   3. Waits for the server to be ready
#   4. Sets up port forwarding (remote 8090 → local 8090)
#   5. Opens http://localhost:8090 in your browser
#   6. Ctrl-C to disconnect
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
RED="\033[31m"
RESET="\033[0m"

echo ""
echo -e "  ${CYAN}${BOLD}image-to-3d${RESET}  remote viewer"
echo -e "  ${DIM}Connecting to ${REMOTE_HOST}...${RESET}"
echo ""

# ── Free local port if occupied ──────────────────────────────────────────
if lsof -i ":${LOCAL_PORT}" -sTCP:LISTEN &>/dev/null; then
    echo -e "  ${YELLOW}!${RESET} Port ${LOCAL_PORT} in use locally — killing previous listener"
    lsof -ti ":${LOCAL_PORT}" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
    sleep 0.5
    echo -e "  ${GREEN}✓${RESET} Port ${LOCAL_PORT} freed"
fi

# ── Start server on remote if not running ────────────────────────────────
echo -e "  ${DIM}Starting server on ${REMOTE_HOST}...${RESET}"

REMOTE_RESULT=$(ssh "${REMOTE_HOST}" bash <<REMOTE_SCRIPT
PORT="${REMOTE_PORT}"
CONDA_ENV="${CONDA_ENV}"
REMOTE_DIR="${REMOTE_DIR}"

# Check if server is already listening
if ss -tlnp 2>/dev/null | grep -q ":\${PORT} " || \
   netstat -tlnp 2>/dev/null | grep -q ":\${PORT} "; then
    echo "ALREADY_RUNNING"
    exit 0
fi

# Find conda
CONDA_PATH=""
for p in ~/miniconda3 ~/anaconda3 ~/opt/anaconda3 /opt/conda; do
    if [ -f "\${p}/etc/profile.d/conda.sh" ]; then
        CONDA_PATH="\${p}"
        break
    fi
done

if [ -z "\${CONDA_PATH}" ]; then
    echo "NO_CONDA"
    exit 1
fi

cd "\${REMOTE_DIR}" || { echo "NO_DIR"; exit 1; }

# Write a launcher script so nohup has a clean environment
cat > /tmp/image-to-3d-start.sh <<LAUNCHER
#!/usr/bin/env bash
source "\${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "\${CONDA_ENV}" 2>/dev/null || true
cd "\${REMOTE_DIR}"
exec python server.py
LAUNCHER
chmod +x /tmp/image-to-3d-start.sh

nohup /tmp/image-to-3d-start.sh > /tmp/image-to-3d-server.log 2>&1 &
SERVER_PID=\$!

# Wait up to 15 seconds for the server to start listening
for i in \$(seq 1 15); do
    if ss -tlnp 2>/dev/null | grep -q ":\${PORT} " || \
       netstat -tlnp 2>/dev/null | grep -q ":\${PORT} "; then
        echo "STARTED"
        exit 0
    fi
    # Check if process died
    if ! kill -0 \${SERVER_PID} 2>/dev/null; then
        echo "FAILED"
        echo "---LOG---"
        tail -20 /tmp/image-to-3d-server.log 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo "TIMEOUT"
echo "---LOG---"
tail -20 /tmp/image-to-3d-server.log 2>/dev/null
exit 1
REMOTE_SCRIPT
)

# Parse the result
if echo "${REMOTE_RESULT}" | head -1 | grep -q "ALREADY_RUNNING"; then
    echo -e "  ${GREEN}✓${RESET} Server already running on ${REMOTE_HOST}:${REMOTE_PORT}"
elif echo "${REMOTE_RESULT}" | head -1 | grep -q "STARTED"; then
    echo -e "  ${GREEN}✓${RESET} Server started on ${REMOTE_HOST}:${REMOTE_PORT}"
elif echo "${REMOTE_RESULT}" | head -1 | grep -q "NO_CONDA"; then
    echo -e "  ${RED}✗${RESET} conda not found on ${REMOTE_HOST}"
    echo -e "  ${DIM}  SSH in and run: python3 setup.py${RESET}"
    exit 1
elif echo "${REMOTE_RESULT}" | head -1 | grep -q "NO_DIR"; then
    echo -e "  ${RED}✗${RESET} Directory not found on ${REMOTE_HOST}: ${REMOTE_DIR}"
    echo -e "  ${DIM}  Set REMOTE_DIR or clone the repo there first${RESET}"
    exit 1
else
    echo -e "  ${RED}✗${RESET} Server failed to start on ${REMOTE_HOST}"
    echo ""
    # Show log output after ---LOG--- marker
    if echo "${REMOTE_RESULT}" | grep -q "---LOG---"; then
        echo -e "  ${DIM}Remote log:${RESET}"
        echo "${REMOTE_RESULT}" | sed -n '/---LOG---/,$ p' | tail -n +2 | sed 's/^/    /'
    fi
    echo ""
    echo -e "  ${DIM}Full log on remote: /tmp/image-to-3d-server.log${RESET}"
    echo -e "  ${DIM}SSH in and check: ssh ${REMOTE_HOST} cat /tmp/image-to-3d-server.log${RESET}"
    exit 1
fi

echo ""

# ── SSH tunnel + open browser ────────────────────────────────────────────
echo -e "  ${DIM}Port forwarding: localhost:${LOCAL_PORT} → ${REMOTE_HOST}:${REMOTE_PORT}${RESET}"
echo -e "  ${DIM}Press Ctrl-C to disconnect${RESET}"
echo ""

# Open browser after a short delay (give tunnel time to establish)
(sleep 1 && {
    if command -v open &>/dev/null; then
        open "http://localhost:${LOCAL_PORT}" 2>/dev/null
    elif command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:${LOCAL_PORT}" 2>/dev/null
    fi
}) &

ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${REMOTE_HOST}"
