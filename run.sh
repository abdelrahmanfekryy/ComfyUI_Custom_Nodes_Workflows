#!/usr/bin/env bash
# Start ComfyUI with the project venv.
#
#   ./run.sh                     # http://127.0.0.1:8188 with live previews
#   ./run.sh --cpu               # any extra flags go straight to main.py
#   PORT=8189 ./run.sh           # different port
#   LISTEN=0.0.0.0 ./run.sh      # reachable from other machines on the network
#   OPEN_BROWSER=1 ./run.sh      # open the UI in the default browser once the server is up
set -euo pipefail

# ComfyUI-Manager loads as a custom node (custom_nodes/comfyui-manager), so --enable-manager is not
# used: that flag needs the separate pip package from manager_requirements.txt.

cd "$(dirname "$(readlink -f "$0")")"

PYTHON=.venv/bin/python
LISTEN="${LISTEN:-127.0.0.1}"
PORT="${PORT:-8188}"

if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PWD/$PYTHON not found. Create the venv first (uv venv, then uv pip install -r requirements.txt)." >&2
    exit 1
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "error: port $PORT is already in use (is ComfyUI already running?). Try PORT=$((PORT + 1)) ./run.sh" >&2
    exit 1
fi

if [[ "${OPEN_BROWSER:-0}" == 1 ]]; then
    url="http://127.0.0.1:$PORT"
    (
        for _ in $(seq 120); do
            if curl -s -o /dev/null "$url/system_stats"; then
                xdg-open "$url" >/dev/null 2>&1
                break
            fi
            sleep 1
        done
    ) &
fi

exec "$PYTHON" main.py \
    --listen "$LISTEN" \
    --port "$PORT" \
    --preview-method auto \
    "$@"
