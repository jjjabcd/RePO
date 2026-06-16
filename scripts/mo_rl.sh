#!/usr/bin/env bash
# MuMo multi-property RL training launcher.
#
# Starts both property API servers (admetModel_api.py, drd2Model_api.py) in the
# background if they are not already running, then launches RL training.
#
# Usage (run from repo root, with momu env active):
#   conda activate momu
#   bash scripts/mo_rl.sh [--output_dir ./output/my_run]
#
# The script forwards any unknown flags to run_RL_training.sh (e.g. --output_dir).

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# 1. Property API servers (README §2)
# ---------------------------------------------------------------------------
ADMET_PORT=10086
DRD2_PORT=10087
SERVER_LOG_DIR="${ROOT_DIR}/logs/servers"
mkdir -p "$SERVER_LOG_DIR"

_server_running() {
  # Returns 0 (true) if something is already listening on the given port.
  python -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', int(sys.argv[1])))
    sys.exit(0)
except Exception:
    sys.exit(1)
" "$1" 2>/dev/null
}

if _server_running "$ADMET_PORT"; then
  echo "[mo_rl.sh] ADMET server already running on :${ADMET_PORT}" >&2
else
  echo "[mo_rl.sh] Starting ADMET server (port ${ADMET_PORT}) ..." >&2
  nohup python multiprop_utils/admetModel_api.py \
    > "$SERVER_LOG_DIR/admetModel_api.log" 2>&1 &
  ADMET_PID=$!
  echo "[mo_rl.sh] ADMET server PID=${ADMET_PID} — log: $SERVER_LOG_DIR/admetModel_api.log" >&2
fi

if _server_running "$DRD2_PORT"; then
  echo "[mo_rl.sh] DRD2 server already running on :${DRD2_PORT}" >&2
else
  echo "[mo_rl.sh] Starting DRD2 server (port ${DRD2_PORT}) ..." >&2
  nohup python multiprop_utils/drd2Model_api.py \
    > "$SERVER_LOG_DIR/drd2Model_api.log" 2>&1 &
  DRD2_PID=$!
  echo "[mo_rl.sh] DRD2 server PID=${DRD2_PID} — log: $SERVER_LOG_DIR/drd2Model_api.log" >&2
fi

# Wait until both servers respond (up to 60 s).
echo "[mo_rl.sh] Waiting for property servers to be ready ..." >&2
for port in "$ADMET_PORT" "$DRD2_PORT"; do
  for i in $(seq 1 60); do
    if _server_running "$port"; then
      echo "[mo_rl.sh]   :${port} ready." >&2
      break
    fi
    if [[ $i -eq 60 ]]; then
      echo "[mo_rl.sh] ERROR: server on :${port} did not start within 60 s." >&2
      exit 1
    fi
    sleep 1
  done
done

# ---------------------------------------------------------------------------
# 2. RL training (README §3)
#    Machine: 2 × RTX 4090 (GPU 0, GPU 1)
#    Layout:  GPU 0 → training process (num_processes=1)
#             GPU 1 → vLLM generation engine (use_vllm: true)
# ---------------------------------------------------------------------------
echo "[mo_rl.sh] Launching MuMo RL training ..." >&2

bash scripts/run_RL_training.sh \
  --gpus       0,1 \
  --num_processes 1 \
  --entry      src/x_r1/repo.py \
  --variant    mumo \
  --config     recipes/MulProp_3B_config.yaml \
  --output_dir ./output/repo_run \
  "$@"
