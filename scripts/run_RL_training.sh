#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# CUDA / DeepSpeed environment setup (runs on every server after conda env create)
#
# nvidia-cuda-nvcc-cu12  installs nvcc  under site-packages/nvidia/cuda_nvcc/bin/
# nvidia-curand-cu12     installs libcurand under site-packages/nvidia/curand/lib/
# DeepSpeed expects both at $CUDA_HOME/bin/nvcc and $CONDA_PREFIX/lib/libcurand.so
# → auto-symlink once if missing so no manual steps are needed on a fresh env.
# ---------------------------------------------------------------------------
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"

  # --- nvcc ---
  if [[ ! -f "${CONDA_PREFIX}/bin/nvcc" ]]; then
    _nvcc="$(python -c "
import os
try:
    import nvidia.cuda_nvcc as m
    p = os.path.join(os.path.dirname(m.__file__), 'bin', 'nvcc')
    print(p) if os.path.isfile(p) else None
except ImportError:
    pass
" 2>/dev/null || true)"
    if [[ -n "${_nvcc:-}" ]]; then
      ln -sf "$_nvcc" "${CONDA_PREFIX}/bin/nvcc"
      echo "[run_RL_training.sh] Linked nvcc: ${CONDA_PREFIX}/bin/nvcc → $_nvcc" >&2
    else
      echo "[run_RL_training.sh] WARNING: nvcc not found — install nvidia-cuda-nvcc-cu12" >&2
    fi
  fi

  # --- libcurand (required by DeepSpeed cpu_adam JIT build) ---
  if [[ ! -f "${CONDA_PREFIX}/lib/libcurand.so" ]]; then
    _curand="$(python -c "
import os
try:
    import nvidia.curand as m
    p = os.path.join(os.path.dirname(m.__file__), 'lib', 'libcurand.so.10')
    print(p) if os.path.isfile(p) else None
except ImportError:
    pass
" 2>/dev/null || true)"
    if [[ -n "${_curand:-}" ]]; then
      ln -sf "$_curand" "${CONDA_PREFIX}/lib/libcurand.so.10" 2>/dev/null || true
      ln -sf "${CONDA_PREFIX}/lib/libcurand.so.10" "${CONDA_PREFIX}/lib/libcurand.so" 2>/dev/null || true
      echo "[run_RL_training.sh] Linked libcurand in ${CONDA_PREFIX}/lib/" >&2
    else
      echo "[run_RL_training.sh] WARNING: libcurand not found — install nvidia-curand-cu12" >&2
    fi
  fi
fi

# rewards.py initialises OpenAI client at import time with api_key=""; openai>=2 rejects empty
# keys unless OPENAI_API_KEY is set. Set a dummy so init succeeds (mumo never calls this reward).
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# Defaults (override via env vars or CLI flags)
# export WANDB_MODE="${WANDB_MODE:-offline}"
GPUS="${GPUS:-0,1}"          # machine has 2 GPUs (index 0,1)
ENTRY="${ENTRY:-src/x_r1/repo.py}"
VARIANT="${VARIANT:-}"
CONFIG="${CONFIG:-recipes/OpenMolIns_3B_config.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-recipes/zero3.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"  # use_vllm=true takes 1 GPU; leave 1 GPU for vLLM
PORT="${PORT:-29500}"
LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"

usage() {
  cat <<'EOF'
Run RL training (GRPO / RePO variants) via accelerate.

Usage:
  bash scripts/run_RL_training.sh [--gpus 0,1] [--entry src/x_r1/grpo.py] [--config recipes/XXX.yaml]
                                 [--variant default|mumo|pure|noisy_demo|random_mask]
                                 [--accelerate_config recipes/zero3.yaml] [--num_processes 1]
                                 [--output_dir /path/to/save_dir]
                                 [--port 29500] [--log_dir ./logs] [--log_file ./logs/your.log]

Examples:
  # Default OpenMolIns 3B GRPO training
  bash scripts/run_RL_training.sh

  # Multi-property training
  bash scripts/run_RL_training.sh --entry src/x_r1/grpo.py --variant mumo --config recipes/MulProp_bbbp-drd2-qed_3B_config.yaml
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --gpus) GPUS="${2:?}"; shift 2 ;;
    --entry) ENTRY="${2:?}"; shift 2 ;;
    --variant) VARIANT="${2:?}"; shift 2 ;;
    --config) CONFIG="${2:?}"; shift 2 ;;
    --accelerate_config) ACCELERATE_CONFIG="${2:?}"; shift 2 ;;
    --num_processes) NUM_PROCESSES="${2:?}"; shift 2 ;;
    --port) PORT="${2:?}"; shift 2 ;;
    --log_dir) LOG_DIR="${2:?}"; shift 2 ;;
    --log_file) LOG_FILE="${2:?}"; shift 2 ;;
    --output_dir) OUTPUT_DIR="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"

if [[ -z "${LOG_FILE}" ]]; then
  entry_base="$(basename "$ENTRY" .py)"
  cfg_base="$(basename "$CONFIG" .yaml)"
  variant_tag="${VARIANT:-default}"
  base="${LOG_DIR}/${entry_base}_${variant_tag}_${cfg_base}"
  candidate="${base}.log"
  if [[ -e "$candidate" ]]; then
    i=2
    while [[ -e "${base}_${i}.log" ]]; do
      i=$((i + 1))
    done
    candidate="${base}_${i}.log"
  fi
  LOG_FILE="$candidate"
fi

mkdir -p "$(dirname "$LOG_FILE")"

echo "[run_RL_training.sh] GPUS=$GPUS" >&2
echo "[run_RL_training.sh] ENTRY=$ENTRY" >&2
echo "[run_RL_training.sh] VARIANT=$VARIANT" >&2
echo "[run_RL_training.sh] CONFIG=$CONFIG" >&2
echo "[run_RL_training.sh] ACCELERATE_CONFIG=$ACCELERATE_CONFIG" >&2
echo "[run_RL_training.sh] NUM_PROCESSES=$NUM_PROCESSES PORT=$PORT" >&2
echo "[run_RL_training.sh] Logging to: $LOG_FILE" >&2

EXTRA_ARGS=()
if [[ -n "$VARIANT" ]]; then
  EXTRA_ARGS+=(--variant "$VARIANT")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output_dir "$OUTPUT_DIR")
fi

# expandable_segments reduces allocator fragmentation, which is PyTorch's own OOM recommendation
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="$GPUS" ACCELERATE_LOG_LEVEL="$ACCELERATE_LOG_LEVEL" \
  accelerate launch \
    --config_file "$ACCELERATE_CONFIG" \
    --main_process_port "$PORT" \
    --num_processes "$NUM_PROCESSES" \
    "$ENTRY" --config "$CONFIG" "${EXTRA_ARGS[@]}" \
    >"$LOG_FILE" 2>&1

echo "[run_RL_training.sh] Done. Log: $LOG_FILE" >&2