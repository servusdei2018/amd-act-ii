#!/usr/bin/env bash
# Production deployment script for running vLLM on AMD hardware (Radeon Pro W7900)

set -euo pipefail
echo "[INFO] Initializing system environment overrides for RDNA3 (gfx1100) ISA..."

# Override HSA GFX version to map gfx1100 to RDNA3 target compatible driver configurations (gfx1100 -> 11.0.0)
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-"11.0.0"}

# Isolate model registry to Hugging Face
export VLLM_USE_MODELSCOPE=${VLLM_USE_MODELSCOPE:-"False"}

# Support environment variable overrides for vllm settings
MODEL=${MODEL:-"cyankiwi/gemma-4-31B-it-AWQ-4bit"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"32768"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"8"}
VLLM_PORT=${VLLM_PORT:-"8000"}

# Only run uv sync if not running in pre-built container environment
if [ "${CONTAINER_PREBUILT:-false}" != "true" ]; then
    echo "[INFO] Aligning virtual environment using uv..."
    uv sync --frozen
fi

# Helper to terminate all background processes on exit/signal
cleanup() {
    echo "[INFO] Cleaning up background processes..."
    kill "$VLLM_PID" "$GATEWAY_PID" 2>/dev/null || true
    exit 0
}

# Trap termination signals
trap cleanup EXIT INT TERM

echo "[INFO] Starting vLLM OpenAI compatibility API server on port ${VLLM_PORT}..."
uv run python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --port "${VLLM_PORT}" &
VLLM_PID=$!

echo "[INFO] Starting Triduum Gateway (Router)..."
uv run python3 main.py &
GATEWAY_PID=$!

# Wait for any of the background processes to terminate
wait -n

echo "[ERROR] A service has exited prematurely. Terminating execution."
