#!/usr/bin/env bash
# Production deployment script for running vLLM on AMD hardware (Radeon Pro W7900)

set -euo pipefail
echo "[INFO] Initializing system environment overrides for RDNA3 (gfx1100) ISA..."

# Override HSA GFX version to map gfx1100 to RDNA3 target compatible driver configurations (gfx1100 -> 11.0.0)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Isolate model registry to Hugging Face
export VLLM_USE_MODELSCOPE=False

echo "[INFO] Aligning virtual environment using uv..."
uv sync --frozen

echo "[INFO] Installing ROCm vLLM wheel..."
uv pip install "https://wheels.vllm.ai/rocm/vllm-0.16.0-cp314-cp314-manylinux_2_35_x86_64.whl"

echo "[INFO] Starting vLLM OpenAI compatibility API server on port 8000..."
exec uv run python3 -m vllm.entrypoints.openai.api_server \
    --model google/gemma-4-31b-it \
    --quantization fp8 \
    --device rocm \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --port 8000
