#!/usr/bin/env bash
# Ling-3.0-tiny NVFP4 -> w2a16_offload (TS4 CPU decode + GPU prefill arm)
set -x
cd /home/lovedheart/sglang-fork-dev
export SGLANG_W2A16_CPU_THREADS=${SGLANG_W2A16_CPU_THREADS:-8}
export SGLANG_W2A16_PREFILL_MIN_TOKENS=${SGLANG_W2A16_PREFILL_MIN_TOKENS:-16}
export SGLANG_W2A16_GPU_MODE=${SGLANG_W2A16_GPU_MODE:-moet}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -m sglang.launch_server \
  --model-path /media/lovedheart/models/Ling-3.0-tiny-NVFP4-w4a4-4o6 \
  --moe-runner-backend w2a16_offload \
  --disable-shared-experts-fusion \
  --mem-fraction-static 0.75 --max-mamba-cache-size 24 --max-total-tokens 4096 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --disable-prefill-cuda-graph \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --host 127.0.0.1 --port 30021
