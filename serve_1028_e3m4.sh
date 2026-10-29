#!/usr/bin/env bash
# serve_1028_e3m4.sh — (FP8-e3m4 attn variant) Qwen3.8-Flash-Next NVFP4 (W4A16 weights + attn FP8 + MTP) on RTX PRO 6000 Blackwell.
# Transcribed verbatim from the 2026-10-23 interactive launch; one flag per line so nothing
# silently fuses together the way a "\<space> continuation in a pasted block does.
set -uo pipefail
cd /home/lovedheart/sglang-fork-dev

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_HC_NORM_FOLD=1
export SGLANG_HC_MIX_TRITON=1
export SGLANG_QSA_USE_FP8_INDEXER=1
export SGLANG_QSA_ATTN_FP4=1
export SGLANG_QSA_FUSED_FP4_GATHER=1
export SGLANG_QSA_FUSED_SPARSE_DECODE=1
export SGLANG_QSA_VECTOR_PREFILL_GATHER=1
export SGLANG_QSA_PREFILL_PACKED_KV=1
export SGLANG_QSA_PREFILL_MULTI_WARP=1
export SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
export SGLANG_MM_FEATURE_CACHE_MB=96
export SGLANG_PPTRACE=0
export SGLANG_SM12X_W4A4_PREFILL_MIN_TOKENS=256

exec .venv/bin/sglang serve \
  --model-path /media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnE3M4b128-mtpV4 \
  --served-model-name Qwen3.8-Flash-Next \
  --host 0.0.0.0 \
  --port 8070 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 16384 \
  --mem-fraction-static 0.918 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size 18 \
  --mamba-max-states-per-path 2 \
  --mamba-track-interval 256 \
  --enable-int8-mamba-checkpoint \
  --page-size 64 \
  --mamba-backend triton \
  --kv-cache-dtype nvfp4 \
  --schedule-policy hrrn \
  --allow-auto-truncate \
  --enable-cache-report \
  --enable-metrics \
  --sleep-on-idle \
  --sampling-backend flashinfer \
  --ple-offload-embedding \
  --moe-runner-backend sm12x_nvfp4_bf16 \
  --reasoning-parser auto \
  --tool-call-parser auto \
  --attention-backend triton \
  --decode-attention-backend trtllm_mha \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-verify-backend flashinfer \
  --enable-linear-replayssm-spec \
  --prefill-kv-cache-dequant-dtype fp8_e4m3 \
  --enable-session-radix-cache \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-draft-kv-cache-dtype fp8_e4m3 \
  --speculative-moe-runner-backend sm12x_nvfp4_bf16 \
  --mm-feature-transport cpu \
  --mm-attention-backend flashinfer_cudnn \
  --image-processor-backend pil \
  --mm-process-config '{"image":{"max_pixels":3686400,"min_pixels":200704}}' \
  --qsa-indexer-dtype fp8_e4m3 \
  --enable-fp8-lm-head \
  --disable-prefill-cuda-graph \
  --enable-fp8-shared-expert \
  --enable-fp8-shared-expert-mtp
