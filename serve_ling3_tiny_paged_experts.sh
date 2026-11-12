#!/usr/bin/env bash
# serve_ling3_tiny_paged_experts.sh — Ling-3.0-tiny NVFP4 (W4A16 experts) under paged experts
# on the RTX PRO 6000, co-tenant with the resident :8070 server.
# Verified 2026-11-12 on branch feature/ig-moe-paged-experts (base 118b80c886):
#   weights 2.33 GB, paged K=33/128 (0.94 GiB tables, 3.7 GiB pinned host store),
#   /generate sane. One flag per line; do not fuse lines.
set -uo pipefail
cd /home/lovedheart/sglang-fork-dev

PORT=${PORT:-8071}

exec .venv/bin/sglang serve \
  --model-path /media/lovedheart/models/Ling-3.0-tiny-NVFP4-w4a4-4o6 \
  --served-model-name Ling-3.0-tiny-NVFP4 \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --enable-paged-experts \
  --disable-shared-experts-fusion \
  --moe-runner-backend marlin \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --mem-fraction-static 0.30 \
  --max-mamba-cache-size 8 \
  --max-running-requests 4 \
  --max-total-tokens 4096 \
  --context-length 4096

# Notes (compat gates found by running it, paged_experts/method.py:486-550):
#  --disable-shared-experts-fusion: Ling fuses shared experts by default; paged experts refuses.
#  --moe-runner-backend marlin: default resolves to flashinfer_trtllm, unsupported; only
#      triton or marlin runners are accepted (runners.py:299-314).
#  CUDA graphs: prefill must be breakable or disabled, decode full or disabled.
#  --mem-fraction-static >= ~0.21 with :8070 up (~11.6 GB avail); the KV/mamba pools need
#      the rest of the budget. --max-mamba-cache-size caps the hybrid linear-attn pool.
# Optional: --paged-experts-enable-disk caches the repacked experts under
#  ~/.cache/sglang/paged_experts (3.7 GiB) so later starts skip load+repack;
#  --paged-experts-num-resident K pins the resident-expert count instead of auto-sizing.
# ig_moe (Vulkan iGPU) integration for this path: see digest/tasks/
#  ig-moe-paged-experts-integration.md — ExpertFormat.runner_method is the hook.
