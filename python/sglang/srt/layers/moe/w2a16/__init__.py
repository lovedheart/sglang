"""w2a16_offload: TS4 2-bit MoE engine with CPU decode + GPU prefill.

Routed expert weights are quantized once (NVFP4 checkpoint -> TS4 2-bit
planes, ``transfer.py``) and live in an on-disk archive that the CPU FFN
kernel (``csrc/cpu_ts4_a16.cpp``, AVX-512 + OpenMP) maps directly.  The GPU
prefill arm runs the same planes on the SM12x GPU (Triton dequant grouped
GEMM, or the vLLM-Moet cubin arm).  Decode batches stay on the host: a
pinned-staging D2H + the CPU kernel + H2D, expressed as a
``cuLaunchHostFunc`` host node so decode CUDA graphs replay it.

Vendored from ~/Documents/ig_moe (CONTRACT.md v1 semantics); this package is
self-contained — ig_moe is not imported at runtime.
"""
