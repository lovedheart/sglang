"""FP8 e3m4 blockwise-128 vision-tower MLP weights (SM120).

The --enable-mm-dtype fp8_e3m4 flag repacks the visual tower MLP projections
(blocks.*.mlp.linear_fc1/2, merger.linear_fc1/2 and the deepstack mergers)
from BF16 into e3m4 codes (1+3+4, bias 3, max 30) with 128x128 block scales at
load time, then releases the BF16 storage. The packed bytes declare themselves
e3m4 (e4m3_only = False) so the block-FP8 linear routes them to the sm120 e3m4
kernels (GEMV at M=1, patched-QMMA tiles above).

Only fc1/fc2 are packed: forward-error emulation on the Qwen3.6 / Flash-Next ViT
measured 29.5 dB / cos 0.9994 merger-output error for this 67% of the matmul
bytes, while qkv/proj sit in the residual feedback loop and want the extra
mantissa bit. The 4304-wide MLP dimension is zero-padded to 4352 = 34 x 128 so
both operands satisfy the 128-block scale contract; the padding is numerically
inert (gelu_tanh(0) == 0) and sliced away in the MLP forward.
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.layers.fp8_lm_head import FP8_E3M4_MAX, _e3m4_tables, _encode_e3m4
from sglang.srt.utils.device import is_sm120_supported

logger = logging.getLogger(__name__)

MM_E3M4_BLOCK = 128


def _pad_to_block(t: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    """Zero-pad dim up to a multiple of the scale block; returns (tensor, pad)."""
    n = t.shape[dim]
    pad = (-n) % MM_E3M4_BLOCK
    if pad == 0:
        return t, 0
    shape = list(t.shape)
    shape[dim] = pad
    return torch.cat([t, torch.zeros(shape, dtype=t.dtype, device=t.device)], dim), pad


def _e3m4_pack_2d(weight: torch.Tensor):
    """(e3m4 codes [n_pad, k_pad], scales [n_pad/128, k_pad/128] fp32)."""
    w, _ = _pad_to_block(weight, 0)
    w, _ = _pad_to_block(w, 1)
    n, k = w.shape
    codes = torch.empty(n, k, dtype=torch.uint8, device=weight.device)
    scales = torch.empty(
        n // MM_E3M4_BLOCK,
        k // MM_E3M4_BLOCK,
        dtype=torch.float32,
        device=weight.device,
    )
    # Row chunks bound the fp64 peak next to the already-loaded model.
    for n0 in range(0, n, 8192):
        rows = min(8192, n - n0)
        blocks = w[n0 : n0 + rows].double().view(rows // 128, 128, k // 128, 128)
        amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
        s = amax / FP8_E3M4_MAX
        codes[n0 : n0 + rows] = _encode_e3m4(blocks / s).reshape(rows, k)
        scales[n0 // 128 : (n0 + rows) // 128] = s.squeeze((1, 3)).float()
    return codes.view(torch.float8_e4m3fn), scales


def _vision_mlps(visual) -> list:
    mods = []
    for b in getattr(visual, "blocks", None) or []:
        if getattr(b, "mlp", None) is not None:
            mods.append(b.mlp)
    merger = getattr(visual, "merger", None)
    if merger is not None:
        mods.append(merger)
    mods.extend(getattr(visual, "deepstack_merger_list", None) or [])
    return [m for m in mods if getattr(m, "linear_fc1", None) is not None]


@torch.no_grad()
def quantize_vision_mlp_to_e3m4_block128(visual) -> tuple[int, int, int]:
    """Pack every vision MLP fc1/fc2 to e3m4 block-128.

    Returns (tensors packed, BF16 bytes dropped, bytes kept by the packing).
    """
    if not is_sm120_supported():
        logger.warning(
            "--enable-mm-dtype fp8_e3m4 is SM120-only; the vision tower stays BF16"
        )
        return 0, 0, 0
    packed = bf16_bytes = packed_bytes = 0
    for mlp in _vision_mlps(visual):
        for name in ("linear_fc1", "linear_fc2"):
            lin = getattr(mlp, name, None)
            if lin is None or getattr(lin, "weight", None) is None:
                continue
            w = lin.weight
            if w.dtype == torch.float8_e4m3fn:
                continue  # already packed, or shipped packed by the checkpoint
            if w.dtype not in (torch.bfloat16, torch.float16) or w.ndim != 2:
                logger.warning(
                    "Skipping mm e3m4 pack of %s: %s %s", name, w.dtype, tuple(w.shape)
                )
                continue
            group = getattr(lin, "tp_group", None)
            ws = getattr(group, "world_size", None)
            if ws is None and group is not None:
                ws = torch.distributed.get_world_size(group=group)
            if ws is not None and ws > 1:
                logger.warning(
                    "Skipping mm e3m4 pack of %s: the block-128 scale layout is defined "
                    "on the full [N, K] weight, which a sharded vision-TP copy lacks",
                    name,
                )
                continue
            n0, k0 = w.shape
            bf16_bytes += w.numel() * w.element_size()
            codes, scales = _e3m4_pack_2d(w)
            n_pad, k_pad = codes.shape
            del w
            lin.weight = torch.nn.Parameter(codes, requires_grad=False)
            # The container is float8_e4m3fn; this says the bytes are e3m4 codes.
            lin.weight.e4m3_only = False
            lin.register_buffer("weight_scale_inv", scales, persistent=False)
            # (input pad = K slack for RowParallel, output pad = N slack for ColumnParallel)
            lin._e3m4_pad = (k_pad - k0, n_pad - n0)
            packed += 1
            packed_bytes += codes.numel() + scales.numel() * 4
    if packed:
        torch.cuda.empty_cache()
    return packed, bf16_bytes, packed_bytes


def _dequant_e3m4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 weights from e3m4 codes + block-128 scales (fallback path only)."""
    pos, _ = _e3m4_tables()
    u = codes.view(torch.uint8).to(torch.int32)
    sign = torch.where(u & 128 > 0, -1.0, 1.0)
    vals = sign * pos[(u & 127).clamp(max=126).cpu()].to(u.device).float()
    n, k = codes.shape
    v = vals.view(n // 128, 128, k // 128, 128)
    return (v * scales[:, None, :, None]).reshape(n, k)


def mm_e3m4_padded_fc(linear, x: torch.Tensor, bias) -> torch.Tensor:
    """Run one packed vision fc: e3m4 block GEMM, drop the padding, add the bias.

    x is this projection activation; a RowParallelLinear input pad is sliced off
    first, an output pad off the result. Returns None when linear is not a packed
    e3m4 fc, so the caller keeps its own path.
    """
    from sglang.srt.layers.quantization.fp8_utils import (
        _e3m4_block_linear,
        uses_e3m4_weight,
    )

    pad = getattr(linear, "_e3m4_pad", None)
    if pad is None or not uses_e3m4_weight(linear.weight):
        return None
    in_pad, out_pad = pad
    if in_pad:
        # RowParallel input: the packed width was rounded up; gelu_tanh(0) == 0, so
        # zero columns contribute nothing to the dot product.
        x = torch.nn.functional.pad(x, (0, in_pad))
    out = _e3m4_block_linear(
        x,
        linear.weight,
        linear.weight_scale_inv,
        input_scale=None,
        bias=None,
        out_shape=[*x.shape[:-1], linear.weight.shape[0]],
    )
    if out is None:
        # Small-M shapes the kernels reject (e.g. K=1152 at M=1, where the GEMV
        # gate needs K %% 512): dequantize and fall back. Correctness-only path.
        from sglang.srt.layers.fp8_mm_visual import _dequant_e3m4

        w = _dequant_e3m4(linear.weight, linear.weight_scale_inv)
        if in_pad:
            w = w[:, : w.shape[1] - in_pad]
        out = x @ w.t()
    if out_pad:
        out = out[..., : out.shape[-1] - out_pad]
    if bias is not None:
        out = out + bias
    return out


@torch.no_grad()
def quantize_vision_mlps_if_enabled(root) -> int:
    """Pack the vision tower of a conditional-generation model if the flag asks.

    Call this from every top-level load_weights: those loaders write the visual
    parameters straight into params_dict, so the vision model's own load_weights
    (which also calls the packer) never runs for them. Safe to call twice -- a
    packed fc is skipped, so the second call packs nothing.
    """
    from sglang.srt.runtime_context import get_exec

    if get_exec().features.enable_mm_dtype != "fp8_e3m4":
        return 0
    if hasattr(root, "blocks") or hasattr(root, "merger"):
        visual = root  # called with the vision model itself
    else:
        visual = getattr(root, "visual", None) or getattr(
            getattr(root, "model", None), "visual", None
        )
    if visual is None or isinstance(visual, torch.nn.Identity):
        return 0
    before = torch.cuda.memory_allocated()
    n, bf16_bytes, packed_bytes = quantize_vision_mlp_to_e3m4_block128(visual)
    if n:
        logger.info(
            "Quantized vision tower mlp linear_fc1 + linear_fc2 (%s: %d tensors) "
            "quantized to fp8_e3m4 blockwise 128x128; %d BF16 MiB released, now "
            "%d MiB of e3m4 codes + scales (device allocated %d -> %d MiB)",
            type(visual).__name__,
            n,
            round(bf16_bytes / 2**20),
            round(packed_bytes / 2**20),
            round(before / 2**20),
            round(torch.cuda.memory_allocated() / 2**20),
        )
    return n
