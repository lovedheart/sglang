"""Online BF16 -> FP8-E3M4 (128x128 block) weight quantization at load time.

``--enable-fp8-weights`` (SM120 only) loads the checkpoint's BF16 dense
(non-MoE) projections normally, then re-encodes each eligible weight in place
as FP8 E3M4 codes with per-128x128-block FP32 scales and releases the BF16
copy. E3M4 (1-3-4, max +/-30, 4-bit mantissa) is ~6 dB more accurate than
E4M3 at equal bytes and is the format the sm120 QMMA kernels decode natively;
this is the same representation ``build_e3m4_ckpt.py`` writes offline, so a
served model matches an E3M4 checkpoint bit-for-bit.

Eligibility: bare BF16/FP16 ``ColumnParallelLinear`` / ``MergedColumnParallelLinear``
/ ``RowParallelLinear`` whose name matches the dense-attention/shared-FFN
patterns and whose (N, K) are both multiples of 128. Deliberately skipped:
MoE expert stacks (W3A16 kernels), the DSA indexer, the DFly draft, the MLA
``kv_b_proj`` (its absorb w_kc/w_vc built in post_load_weights have no E3M4
dequant path), and anything already quantized.
"""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.fp8_lm_head import FP8_E3M4_MAX, _e3m4_tables, _encode_e3m4
from sglang.srt.layers.fp8_shared_expert import Fp8Block128LinearMethod
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.utils.device import is_sm120_supported
from torch import nn

logger = logging.getLogger(__name__)


@torch.no_grad()
def dequant_e3m4(codes_u8: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Exact fp32 dequant of e3m4 codes + [N/128, K/128] scales (validation aid)."""
    pos, _ = _e3m4_tables()
    codes = codes_u8.to(torch.uint8)
    idx = (codes.cpu() & 127).to(torch.long)
    idx = torch.where(idx > 126, torch.zeros_like(idx), idx)  # NaN codes -> 0
    sign = torch.where((codes.cpu() & 128) > 0, -1.0, 1.0)
    w = (pos[idx].float() * sign).reshape(codes.shape)
    s = scale_inv.float().cpu().repeat_interleave(128, 0).repeat_interleave(128, 1)
    return (w * s).to(codes.device)


def quantize_linear_to_fp8_e3m4_block128(linear: nn.Module) -> bool:
    """Re-encode ``linear.weight`` (BF16/FP16, N%128==0, K%128==0) in place as
    FP8 E3M4 codes with 128x128 block scales; the BF16 storage is released.

    Returns True when the layer was swapped. Tagging ``weight.e4m3_only =
    False`` is what routes apply through the sm120 e3m4 kernels (the
    ``e3m4_block_fp8_linear`` intercept inside the block-linear method).
    """
    if not is_sm120_supported():
        return False
    # The BF16/FP16 dtype gate is what distinguishes an untouched weight
    # from a checkpoint-quantized one (fp8 containers, packed uint8, ...);
    # every linear carries *a* quant_method (unquantized included), so the
    # method itself says nothing here.
    weight = getattr(linear, "weight", None)
    if weight is None or weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    n, k = weight.shape
    if n % 128 != 0 or k % 128 != 0:
        return False

    with torch.no_grad():
        b = weight.detach().float().reshape(n // 128, 128, k // 128, 128).double()
        am = b.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-10)
        scale = am / FP8_E3M4_MAX
        codes = _encode_e3m4(b / scale).reshape(n, k)
        quantized = codes.view(torch.float8_e4m3fn)
        scale_inv = scale.squeeze((1, 3)).float().contiguous()
    # Rebind the parameter first so the old BF16 storage, now unreferenced,
    # returns to the caching allocator.
    linear.weight = nn.Parameter(quantized, requires_grad=False)
    linear.weight.e4m3_only = False
    linear.register_buffer("weight_block_scale", scale_inv, persistent=False)
    linear.quant_method = Fp8Block128LinearMethod()
    return True


# Dense linears worth quantizing, by leaf attribute name under each decoder
# layer. kv_b_proj is intentionally absent: the MLA absorb weights (w_kc/w_vc)
# derived from it in post_load_weights dequant with a plain float multiply and
# would silently misread E3M4 codes.
_DENSE_LEAF_ALLOWLIST = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "o_proj",
        "gate_up_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
)
_SKIP_NAME_FRAGMENTS = (
    "dfly",
    "indexer",
    "visual",
    "engram",
    ".experts.",  # FusedMoE stack; shared_experts stays eligible
    "nextn",
    "mtp",
)
_LINEAR_CLASSES = (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


def _is_eligible(name: str, module: nn.Module) -> bool:
    if not isinstance(module, _LINEAR_CLASSES):
        return False
    leaf = name.rsplit(".", 1)[-1]
    if leaf not in _DENSE_LEAF_ALLOWLIST:
        return False
    lname = name.lower()
    return not any(f in lname for f in _SKIP_NAME_FRAGMENTS)


def quantize_hy4_dense_weights_fp8_e3m4(model: nn.Module) -> int:
    """Quantize every eligible dense linear in ``model`` to FP8 E3M4 b128.

    Called from the model's post-load hook, i.e. before CUDA-graph capture,
    so the decode graphs bake the e3m4 kernels just like an offline E3M4
    checkpoint does. Returns the number of layers swapped.
    """
    if not is_sm120_supported():
        logger.warning(
            "--enable-fp8-weights is SM120-only and this device is unsupported; "
            "weights left in their original dtype."
        )
        return 0
    quantized = 0
    skipped_shapes = []
    for name, module in model.named_modules():
        if not _is_eligible(name, module):
            continue
        w = getattr(module, "weight", None)
        if w is None or w.dtype not in (torch.bfloat16, torch.float16):
            continue
        if quantize_linear_to_fp8_e3m4_block128(module):
            quantized += 1
        else:
            skipped_shapes.append(f"{name} {tuple(w.shape)}")
    if skipped_shapes:
        logger.info(
            "--enable-fp8-weights: skipped %d layers with non-128-multiple shapes: %s",
            len(skipped_shapes),
            "; ".join(skipped_shapes[:8]),
        )
    if quantized:
        logger.info(
            "--enable-fp8-weights: re-encoded %d BF16 layers as FP8 E3M4 "
            "128x128 block (bf16 copies released).",
            quantized,
        )
    return quantized
