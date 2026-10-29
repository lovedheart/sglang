"""MXFP8 (e4m3 + 1x32 UE8M0) storage for the HC mix projections.

The per-layer mHC mix pair is the largest non-MoE BF16 weight group in
Qwen3.8-Flash-Next: ``input_mix_weight_down`` [hc_lowrank, hc*hs] and
``input_mix_weight_up`` [hc*hs, hc_lowrank] carry ~6.6M params per branch,
~1.27 GB across 48 layers x {attn, mlp}. Decode-size mix GEMMs (rows <= 16)
are weight-bandwidth bound, so halving their footprint with MXFP8 cuts
~614 MB of per-step traffic with ~1.8% mix-gate relative error.

Same load-time contract as ``fp8_lm_head`` / ``fp8_shared_expert``: quantize
once at weight load, never lazily, so the swap always precedes CUDA-graph
capture. The persistent mix kernel dequantizes the e4m3 x 2^e tiles to bf16
in registers (exact: e4m3 mantissas fit in bf16), so the W8A16 forward is
bit-comparable to running the BF16 kernel on quantized weights.

Deliberately excluded (kept BF16): the last main layer, the model-level
final ``hyper_connection_mixer``, ``block_inject_weight`` (only ~40K
params/branch), and everything on the NEXTN/MTP draft runner — its own
load path never calls here, mirroring the shared-expert flag split.
"""

from __future__ import annotations

import logging
import re

import torch

logger = logging.getLogger(__name__)

MXFP8_BLOCK = 32
FP8_E4M3_MAX = 448.0
# uint8 exponent bias shared with _ue8m0_to_fp32 in mxfp8_block_convert.
_UE8M0_BIAS = 127

_LAYER_HC_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:attn|mlp)_hyper_connection$")


def mxfp8_quantize_rowmajor(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [N, K] (K % 32 == 0) to (e4m3 [N, K], ue8m0 [N, K//32]).

    Scale per 32-element group along K: ``s = 2^ceil(log2(amax/448))`` so no
    value can overflow e4m3. Zero groups keep exponent 0 (scale 1.0). Row
    major and plain torch so the result layout is deterministic at load time
    without depending on triton_kernels' swizzled scale tiles.
    """
    assert weight.dim() == 2 and weight.is_contiguous()
    n, k = weight.shape
    if k % MXFP8_BLOCK != 0:
        raise ValueError(f"K={k} must be divisible by {MXFP8_BLOCK} for MXFP8")
    w = weight.detach().float()
    g = w.view(n, k // MXFP8_BLOCK, MXFP8_BLOCK)
    amax = g.abs().amax(dim=-1)
    e = torch.ceil(torch.log2(amax.clamp(min=1e-30) / FP8_E4M3_MAX))
    e = e.clamp(-126.0, 126.0)
    e = torch.where(amax > 0, e, torch.zeros_like(e))
    scale = torch.exp2(e)
    q = (g / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).view(n, k)
    scale_u8 = (e + _UE8M0_BIAS).to(torch.uint8)
    return q.contiguous(), scale_u8.contiguous()


def mxfp8_dequantize_to_bf16(
    weight: torch.Tensor, scale_u8: torch.Tensor
) -> torch.Tensor:
    """e4m3 [N, K] + ue8m0 [N, K//32] -> bf16 [N, K] (exact up to bf16)."""
    n, k = weight.shape
    descale = (scale_u8.to(torch.int32) << 23).view(torch.float32)
    deq = weight.to(torch.float32).view(n, k // MXFP8_BLOCK, MXFP8_BLOCK)
    return (deq * descale.unsqueeze(-1)).view(n, k).to(torch.bfloat16)


@torch.no_grad()
def quantize_linear_to_mxfp8(linear: torch.nn.Linear) -> bool:
    """In-place MXFP8 swap of a bias-free nn.Linear weight. False if ineligible."""
    w = linear.weight
    if w.dtype != torch.bfloat16 or w.dim() != 2 or w.shape[1] % MXFP8_BLOCK != 0:
        return False
    q, s = mxfp8_quantize_rowmajor(w.detach())
    linear.weight = torch.nn.Parameter(q, requires_grad=False)
    # Buffer (not a bare tensor attribute) so _apply / device moves keep it
    # in sync with the packed weight.
    linear.register_buffer("_mxfp8_scale", s, persistent=False)
    linear.mxfp8_weight = True
    return True


@torch.no_grad()
def quantize_hc_mix_to_mxfp8(
    model: torch.nn.Module, last_layer: int | None = None
) -> int:
    """Swap every per-layer HC mix pair except the last layer's.

    The final ``hyper_connection_mixer`` never matches the layer pattern, so
    it (and the combine-side ``block_inject_weight``) stays BF16.
    ``last_layer`` is the main model's final layer id (config-derived so PP
    shards skip the right one); when omitted it falls back to the highest id
    found in ``model``. Returns the number of GatedResidual modules converted.
    """
    from sglang.srt.layers.hyperconnection import GatedResidual

    candidates: list[tuple[int, str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        m = _LAYER_HC_RE.search(name)
        if (
            m
            and isinstance(module, GatedResidual)
            and not getattr(module, "_mxfp8_mix", False)
            and hasattr(module, "input_mix_weight_down")
        ):
            candidates.append((int(m.group(1)), name, module))
    if not candidates:
        return 0
    skip = (
        last_layer if last_layer is not None else max(idx for idx, _, _ in candidates)
    )
    n = 0
    for idx, name, module in candidates:
        if idx == skip:
            continue
        ok_down = quantize_linear_to_mxfp8(module.input_mix_weight_down)
        ok_up = quantize_linear_to_mxfp8(module.input_mix_weight_up)
        if ok_down and ok_up:
            module._mxfp8_mix = True
            # The mix dispatch prefers the packed MXFP8 fold kernel at decode
            # sizes (gain in-register, no BF16 copy anywhere) and falls back
            # to on-the-fly dequant for larger rows. The CuTe split-K pair is
            # bf16-only, so never let it pad the up weight here.
            module._mix_up_weight_padded = None
            n += 1
        else:
            logger.warning(
                "HC mix at %s ineligible for MXFP8 (shapes %s/%s); left BF16",
                name,
                tuple(module.input_mix_weight_down.weight.shape),
                tuple(module.input_mix_weight_up.weight.shape),
            )
    if n:
        logger.info(
            "MXFP8: quantized %d HC mix pairs (down+up) to e4m3 + 1x32 UE8M0 "
            "scales (last layer and final mixer left BF16)",
            n,
        )
    return n
