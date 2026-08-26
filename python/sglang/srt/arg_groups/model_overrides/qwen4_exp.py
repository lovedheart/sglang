"""Config-time override declarations for qwen4_exp.

Architectures: Qwen4ExpForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


@_register_for("Qwen4ExpForConditionalGeneration")
def _qwen4_exp_overrides(server_args: Any, hf_config: Any) -> dict:
    """Qwen4-Exp keeps the MoE config under ``text_config``; every layer is
    sparse, so a dense-MLP TP size of 1 only stalls the DP MoE path.

    Compressed QSA additionally pins page_size=64 (overriding the hybrid
    family's triton default of 1): its compressed cache is addressed as
    ``full_slot // compress_ratio`` (the DSV4 scheme), which requires
    page-aligned full-KV allocation with the page a multiple of the compress
    ratio, and page-granular prefix sharing so shared pages share their
    compressed slots. MambaRadixCache supports page_size > 1 only with the
    mamba extra-buffer strategy, so fall back to the family default when
    neither that nor --disable-radix-cache holds (the QSA pool then fails
    fast at boot).
    """
    import torch

    from sglang.srt.arg_groups.model_overrides.qwen3_5 import (
        _qwen3_5_hybrid_overrides,
    )

    # This arch is not in the hybrid family list (the split test forbids two
    # modules declaring page_size for it), so the hybrid attention/page-size
    # defaults are borrowed here first; the compressed-QSA override below
    # stays the unconditional last writer.
    overrides: Dict[str, Any] = dict(_qwen3_5_hybrid_overrides(server_args, hf_config))

    cfg = resolving_view(server_args)
    if cfg.ple_offload_embedding is None:
        overrides["ple_offload_embedding"] = (
            get_platform().is_cuda
            and model_config_of(server_args).dtype == torch.bfloat16
        )

    text_config = getattr(hf_config, "text_config", hf_config)
    if (
        getattr(text_config, "num_experts", None) is not None
        and cfg.moe_dense_tp_size == 1
    ):
        overrides["moe_dense_tp_size"] = None

    from sglang.srt.layers.attention.qsa.config import (
        QSA_VARIANT_COMPRESSED,
        parse_qsa_profile,
    )

    profile = parse_qsa_profile(hf_config)
    if profile is not None and profile.variant == QSA_VARIANT_COMPRESSED:
        # Unconditional, like DeepSeek-V4's page-256 declaration: compressed
        # addressing is full_slot // ratio and requires page-aligned
        # allocation on every backend. Do not gate this on
        # mamba_radix_cache_strategy — that field resolves in a later pass
        # (mid-resolution it still holds the unresolved default, which made
        # this declaration silently skip on non-SM100 boxes); if the finally
        # resolved strategy cannot support page > 1, MambaRadixCache's own
        # boot assertion reports it.
        overrides["page_size"] = 64
        logger.info(
            "Setting page size to 64 for compressed QSA "
            "(full//ratio compressed addressing)."
        )
    return overrides
