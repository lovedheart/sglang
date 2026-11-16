# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the Mamba / linear-attention backends."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def handle_mamba_backend(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.mamba_cache_philox_rounds < 0:
        raise ValueError("--mamba-cache-philox-rounds must be non-negative.")

    cap = cfg.mamba_max_states_per_path
    if cap is not None and (cap == 0 or cap < -1):
        raise ValueError(
            "--mamba-max-states-per-path must be -1 (unlimited) or a positive "
            f"integer, got {cfg.mamba_max_states_per_path}."
        )

    if cfg.enable_mamba_cache_stochastic_rounding:
        if cfg.mamba_ssm_dtype != "float16":
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache requires "
                f"--mamba-ssm-dtype float16, got {cfg.mamba_ssm_dtype!r}. "
                "Run with --mamba-ssm-dtype float16 or disable "
                "--enable-mamba-cache-stochastic-rounding."
            )
        if not get_platform().is_cuda:
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache is only "
                "supported on NVIDIA CUDA platforms. Disable "
                "--enable-mamba-cache-stochastic-rounding on this platform."
            )
        if cfg.mamba_backend == "triton" and not get_platform().is_sm100:
            raise ValueError(
                "Stochastic rounding for the Mamba SSM cache with "
                "--mamba-backend triton requires SM100 with CUDA >= 12.8 "
                "because it uses the cvt.rs.f16x2.f32 PTX instruction. On "
                "H100/SM90, run with --mamba-backend flashinfer "
                "--mamba-ssm-dtype float16, or disable "
                "--enable-mamba-cache-stochastic-rounding."
            )

    if cfg.mamba_backend == "flashinfer":
        flashinfer_error = (
            "FlashInfer mamba module not available, please check the "
            "FlashInfer installation."
        )
        if cfg.enable_mamba_cache_stochastic_rounding:
            flashinfer_error += (
                " Stochastic rounding with --mamba-backend flashinfer "
                "requires FlashInfer Mamba and --mamba-ssm-dtype float16."
            )
        if get_platform().has_flashinfer:
            try:
                import flashinfer.mamba  # noqa: F401

                logger.info("Successfully imported FlashInfer mamba module")
            except (ImportError, AttributeError):
                raise ValueError(flashinfer_error)
        else:
            raise ValueError(flashinfer_error)


def handle_int8_mamba_checkpoint(server_args: Any):
    # The host-offload path (enabled by --enable-hierarchical-cache) and
    # custom radix-cache backends are NOT int8-aware: they would read int8
    # checkpoint slots as bf16 active slots (wrong pool / out-of-range).
    # Reject the combination up front rather than silently corrupting state.
    cfg = resolving_view(server_args)
    if not cfg.enable_int8_mamba_checkpoint and (
        cfg.enable_int8_mamba_checkpoint_offload
        or cfg.int8_mamba_ckpt_host_size is not None
    ):
        raise ValueError(
            "--enable-int8-mamba-checkpoint-offload / "
            "--int8-mamba-ckpt-host-size require --enable-int8-mamba-checkpoint: "
            "the RAM tier is an extension of the int8 checkpoint pool."
        )
    if not cfg.enable_int8_mamba_checkpoint:
        return
    if cfg.enable_hierarchical_cache:
        raise ValueError(
            "--enable-int8-mamba-checkpoint is not supported together with "
            "--enable-hierarchical-cache: the host-offload path "
            "is not int8-aware. Disable one of them."
        )
    if cfg.radix_cache_backend is not None:
        raise ValueError(
            "--enable-int8-mamba-checkpoint only supports the built-in mamba "
            f"radix cache; --radix-cache-backend={cfg.radix_cache_backend!r} "
            "is not int8-aware. Omit --radix-cache-backend."
        )
    if cfg.enable_lmcache:
        raise ValueError(
            "--enable-int8-mamba-checkpoint is not supported together with "
            "--enable-lmcache: LMCache is not int8-aware. Disable one of them."
        )
    if cfg.enable_unified_memory:
        raise ValueError(
            "--enable-int8-mamba-checkpoint is not supported together with "
            "--enable-unified-memory: the unified pool keeps no int8 checkpoint "
            "pool, so the flag would have no effect. Disable one of them."
        )
    if cfg.enable_int8_mamba_checkpoint_offload:
        if cfg.int8_mamba_ckpt_host_size is None or cfg.int8_mamba_ckpt_host_size <= 0:
            raise ValueError(
                "--enable-int8-mamba-checkpoint-offload requires a positive "
                "--int8-mamba-ckpt-host-size (number of pinned-RAM checkpoint "
                "slots for the checkpoint RAM tier)."
            )
        if cfg.mamba_radix_cache_strategy not in ("auto", "no_buffer", "extra_buffer"):
            raise ValueError(
                "--enable-int8-mamba-checkpoint-offload does not support "
                f"--mamba-radix-cache-strategy={cfg.mamba_radix_cache_strategy!r}."
            )
    elif cfg.int8_mamba_ckpt_host_size is not None:
        raise ValueError(
            "--int8-mamba-ckpt-host-size requires "
            "--enable-int8-mamba-checkpoint-offload."
        )


def handle_mamba_state_swap(server_args: Any):
    cfg = resolving_view(server_args)
    if not cfg.enable_mamba_state_swap:
        if cfg.mamba_swap_size is not None:
            raise ValueError("--mamba-swap-size requires --enable-mamba-state-swap.")
        return
    if cfg.mamba_swap_size is None or cfg.mamba_swap_size <= 0:
        raise ValueError(
            "--enable-mamba-state-swap requires a positive --mamba-swap-size "
            "(number of pinned-RAM parked-state slots)."
        )
    if cfg.enable_unified_memory:
        raise ValueError(
            "--enable-mamba-state-swap is not supported together with "
            "--enable-unified-memory: the unified pool keeps no separate "
            "active-slot allocator to park into. Disable one of them."
        )
    if cfg.radix_cache_backend is not None:
        raise ValueError(
            "--enable-mamba-state-swap only supports the built-in mamba radix "
            f"cache; --radix-cache-backend={cfg.radix_cache_backend!r} is not "
            "swap-aware. Omit --radix-cache-backend."
        )
    if not cfg.disable_radix_cache:
        # v1 slot economics: a parked request releases its MAIN slot only;
        # a request holding extra ping-pong tracking slots (radix state
        # caching) cannot free enough of the active pool for the rotation
        # to pay. The 1-slot-per-request config is radix-off.
        raise ValueError(
            "--enable-mamba-state-swap currently requires "
            "--disable-radix-cache: parked requests must be able to release "
            "their whole active state footprint."
        )
    if cfg.speculative_algorithm is not None:
        raise ValueError(
            "--enable-mamba-state-swap is not supported with speculative "
            "decoding: draft-state slots are not parked. Disable one of them."
        )
    if cfg.enable_linear_replayssm or cfg.enable_linear_replayssm_spec:
        raise ValueError(
            "--enable-mamba-state-swap is not supported with the ReplaySSM "
            "ring buffers: park/wake round-trips raw state rows only. "
            "Disable one of them."
        )
    if cfg.disaggregation_mode != "null":
        raise ValueError(
            "--enable-mamba-state-swap is not supported with PD "
            "disaggregation (--disaggregation-mode="
            f"{cfg.disaggregation_mode!r}). Disable one of them."
        )


def _platform_supports_mamba_cache_extra_buffer() -> bool:
    if current_platform.is_out_of_tree():
        return current_platform.support_mamba_cache_extra_buffer()
    platform = get_platform()
    return (
        platform.is_cuda
        or platform.is_musa
        or platform.is_npu
        or platform.is_hip
        or platform.is_xpu
    )


def validate_mamba_extra_buffer(view, hf_config: Any, *, mamba_cache_chunk_size_of):

    assert supports_mamba_cache_extra_buffer(view, hf_config), (
        f"extra_buffer is not supported for {hf_config.architectures[0]}; use no_buffer."
    )
    assert _platform_supports_mamba_cache_extra_buffer(), (
        "extra_buffer requires platform support for Mamba cache state snapshots."
    )
    if view.mamba_radix_cache_strategy == "extra_buffer_lazy":
        # The PD-disagg decode pool is not wired for lazy slots.
        assert view.disaggregation_mode == "null", (
            "extra_buffer_lazy unsupported under PD disaggregation; use "
            "--mamba-radix-cache-strategy extra_buffer."
        )
        # eagle/ngram/dspark/dflash all verify through
        # prepare_mamba_track_for_verify (lazy plan wired); dflash gained
        # the hook in DFlashVerifyInput.prepare_for_verify.
    if view.speculative_num_draft_tokens is not None:
        assert view.mamba_track_interval >= view.speculative_num_draft_tokens
    if view.page_size is not None:
        assert view.mamba_track_interval % view.page_size == 0
        # Called here and not passed in: `mamba_cache_chunk_size` derives from
        # `page_size`, which resolution writes after this validator runs, so
        # evaluating it at the call site raises on the unresolved `None`.
        mamba_cache_chunk_size = mamba_cache_chunk_size_of()
        assert mamba_cache_chunk_size is not None

        if (
            view.chunked_prefill_size is not None
            and 0 < view.chunked_prefill_size < mamba_cache_chunk_size
        ):
            logger.warning(
                "Mamba radix extra-buffer is enabled with chunked_prefill_size=%s "
                "smaller than mamba_cache_chunk_size=%s. This can make "
                "mamba_track_mask false for unfinished chunked-prefill handoff "
                "and skip Mamba state checkpoints.",
                view.chunked_prefill_size,
                mamba_cache_chunk_size,
            )


def validate_mamba_no_buffer(view, model_arch: str):
    assert view.page_size in (1, None), "no_buffer only supports page_size=1."
    assert view.disable_overlap_schedule, (
        "no_buffer do not support overlap schedule. Try to set disable_overlap_schedule=True."
    )
    assert view.attention_backend != "trtllm_mha", (
        "no_buffer do not support trtllm_mha attention backend."
    )
