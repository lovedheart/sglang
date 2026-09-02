"""Weight-free TileLang MQA operators for the simple QSA indexer.

The CUDA kernels are reduced versions of the previously validated Qwen MQA
kernels: the per-head weight input and all unrelated feature branches are
removed. Torch implementations are kept as the only fallback and reference.
"""

import math
from typing import Optional, Tuple

import torch

try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False

try:
    import deep_gemm

    HAS_DEEPGEMM = True
except ImportError:
    deep_gemm = None
    HAS_DEEPGEMM = False


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


# DeepGEMM's SM120 packed fp8 MQA kernel requires a head count in
# {8, 16, 32, 64}; the weight-free scorer zeroes the padded query heads'
# weights so they contribute nothing.
_DEEPGEMM_ALLOWED_HEADS = (8, 16, 32, 64)



def _require_deepgemm(fp4: bool = False) -> None:
    if not HAS_DEEPGEMM:
        raise RuntimeError(
            "SGLANG_QSA_USE_FP8_INDEXER requires the deep_gemm package "
            "(with SM120 fp8 MQA-logits kernels); install deep_gemm or unset "
            "the flag to keep the BF16 indexer."
        )
    if fp4 and not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        raise RuntimeError(
            "SGLANG_QSA_USE_FP4_INDEXER requires a deep_gemm build with the "
            "SM120 FP4 MQA-logits kernels (fp8_fp4_mqa_logits); this install "
            "only exposes FP8 kernels."
        )


def _qsa_fp8_query(
    q: torch.Tensor, score_scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[rows, heads, dim] bf16 -> padded fp8 q + per-head scorer weights.

    The torch scorer computes ``sum_h relu(q_h . k) / score_scale``; DeepGEMM
    applies per-head weights instead, so real heads weigh ``1/score_scale``
    and padded heads weigh zero.
    """

    _require_deepgemm()
    rows, heads, head_dim = q.shape
    padded_heads = next((h for h in _DEEPGEMM_ALLOWED_HEADS if h >= heads), None)
    if padded_heads is None:
        raise ValueError(f"QSA fp8 MQA cannot pad {heads} query heads")
    q_fp8 = q
    if padded_heads != heads:
        q_fp8 = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))
    weights = q.new_zeros((rows, padded_heads), dtype=torch.float32)
    weights[:, :heads] = 1.0 / (score_scale or math.sqrt(head_dim))
    return q_fp8.to(torch.float8_e4m3fn).contiguous(), weights


def deepgemm_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """FP8 packed prefill scores on fp8_e4m3 keys via DeepGEMM.

    Column semantics match ``torch_qsa_mqa_prefill`` (column n is key n,
    ``row_starts``-relative absolute numbering); out-of-window columns are
    kernel scratch and must never reach a consumer -- top-k masks by the same
    ``row_starts``/``row_ends`` windows.
    """

    _require_deepgemm()
    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    keys = k.shape[0]
    k_fp8 = k.reshape(keys, k.shape[-1])
    # RMS-normalized keys keep the constant unit scale DeepGEMM expects.
    k_scale = torch.ones(keys, dtype=torch.float32, device=q.device)
    q_fp8, weights = _qsa_fp8_query(q, score_scale)
    return deep_gemm.fp8_mqa_logits(
        q_fp8,
        (k_fp8, k_scale),
        weights,
        row_starts.to(torch.int32),
        row_ends.to(torch.int32),
    )


def deepgemm_qsa_mqa_prefill_fp4(
    q: torch.Tensor,
    k: Tuple[torch.Tensor, torch.Tensor],
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    head_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 packed prefill scores; ``k`` is a ``(codes int8, sf int32)`` tuple.

    Same column semantics as ``deepgemm_qsa_mqa_prefill``.  The query is
    e2m1-quantized here (group-32 ue8m0 scales) through the DeepSeek-V4
    indexer quantization kernel; keys arrive pre-quantized (BF16 pools
    quantize the slab once per forward at the call site, fp4 index caches
    gather it straight out of the pages).  ``head_weights`` ([rows, heads])
    carries the tokenwise indexer's per-head weights (folded as
    ``head_weights / score_scale``, mirroring the fp8 weighted path);
    without it all heads weigh uniformly ``1 / score_scale`` (the
    weight-free compressed indexer).
    """

    _require_deepgemm(fp4=True)
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        quantize_fp4_indexer_tensor,
    )

    _validate_q(q)
    rows, heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError("QSA FP4 MQA requires head_dim 128 (packed 64 B/token)")
    k_fp4, k_sf = k
    if k_fp4.shape[-1] != head_dim // 2:
        raise ValueError("QSA FP4 key codes must be head_dim/2 bytes per token")
    keys = k_fp4.shape[0]
    padded_heads = next((h for h in _DEEPGEMM_ALLOWED_HEADS if h >= heads), None)
    if padded_heads is None:
        raise ValueError(f"QSA FP4 MQA cannot pad {heads} query heads")
    q_pad = q if padded_heads == heads else torch.nn.functional.pad(
        q, (0, 0, 0, padded_heads - heads)
    )
    weights = q.new_zeros((rows, padded_heads), dtype=torch.float32)
    if head_weights is None:
        weights[:, :heads] = 1.0 / (score_scale or math.sqrt(head_dim))
    else:
        if head_weights.shape != (rows, heads):
            raise ValueError("QSA FP4 head_weights must be [rows, heads]")
        weights[:, :heads] = (
            head_weights.float() / (score_scale or math.sqrt(head_dim))
        )
    q_codes, q_sf = quantize_fp4_indexer_tensor(
        q_pad.contiguous().reshape(rows * padded_heads, head_dim)
    )
    return deep_gemm.fp8_fp4_mqa_logits(
        (
            q_codes.view(torch.int8).view(rows, padded_heads, head_dim // 2),
            q_sf.view(rows, padded_heads),
        ),
        (k_fp4.contiguous(), k_sf.contiguous()),
        weights,
        row_starts.to(torch.int32),
        row_ends.to(torch.int32),
    )


def deepgemm_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """FP8 paged decode scores via gather + the packed fp8 kernel.

    The SM120 *paged* kernel hard-requires ``block_kv == 64`` while the
    compressed cache's pages are compress-ratio shrunken, so decode gathers
    each row's page table into one flat fp8 slab and feeds the packed kernel
    with ``ks = row * total``/``ke = ks + context_len`` plus
    ``max_seqlen_k = total``.  That compact mode returns ``[rows, total]``
    logits whose column j is row-relative key j -- exactly the layout the
    paged torch reference exposes (beyond ``context_lens`` is scratch).
    Rows are processed in blocks to bound the gathered slab.
    """

    _require_deepgemm()
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    rows, heads, head_dim = q.shape
    pages, page_size, _, _ = k_cache.shape
    total = page_table.shape[1] * page_size
    if total == 0:
        return torch.full((rows, 0), -float("inf"), dtype=torch.float32, device=q.device)
    q_fp8, weights = _qsa_fp8_query(q, score_scale)
    cache_flat = k_cache.reshape(-1, head_dim)
    context = context_lens.to(torch.int32).reshape(-1)
    offsets = torch.arange(page_size, dtype=torch.int64, device=q.device)
    logits = q.new_zeros((rows, total), dtype=torch.float32)
    # Bound the gathered slab (~128 MiB): each row materializes ``total``
    # fp8 keys, so long contexts shrink the rows scored per kernel call.
    block = max(1, min(rows, (128 << 20) // max(total * head_dim, 1)))
    for row_begin in range(0, rows, block):
        row_end = min(row_begin + block, rows)
        slots = (
            page_table[row_begin:row_end]
            .to(torch.int64)
            .clamp_min(0)[:, :, None]
            * page_size
            + offsets
        ).reshape(row_end - row_begin, total)
        slab = cache_flat.index_select(0, slots.reshape(-1))
        k_scale = torch.ones(slab.shape[0], dtype=torch.float32, device=q.device)
        row_starts = (
            torch.arange(row_end - row_begin, dtype=torch.int32, device=q.device)
            * total
        )
        logits[row_begin:row_end] = deep_gemm.fp8_mqa_logits(
            q_fp8[row_begin:row_end],
            (slab, k_scale),
            weights[row_begin:row_end],
            row_starts,
            row_starts + context[row_begin:row_end],
            max_seqlen_k=total,
        )
    return logits


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        # The validated MMA layout wants 64 GEMM rows; pages narrower than
        # that (full_page // ratio compressed views) are packed in sub-page
        # quadrants of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        # Python-level unroll: sub_pages is a compile-time
                        # constant and TileLang's pipeliner rejects dynamic
                        # inner loops around shared-memory copies.
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[
                                        sp * page_size : (sp + 1) * page_size, :
                                    ],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0)
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = reduced[token] / Scale

        return kernel


def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # Allocate the padded output once. Appending even a few padding rows with
    # torch.cat would allocate and copy the entire [rows, keys] FP32 matrix,
    # temporarily doubling the dominant prefill buffer for long contexts.
    logits = torch.zeros(
        (padded_rows, keys), dtype=torch.float32, device=q.device
    )
    q_padded = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(heads=heads, head_dim=head_dim, block_q=block_q)(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.full(
        (q.shape[0], max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if not q.shape[0] or not max_model_len:
        return logits
    # The validated MMA layout requires N (the Q-head dimension) to be a
    # multiple of eight. Zero-padding preserves the weight-free head sum.
    query_heads, head_dim = q.shape[1:]
    kernel_heads = max(8, ((query_heads + 7) // 8) * 8)
    q_kernel = q.to(torch.bfloat16)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    _tilelang_qsa_mqa_decode_kernel(
        heads=kernel_heads, head_dim=head_dim, page_size=page_size
    )(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if isinstance(k, tuple):
        # FP4 keys arrive as a (codes, scales) pair; the packed scorer
        # quantizes the query itself.
        return deepgemm_qsa_mqa_prefill_fp4(q, k, row_starts, row_ends, score_scale)
    if k.dtype == torch.float8_e4m3fn:
        return deepgemm_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if k_cache.dtype == torch.float8_e4m3fn:
        return deepgemm_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, score_scale
        )
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "deepgemm_qsa_mqa_decode",
    "deepgemm_qsa_mqa_prefill",
    "deepgemm_qsa_mqa_prefill_fp4",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
