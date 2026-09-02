# QSA + NEXTN/MTP Speculative Decoding on `rebase/qsa-on-main` (HEAD `4ae3fe9d52`)

Read-only survey. All paths relative to `python/sglang/srt/`. No files modified.

---

## 0. Shape of the stack

Qwen Sparse Attention (QSA) is a DSA-shaped two-stage sparsifier: an MQA **indexer** scores
`compress_ratio`-compressed key "pages", `block_topk ∈ {512, 2048}` pages are chosen, then
`expand_qsa_block_indices` expands pages to a token top-k of width
`final_topk = token_topk + compress_ratio - 1` and appends the causal tail. Spec decode is spec-v2
(EAGLE v2 / MTP) with **chain speculation only** (`topk == 1`).

Layer placement: Qwen4-Exp is a hybrid GDN model, so the live object is
`HybridLinearAttnBackend` wrapping `QwenSparseAttnBackend` as `full_attn_backend`
(`layers/attention/attention_registry.py:520-541` → `hybrid_backend_cls(full_attn_backend,
linear_attn_backend, full_attn_layers)`). Child metadata hooks are fanned out by the wrapper
(`hybrid_linear_attn_backend.py:1130` loops `init_forward_metadata_out_graph`,
`get_indexer_metadata` at `:1146`).

---

## 1. Forward-mode handling, and where `raw_indices`/top-k come from at verify

### 1.1 Mode → method routing

`base_attn_backend.py:265-292`: everything that is not `DECODE` routes to `forward_extend`.
So `TARGET_VERIFY` and `DRAFT_EXTEND_V2` both land in
`QwenSparseAttnBackend.forward_extend` (`qwen_sparse_attn_backend.py:1414`), **not** in
`forward_decode`. Inside `forward_extend` there is an early bail for speculative-paged modes
(`:1441-1446`):

```python
if self._is_speculative_paged_mode(forward_batch.forward_mode):
    output = self._forward_paged_attention(q, layer, forward_batch, topk_indices)
    return self._pad_extend_output(output, num_output_rows)
```

`_is_speculative_paged_mode` (`:357`) is exactly
`is_target_verify() or is_draft_extend_v2()`. So verify and draft-extend share the *paged
decode-style* gather path (`_forward_paged_attention`, `:1884`) rather than the varlen prefill
kernels — the prefill kernels below that bail (`sparse_gqa_fwd_interface_triton`,
`SGLANG_QSA_VECTOR_PREFILL_GATHER`) are never reached under spec.

`DECODE` (draft steps) goes to `forward_decode` (`:1848`) → same `_forward_paged_attention`.

### 1.2 The row contract: one metadata row **per query token** under spec

This is the single most important difference from plain EXTEND. `layers/attention/qsa/metadata.py:46-60`
documents it. Implementation in `_metadata_from_forward_batch` (`:678`), spec branch `:700-726`:

```python
logical_positions = forward_batch.positions          # [num_rows], one per token
sequence_lengths  = logical_positions + 1           # row "length" = its position + 1
token_slot_table  = req_to_token[row_req_pool_indices, :max_length]
```

vs. the decode branch `:753` which asserts one row per request, and the plain-EXTEND branch which
builds one row per sequence with `cu_seqlens`. `row_req_pool_indices` is
`repeat_interleave(req_pool_indices, extend_seq_lens)` (`:408 _speculative_row_to_request`, or the
`num_rows % batch_size` uniform fast path).

**Query tokens per row at `speculative_num_draft_tokens = 4` (chain, topk=1):**
- `TARGET_VERIFY`: `num_rows = bs * 4`. Each row is one draft token; row `i` has
  `sequence_length = position_i + 1`, so rows of one request form 4 consecutive lengths.
  Gather width = `final_topk = token_topk + ratio - 1` (from `parse_qsa_profile`, `indexer_budget`
  = `token_topk`, `block_topk = budget / ratio`).
- `DRAFT_EXTEND_V2`: `num_rows = bs * (num_draft_tokens + front)` (`eagle_worker_common.py:105
  prepare_for_draft_extend`), `forward_batch.seq_lens += num_draft_tokens` at the same place.
- `DECODE` draft step `k`: `num_rows = bs`, `seq_lens + k + 1`
  (`QwenSparseMultiStepDraftBackend._make_step_forward_batch`, `:2104`).

`spec_utils.py:98 resolve_num_tokens_per_req` is the authority for the multiplier: both
`target_verify` and `draft_extend` resolve to `num_draft_tokens`.

### 1.3 Where the top-k comes from at verify

Two possibilities, decided per layer in `models/qwen4_exp.py:1495 _compute_qsa_topk_indices`:

**(a) Indexer runs live (default).** `:1518` calls
`QSAIndexer.forward_cuda` (`qsa/qsa_indexer.py:649`). For verify/draft-extend it takes the
**decode-style** selection branch (`:736-748` — same branch as DECODE) with
`decode_logical_positions` (`:663/:668`) → `select_decode_tokens` (`:582`) →
`qsa_mqa_decode` over the compressed page table → `fast_topk`/`qsa_fast_topk` →
`expand_qsa_block_indices` (`qsa/kernel.py:98`). So at verify, `raw_indices` are *recomputed per
verify row*, not read from a buffer.

**(b) Index-share (MTP) is on.** `:1508-1513` — if
`backend.should_reuse_mtp_sparse_indices(forward_batch)` (`qwen_sparse_attn_backend.py:1277`, true
only for `forward_mode.is_decode()`), the indexer is **skipped entirely** and
`lookup_mtp_sparse_indices` (`:1377`) serves indices from `QSAMTPSharedSparseIndices`
(`:114-170`). Capture happens in draft-extend:
`should_capture_mtp_sparse_indices` (`:1284`) is true for `DRAFT_EXTEND_V2` (and the draft runner's
plain post-prefill EXTEND), and `capture_mtp_sparse_indices` (`:1294`) →
`_capture_mtp_sparse_indices_from_extend_lens` (`:1338`) anchors on the **final accepted row** of
each request (`anchor row = block_ends - (extend_len - front - num_accept)`). Gated in
`speculative/eagle_worker_v2.py:390 _configure_qsa_mtp_index_share` (needs `topk==1`, `steps>1`,
both backends QSA, no adaptive spec); `expanded_width = token_topk + compress_ratio - 1` (`:424`).

Note the *target verify* always runs the indexer; index-share only affects the **draft decode**
steps.

### 1.4 Attention compute at verify

`_forward_paged_attention` (`:1884`) → trtllm-gen paged decode via
`_forward_trtllm_sparse` (`:1649`, page packing + valid counts), or the FA2-varlen fallback which
uses `qwen_sparse_fa2_cu_seqlens_triton` on the per-row top-k (`:1953`) plus
`qwen_sparse_kv_extraction_gathered_rows_triton` (`:2010`) — **row-gathered** KV keyed by
`metadata.row_req_pool_indices` (`:2005-2009`, and `:1930-1944` in the trtllm path). That index
source is the hook surface: it is *row-major over tokens*, not `forward_batch.req_pool_indices`
(the `else` fallback is only for non-spec).

---

## 2. The indexer under draft-extend vs verify: what gets written, how it is rolled back

### 2.1 What it writes

`QSAIndexer.update_key_state_and_compress` (`qsa_indexer.py:321`) does two things on every forward:

1. **Pending ring store** — per *token*: `pool.set_qsa_key_state_buffer(layer, state_slots, k)` +
   `set_qsa_rope_position_buffer` (`:335-341`). Slots from
   `build_pending_ring_slots` (`qsa/metadata.py:313`):
   `slot = req_pool_idx * stride + position % stride`, `stride = qsa_ring_stride(ratio) = 2 * ratio`
   (`:297`).
2. **Compressed-K store** — per *completed group*: `set_qsa_compressed_k_buffer(layer,
   write_locs, compressed)` (`:400`, graph version `_compress_decode_cuda_graph` `:402`).

For **paged** modes (decode, verify, draft-extend) the write plan is
`_qsa_build_write_plan` (`qwen_sparse_attn_backend.py:633`) with
`start_blocks = end_blocks - (lengths % ratio == 0)` — i.e. **each paged row compresses at most
the one group its length completes**, and members are sourced from the ring (`:376-390`).
For plain EXTEND, members come from the chunk itself (`compress_member_rows`, `:365-374`).

So: **yes, index-K (both the ring and, at group boundaries, the compressed page) is written for
draft tokens** during DRAFT_EXTEND_V2, and again for verify rows (the verify rows re-store the same
ring slots).

### 2.2 There is NO indexer rollback — and why that is safe

There is no rollback code anywhere in `qsa/`. Safety is structural, documented verbatim in
`qsa/metadata.py:297-309`:

> The stride must exceed one group: a speculative verify forward stores its whole draft window into
> the ring BEFORE compressing the boundary group, so under a `% ratio` stride the window's later rows
> clobber the still needed members of that group (same class of bug DSWA fixed as
> `ring_stride = window + max_spec_steps`). A stride of `2 * ratio` keeps one full group plus one
> speculation window apart.

Enforced at `qwen_sparse_attn_backend.py:362 _require_chain_speculation`:

```python
if topk != 1: raise NotImplementedError(...)
if draft_tokens > 2 * self.compress_ratio: raise ...   # :370
```

Consequence for an offload adapter: **rejected draft tokens are inert for the indexer.** Their ring
slots are simply overwritten by the next iteration's rows (`pos % stride`), and their compressed
slots (if a boundary was crossed) sit past `kv_committed_len` and are never addressed by a
`compressed_page_table` rebuilt from committed lengths. Nothing needs to be undone; the adapter must
likewise treat post-commit compressed slots as *undefined, not deleted*.

---

## 3. KV commit / rollback for rejected draft tokens

### 3.1 Allocation: reserve 2×, commit only what is accepted

- `mem_cache/allocation_sizing.py`: `get_alloc_len_per_decode = max(steps*topk, spec_tokens)`,
  `get_alloc_reserve_per_decode = 2×` that.
- `mem_cache/allocation.py:725 alloc_for_spec_decode` bumps only
  `req.kv.kv_allocated_len = max(..., nxt)` (`:772`); `kv_committed_len` is untouched. Decode-time
  allocation (`:641`) increments both.

### 3.2 Verify writes into slots that were already reserved

`eagle_utils.py:513 eagle_prepare_for_verify` →
`assign_extend_cache_locs_uniform_func` (`:562`): verify `out_cache_loc` is read out of
`req_to_token[req, committed : committed + N]` — **no new allocation**, the window was reserved by
`eagle_prepare_for_decode`'s double-alloc (`:1004`). Sets `ForwardMode.TARGET_VERIFY` and calls
`decode_cuda_graph_runner.load_batch`.

### 3.3 The commit

`managers/scheduler_components/batch_result_processor.py:744 _resolve_spec_v2_tokens`,
`:803`:

```python
req.kv.kv_committed_len += num_accept_tokens
```

`speculative/eagle_worker_common.py:599` computes the mirror image on GPU:
`new_seq_lens = batch.seq_lens + accept_lens`.

### 3.4 The rollback — and the important negative result

**There is no per-verify free of rejected tokens.** The rejected KV simply remains "allocated but
uncommitted" and is released once, at request teardown:
`mem_cache/common.py:265 release_kv_cache` → `:293-294`:

```python
_release_overallocated_kv_indices(req, req.kv.kv_committed_len, req.kv.kv_allocated_len)
```

Radix/chunk paths trim the same way (`mem_cache/unified_radix_cache.py:945-1052` with
`tail_free_start`, `chunk_cache.py:84`, `base_prefix_cache.py:414 free_kv_row`).

`spec_utils.py:703 move_accept_tokens_to_target_kvcache` (`:763 move_kv_cache`) is the *compaction*
path and it is **dead under QSA**: only reachable from
`eagle_worker_common.py:640 if ... and topk > 1` ("topk == 1 needs nothing here: the accepted path
is already the front chain, so the whole compaction is an identity transform"). Same for the tree
finalize at `:640`.

**Net:** accepted draft tokens never move; rejected ones are garbage-in-place. Slot → position
mapping is therefore only meaningful when re-derived from
`req_to_token[req, : kv_committed_len]`.

---

## 4. CUDA graph handling of target-verify

### 4.1 Which runner, which shapes

TARGET_VERIFY is captured by the **decode** runner
(`model_executor/runner/decode_cuda_graph_runner.py:286-301`):

```python
self.capture_forward_mode = ForwardMode.TARGET_VERIFY   # when spec
captured_req_width = decode_num_tokens_per_req(...)     # :292  == num_draft_tokens
max_num_token = max_bs * captured_req_width             # :345
```

`can_run_graph` (`:633`) replays only when `spec_info.num_tokens_per_req == captured_req_width`,
otherwise it falls back to eager. Ragged verify (`:315 ragged_verify_mode`,
`SGLANG_RAGGED_VERIFY_MODE`) requires `supports_ragged_verify_graph`
(`base_attn_backend.py:64`) and is **dspark-only — QSA does not set it**, so non-uniform verify is
unavailable for QSA.

DRAFT_EXTEND_V2 has its own runner: `eagle_draft_extend_cuda_graph_runner.py:91/109/150/152`
(shape `(bs, bs*num_draft_tokens)`). Draft decode steps use the QSA multi-step draft backend
(`qwen_sparse_attn_backend.py:2053 QwenSparseMultiStepDraftBackend`, `topk != 1` raises at `:2059`).

### 4.2 QSA's graph metadata is rebuilt on GPU

`QwenSparseAttnBackend.init_forward_metadata_out_graph` (`:882`) splits capture vs replay; capture
computes `num_tokens = input_ids.shape[0]` when `_is_speculative_paged_mode` (else `bs`) — i.e.
the metadata row count is `bs * N` for verify, not `bs` (`:1002 metadata_rows`).
Buffers are all sized `max_num_tokens` in `init_cuda_graph_state` (`:913`):
`_graph_seq_lens`, `_graph_row_req_pool_indices`, `_graph_logical_positions`, `_graph_state_slots`,
`_graph_ring_group_locs (max_num_tokens, ratio)`, `_graph_compressed_page_table
(max_num_tokens, max_pages)`, `_graph_write_locs`, FA2 valid-counts / cu_seqlens, plus a 2-half
pinned `_graph_extend_lens_pin` for the async HtoD.

Replay refresh: `_replay_cuda_graph_metadata` (`:1066`) → `_can_replay_with_gpu_kernels` (`:1125`)
→ `_replay_cuda_graph_metadata_gpu` (`:1156`, "nothing here may read back to the host") →
`launch_graph_metadata` (`qsa/graph_metadata.py:182`) with
`_qsa_graph_layout_kernel` MODE `0=DECODE, 1=TARGET_VERIFY (uniform extend_len),
2=DRAFT_EXTEND (per-req extend lens)`. All accept-dependent lengths are derived from `req_to_token`
arithmetic on device — no host round-trip. Fallback for pools without the kernels:
`_update_qsa_cuda_graph_metadata` (`:1210`).

`init_forward_metadata_in_graph` (`:2184`) is a **no-op for QSA** (the whole rebuild happens in the
out-of-graph replay-prep), and the base contract for the in-graph hook is
`base_attn_backend.py:113-131` (no `.item()`, no `.cpu()`, no dynamic `torch.empty`).

Layout for capture-time is built by `_graph_speculative_layout` (`:467`): per request
`extend_len = spec_info.draft_token_num` (verify) or per-req `extend_seq_lens_cpu` (draft extend),
then per-row `lengths = arange(prefix+1, prefix+len+1).clamp_max(seq_len)` and
`row_req_pool_indices = repeat_interleave(req_pool_indices, extend_lengths)`; padding rows get
`length 1 / prefix 0` on the **inert request slot** so they are harmless.

---

## 5. Exact place a per-layer "selected `raw_indices`" hook would fire

This is the seam the hisparse-style adapter wants:

| Need | File:line | Note |
|---|---|---|
| Model-level per-layer top-k producer | `models/qwen4_exp.py:1495` (`:1508` reuse, `:1518` indexer, `:1525-1529` capture) | the one place where "selected raw_indices" exists as a tensor, for **every** mode incl. verify |
| Backend paged compute (verify + draft-extend) | `qwen_sparse_attn_backend.py:1441-1446` → `_forward_paged_attention :1884` | sees verify-shaped batches; `topk_indices` is `[bs*N, final_topk]` |
| Backend paged compute (draft decode) | `:1848 forward_decode` → `:1884` | `[bs, final_topk]` |
| Row→request resolution inside gather | `:1930` (trtllm), `:2005-2009` (FA2), `:1868-1874` (FP4) | `metadata.row_req_pool_indices` is the correct key |
| Indexer-side selection | `qsa/qsa_indexer.py:582 select_decode_tokens` return site (`expand_qsa_block_indices`) | block indices are available one line earlier |
| Graph-recordable per-iter hook | `base_attn_backend.py:113 init_forward_metadata_in_graph` / `:83 init_forward_metadata_out_graph` | QSA currently uses the out-graph variant only |
| Pool-side compressed-K + ring | `mem_cache/qsa_kv_pool.py:150-190` | `qsa_compressed_k_buffer_pool`, rope position buffer, ring |

Reference for the shape of such a hook on other branches: `layers/attention/dsv4/indexer.py:834-846`
and `:986-1002`:

```python
hisparse_decode = coordinator is not None and forward_mode.is_decode()
...
coordinator.swap_in_selected_pages(req_pool_indices, compressed_seq_lens, top_k_result, layer_id)
```

That `is_decode()`-only gate is precisely what breaks under spec decode: TARGET_VERIFY /
DRAFT_EXTEND_V2 are not `is_decode()`, so the adapter silently takes the non-hisparse
`translate_loc_to_hisparse_device` path and either reads host-resident KV or misses the swap-in.
`managers/hisparse_coordinator.py` **is** on this branch (`:248 raw_indices_buffer`,
`:1005 swap_in_selected_pages`) but is DSA/DSv4-only; `validate_hisparse`
(`arg_groups/hisparse_hook.py:94`) still asserts `is_deepseek_dsa or is_deepseek_v4`. No QSA
hisparse code exists here — it lives on `qsa-hisparse-v2`, `tmp/qsa-hisparse-port`,
`qsa-on-main-hisparse` (`qsa/hisparse_graph.py`, `mem_cache/qsa_hisparse_{v3,p2,slots}.py`).
`layers/attention/dsa_backend.py:2259` is the DSA decode-side equivalent.

---

## 6. Existing spec + QSA restrictions (all hard `raise`/`assert`)

1. **Chain only.** `_require_chain_speculation` `qwen_sparse_attn_backend.py:362` —
   `speculative_eagle_topk != 1` → `NotImplementedError`. Also `QwenSparseMultiStepDraftBackend`
   `:2059`. Called from metadata init at `:991` and `:1077`.
2. **`speculative_num_draft_tokens <= 2 * compress_ratio`** (`:370`) — the ring-stride bound from §2.2.
3. **`page_size % compress_ratio == 0`** — `mem_cache/qsa_kv_pool.py:90` (compressed slot =
   full slot // ratio).
4. **Index-share disable conditions** — `eagle_worker_v2.py:390-424`: needs `index_share_for_mtp_iteration`
   in `hf_config`/`text_config` (`:143`), `topk==1`, `steps>1`, both backends QSA, and **no adaptive
   spec**.
5. **No ragged verify** for QSA (§4.1) — dspark-only flag.
6. **No breakable/breakable-prefill CUDA graph** for Qwen4-Exp: `model_config.py:2214`
   ("Qwen4-Exp is intentionally absent: QSA builds host-side sparse metadata").
7. **Hisparse is not selectable for QSA at all** today
   (`arg_groups/hisparse_hook.py:94`), and `--dcp-size > 1` is rejected with hisparse
   (`arg_groups/hicache_hook.py:108`).
8. Indexer overlap only under 1024 tokens (`models/qwen4_exp.py:76
   _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024`, used at `:1543`), and only on the
   `plan_stream` when `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` (`spec_utils.py:1235`).

---

## 7. Integration points for a KV-offload (hisparse) adapter under spec decode

### (a) Multi-token-per-row query layouts — the #1 correctness trap

Any per-request gather must be keyed by **row**, not by request index:

```python
req_rows = metadata.row_req_pool_indices            # [num_rows]; length == topk_indices.shape[0]
# or equivalently metadata.get_token_to_batch_idx() -> token -> batch slot
```

`forward_batch.req_pool_indices` is `[bs]` and will *silently broadcast wrong* on
`[bs*4, final_topk]` top-k. `dsv4/indexer.py:986`'s `swap_in_selected_pages(req_pool_indices, ...)`
signature is req-major and must be widened to `(row_req_pool_indices, row_seq_lens)` where
`row_seq_lens = metadata.sequence_lengths` (= `position + 1`). Expect three call sites:
verify (`bs*4` rows), draft-extend (`bs*(4+front)` rows), draft decode (`bs` rows).

### (b) Rejected-token KV rollback — nothing to hook; re-resolve instead

There is no free/rollback callback at verify end (§3.4). The adapter must:
- consider every slot `req_to_token[req, kv_committed_len : kv_allocated_len]` **reclaimable**
  (it will be freed in bulk at teardown by `_release_overallocated_kv_indices`), and
- rebuild slot→host mapping from `req_to_token[req, :committed]` each iteration rather than
  incrementally, because accepted tokens never move but *positions* of a row change with
  `accept_len`.

`qsa/graph_metadata.py`'s device-only derivation of lengths/page tables from `req_to_token` is the
exact template — accept counts never touch the host, so the adapter's mapping refresh must stay on
GPU too.

### (c) Indexer state for draft rows

Three separate states, with three different lifetimes:

| State | Draft-row behavior | Adapter duty |
|---|---|---|
| Pending ring (`req*stride + pos%stride`, `stride=2*ratio`) | Written, then harmlessly overwritten | None (inert) |
| Compressed K pages (`write_locs`) | Written at group boundaries; garbage past `kv_committed_len` | Must never be swapped in / counted as valid; validate against committed length |
| `QSAMTPSharedSparseIndices` buffers (`qwen_sparse_attn_backend.py:114-170`, `tail_width = num_steps+1`, trash row `== num_requests`) | Captured at DRAFT_EXTEND_V2, consumed at draft DECODE | Buffers must live on the accelerator-side or be graph-pinned; a swap-in per draft step is the *point* of index-share (it removes indexer work, so the adapter must not re-run the indexer there) |

### (d) Graph-capture shapes

Pre-size every adapter buffer at capture time to `max_num_tokens = max_bs * num_draft_tokens`
(§4.1) and make sure the per-iter work is recordable:
- All state must be **pre-allocated static tensors** in `init_cuda_graph_state` (`:913`) like
  QSA's `_graph_*` set; no `torch.empty` with a data-dependent shape, no `.item()/.cpu()/.tolist()`
  in `init_forward_metadata_in_graph` (`base_attn_backend.py:113-131`).
- If the adapter's swap-in depends on accept counts, it must run either (i) inside the GPU metadata
  rebuild (`launch_graph_metadata`, adding a MODE for the adapter) or (ii) as an in-graph kernel
  reading device-side counters — `hisparse_coordinator.py`'s `num_real_reqs`-style device counter is
  the existing pattern for "how much of this padded buffer is real".
- Capture-time padding rows must land on an inert slot (request slot 0 is already the inert dump in
  `qsa_kv_pool`; `_graph_speculative_layout` reuses `req_pool_indices[0]`) so an adapter swap-in for
  padding is a no-op, not a wild read.
- Pinned double-buffering (`_graph_extend_lens_pin` × 2) is the pattern if any HtoD is unavoidable —
  a single pinned buffer races the previous replay's queued copy.

### (e) Suggested minimum patch surface

1. New gate replacing `is_decode()`:
   `is_decode() or is_target_verify() or is_draft_extend_v2()` — mirror
   `_is_speculative_paged_mode` (`:357`) exactly so adapter and backend agree.
2. Row-major swap-in API taking `row_req_pool_indices` + row lengths (§7a) called from
   `_forward_paged_attention` (`:1884`) or `select_decode_tokens` return (`qsa_indexer.py:630`).
3. Ring-stride-aware window clamp: only pages covering `[0, committed)` are offloadable; the
   trailing `2*ratio` positions are always device-resident (§2.2, §7c).
4. Buffer sizing + graph hooks in `init_cuda_graph_state`/`_capture_cuda_graph_metadata`/
   `_replay_cuda_graph_metadata*` (`:913/:981/:1066`).
5. Relax `arg_groups/hisparse_hook.py:94` model gate and add the QSA spec validation (draft tokens
   ≤ `2*ratio` already asserted at `:370`, so the adapter can rely on the window bound).
