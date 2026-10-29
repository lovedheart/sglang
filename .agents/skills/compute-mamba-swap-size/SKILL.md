---
name: compute-mamba-swap-size
description: Size --mamba-swap-size / --max-mamba-cache-size / --max-running-requests for the mamba state-swap tier (parked-request host records), from the model's state-row bytes and the desired live-request count. Use when a user asks how many swap records to allocate, how much pinned host RAM swap needs, how many concurrent requests a mamba pool config supports, or whether to enable swap at all. Complements compute-mamba-ratio (KV/state pool split) and hybrid-cache-checkpoints (int8 prefix-checkpoint tiers).
---

# Sizing the mamba state-swap tier (`--mamba-swap-size`)

The swap tier (`--enable-mamba-state-swap`) is virtual memory for the **mamba
state pool**: a request that loses or never wins a GPU row is *parked* — its
recurrent state (the full footprint: ping-pong pair + tracking rows) is copied
to one host **record** and woken back losslessly when a row frees. It decouples
the number of **live** requests from the number of GPU state rows; nothing is
re-prefilled.

Key semantics (all verified in `kv_cache_configurator.py` /
`managers/mamba_swap_scheduler.py`):

- `--mamba-swap-size S` counts **records, one per parked request** — not rows.
  Each record's width = the request's row footprint `F` (variable width, v2.5).
- The live-request cap is exactly `M // ratio + S` (`ratio = F` at pricing;
  logged at boot as `swap v2: radix-on mode ... %d live`).
- Parked requests **hold on to their record** until they are woken; wake needs
  free GPU rows ≥ the record's parked width.
- Records are **bf16-exact** regardless of int8 checkpoints. The int8
  checkpoint tiers (`--enable-int8-mamba-checkpoint[-offload]`,
  `--int8-mamba-ckpt-host-size`) are a **separate pool** for radix-tree prefix
  checkpoints (~`B_int8_row ≈ B_fp/2` bytes/slot) and never share the swap
  budget.

## Inputs to collect

1. `B_fp` — bytes of one live (bf16) state row. From the model config:
   `B_fp = n_linear_layers · (n_v_heads · d_k · d_v + conv_kernel · conv_channels) · dtype_bytes`
   (temporal state + conv window). Printed by the pool at boot; sanity-check
   against config.json.
2. Serving strategy: `--mamba-radix-cache-strategy` and whether
   `--enable-int8-mamba-checkpoint` is on (the table below prices both).
3. `N` = live requests wanted (`--max-running-requests`), and either `M`
   (`--max-mamba-cache-size`) or the VRAM/RAM budgets to derive it.

## Footprint / pricing (GPU rows per live request)

| Config | boots | guaranteed (use this) |
|---|---|---|
| radix off (`--disable-radix-cache`) | 1 | **1** |
| int8 + `no_buffer` | 1 | **1** |
| int8 + `extra_buffer_lazy` | 2 | **3** |
| int8 + `extra_buffer` | 3 | **3** |
| fp radix + `no_buffer` | 3 | **3** |
| fp radix + `extra_buffer_lazy` | 4 | **5** |
| fp radix + `extra_buffer` | 5 | **5** |

(int8 rows: `1 + tracking slots`; fp rows: base ratio 3 + tracking slots —
`mem_cache/allocation_sizing.py::get_mamba_tracking_slots`. `lazy` *grows* to
the guaranteed number under radix pressure; growth is admission-gated by
`state_slots_fit`, so sizing on the boot number **throttles to the waiting
queue**, never overcommits.)

## The formulas

```
ratio  = guaranteed footprint from the table (int8 + either extra_buffer* → 3)
S      = max(0, N − ⌊M / ratio⌋)              # minimum records for N live
N      = ⌊M / ratio⌋ + S                       # inverse: live capacity given M,S
M      = ratio · (N − S)  ... to hit exactly N with S records
RAM    = S · F · B_fp            +  ckpt_host_slots · B_int8_row   # pinned host
GPU    = M · B_fp  (+ spec-decode scratch, charged automatically)
```

Undersizing `S` is a throttle, not a crash: the configurator clamps the live
cap and excess requests queue. Records themselves are allocated on demand from
the preallocated record pool — size `S` to the target parked count, no penalty
for records that stay empty beyond pool memory.

## Worked example — Qwen3.8-Flash-Next (GDN, 36 linear + 12 full layers)

`B_fp`: temporal 36·48·128·128·2 B = 56.6 MB + conv 36·3·10240·2 B = 2.2 MB →
**≈59 MB/row bf16**; int8 ckpt row ≈31 MB. With int8 + extra_buffer(_lazy),
ratio = 3 → 177 MB per swap record:

| N | M | residents ⌊M/3⌋ | S | swap RAM | + ckpt host 128 slots |
|---|---|---|---|---|---|
| 8 | 12 | 4 | **4** | 0.71 GB | +4 GB |
| 8 | 9 | 3 | **5** | 0.88 GB | +4 GB |
| 16 | 12 | 4 | **12** | 2.1 GB | — |
| 64 | 24 | 8 | **56** | 9.9 GB | — |

## When swap does NOT pay

Swap is time-slicing: only residents (`M//ratio`) decode per step; a parked
request waits for a wake (~4 ms per 3-row record on PCIe 5, one D2H/H2D pair
per park/wake). Per-request decode TPS ≈ `resident_batch · residents / N`.
If `N ≲ 4 × residents` (i.e. a no-swap `M ≥ ratio·N` fits in VRAM), prefer
no swap at all: all requests stay resident, best latency, similar total
memory. Enable swap when `N ≫ residents` and the pinned-RAM budget allows —
S is the cheap capacity knob; raise M for more residents and rotation quality.

## Gotchas

- `extra_buffer_lazy` looks cheaper at boot (ratio 2) but guarantees only
  ⌊M/3⌋ residents — size S on 3 or accept queue throttling after growth.
- Speculative decoding is supported (warning at boot): verify scratch is GPU
  fixed and charged automatically (`_swap_extra()`); drafts that own recurrent
  state resume stale after wake = acceptance-rate cost only (target verifies).
  `--enable-linear-replayssm-spec` is admitted with swap in the static verify
  layout (`SGLANG_RAGGED_VERIFY_MODE=static`, the default): GDN's ring +
  cursors ride the parked request's request-pool row, KDA's fold ring rides
  the parked slot. Ragged modes stay rejected.
- Radix-on swap requires `--enable-int8-mamba-checkpoint` (plain
  `--enable-hierarchical-cache` is rejected with int8).
- Current-server audit: compute the *existing* `M//ratio` first — eager
  `extra_buffer` with `M = N` silently leaves only `N//3` residents.

## Verify

- Boot log: `swap v2: radix-on mode, active=M swap=S, N live requests...`
- Scheduler swap counters (park/wake counts, `parked_num_rows` per record);
  `free -g` delta for pinned records; `nvidia-smi` for the GPU pool.
- GPU acceptance driver: `.scratch/accept/spec_swap_accept.py` (spec+swap
  logprob + acceptance-length parity vs the no-swap arm).
