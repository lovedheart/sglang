"""In-graph device-side expert paging for the SM12x disk MoE runner.

Decode with a FULL (unbreakable) CUDA graph cannot call the host fill path
(no ``pread``, no D2H sync, no Python LRU). This module moves the entire
miss/fill decision onto the GPU so the whole decode step captures:

  * every routed expert payload is mirrored once into a pinned host buffer
    at cache-build time (host RAM stands in for the disk during replay);
  * a resolve kernel replicates the host LRU (free-slot / oldest-slot CAS
    allocation) over ``(layer, expert) -> slot`` device tables;
  * a fill kernel gathers missed experts pinned -> VRAM (payloads from the
    mirror, scales and ``scale_2`` scalars from the resident quant_info
    tensors), all with fixed kernel arguments and device-computed offsets,
    which is exactly what graph replay requires.

Only checkpoints whose scale factors are VRAM-resident (layout A) are
supported; streamed-scale layouts still need the breakable path.
"""

from __future__ import annotations

import logging
import threading
from array import array
from collections import deque

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ACTIVE_PAGER = None


def get_ingraph_pager():
    """The live InGraphPager, or None when in-graph paging is inactive."""
    return _ACTIVE_PAGER


def _mem_available() -> int:
    with open("/proc/meminfo") as fh:
        return (
            next(int(line.split()[1]) for line in fh if line.startswith("MemAvailable"))
            << 10
        )


def _has_inplace_recurrent_state() -> bool:
    # True when the served model updates recurrent state in place during
    # decode (mamba / KDA / GDN layers): then a replayed step would
    # advance that state twice, so budgeted mirroring is not allowed.
    try:
        from sglang.srt.runtime_context import get_exec

        mb = get_exec().mamba
        return bool(
            getattr(mb, "uses_mamba_radix_cache", False)
            or getattr(mb, "linear_attn_backend", None) not in (None, "none")
        )
    except (ValueError, AttributeError):
        # Harnesses without a populated runtime context (unit tests): no
        # model runs here, so replay-idempotence is trivially true.
        return False


@triton.jit
def _resolve(
    present,  # [L*E] int32, 1 at (layer, expert) keys in this launch's ids
    miss,  # [E] int32 out: 1 if this launch allocated+filled the expert
    gslot,  # [L*E] int32, (layer, expert) -> slot or -1
    owner,  # [K] int32, slot -> (layer, expert) or -1
    age,  # [K] int64, last-use clock
    clock,  # [1] int64, bumped once per launch before this kernel
    idmap,  # [E+1] int32 out, expert -> slot (row E / absent = K)
    mslot,  # [L*E] int32, (layer, expert) -> pinned mirror row or -1
    ring,  # host-mapped int32 ring: [0] cursor, [1..R] starved keys
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    R: tl.constexpr,
    LAYER: tl.constexpr,
):
    # Single program, vectorized over slots: one writer per launch makes
    # the allocation race-free and reproduces the host LRU exactly
    # (free-slot-first, then oldest-age, never evicting a key this same
    # launch also needs — impossible anyway since distinct present keys
    # never exceed K >= E). With a budgeted mirror, victims prefer slots
    # whose key still has a mirror row (refill = PCIe only), and a miss
    # whose key is not mirrored at all "starves": the lane parks on the
    # sentinel slot and the key goes into the host ring for the refiller
    # thread; the whole step is then replayed (see InGraphPager.end_step).
    if tl.program_id(0) != 0:
        return
    clk = tl.load(clock)
    kp = tl.arange(0, KP)
    in_k = kp < K
    tl.store(idmap + E, K)  # padding row -> sentinel slot
    for e in tl.range(0, E):
        gk = LAYER * E + e
        s = tl.load(gslot + gk)
        if tl.load(present + gk) == 0:
            tl.store(idmap + e, K)
            tl.store(miss + e, 0)
        elif s >= 0:
            tl.store(miss + e, 0)
            tl.store(age + s, clk)
            tl.store(idmap + e, s)
        elif tl.load(mslot + gk) < 0:
            # Starved: no VRAM slot and no mirror row. Park the lane and
            # tell the host; do not touch gslot/owner/age.
            tl.store(miss + e, 0)
            tl.store(idmap + e, K)
            pos = tl.atomic_add(ring, 1)
            tl.store(ring + 1 + (pos % R), gk)
        else:
            own = tl.load(owner + kp, mask=in_k, other=-2)
            v = tl.min(tl.where(own == -1, kp, K), axis=0)
            if v >= K:  # no free slot: evict the oldest unprotected key
                prot = tl.load(present + own, mask=own >= 0, other=1)
                live = in_k & (own >= 0) & (prot == 0)
                # Prefer victims whose key can be refetched from the
                # mirror; with a full mirror every live slot qualifies,
                # so this reduces to the plain oldest-unprotected pick.
                mir = tl.load(mslot + own, mask=live, other=-1)
                live0 = live & (mir >= 0)
                if tl.max(live0.to(tl.int32), axis=0) == 1:
                    a0 = tl.load(age + kp, mask=live0, other=9223372036854775807)
                    amin = tl.min(a0, axis=0)
                    v = tl.min(tl.where(live0 & (a0 == amin), kp, K), axis=0)
                else:
                    a = tl.load(age + kp, mask=live, other=9223372036854775807)
                    amin = tl.min(a, axis=0)
                    v = tl.min(tl.where(live & (a == amin), kp, K), axis=0)
                old = tl.load(owner + v)
                if old >= 0:
                    tl.store(gslot + old, -1)
            else:
                old = tl.load(owner + v)  # -1
            tl.store(owner + v, gk)
            tl.store(gslot + gk, v)
            tl.store(age + v, clk)
            tl.store(miss + e, 1)
            tl.store(idmap + e, v)


@triton.jit
def _fill(
    mirror,  # [L*E*PAY] int64 pinned host mirror (UVA)
    slot_w13,  # [K, W13] int64
    slot_w2,  # [K, W2] int64
    slot_s13,  # [K, SF13] int64 (resident scales destination)
    slot_s2,  # [K, SF2] int64
    res_s13,  # [E, SF13] int64 resident scales source
    res_s2,  # [E, SF2] int64
    g13,  # [E] fp32 scale_2 source
    g2,  # [E] fp32
    slot_g13,  # [K] fp32 destination
    slot_g2,  # [K] fp32
    miss,  # [E] int32
    gslot,  # [L*E] int32
    mslot,  # [L*E] int32, (layer, expert) -> pinned mirror row
    W13: tl.constexpr,
    W2: tl.constexpr,
    SF13: tl.constexpr,
    SF2: tl.constexpr,
    PAY: tl.constexpr,  # W13 + W2
    E: tl.constexpr,
    LAYER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    e = tl.program_id(0)
    c = tl.program_id(1)
    if tl.load(miss + e) == 0:
        return
    s = tl.load(gslot + LAYER * E + e).to(tl.int64)
    e64 = e.to(tl.int64)
    moff = tl.load(mslot + LAYER * E + e).to(tl.int64) * PAY
    lane = c * BLOCK + tl.arange(0, BLOCK)
    # Payload regions come from the pinned mirror.
    o = lane
    m = o < W13
    tl.store(slot_w13 + s * W13 + o, tl.load(mirror + moff + o, mask=m), mask=m)
    o = lane - W13
    m = (o >= 0) & (o < W2)
    tl.store(slot_w2 + s * W2 + o, tl.load(mirror + moff + W13 + o, mask=m), mask=m)
    # Scale-factor regions come from the resident tensors (layout A).
    o = lane - PAY
    m = (o >= 0) & (o < SF13)
    tl.store(slot_s13 + s * SF13 + o, tl.load(res_s13 + e64 * SF13 + o, mask=m), mask=m)
    o = lane - PAY - SF13
    m = (o >= 0) & (o < SF2)
    tl.store(slot_s2 + s * SF2 + o, tl.load(res_s2 + e64 * SF2 + o, mask=m), mask=m)
    if c == 0:
        tl.store(slot_g13 + s, tl.load(g13 + e))
        tl.store(slot_g2 + s, tl.load(g2 + e))


def _as_lanes(t: torch.Tensor) -> torch.Tensor:
    """uint8/other tensor -> int64 view for 8-byte gather lanes."""
    flat = t.reshape(-1)
    assert flat.numel() % 8 == 0, "region sizes must be int64-multiples"
    return flat.view(torch.int64)


class InGraphPager:
    """Device-side LRU + pinned-mirror fill; see module docstring."""

    def __init__(self, cache, registry, quant_info, num_experts: int, device):
        e = num_experts
        self.e = e
        self.k = cache.k
        self.device = device
        self.layer_ids = sorted({layer for layer, _ in registry._payloads})
        self.nlayers = max(self.layer_ids) + 1

        payload0 = registry.payload(self.layer_ids[0], 0)
        pay_bytes = sum(s.nbytes for s in payload0.w13 + payload0.w2)
        assert pay_bytes % 8 == 0
        self.pay = pay_bytes // 8  # int64 lanes
        self.w13 = cache.slot_w13.reshape(-1).numel() // 8 // cache.k
        self.w2 = cache.slot_w2.reshape(-1).numel() // 8 // cache.k

        # Mirror sizing: full (one row per expert) unless a budget is set.
        # Budgeted mode is only safe when a decode step is replay-
        # idempotent (pure attention state); models with in-place
        # recurrent state (mamba/KDA/GDN) silently keep the full mirror.
        total_rows = self.nlayers * e
        need = total_rows * pay_bytes
        budget = int(envs.SGLANG_SM12X_DISK_MIRROR_MB.get()) << 20
        if budget and _has_inplace_recurrent_state():
            logger.info(
                "sm12x disk: mirror budget ignored — this model keeps "
                "in-place recurrent state, so a starved step could not be "
                "replayed; mirroring all %.2f GiB instead",
                need / (1 << 30),
            )
            budget = 0
        if not budget and need + (2 << 30) > _mem_available():
            raise RuntimeError(
                f"in-graph paging needs {need / (1 << 30):.1f} GiB pinned of "
                f"{_mem_available() / (1 << 30):.1f} GiB available host "
                "memory; run without SGLANG_SM12X_DISK_INGRAPH (eager fill) "
                "or set SGLANG_SM12X_DISK_MIRROR_MB to a smaller budget."
            )
        rows = (
            total_rows if not budget else max(min(budget // pay_bytes, total_rows), 2)
        )
        dbg_rows = int(envs.SGLANG_SM12X_DISK_DEBUG_MIRROR_ROWS.get())
        if dbg_rows:
            rows = max(min(dbg_rows, total_rows), 1)
            if _has_inplace_recurrent_state():
                logger.warning(
                    "sm12x disk: DEBUG_MIRROR_ROWS on a model with in-place "
                    "recurrent state — starved-step replays are numerically "
                    "invalid; benchmark use only"
                )
        self.budgeted = rows < total_rows
        self.nrows = rows
        logger.info(
            "sm12x disk: %s pinned expert mirror in RAM, %.2f GiB "
            "(%d/%d rows, %.1f%% coverage)%s",
            "budgeted LRU" if self.budgeted else "full",
            rows * pay_bytes / (1 << 30),
            rows,
            total_rows,
            100.0 * rows / total_rows,
            " — starved steps replay after refill" if self.budgeted else "",
        )
        vram_bytes = sum(
            t.numel() * t.element_size()
            for t in (
                cache.slot_w13,
                cache.slot_w2,
                cache.slot_w13_sf,
                cache.slot_w2_sf,
                cache.slot_w13_gs,
                cache.slot_w2_gs,
            )
        )
        _sf_gs = cache.slot_vram_bytes - cache.slot_bytes
        req_slots = int(envs.SGLANG_SM12X_DISK_VRAM_EXPERT_SLOTS.get())
        if cache.k != req_slots:
            logger.info(
                "sm12x disk: requested %d expert slots -> %d slots "
                "(pow-2 round-up, floor E=%d)",
                req_slots,
                cache.k,
                self.e,
            )
        logger.info(
            "sm12x disk: expert slots in VRAM: %d slots x %.3f MiB/slot "
            "(payload %.3f + scales %.3f) = %.2f GiB vram total, "
            "covering %.1f%% of the %d experts; "
            "contents LRU-chosen per step",
            cache.k,
            cache.slot_vram_bytes / (1 << 20),
            cache.slot_bytes / (1 << 20),
            _sf_gs / (1 << 20),
            vram_bytes / (1 << 30),
            100.0 * cache.k / total_rows,
            total_rows,
        )

        self.mirror = torch.empty(rows * pay_bytes, dtype=torch.uint8, pin_memory=True)
        self._m64 = self.mirror.view(torch.int64)
        self._registry = registry
        # Host-side mirror tables (device mslot is published from these).
        self._h_ms = array("i", [-1] * total_rows)  # key -> row
        self._row_key = [-1] * rows  # row -> key
        self._row_of = array("i", [-1] * total_rows)  # key -> row (inverse)
        self._free_rows = deque(range(rows))
        self._fifo = deque()
        self._pinned: set[int] = set()
        self._pend_idx: list[int] = []
        self._pend_val: list[int] = []
        if not self.budgeted:
            for layer in self.layer_ids:  # dense layers have no payloads
                for ex in range(e):
                    self._load_row(layer * e + ex, row=layer * e + ex)

        # Device mirror map: key -> pinned row or -1.
        self.mslot = torch.tensor(self._h_ms, dtype=torch.int32, device=device)
        self._pend_idx.clear()
        self._pend_val.clear()

        # Starve ring in host-mapped pinned memory: [0] = write cursor,
        # [1..R] = keys the device resolver could not serve.
        self._R = 16384
        self.ring = torch.zeros(1 + self._R, dtype=torch.int32, pin_memory=True)
        self._ring_np = self.ring.numpy()
        self._seen = 0
        self._mu = threading.Lock()

        # Slot pool device state.
        self.gslot = torch.full(
            (self.nlayers * e,), -1, dtype=torch.int32, device=device
        )
        self.owner = torch.full((self.k,), -1, dtype=torch.int32, device=device)
        self.age = torch.zeros((self.k,), dtype=torch.int64, device=device)
        self.clock = torch.zeros((1,), dtype=torch.int64, device=device)
        self.present = torch.zeros(
            (self.nlayers * e,), dtype=torch.int32, device=device
        )
        self.ones = torch.ones((4096,), dtype=torch.int32, device=device)
        self._state: dict[int, dict] = {}
        self._calls = 0
        self.n_replays = 0
        self.n_starved_keys = 0
        self._watch = -1
        global _ACTIVE_PAGER
        _ACTIVE_PAGER = self

        # Region lane sizes.
        if quant_info.w13_sf.device.type == "meta":
            raise NotImplementedError(
                "in-graph paging supports layout A (resident scales) only"
            )
        self.sf13 = quant_info.w13_sf.reshape(-1).numel() // 8 // e
        self.sf2 = quant_info.w2_sf.reshape(-1).numel() // 8 // e
        assert quant_info.w13_gs.ndim == 1 and quant_info.w2_gs.ndim == 1
        self.total = self.pay + self.sf13 + self.sf2
        self.block = 4096
        self.chunks = (self.total + self.block - 1) // self.block
        # Slot destinations are pool-wide; the scale SOURCES are per-layer
        # (each layer's own quant_info) and bind lazily in fill().
        self._dst = (
            cache.slot_w13,
            cache.slot_w2,
            cache.slot_w13_sf,
            cache.slot_w2_sf,
            cache.slot_w13_gs,
            cache.slot_w2_gs,
        )

    def _load_row(self, key: int, row: int | None = None) -> None:
        """pread an expert payload into a pinned mirror row (caller holds the
        lock, or is in single-threaded init)."""
        if row is None:
            if self._free_rows:
                row = self._free_rows.popleft()
            else:
                victim = self._fifo.popleft()
                while self._fifo and victim in self._pinned:
                    # Rows needed by an in-flight starve retry must not be
                    # stolen mid-step, or the retry never converges.
                    self._fifo.append(victim)
                    victim = self._fifo.popleft()
                if victim in self._pinned:
                    raise RuntimeError(
                        "sm12x disk: mirror budget too small for one step's "
                        "expert working set — raise SGLANG_SM12X_DISK_MIRROR_MB"
                    )
                row = self._row_of[victim]
                self._row_of[victim] = -1
                self._row_key[row] = -1
                self._h_ms[victim] = -1
                # The device MUST learn the row was stolen before its
                # next resolve, or it re-serves the stale (now overwritten)
                # row: silent corruption, never a re-starve.
                self._pend_idx.append(victim)
                self._pend_val.append(-1)
        layer, ex = divmod(key, self.e)
        p = self._registry.payload(layer, ex)
        base = row * self.pay
        # dst_offset is per-TENSOR (w13 and w2 slot views start at 0); the
        # mirror row is [w13 | w2], so w2 segments land after the w13 window.
        for lane0, segs in ((0, p.w13), (self.w13, p.w2)):
            for seg in segs:
                blob = self._registry.pread(seg.shard, seg.src_offset, seg.nbytes)
                dst = self._m64[base + lane0 + seg.dst_offset // 8 :]
                dst[: seg.nbytes // 8].copy_(
                    torch.frombuffer(blob, dtype=torch.uint8).view(torch.int64)
                )
        self._h_ms[key] = row
        self._row_of[key] = row
        self._row_key[row] = key
        self._fifo.append(key)
        self._pend_idx.append(key)
        self._pend_val.append(row)

    def publish_pending(self) -> None:
        """H2D the mirror-map edits accumulated since the last step. Must
        run between replays (never inside a capture)."""
        if not self._pend_idx:
            return
        idx = torch.tensor(self._pend_idx, dtype=torch.long, device=self.device)
        val = torch.tensor(self._pend_val, dtype=torch.int32, device=self.device)
        self.mslot[idx] = val
        self._pend_idx.clear()
        self._pend_val.clear()

    def _drain_starve(self) -> int:
        """Process starve-ring records: pread missing mirror rows (caller
        thread, under the lock, one budget-sized chunk) and publish the
        mirror-map edits. Returns #starved keys seen; >0 means the caller
        must relaunch so the resolver can land the new rows into slots —
        keys that still lack a row re-park on the ring, so repeating
        (drain, relaunch) converges for ANY mirror budget. Caller must not
        be inside a capture."""
        torch.cuda.current_stream().synchronize()
        self.publish_pending()
        cur = int(self._ring_np[0])
        if cur == self._seen:
            return 0
        keys = sorted(
            {int(self._ring_np[1 + (i % self._R)]) for i in range(self._seen, cur)}
        )
        if cur - self._seen > self._R - 2:
            logger.warning("sm12x disk: starve ring overflow, entries dropped")
        self._seen = cur
        pending = [k for k in keys if 0 <= k < len(self._h_ms)]
        self.n_starved_keys += len(pending)
        resident = set(self.owner.cpu().tolist())
        pending = [k for k in pending if k not in resident]
        missing = [k for k in pending if self._h_ms[k] < 0]
        if bool(envs.SGLANG_SM12X_DISK_DEBUG.get()) and pending:
            gs = self.gslot.cpu()
            self._watch = missing[0] if missing else pending[0]
            print(
                f"P12DRAIN n={len(pending)} miss={len(missing)} "
                f"fifo={len(self._fifo)} sample="
                + " ".join(f"{k}:g{int(gs[k])}/h{self._h_ms[k]}" for k in pending[:8]),
                flush=True,
            )
        if missing:
            chunk = missing[: max(1, self.nrows)]
            with self._mu:
                # Pin the chunk so burst evictions cannot steal rows we
                # are about to fill (the refiller honours _pinned too).
                self._pinned = set(chunk)
                try:
                    for k in chunk:
                        if self._h_ms[k] < 0:
                            self._load_row(k)
                finally:
                    self._pinned = set()
            self.publish_pending()
        return len(pending)

    def end_step(self, rerun) -> object:
        """Post-forward hook: if the step starved, refill and re-run it
        until every expert the step touched is served from a slot.

        Only budgeted mode pays anything here (stream sync + cursor peek);
        full mirrors never write the ring, so the check is a cached int
        compare. rerun() re-executes the identical forward (valid because
        budgeted mode is restricted to replay-idempotent models).
        """
        if not self.budgeted:
            return None
        out = None
        for _ in range(512):
            if bool(envs.SGLANG_SM12X_DISK_DEBUG.get()):
                print(
                    f"P12STEP r{_} cur={int(self._ring_np[0])} "
                    f"seen={self._seen} "
                    f"gs={int((self.gslot >= 0).sum())} "
                    f"own={int((self.owner >= 0).sum())} "
                    f"ms={int((self.mslot >= 0).sum())} "
                    f"free={len(self._free_rows)}"
                    + (
                        f" w{self._watch}:g{int(self.gslot[self._watch].item())}"
                        f"/m{int(self.mslot[self._watch].item())}"
                        f"/h{self._h_ms[self._watch]}"
                        if getattr(self, "_watch", -1) >= 0
                        else ""
                    ),
                    flush=True,
                )
            if not self._drain_starve():
                return out
            self.n_replays += 1
            out = rerun()
        raise RuntimeError(
            "sm12x disk: starve retry did not converge — check that the "
            "slot pool (K) can hold one step's distinct experts"
        )

    def reset(self) -> None:
        self.gslot.fill_(-1)
        self.owner.fill_(-1)
        self.age.zero_()
        self._state.clear()
        # Drop starve records from discarded steps (capture warmups run
        # the resolver on dummy ids; those rows must not trigger refills).
        self._seen = int(self._ring_np[0])

    def fill(self, layer_id: int, topk_ids: torch.Tensor, quant_info) -> torch.Tensor:
        """Resolve + fill for this launch; returns the [E+1] expert->slot map."""
        st = self._state.get(layer_id)
        if st is None:
            # Bind this layer's resident scale tensors (persistent, so the
            # captured graph may reference their addresses on replay).
            self._state[layer_id] = st = {
                "idmap": torch.full(
                    (self.e + 1,), self.k, dtype=torch.int32, device=self.device
                ),
                "miss": torch.zeros((self.e,), dtype=torch.int32, device=self.device),
                "res13": _as_lanes(quant_info.w13_sf),
                "res2": _as_lanes(quant_info.w2_sf),
                "g13": quant_info.w13_gs,
                "g2": quant_info.w2_gs,
            }
        ids = topk_ids.reshape(-1)
        n = ids.numel()
        self.present.zero_()
        keys = (layer_id * self.e + ids.long()).clamp(max=self.nlayers * self.e - 1)
        ones = (
            self.ones[:n]
            if n <= 4096
            else torch.ones(n, dtype=torch.int32, device=self.device)
        )
        self.present.scatter_reduce_(0, keys, ones, reduce="amax")
        self.clock.add_(1)
        (
            slot_w13,
            slot_w2,
            slot_s13,
            slot_s2,
            slot_g13,
            slot_g2,
        ) = self._dst

        def _launch():
            _resolve[(1,)](
                self.present,
                st["miss"],
                self.gslot,
                self.owner,
                self.age,
                self.clock,
                st["idmap"],
                self.mslot,
                self.ring,
                E=self.e,
                K=self.k,
                KP=max(triton.next_power_of_2(self.k), 2),
                R=self._R,
                LAYER=layer_id,
            )
            _fill[(self.e, self.chunks)](
                self.mirror.view(torch.int64),
                slot_w13.view(torch.int64),
                slot_w2.view(torch.int64),
                slot_s13.view(torch.int64)
                if slot_s13.dtype != torch.int64
                else slot_s13,
                slot_s2.view(torch.int64) if slot_s2.dtype != torch.int64 else slot_s2,
                st["res13"],
                st["res2"],
                st["g13"],
                st["g2"],
                slot_g13,
                slot_g2,
                st["miss"],
                self.gslot,
                self.mslot,
                W13=self.w13,
                W2=self.w2,
                SF13=self.sf13,
                SF2=self.sf2,
                PAY=self.pay,
                E=self.e,
                LAYER=layer_id,
                BLOCK=self.block,
            )

        _launch()
        if self.budgeted and not torch.cuda.is_current_stream_capturing():
            # Eager launch (prefill/extend, uncaptured sizes): Python owns
            # the launch, so resolve starvation inline — load missing rows
            # and relaunch resolve+fill — without runner-level hooks.
            for _r in range(512):
                if not self._drain_starve():
                    break
                _launch()
            else:
                raise RuntimeError(
                    "sm12x disk: starve retry did not converge — check "
                    "that the slot pool (K) can hold one step's distinct "
                    "experts"
                )
        if (
            bool(envs.SGLANG_SM12X_DISK_DEBUG.get())
            and self._calls < 1000000
            and not torch.cuda.is_current_stream_capturing()
        ):
            self._calls += 1
            im = st["idmap"].cpu()
            import hashlib

            dig = []
            for ex in range(self.e):
                sl = int(im[ex].item())
                if sl >= self.k:
                    continue
                h = hashlib.md5(
                    b"".join(
                        bytes(t[sl].view(torch.uint8).cpu().numpy())
                        for t in (slot_w13, slot_w2, slot_s13, slot_s2)
                    )
                    + bytes(slot_g13[sl : sl + 1].view(torch.uint8).cpu().numpy())
                    + bytes(slot_g2[sl : sl + 1].view(torch.uint8).cpu().numpy())
                ).hexdigest()[:8]
                dig.append(f"{ex}:{sl}:{h}")
            logger.info("IGDBG L%d n=%d %s", layer_id, n, " ".join(dig[:6]))
        return st["idmap"]
