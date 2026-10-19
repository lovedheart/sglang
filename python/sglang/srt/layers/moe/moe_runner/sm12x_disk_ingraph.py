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

import torch
import triton
import triton.language as tl
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

@triton.jit
def _resolve(
    present,  # [L*E] int32, 1 at (layer, expert) keys in this launch's ids
    miss,  # [E] int32 out: 1 if this launch allocated+filled the expert
    gslot,  # [L*E] int32, (layer, expert) -> slot or -1
    owner,  # [K] int32, slot -> (layer, expert) or -1
    age,  # [K] int64, last-use clock
    clock,  # [1] int64, bumped once per launch before this kernel
    idmap,  # [E+1] int32 out, expert -> slot (row E / absent = K)
    E: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    LAYER: tl.constexpr,
):
    # Single program, vectorized over slots: one writer per launch makes
    # the allocation race-free and reproduces the host LRU exactly
    # (free-slot-first, then oldest-age, never evicting a key this same
    # launch also needs — impossible anyway since distinct present keys
    # never exceed K >= E).
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
        else:
            own = tl.load(owner + kp, mask=in_k, other=-2)
            v = tl.min(tl.where(own == -1, kp, K), axis=0)
            if v >= K:  # no free slot: evict the oldest unprotected key
                prot = tl.load(present + own, mask=own >= 0, other=1)
                live = in_k & (own >= 0) & (prot == 0)
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
    moff = (LAYER * E + e).to(tl.int64) * PAY
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

        # Host RAM guard: refuse to pin more than a safe share of what is
        # currently available.
        with open("/proc/meminfo") as fh:
            avail = next(
                int(line.split()[1]) for line in fh if line.startswith("MemAvailable")
            ) << 10
        need = self.nlayers * e * pay_bytes
        if need + (2 << 30) > avail:
            raise RuntimeError(
                f"in-graph paging needs {need / (1 << 30):.1f} GiB pinned of "
                f"{avail / (1 << 30):.1f} GiB available host memory; run "
                "without SGLANG_SM12X_DISK_INGRAPH (breakable graphs)."
            )
        logger.info(
            "sm12x disk: building %.2f GiB pinned expert mirror "
            "(in-graph full-graph-decode mode)",
            need / (1 << 30),
        )

        self.mirror = torch.empty(need, dtype=torch.uint8, pin_memory=True)
        m64 = self.mirror.view(torch.int64)
        for layer in self.layer_ids:
            for ex in range(e):
                p = registry.payload(layer, ex)
                base = (layer * e + ex) * self.pay
                # dst_offset is per-TENSOR (w13 and w2 slot views start at
                # 0); the mirror row is [w13 | w2], so w2 segments land
                # after the w13 window.
                for seg in p.w13:
                    assert seg.dst_offset % 8 == 0 and seg.nbytes % 8 == 0
                    blob = registry.pread(seg.shard, seg.src_offset, seg.nbytes)
                    dst = m64[base + seg.dst_offset // 8 :]
                    dst[: seg.nbytes // 8].copy_(
                        torch.frombuffer(blob, dtype=torch.uint8).view(torch.int64)
                    )
                for seg in p.w2:
                    assert seg.dst_offset % 8 == 0 and seg.nbytes % 8 == 0
                    blob = registry.pread(seg.shard, seg.src_offset, seg.nbytes)
                    dst = m64[base + self.w13 + seg.dst_offset // 8 :]
                    dst[: seg.nbytes // 8].copy_(
                        torch.frombuffer(blob, dtype=torch.uint8).view(torch.int64)
                    )

        # Slot pool device state.
        self.gslot = torch.full((self.nlayers * e,), -1, dtype=torch.int32, device=device)
        self.owner = torch.full((self.k,), -1, dtype=torch.int32, device=device)
        self.age = torch.zeros((self.k,), dtype=torch.int64, device=device)
        self.clock = torch.zeros((1,), dtype=torch.int64, device=device)
        self.present = torch.zeros((self.nlayers * e,), dtype=torch.int32, device=device)
        self.ones = torch.ones((4096,), dtype=torch.int32, device=device)
        self._state: dict[int, dict] = {}
        self._calls = 0

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

    def reset(self) -> None:
        self.gslot.fill_(-1)
        self.owner.fill_(-1)
        self.age.zero_()
        self._state.clear()

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
        ones = self.ones[: n] if n <= 4096 else torch.ones(
            n, dtype=torch.int32, device=self.device
        )
        self.present.scatter_reduce_(0, keys, ones, reduce="amax")
        self.clock.add_(1)
        _resolve[(1,)](
            self.present,
            st["miss"],
            self.gslot,
            self.owner,
            self.age,
            self.clock,
            st["idmap"],
            E=self.e,
            K=self.k,
            KP=max(triton.next_power_of_2(self.k), 2),
            LAYER=layer_id,
        )
        (
            slot_w13,
            slot_w2,
            slot_s13,
            slot_s2,
            slot_g13,
            slot_g2,
        ) = self._dst
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
            W13=self.w13,
            W2=self.w2,
            SF13=self.sf13,
            SF2=self.sf2,
            PAY=self.pay,
            E=self.e,
            LAYER=layer_id,
            BLOCK=self.block,
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
