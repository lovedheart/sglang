# SPDX-License-Identifier: Apache-2.0
"""Decode launcher (split-K / fused) on Hy4-Flash SM120 shapes: the rope tail
is padded to ``next_pow2(D_TAIL)`` inside the kernels, so a D_TAIL=320 model
with d_v=256 stages 768 elements per key -- more than the 576-wide cache row
the tile heuristic budgets -- and the launch used to die with OutOfResources
(or an illegal access when captured in a CUDA graph). These tests pin that the
launcher now steps the KV tile down and still matches the fp32 reference.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SM_SCALE = 0.03
_D_V, _D_TAIL, _TOPK = 256, 320, 2048  # Hy4-Flash: v_head 256 + rope tail 320
_H = 32  # heads at TP1


def _qkv(bs, S, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_nope = (
        torch.randn(bs, _H, _D_V, dtype=torch.bfloat16, device="cuda", generator=g)
        * 0.2
    )
    q_rope = (
        torch.randn(bs, _H, _D_TAIL, dtype=torch.bfloat16, device="cuda", generator=g)
        * 0.2
    )
    kv = (torch.randn(S, _D_V + _D_TAIL, device="cuda", generator=g) * 0.05).to(
        torch.bfloat16
    )
    return q_nope, q_rope, kv, g


def _reference(q_nope, q_rope, kv, idx, length, sink):
    rows = idx[0, : length[0].item()].long()
    k = kv[rows].float()
    q = torch.cat([q_nope[0], q_rope[0]], dim=-1).float()
    s = (q @ k.t()) * SM_SCALE
    if sink is not None:
        s = torch.cat([s, sink.view(_H, 1).to(s.dtype)], dim=-1)
    p = torch.softmax(s, dim=-1)[:, : rows.numel()]
    return p @ k[:, :_D_V]


class TestTritonSparseMLADecodeHy4Shape(CustomTestCase):
    def test_length_limited_splitk_matches_reference(self):
        q_nope, q_rope, kv, g = _qkv(1, 8192, seed=13)
        sink = torch.randn(_H, dtype=torch.float32, device="cuda", generator=g) * 2
        for ctx in (257, 2048):
            with self.subTest(ctx=ctx):
                idx = torch.full((1, _TOPK), -1, dtype=torch.int32, device="cuda")
                idx[0, :ctx] = torch.arange(ctx, dtype=torch.int32, device="cuda")
                length = torch.tensor([ctx], dtype=torch.int32, device="cuda")
                out = triton_sparse_mla_decode_splitk(
                    q_nope,
                    q_rope,
                    kv,
                    idx.unsqueeze(1),
                    SM_SCALE,
                    _D_V,
                    workspace=[],
                    lengths=length,
                    attn_sink=sink,
                )
                ref = _reference(q_nope, q_rope, kv, idx, length, sink)
                rel = (out[0].float() - ref).norm() / ref.norm()
                self.assertLess(rel.item(), 2e-2, f"ctx={ctx} rel={rel.item():.3e}")

    def test_exact_topk_two_rows_no_sink(self):
        q_nope, q_rope, kv, g = _qkv(2, 8192, seed=17)
        idx = (
            torch.randperm(8192, device="cuda", generator=g)[: 2 * _TOPK]
            .view(2, _TOPK)
            .to(torch.int32)
            .contiguous()
        )
        out = triton_sparse_mla_decode_splitk(
            q_nope, q_rope, kv, idx.unsqueeze(1), SM_SCALE, _D_V, workspace=[]
        )
        self.assertTrue(torch.isfinite(out.float()).all())
        ref = _reference(
            q_nope, q_rope, kv, idx, torch.tensor([_TOPK], dtype=torch.int32), None
        )
        rel = (out[0, 0].float() - ref).norm() / ref.norm()
        self.assertLess(rel.item(), 2e-2)


if __name__ == "__main__":
    unittest.main()
