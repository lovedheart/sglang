import sys

import numpy as np
import torch

sys.path.insert(0, "/home/lovedheart/sglang-fork-dev/python")

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
from sglang.kernels.ops.elementwise.fast_topk import _jit_fast_topk_module

mod = _jit_fast_topk_module(512)
x = np.random.default_rng(0).random((3, 16384)).astype(np.float32)
x_gpu = torch.from_numpy(x).cuda()
lengths = torch.full((3,), 16384, device="cuda", dtype=torch.int32)
outs = []
for _ in range(4):
    out = mod.fast_topk(x_gpu, lengths, 512)
    outs.append(np.sort(out[0].cpu().numpy()))
ref = np.sort(np.argsort(-x[0])[:512])
print("all identical:", all(np.array_equal(o, outs[0]) for o in outs))
print("vs ref:", np.array_equal(outs[0], ref))
print("diff:", np.setdiff1d(outs[0], ref)[:5], np.setdiff1d(ref, outs[0])[:5])
