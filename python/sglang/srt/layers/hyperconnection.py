from typing import Optional

import msgspec
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.gemm.hc_mix import (
    fused_hc_mix,
    fused_hc_mix_fold,
    fused_hc_mix_fold_supported,
    fused_hc_mix_supported,
    hc_norm_fold_enabled,
)


class HyperConnectionConfig(msgspec.Struct, frozen=True):
    hc_count: int = 4
    hidden_size: int = 64
    params_dtype: torch.dtype = torch.bfloat16
    mtp_hc: bool = False
    hc_lowrank: int = 16
    rms_norm_eps: float = 1e-6
    hc_per_branch_norm: bool = False


class GroupedGemmaRMSNorm(nn.Module):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, group_size: Optional[int] = None
    ):
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight.weight_loader = self._weight_loader
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def _weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param.size() == loaded_weight.size()
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(
                x, self.weight, self._jit_group_size, self.variance_epsilon
            )
        input_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            x_norm = x_float * torch.rsqrt(variance + self.variance_epsilon)
        else:
            x_grouped = x_float.reshape(
                *x_float.shape[:-1],
                x_float.shape[-1] // self.group_size,
                self.group_size,
            )
            variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
            x_norm = (
                x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            ).flatten(-2)
        return (x_norm * (1.0 + self.weight.float())).to(input_dtype)


class HyperConnectionBase(nn.Module):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__()

        self.config = config
        self.hc_count = config.hc_count
        if config.mtp_hc and role is not None and "mtp" in role:
            self.hc_count = self.hc_count + 1
        self.hidden_size = config.hidden_size
        self.params_dtype = config.params_dtype

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        mixed_input = hyper_input.view(
            *hyper_input.shape[:-1], self.hc_count, self.hidden_size
        ).mean(dim=-2)
        return mixed_input, hyper_input

    def combine(
        self, block_output: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        assert residual.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        residual_reshaped = residual.view(
            *residual.shape[:-1], self.hc_count, self.hidden_size
        )
        combined_output = residual_reshaped + block_output.unsqueeze(-2)
        combined_output = combined_output.view(
            *residual.shape[:-1], self.hc_count * self.hidden_size
        )
        return combined_output


class GatedResidual(HyperConnectionBase):
    def __init__(
        self,
        config: HyperConnectionConfig,
        use_mix: bool = True,
        use_combine: bool = True,
        role: Optional[str] = None,
    ):
        super().__init__(config, use_mix, use_combine, role)

        norm_dim = (
            self.config.hidden_size * self.hc_count
            if self.config.hc_per_branch_norm
            else self.config.hidden_size
        )
        norm_group_size = (
            self.config.hidden_size if self.config.hc_per_branch_norm else None
        )
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_dim, eps=self.config.rms_norm_eps, group_size=norm_group_size
        )

        if use_mix:
            self.input_mix_weight_down = nn.Linear(
                self.hidden_size * self.hc_count,
                self.config.hc_lowrank,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            self.input_mix_weight_up = nn.Linear(
                self.config.hc_lowrank,
                self.hc_count * self.hidden_size,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            lowrank = self.config.hc_lowrank
            self._jit_mix_ok = (
                torch.cuda.is_available()
                # The CuTe split-K pair is tcgen05 (sm_100 family) only.
                and torch.cuda.get_device_capability()[0] == 10
                and (self.hc_count * self.hidden_size) % 2048 == 0
                and self.hidden_size % 8 == 0
                and lowrank > 0
                and lowrank % 8 == 0
            )
            self._mix_up_weight_padded = None

        if use_combine:
            self.block_inject_weight = nn.Linear(
                self.hidden_size * self.hc_count,
                self.hc_count,
                bias=False,
                device=torch.get_device_module().current_device(),
                dtype=config.params_dtype,
            )
            # hc_combine rejects other shapes; device and dtype are checked per call.
            self._jit_combine_ok = (
                self.hidden_size % 8 == 0
                and (self.hc_count * self.hidden_size) % 2048 == 0
            )
            vecs = self.hc_count * self.hidden_size // 8
            self._split_combine_ok = (
                self._jit_combine_ok
                and vecs % (8 * 160) == 0
                and (self.hidden_size // 8) % (vecs // 8) == 0
            )

        # Norm-weight folding (TileKernels-style): the RMSNorm weight and the
        # per-branch rsqrt factors are folded into the down/inject projections
        # and the gate-apply, so the normed residual is never materialized.
        self._use_mix = use_mix
        self._use_combine = use_combine
        self._norm_fold_ok = (
            use_mix
            and config.hc_per_branch_norm
            and hc_norm_fold_enabled()
            and torch.cuda.is_available()
        )
        self._wn = None
        self._w_down_fold = None
        self._w_inject_fold = None

        def _mix_compute(
            hyper_input_normed: torch.Tensor,
            input_mix_weight_down: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            input_mix_weight = F.silu(
                F.linear(hyper_input_normed, input_mix_weight_down) / hc
            )
            input_mix_weight = F.linear(input_mix_weight, input_mix_weight_up)
            input_mix_weight = torch.sigmoid(input_mix_weight)
            input_mix_weight = input_mix_weight.unflatten(-1, (hc, hs))
            output = (
                input_mix_weight * hyper_input_normed.unflatten(-1, (hc, hs))
            ).mean(dim=-2)
            return output

        def _combine_compute(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            normed_residual: torch.Tensor,
            block_inject_weight: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            block_inject_weight_out = 2 * torch.sigmoid(
                F.linear(normed_residual, block_inject_weight) / hc
            )
            injection = block_output.unsqueeze(-2) * block_inject_weight_out.unsqueeze(
                -1
            )
            return (R + injection).flatten(-2)

        def _mix_folded(
            hyper_input: torch.Tensor,
            w_down_fold: torch.Tensor,
            input_mix_weight_up: torch.Tensor,
            wn: torch.Tensor,
            hc: int,
            hs: int,
            eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Same math as _mix_compute with norm(x) = x * wn * g expanded:
            # per-branch partial down-projections, g applied after the GEMM.
            x3 = hyper_input.unflatten(-1, (hc, hs))
            xf = x3.float()
            g = torch.rsqrt(xf.square().mean(-1) + eps)
            partials = torch.matmul(
                x3.transpose(0, 1),
                w_down_fold.view(w_down_fold.shape[0], hc, hs).permute(1, 2, 0),
            )
            down = torch.einsum("bnl,nb->nl", partials.float(), g) / hc
            h = F.silu(down).to(hyper_input.dtype)
            input_mix_weight = F.linear(h, input_mix_weight_up).unflatten(-1, (hc, hs))
            gate = torch.sigmoid(input_mix_weight)
            output = (gate * (xf * wn.unflatten(-1, (hc, hs)) * g.unsqueeze(-1))).sum(
                dim=-2
            ) / hc
            return output.to(hyper_input.dtype), g

        def _combine_folded(
            block_output: torch.Tensor,
            residual: torch.Tensor,
            g: torch.Tensor,
            w_inject_fold: torch.Tensor,
            hc: int,
            hs: int,
        ) -> torch.Tensor:
            R = residual.unflatten(-1, (hc, hs))
            partials = torch.matmul(
                R.transpose(0, 1), w_inject_fold.view(hc, hc, hs).permute(1, 2, 0)
            )
            gate_in = torch.einsum("bnc,nb->nc", partials.float(), g)
            a = 2 * torch.sigmoid(gate_in / hc)
            return (R + block_output.unsqueeze(-2) * a.unsqueeze(-1)).flatten(-2)

        self._mix_compute = torch.compile(_mix_compute)
        self._combine_compute = torch.compile(_combine_compute)
        self._mix_folded = torch.compile(_mix_folded)
        self._combine_folded = torch.compile(_combine_folded)

    def _ensure_fold_weights(self) -> None:
        # First eager call (weights are loaded before any forward), so folding
        # once here is CUDA-graph safe.
        if self._wn is not None:
            return
        wn = (1.0 + self.hc_norm.weight.float()).contiguous()
        self._wn = wn
        if self._use_mix:
            w_down = self.input_mix_weight_down.weight
            self._w_down_fold = (w_down.float() * wn).to(w_down.dtype).contiguous()
        if self._use_combine:
            w_inject = self.block_inject_weight.weight
            self._w_inject_fold = (
                (w_inject.float() * wn).to(w_inject.dtype).contiguous()
            )

    def mix(self, hyper_input: torch.Tensor):
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        if hyper_input.shape[0] == 0:
            mixed_input = hyper_input.new_empty(
                (*hyper_input.shape[:-1], self.hidden_size), dtype=self.params_dtype
            )
            return mixed_input, (hyper_input, hyper_input, None)

        # The CuTe split-K pair wants the normed residual and only exists on
        # sm_100 at decode sizes; there, keep the unfused norm.
        fold = (
            self._norm_fold_ok
            and hyper_input.is_cuda
            and hyper_input.dtype in (torch.bfloat16, torch.float16)
            and not (self._jit_mix_ok and hyper_input.shape[0] <= 24)
        )
        if fold:
            self._ensure_fold_weights()
            if fused_hc_mix_fold_supported(
                hyper_input, self._w_down_fold, self.input_mix_weight_up.weight
            ):
                mixed_input, norm_g = fused_hc_mix_fold(
                    hyper_input,
                    self._w_down_fold,
                    self.input_mix_weight_up.weight,
                    self._wn,
                    self.hc_count,
                    self.hidden_size,
                    self.config.rms_norm_eps,
                )
            else:
                mixed_input, norm_g = self._mix_folded(
                    hyper_input,
                    self._w_down_fold,
                    self.input_mix_weight_up.weight,
                    self._wn,
                    self.hc_count,
                    self.hidden_size,
                    self.config.rms_norm_eps,
                )
            return mixed_input.to(self.params_dtype), (hyper_input, None, norm_g)

        if self.config.hc_per_branch_norm:
            hyper_input_normed = self.hc_norm(hyper_input)
        else:
            hyper_input_normed = self.hc_norm(
                hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
            ).flatten(-2)
        if (
            self._jit_mix_ok
            and hyper_input_normed.is_cuda
            and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
            and hyper_input_normed.shape[0] <= 24
        ):
            from sglang.kernels.ops.elementwise.hc_mix import (
                hc_mix,
                permute_pad_up_weight,
            )

            if self._mix_up_weight_padded is None:
                self._mix_up_weight_padded = permute_pad_up_weight(
                    self.input_mix_weight_up.weight, self.hc_count
                )
            mixed_input = hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight.data,
                self._mix_up_weight_padded,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        elif fused_hc_mix_supported(
            hyper_input_normed,
            self.input_mix_weight_down.weight,
            self.input_mix_weight_up.weight,
        ):
            mixed_input = fused_hc_mix(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        else:
            mixed_input = self._mix_compute(
                hyper_input_normed,
                self.input_mix_weight_down.weight,
                self.input_mix_weight_up.weight,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)
        return mixed_input, (hyper_input, hyper_input_normed, None)

    def combine(self, block_output: torch.Tensor, residuals) -> torch.Tensor:
        hyper_input, hyper_input_normed, norm_g = residuals
        assert hyper_input.shape[-1] == self.hc_count * self.hidden_size
        assert block_output.shape[-1] == self.hidden_size
        if block_output.shape[0] == 0:
            return hyper_input.to(self.params_dtype)

        if norm_g is not None:
            self._ensure_fold_weights()
            if (
                self._jit_combine_ok
                and block_output.is_cuda
                and block_output.dtype in (torch.bfloat16, torch.float16)
                and hyper_input.dtype == block_output.dtype
                and self._w_inject_fold.dtype == block_output.dtype
            ):
                from sglang.kernels.ops.elementwise.hc_combine import (
                    hc_combine_fold,
                    hc_combine_split_fold,
                )

                if self._split_combine_ok and block_output.shape[0] <= 32:
                    return hc_combine_split_fold(
                        block_output,
                        hyper_input,
                        norm_g,
                        self._w_inject_fold,
                        self.hc_count,
                        self.hidden_size,
                    )
                return hc_combine_fold(
                    block_output,
                    hyper_input,
                    norm_g,
                    self._w_inject_fold,
                    self.hc_count,
                    self.hidden_size,
                )
            return self._combine_folded(
                block_output,
                hyper_input,
                norm_g,
                self._w_inject_fold,
                self.hc_count,
                self.hidden_size,
            ).to(self.params_dtype)

        if (
            self._jit_combine_ok
            and block_output.is_cuda
            and block_output.dtype in (torch.bfloat16, torch.float16)
            and hyper_input.dtype == block_output.dtype
            and hyper_input_normed.dtype == block_output.dtype
            and self.block_inject_weight.weight.dtype == block_output.dtype
        ):
            if self._split_combine_ok and block_output.shape[0] <= 32:
                from sglang.kernels.ops.elementwise.hc_combine import (
                    hc_combine_split,
                )

                return hc_combine_split(
                    block_output,
                    hyper_input,
                    hyper_input_normed,
                    self.block_inject_weight.weight.data,
                    self.hc_count,
                    self.hidden_size,
                )
            from sglang.kernels.ops.elementwise.hc_combine import hc_combine

            return hc_combine(
                block_output,
                hyper_input,
                hyper_input_normed,
                self.block_inject_weight.weight,
                self.hc_count,
                self.hidden_size,
            )

        updated_residuals = self._combine_compute(
            block_output,
            hyper_input,
            hyper_input_normed,
            self.block_inject_weight.weight,
            self.hc_count,
            self.hidden_size,
        ).to(self.params_dtype)
        return updated_residuals


HYPERCONNECTION_CLASS_DICT = {
    "hyperconnection_average": HyperConnectionBase,
    "gated_residual_simple": GatedResidual,
}
