"""Fused Add-residual + RMSNorm + 1x128 fp8 block-quant producer (SM90, bf16).

JIT port of the standalone kernel validated for the deepseek_v2-family fp8 path. Replaces the
stock 2-kernel producer (fused_add_rmsnorm then per_token_group_quant_fp8) with a single kernel
that emits, in one pass: fp8 e4m3 activation, a column-major TMA-aligned block-scale, the pre-norm
residual (hidden+residual) and the bf16 normed output. The (fp8, scale) pair is the exact MN-major
layout the DeepGEMM / figemm pre-quant path consumes, so downstream fp8 linears skip re-quant.

Dispatch is by hidden size H only (memory-bound -> more threads / smaller VPT wins at every M):
VPT=8 for H<=8192, VPT=16 for 8192<H<=16384. Requires H % (32*VPT) == 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_FP8 = torch.float8_e4m3fn


def is_supported_fused_add_rmsnorm_quant_hidden_size(hidden_size: int) -> bool:
    """VPT8 covers H%256==0 & H<=8192; VPT16 covers H%512==0 & 8192<H<=16384."""
    return (hidden_size % 256 == 0 and 0 < hidden_size <= 8192) or (
        hidden_size % 512 == 0 and 8192 < hidden_size <= 16384
    )


def _vpt_for_hidden_size(hidden_size: int) -> int:
    return 8 if hidden_size <= 8192 else 16


@cache_once
def _jit_module(vpt: int) -> "Module":
    args = make_cpp_args(vpt)
    return load_jit(
        "fused_add_rmsnorm_quant",
        *args,
        cuda_files=["elementwise/fused_add_rmsnorm_quant.cuh"],
        cuda_wrappers=[
            ("fused_add_rmsnorm_quant", f"FusedAddRMSNormQuantKernel<{args}>::run")
        ],
        # No --use_fast_math: the fp8 quant must match the figemm/DeepGEMM precise-math path
        # bit-for-bit, so downstream linears see the same activation they would have re-quantized.
    )


@register_custom_op(
    op_name="fused_add_rmsnorm_quant",
    mutates_args=["x_fp8", "x_scale", "residual_out", "normed_out"],
)
def _fused_add_rmsnorm_quant_custom_op(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    residual_out: torch.Tensor,
    normed_out: torch.Tensor,
    vpt: int,
    eps: float,
) -> None:
    """Opaque custom-op boundary around the tvm-ffi kernel (keeps torch.compile / piecewise
    CUDA graph from tracing into Function.__call__)."""
    _jit_module(vpt).fused_add_rmsnorm_quant(
        hidden, residual, weight, x_fp8, x_scale, residual_out, normed_out, eps
    )


@debug_kernel_api
def fused_add_rmsnorm_quant(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    residual_out: Optional[torch.Tensor] = None,
    normed_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden[M,H] bf16, residual[M,H] bf16, weight[H] bf16 (all contiguous) ->
    (x_fp8[M,H] e4m3, x_scale[M,H/128] fp32 column-major TMA-aligned view, residual_out[M,H] bf16,
     normed_out[M,H] bf16). Inputs are untouched (NOT in-place). Caller must ensure H is supported
    (is_supported_fused_add_rmsnorm_quant_hidden_size).

    residual_out / normed_out may be passed pre-allocated so the caller controls their memory pool:
    these two bf16 outputs feed the post-attention TP all-reduce, so a symm-mem-registered caller
    (LayerCommunicator) allocates them inside use_symmetric_memory(get_tp_group()) to keep the
    NCCL multicast all-reduce (Fix A). x_fp8 / x_scale feed column-parallel GEMMs (no all-reduce)
    and always come from the default pool."""
    M, H = hidden.shape
    vpt = _vpt_for_hidden_size(H)
    m_pad = (M + 3) & ~3
    x_fp8 = torch.empty((M, H), dtype=_FP8, device=hidden.device)
    x_scale = torch.empty((H // 128, m_pad), dtype=torch.float32, device=hidden.device)
    if residual_out is None:
        residual_out = torch.empty((M, H), dtype=torch.bfloat16, device=hidden.device)
    if normed_out is None:
        normed_out = torch.empty((M, H), dtype=torch.bfloat16, device=hidden.device)
    _fused_add_rmsnorm_quant_custom_op(
        hidden, residual, weight, x_fp8, x_scale, residual_out, normed_out, int(vpt), float(eps)
    )
    # x_scale buffer is (H/128, Mpad) MN-major; expose the (M, H/128) column-major view the
    # w8a8 block-fp8 linears expect (stride == (1, Mpad)).
    return x_fp8, x_scale[:, :M].transpose(0, 1), residual_out, normed_out
