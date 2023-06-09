# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines a Triton kernel for the RWKV forward and backward passes."""

import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx


@triton.jit
def _forward_kernel(
    w_ptr,
    u_ptr,
    k_ptr,
    v_ptr,
    last_num_ptr,
    last_den_ptr,
    k_stride_b,
    v_stride_b,
    tsz,
    y_ptr,
    num_ptr,
    den_ptr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Parallelize over the batch dimension.
    b_idx = tl.program_id(0)

    # Start pointers for the current batch.
    k_start_ptr = k_ptr + b_idx * k_stride_b
    v_start_ptr = v_ptr + b_idx * v_stride_b

    # Time 
    t = tl.arange(0, tsz + 1)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape
    BLOCK_T = triton.next_power_of_2(tsz)
    BLOCK_C = triton.next_power_of_2(chans)
    y = k.new_empty(k.shape)
    num = k.new_empty(k.shape)
    den = k.new_empty(k.shape)
    _forward_kernel[(bsz,)](
        w,
        u,
        k,
        v,
        last_num,
        last_den,
        k.stride(0),
        v.stride(0),
        tsz,
        y,
        num,
        den,
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
    )
    return y


def _backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
    gy: Tensor,
    gn: Tensor,
    gd: Tensor,
) -> tuple[Tensor, ...]:
    raise NotImplementedError


class _WKV(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        last_num: Tensor,
        last_den: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        breakpoint()
        y, num, den = _forward(w, u, k, v, last_num, last_den)
        ctx.save_for_backward(w, u, k, v, last_num, last_den)
        return y, num, den

    @staticmethod
    def backward(ctx: FunctionCtx, gy: Tensor, gn: Tensor, gd: Tensor) -> tuple[Tensor, ...]:
        w, u, k, v, last_num, last_den = ctx.saved_tensors
        gw, gu, gk, gv, gn, gd = _backward(w, u, k, v, last_num, last_den, gy, gn, gd)
        return gw, gu, gk, gv, gn, gd


def triton_wkv(w: Tensor, u: Tensor, k: Tensor, v: Tensor, last_num: Tensor, last_den: Tensor) -> Tensor:
    return _WKV.apply(w, u, k, v, last_num, last_den)
