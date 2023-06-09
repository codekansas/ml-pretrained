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
    k_stride_c,
    v_stride_b,
    v_stride_c,
    tsz,
    chans,
    y_ptr,
    num_ptr,
    den_ptr,
    BLOCK_T: tl.constexpr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    # Start pointers for the current time row.
    k_start_ptr = k_ptr + b_idx * k_stride_b + c_idx * k_stride_c
    v_start_ptr = v_ptr + b_idx * v_stride_b + c_idx * v_stride_c

    # Loads a row of times.
    t = tl.arange(0, tsz)
    k = tl.load(k_start_ptr, t)
    v = tl.load(v_start_ptr, t)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    # (B, T, C) -> (B, C, T)
    k, v = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    BLOCK_T = triton.next_power_of_2(tsz)
    y = k.new_empty(k.shape)
    num = k.new_empty(k.shape)
    den = k.new_empty(k.shape)
    _forward_kernel[(bsz, chans)](
        w,
        u,
        k,
        v,
        last_num,
        last_den,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        tsz,
        y,
        num,
        den,
        BLOCK_T=BLOCK_T,
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
