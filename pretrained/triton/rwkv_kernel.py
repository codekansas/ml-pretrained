# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Defines a Triton kernel for the RWKV forward and backward passes."""

import torch
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
    stride_b,
    stride_c,
    tsz,
    y_ptr,
    num_ptr,
    den_ptr,
):
    # Parallelize over the batch and channel dimensions.
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)

    # Start pointers for the current time row.
    k_start_ptr = k_ptr + b_idx * stride_b + c_idx * stride_c
    v_start_ptr = v_ptr + b_idx * stride_b + c_idx * stride_c
    y_start_ptr = y_ptr + b_idx * stride_b + c_idx * stride_c
    num_start_ptr = num_ptr + b_idx * stride_b + c_idx * stride_c
    den_start_ptr = den_ptr + b_idx * stride_b + c_idx * stride_c

    # Loads mixing parameters.
    num = tl.load(last_num_ptr + b_idx * stride_b + c_idx * stride_c)
    den = tl.load(last_den_ptr + b_idx * stride_b + c_idx * stride_c)
    w = -tl.exp(tl.load(w_ptr + c_idx))
    u = tl.load(u_ptr + c_idx)

    for t in range(tsz):
        # Load the current key and value.
        k = tl.load(k_start_ptr + t)
        v = tl.load(v_start_ptr + t)

        # Compute the new output.
        c = tl.exp(u + k)
        y = (num + (c * v)) / (den + c)

        # Compute the new numerator and denominator.
        num = tl.exp(w) * num + tl.exp(w * v)
        den = tl.exp(w) * den + tl.exp(w)

        # Store the results.
        tl.store(y_start_ptr + t, y)
        tl.store(num_start_ptr + t, num)
        tl.store(den_start_ptr + t, den)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    k = k.transpose(1, 2).contiguous()  # (B, T, C) -> (B, C, T)
    v = v.transpose(1, 2).contiguous()  # (B, T, C) -> (B, C, T)

    # New tensors to output.
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
        tsz,
        y,
        num,
        den,
    )

    y = y.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)
    num = num.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)
    den = den.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)

    # # REFERENCE IMPLEMENTATION - DELETE LATER
    # k, v = k.transpose(1, 2), v.transpose(1, 2)
    # t = torch.arange(tsz + 1, device=w.device)[None, :, None]  # (1, T, 1)
    # wt = t[:, None, :-1, :] - t[:, :-1, None, :]  # (1, T, T, 1)
    # w = -torch.exp(w)  # (D)
    # tw = w * t[:, 1:]  # (1, T, 1)
    # twt = w * wt  # (1, T, T, 1)
    # ktw = twt + k[:, :, None]  # (B, T, T, D)
    # device, dtype = tw.device, tw.dtype
    # mask = torch.empty(tsz, tsz, device=device, dtype=dtype)
    # mask.fill_(float("-inf"))
    # # mask.triu_(1)
    # mask.tril_(-1)
    # ktw = ktw + mask[None, :tsz, :tsz, None]
    # etw, ektw = torch.exp(tw), torch.exp(ktw)  # (1, T, 1), (B, T, T, D)
    # ref_num = etw * last_num + (ektw * v[:, :, None]).sum(1)  # (B, T, D)
    # ref_den = etw * last_den + ektw.sum(1)  # (B, T, D)

    # last_num = torch.cat((last_num, ref_num[..., :-1, :]), dim=-2)  # (B, T, D)
    # last_den = torch.cat((last_den, ref_den[..., :-1, :]), dim=-2)  # (B, T, D)

    # out = (last_num + torch.exp(u + k) * v) / (last_den + torch.exp(u + k))  # (B, T, D)

    # breakpoint()

    return y, num, den


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
