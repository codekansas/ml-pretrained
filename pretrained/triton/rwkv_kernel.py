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
    BLOCK_T: tl.constexpr,
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
    w_start_ptr = w_ptr + c_idx
    u_start_ptr = u_ptr + c_idx
    last_num_start_ptr = last_num_ptr + b_idx * stride_b + c_idx * stride_c
    last_den_start_ptr = last_den_ptr + b_idx * stride_b + c_idx * stride_c

    # Loads a row of times.
    t = tl.arange(0, BLOCK_T)
    t_mask = t < tsz
    t_mask_2 = (t + 1) < tsz
    k = tl.load(k_start_ptr + t, mask=t_mask)  # T
    v = tl.load(v_start_ptr + t, mask=t_mask)  # T
    w = -tl.exp(tl.load(w_start_ptr))  # 1
    u = tl.load(u_start_ptr)  # 1
    last_num = tl.load(last_num_start_ptr)  # 1
    last_den = tl.load(last_den_start_ptr)  # 1

    # Runs the forward pass computation.
    tw = w * (t + 1)  # T
    tt = t[None, :] - t[:, None]
    twt = w * tt  # (T, T)
    ktw = twt + k[:, None]  # (T, T)
    ktw = tl.where(tt < 0, float("-inf"), ktw)
    etw, ektw = tl.exp(tw), tl.exp(ktw)  # T, (T, T)
    num = etw * last_num + tl.sum(ektw * v[:, None], 0)  # T
    den = etw * last_den + tl.sum(ektw, 0)  # T

    # last_out = (last_num + tl.exp(u + k) * v) / (last_den + tl.exp(u + k))  # 1
    out = (num + tl.exp(u + k) * v) / (den + tl.exp(u + k))  # T

    tl.store(num_start_ptr + t, num, mask=t_mask)
    tl.store(den_start_ptr + t, den, mask=t_mask)
    # tl.store(y_start_ptr, last_out)
    tl.store(y_start_ptr + 1 + t, out, mask=t_mask_2)


def _forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    last_num: Tensor,
    last_den: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    BLOCK_T = max(triton.next_power_of_2(tsz), 512)

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
        BLOCK_T=BLOCK_T,
    )

    y = y.transpose(1, 2).contiguous()
    num = num.transpose(1, 2).contiguous()
    den = den.transpose(1, 2).contiguous()

    # REFERENCE IMPLEMENTATION - DELETE LATER
    k, v = k.transpose(1, 2), v.transpose(1, 2)
    t = torch.arange(tsz + 1, device=w.device)[None, :, None]  # (1, T, 1)
    wt = t[:, None, :-1, :] - t[:, :-1, None, :]  # (1, T, T, 1)
    w = -torch.exp(w)  # (D)
    tw = w * t[:, 1:]  # (1, T, 1)
    twt = w * wt  # (1, T, T, 1)
    ktw = twt + k[:, :, None]  # (B, T, T, D)
    device, dtype = tw.device, tw.dtype
    mask = torch.empty(tsz, tsz, device=device, dtype=dtype)
    mask.fill_(float("-inf"))
    # mask.triu_(1)
    mask.tril_(-1)
    ktw = ktw + mask[None, :tsz, :tsz, None]
    etw, ektw = torch.exp(tw), torch.exp(ktw)  # (1, T, 1), (B, T, T, D)
    ref_num = etw * last_num + (ektw * v[:, :, None]).sum(1)  # (B, T, D)
    ref_den = etw * last_den + ektw.sum(1)  # (B, T, D)

    last_num = torch.cat((last_num, ref_num[..., :-1, :]), dim=-2)  # (B, T, D)
    last_den = torch.cat((last_den, ref_den[..., :-1, :]), dim=-2)  # (B, T, D)

    out = (last_num + torch.exp(u + k) * v) / (last_den + torch.exp(u + k))  # (B, T, D)

    breakpoint()

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
